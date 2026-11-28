// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// MainPageController: personalize training. Split out of MainPage.cpp (#549); shared
// includes and helpers are in mainpage-internal.h.

#include "mainpage-internal.h"
#include "xllama/model_write.h"

#ifdef XLLAMA_UWP

namespace xllama {

// ---------------------------------------------------------------------------
// Phase 11 personalize (#116) — train on samples.jsonl, publish merged GGUF
// ---------------------------------------------------------------------------

std::string MainPageController::ResolvePersonalizeBase() const {
    // Prefer the operator-uploaded Lane B base (same path as device-train harness).
    const std::wstring base_w = local_wpath(L"training\\base-f16.gguf");
    if (GetFileAttributesW(base_w.c_str()) != INVALID_FILE_ATTRIBUTES)
        return ::xllama::kPersonalizeDefaultBase;

    // Else: if the current catalogue model is GGUF and provisioned, use it.
    auto manifest = ::xllama::LoadModelManifest();
    if (const auto* e = ::xllama::FindManifestEntry(manifest, m_model_filename)) {
        if (e->kind == L"gguf" && ::xllama::IsModelProvisioned(m_model_filename)) {
            const std::string id = ::xllama::wstring_to_utf8(m_model_filename);
            if (::xllama::guess_last_block_from_model_id(id) >= 0)
                return std::string("models/") + id;
        }
    }
    // Any provisioned smollm2 GGUF in the catalogue (last-block known).
    for (const auto& e : manifest) {
        if (e.kind != L"gguf")
            continue;
        const std::string id = ::xllama::wstring_to_utf8(e.name);
        if (::xllama::guess_last_block_from_model_id(id) < 0)
            continue;
        if (::xllama::IsModelProvisioned(e.name))
            return std::string("models/") + id;
    }
    return {};
}

bool MainPageController::PublishPersonalizedModel(const std::string& merged_gguf_path,
                                                  std::string* err) {
    auto permit = ModelDownloader::WriterGate().try_acquire();
    if (!permit.owns_lock()) {
        if (err)
            *err = "Another model write is in progress";
        return false;
    }
    const auto root = winrt::Windows::Storage::ApplicationData::Current().LocalFolder().Path();
    if (!publish_personalized_model(std::filesystem::path(root.c_str()),
                                    std::filesystem::u8path(merged_gguf_path), err))
        return false;
    InvalidateManifestCache(); // catalogue changed — StartInference reads the cache
    return true;
}

void MainPageController::StartPersonalizeTrain() {
    if (m_is_running.load() || m_train_running.load() || m_diffuse_running.load()) {
        SetStatus(L"Busy — wait for the current job to finish", StatusKind::Error);
        return;
    }
    #ifndef XLLAMA_DEVICE_TRAIN
    SetStatus(L"This build has no on-device training engine", StatusKind::Error);
    return;
    #else
    const std::string samples = ::xllama::wstring_to_utf8(local_wpath(L"training\\samples.jsonl"));
    if (::xllama::count_usable_preference_samples(samples) < 1) {
        SetStatus(L"No usable preference samples (Like/Correct first)", StatusKind::Error);
        return;
    }
    const std::string base = ResolvePersonalizeBase();
    if (base.empty()) {
        SetStatus(L"No base GGUF — place training\\base-f16.gguf first", StatusKind::Error);
        return;
    }

    ::xllama::PersonalizeSpec spec;
    spec.base_model = base;
    spec.dataset_path = ::xllama::kPersonalizeDefaultDataset;
    spec.out_dir = ::xllama::kPersonalizeDefaultOutDir;
    ::xllama::TrainingJob job;
    std::string err;
    if (!::xllama::build_personalize_job(spec, job, &err)) {
        SetStatus(L"Train setup failed: " + ::xllama::utf8_to_wstring(err), StatusKind::Error);
        return;
    }

    auto compute_permit = ::xllama::try_acquire_compute_job();
    if (!compute_permit) {
        SetStatus(L"Busy — another compute job is active", StatusKind::Error);
        return;
    }

    // Persist job.json for harness parity / debugging.
    CreateDirectoryW(local_wpath(L"training").c_str(), nullptr);
    ::xllama::write_local_text("training\\job.json", ::xllama::format_personalize_job_json(job));
    _wremove(local_wpath(L"training\\result.done").c_str());
    ::xllama::write_local_text("training\\progress.json", ::xllama::format_train_progress_json(
                                                              "prepare", 0, job.epochs, 0, 0, 0));

    // Free chat RSS before train (Phase 10 peak gate ~1.2 GB).
    {
        auto& hub = ::xllama::session_hub();
        std::lock_guard<std::mutex> hub_lk(hub.mtx);
        hub.reset_locked();
    }
    m_kv_valid = false;

    m_train_abort.store(false);
    m_train_running.store(true);
    SetRunning(true);
    m_loadingBar.IsIndeterminate(true);
    SetStatus(L"Personalizing model (on-device train)...", StatusKind::Working);

    auto self = shared_from_this();
    auto dispatcher = m_root.Dispatcher();
    std::thread([self, dispatcher, job, compute_permit]() {
        (void)compute_permit;
        try {
            winrt::init_apartment(); // MTA
            ::xllama::DeviceTrainCallbacks cb;
            cb.abort_flag = &self->m_train_abort;
            cb.on_progress = [self, dispatcher](const ::xllama::DeviceTrainProgress& p) {
                const char* st = ::xllama::training_stage_name(p.stage);
                wchar_t line[192];
                swprintf(line, 192, L"Training %hs — epoch %d/%d loss=%.4f", st ? st : "train",
                         p.epoch, p.epochs, p.loss);
                dispatcher.RunAsync(CoreDispatcherPriority::Normal,
                                    [self, msg = std::wstring(line)]() {
                                        self->SetStatus(msg, StatusKind::Working);
                                    });
            };
            const ::xllama::TrainingResult r = ::xllama::bridge::run_train_job_localized(job, cb);

            // result.done for API / autopilot pollers
            ::xllama::write_local_text("training\\result.done", r.success ? "ok\n" : "fail\n");

            if (r.success) {
                std::string pub_err;
                const bool published = self->PublishPersonalizedModel(r.merged_gguf_path, &pub_err);
                if (published) {
                    auto& hub = ::xllama::session_hub();
                    std::lock_guard<std::mutex> hub_lk(hub.mtx);
                    hub.reset_locked();
                }
                dispatcher.RunAsync(CoreDispatcherPriority::Normal, [self, published, pub_err,
                                                                     path = r.merged_gguf_path]() {
                    self->m_train_running.store(false);
                    self->SetRunning(false);
                    self->m_loadingBar.IsIndeterminate(false);
                    if (!published) {
                        self->SetStatus(L"Train OK but publish failed: " +
                                            ::xllama::utf8_to_wstring(pub_err),
                                        StatusKind::Error);
                        return;
                    }
                    self->m_model_filename =
                        ::xllama::utf8_to_wstring(::xllama::kPersonalizedModelId);
                    self->m_modelText.Text(self->m_model_filename);
                    self->m_kv_valid = false;
                    self->m_active_model.clear();
                    self->SaveSettings();
                    self->m_model_ready.store(false);
                    self->m_runButton.IsEnabled(false);
                    self->SetStatus(L"Personalized model ready", StatusKind::Success);
                    self->EnsureModelNamedAsync(self->m_model_filename, true);
                    (void)path;
                });
            } else {
                dispatcher.RunAsync(CoreDispatcherPriority::Normal, [self, err = r.error_msg]() {
                    self->m_train_running.store(false);
                    self->SetRunning(false);
                    self->m_loadingBar.IsIndeterminate(false);
                    self->SetStatus(L"Training failed: " + ::xllama::utf8_to_wstring(err),
                                    StatusKind::Error);
                });
            }
        } catch (const std::exception& ex) {
            dispatcher.RunAsync(
                CoreDispatcherPriority::Normal, [self, msg = std::string(ex.what())]() {
                    self->m_train_running.store(false);
                    self->SetRunning(false);
                    self->m_loadingBar.IsIndeterminate(false);
                    self->SetStatus(L"Training exception: " + ::xllama::utf8_to_wstring(msg),
                                    StatusKind::Error);
                });
        } catch (...) {
            dispatcher.RunAsync(CoreDispatcherPriority::Normal, [self]() {
                self->m_train_running.store(false);
                self->SetRunning(false);
                self->m_loadingBar.IsIndeterminate(false);
                self->SetStatus(L"Training failed (unknown exception)", StatusKind::Error);
            });
        }
    }).detach();
    #endif
}

} // namespace xllama

#endif // XLLAMA_UWP
