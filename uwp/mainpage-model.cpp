// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// MainPageController: model selection, download and session. Split out of MainPage.cpp (#549);
// shared includes and helpers are in mainpage-internal.h.

#include "mainpage-internal.h"

#ifdef XLLAMA_UWP

namespace xllama {

const std::vector<::xllama::ManifestEntry>& MainPageController::CachedManifest() const {
    if (!m_manifest_cached) {
        m_manifest_cache = ::xllama::LoadModelManifest();
        m_manifest_cached = true;
    }
    return m_manifest_cache;
}

void MainPageController::ApplyCatalogueModelKnobs(const std::wstring& model) {
    // Catalogue optional n_predict (thinking tier uses 1024 — #223). 0 / absent
    // leaves the Settings slider alone so chat/coding stay at the user's value.
    const auto& manifest = CachedManifest();
    const auto* e = ::xllama::FindManifestEntry(manifest, model);
    if (!e || e->n_predict <= 0)
        return;
    m_n_predict = std::clamp(e->n_predict, ::xllama::settings_limits::kNPredictMin,
                             ::xllama::settings_limits::kNPredictMax);
}

::xllama::ResolvedSampling MainPageController::ModelSampling(const std::wstring& model) const {
    const auto* e = ::xllama::FindManifestEntry(CachedManifest(), model);
    return ::xllama::resolve_sampling(e ? e->sampling : ::xllama::SamplingOverrides{}, m_sampling);
}

// ---------------------------------------------------------------------------
// Settings and autopilot select through the same resident/provisioning policy.
bool MainPageController::SelectModel(const std::wstring& name) {
    if (name == m_model_filename)
        return false;
    m_model_filename = name;
    m_modelText.Text(name);
    m_active_model.clear();
    ApplyCatalogueModelKnobs(name);
    {
        auto& hub = ::xllama::session_hub();
        std::lock_guard<std::mutex> lock(hub.mtx);
        hub.reset_locked();
    }
    m_kv_valid = false;
    m_model_ready.store(false);
    m_runButton.IsEnabled(false);
    SetStatus(L"Loading model...", StatusKind::Working);
    EnsureModelNamedAsync(name, true);
    return true;
}

// ---------------------------------------------------------------------------
// Model provisioning — catalogue / USB / bundled. EnsureModelAsync loads the
// selected chat model; EnsureGpuModelIfNeeded queues gpu_model when routing≠0.
// ---------------------------------------------------------------------------

void MainPageController::EnsureGpuModelIfNeeded() {
    // #91/#95: if the configured gpu_model is not a parity-validated DML text
    // asset, routing can never pick it — don't background-download 725 MB that
    // cannot be used.
    if (!::xllama::dml_text_model_ok(m_gpu_model))
        return;
    if (m_routing == 0)
        return;
    const std::wstring gpu = ::xllama::utf8_to_wstring(m_gpu_model);
    // Expected-aware so a stale-quant GPU model is upgraded like the chat model
    // (EnsureModelNamedAsync would otherwise be short-circuited here). Empty
    // expected → loose fallback. Names alone are not enough: a LocalState
    // GPU dir with every filename but no `.complete` still needs a write.
    std::vector<std::wstring> gpu_expected;
    {
        auto manifest = ::xllama::LoadModelManifest();
        if (const auto* e = ::xllama::FindManifestEntry(manifest, gpu))
            for (const auto& f : e->files)
                gpu_expected.push_back(f.filename);
    }
    auto local_folder = winrt::Windows::Storage::ApplicationData::Current().LocalFolder();
    const std::wstring local_gpu_dir =
        std::wstring(local_folder.Path().c_str()) + L"\\models\\" + gpu;
    const bool names_present = ::xllama::IsModelProvisioned(gpu, gpu_expected);
    const bool marker = ::xllama::ModelDownloader::IsComplete(local_gpu_dir);
    if (!::xllama::gpu_needs_provision(names_present, marker))
        return;
    log_output(
        ("[xllama] EnsureModel: gpu_model '" + m_gpu_model + "' missing — background provision\n")
            .c_str());
    EnsureModelNamedAsync(gpu, false);
}

fire_and_forget MainPageController::EnsureModelAsync() {
    auto self = shared_from_this();
    std::wstring model_name =
        self->m_model_filename.empty() ? DefaultChatModelId() : self->m_model_filename;
    self->EnsureModelNamedAsync(model_name, true);
    // Without a co_ statement this is an ordinary function that runs off its
    // end with a non-void return type, which is undefined behaviour.
    co_return;
}

fire_and_forget MainPageController::EnsureModelNamedAsync(std::wstring model_name,
                                                          bool set_app_ready) {
    auto self = shared_from_this();
    auto dispatcher = self->m_root.Dispatcher();

    std::optional<std::wstring> err;
    try {
        log_output(("[xllama] EnsureModel: begin '" + ::xllama::wstring_to_utf8(model_name) + "'\n")
                       .c_str());
        co_await resume_background();

        // Load the catalogue up-front so every provisioning check can compare against
        // the entry's CURRENT expected files (auto-upgrade a stale quant) instead of
        // accepting any gguf. Empty expected → loose fallback (USB/WDP/no entry).
        auto manifest = ::xllama::LoadModelManifest();
        const ::xllama::ManifestEntry* entry = ::xllama::FindManifestEntry(manifest, model_name);
        std::vector<std::wstring> expected_files;
        if (entry)
            for (const auto& f : entry->files)
                expected_files.push_back(f.filename);

        // Check 1: LocalState model complete?
        auto local_folder = winrt::Windows::Storage::ApplicationData::Current().LocalFolder();
        std::wstring local_models_root = std::wstring(local_folder.Path().c_str()) + L"\\models";
        std::wstring local_model_dir = local_models_root + L"\\" + model_name;

        const bool marker = ModelDownloader::IsComplete(local_model_dir);
        const bool interrupted_usb = usb_import_pending(local_model_dir);
        const bool skip_early =
            interrupted_usb || skip_provision_early_exits(set_app_ready, marker);

        if (!skip_early && IsModelProvisioned(model_name, expected_files)) {
            log_output(("[xllama] EnsureModel: '" + ::xllama::wstring_to_utf8(model_name) +
                        "' already provisioned\n")
                           .c_str());
            co_await resume_foreground(dispatcher);
            if (set_app_ready) {
                self->LoadModelName();
                self->SetStatus(L"Ready", StatusKind::Success);
                self->m_runButton.IsEnabled(true);
                self->m_model_ready.store(true);
                self->EnsureGpuModelIfNeeded();
                self->PreloadSessionAsync();
            }
            co_return;
        }

        // Weights already match their pins; only catalogue sidecars (LICENSE /
        // NOTICE / …) are missing. Do not block Run — mark ready, then write
        // in the background. had_marker = IsComplete on entry: sidecars only
        // when the marker was already published; otherwise pass every
        // catalogue file so weights are hashed and complete() runs only if
        // every pin is Ok (#370 B5). GPU provision with no marker skips this
        // path (skip_early) and falls through to DownloadAsync(entry->files).
        if (!skip_early && entry && !entry->files.empty()) {
            std::vector<PresentFile> present;
            std::vector<ManifestFilePin> pins;
            present.reserve(entry->files.size());
            pins.reserve(entry->files.size());
            for (const auto& f : entry->files) {
                uint64_t sz = 0;
                std::error_code ec;
                const auto p = std::filesystem::path(local_model_dir) / f.filename;
                if (std::filesystem::is_regular_file(p, ec)) {
                    const auto n = std::filesystem::file_size(p, ec);
                    if (!ec)
                        sz = static_cast<uint64_t>(n);
                }
                present.push_back({f.filename, sz, {}});
                pins.push_back(make_file_pin(f.filename, f.approx_bytes,
                                             ::xllama::wstring_to_utf8(f.sha256),
                                             ::xllama::wstring_to_utf8(f.role)));
            }
            if (only_sidecars_missing(present, pins)) {
                const bool had_marker = marker;
                log_output(("[xllama] EnsureModel: '" + ::xllama::wstring_to_utf8(model_name) +
                            "' weights present; sidecar fetch in background\n")
                               .c_str());
                co_await resume_foreground(dispatcher);
                if (set_app_ready) {
                    self->LoadModelName();
                    self->SetStatus(L"Ready", StatusKind::Success);
                    self->m_runButton.IsEnabled(true);
                    self->m_model_ready.store(true);
                    self->PreloadSessionAsync();
                }
                if (!entry->hf_base_url.empty()) {
                    const auto want = sidecar_background_write_files(pins, had_marker);
                    std::vector<ModelFile> write_files;
                    std::vector<std::wstring> publish;
                    write_files.reserve(want.size());
                    publish.reserve(entry->files.size());
                    for (const auto& f : entry->files)
                        publish.push_back(f.filename);
                    for (const auto& name : want) {
                        for (const auto& f : entry->files) {
                            if (f.filename == name) {
                                write_files.push_back(f);
                                break;
                            }
                        }
                    }
                    if (!write_files.empty()) {
                        co_await ModelDownloader::DownloadAsync(
                            entry->hf_base_url, local_model_dir, std::move(write_files), dispatcher,
                            [](uint64_t, uint64_t) {},
                            [self, set_app_ready](bool ok, std::wstring e) {
                                if (!ok)
                                    log_output(("[xllama] EnsureModel: sidecar fetch failed: " +
                                                ::xllama::wstring_to_utf8(e) + "\n")
                                                   .c_str());
                                else
                                    log_output("[xllama] EnsureModel: sidecar fetch complete\n");
                                if (set_app_ready)
                                    self->EnsureGpuModelIfNeeded();
                            },
                            std::move(publish));
                    } else if (set_app_ready) {
                        self->EnsureGpuModelIfNeeded();
                    }
                } else if (set_app_ready) {
                    self->EnsureGpuModelIfNeeded();
                }
                co_return;
            }
        }

        // Check 2: model bundled in InstalledPath (MSIX)?
        auto pkg = winrt::Windows::ApplicationModel::Package::Current();
        std::wstring installed_model_dir =
            std::wstring(pkg.InstalledPath().c_str()) + L"\\models\\" + model_name;
        {
            std::error_code ec;
            bool bundled = std::filesystem::exists(
                std::filesystem::path(installed_model_dir) / L"genai_config.json", ec);
            if (!interrupted_usb && bundled) {
                co_await resume_foreground(dispatcher);
                if (set_app_ready) {
                    self->LoadModelName();
                    self->SetStatus(L"Ready", StatusKind::Success);
                    self->m_runButton.IsEnabled(true);
                    self->m_model_ready.store(true);
                    self->EnsureGpuModelIfNeeded();
                    self->PreloadSessionAsync();
                }
                co_return;
            }
        }

    #ifndef XLLAMA_STORE_SKU
        // Check 3: USB removable storage via KnownFolders.RemovableDevices
        // (requires <uap:Capability Name="removableStorage" /> in manifest).
        // Enumerates all removable drives; looks for xllama/models/<name>/genai_config.json.
        // Store SKU: no removableStorage capability — download / LocalState only.
        co_await resume_foreground(dispatcher);
        {
            using winrt::Windows::Storage::StorageFile;
            using winrt::Windows::Storage::StorageFolder;
            StorageFolder usb_model_folder{nullptr};
            std::wstring usb_root;
            const bool expects_config =
                std::any_of(expected_files.begin(), expected_files.end(), [](const auto& file) {
                    return normalize_model_path(file) == L"genai_config.json";
                });
            try {
                auto drives = co_await winrt::Windows::Storage::KnownFolders::RemovableDevices()
                                  .GetFoldersAsync();
                for (auto const& drive : drives) {
                    // Hold the selected folder through verification and copy. Re-enumerating
                    // volumes could select a different folder with the same model name.
                    try {
                        auto app_folder = co_await drive.GetFolderAsync(L"xllama");
                        auto models = co_await app_folder.GetFolderAsync(L"models");
                        auto candidate = co_await models.GetFolderAsync(model_name);
                        auto config = co_await candidate.TryGetItemAsync(L"genai_config.json");
                        const bool found_config = config && config.try_as<StorageFile>();
                        if (found_config || (expects_config && !usb_model_folder)) {
                            usb_model_folder = candidate;
                            usb_root = drive.Path().c_str();
                        }
                        if (found_config)
                            break;
                    } catch (winrt::hresult_error const&) {
                        // This drive is inaccessible or has no folder for this model.
                    }
                }
            } catch (winrt::hresult_error const&) {
                log_output("[xllama] USB probe: RemovableDevices enumeration failed\n");
            }
            if (usb_model_folder) {
                // ORT GenAI uses Win32 I/O (blocked on USB by AppContainer).
                // Copy model files from USB to LocalState via WinRT StorageFile.CopyAsync,
                // then load from LocalState like the bundled model.
                co_await resume_foreground(dispatcher);
                self->SetStatus(L"Copying model from USB...", StatusKind::Working);
                self->m_loadingBar.IsIndeterminate(true);
                self->m_loadingBar.Visibility(winrt::Windows::UI::Xaml::Visibility::Visible);
                self->m_runButton.IsEnabled(false);

                ModelWriteOperation writer(ModelDownloader::WriterGate(), local_model_dir);
                if (!writer.owns_lock()) {
                    self->m_loadingBar.IsIndeterminate(false);
                    self->m_loadingBar.Visibility(winrt::Windows::UI::Xaml::Visibility::Collapsed);
                    self->m_runButton.IsEnabled(true);
                    self->SetStatus(ModelDownloader::kBusyError, StatusKind::Error);
                    co_return;
                }
                bool copy_ok = false;
                std::wstring copy_err;
                try {
                    using winrt::Windows::Storage::CreationCollisionOption;
                    // Inspect the source before invalidating the installed model. A stale
                    // destination file must never mask an incomplete removable source.
                    std::vector<std::pair<std::wstring, StorageFile>> copies;
                    std::vector<std::wstring> present;
                    for (auto const& file : co_await usb_model_folder.GetFilesAsync()) {
                        const std::wstring name(file.Name().c_str());
                        const auto normalized = normalize_model_path(name);
                        if (normalized == L".complete" ||
                            normalized == utf8_to_wstring(kUsbImportPending))
                            continue;
                        present.push_back(name);
                        copies.emplace_back(name, file);
                    }
                    for (const auto& name : expected_files) {
                        if (name.find_first_of(L"/\\") == std::wstring::npos)
                            continue;
                        auto relative = std::filesystem::path(name);
                        relative.make_preferred();
                        if (relative.has_root_path() ||
                            std::find(relative.begin(), relative.end(), L"..") != relative.end())
                            throw std::runtime_error("Invalid relative model filename");
                        StorageFolder parent = usb_model_folder;
                        for (const auto& dir : relative.parent_path()) {
                            auto item = co_await parent.TryGetItemAsync(dir.wstring());
                            parent = item.try_as<StorageFolder>();
                            if (!parent)
                                break;
                        }
                        if (!parent)
                            continue;
                        auto item = co_await parent.TryGetItemAsync(relative.filename().wstring());
                        if (auto file = item.try_as<StorageFile>()) {
                            present.push_back(name);
                            copies.emplace_back(name, file);
                        }
                    }
                    const auto missing = missing_usb_model_file(present, expected_files);
                    if (!missing.empty())
                        throw std::runtime_error("USB model is missing " +
                                                 wstring_to_utf8(missing));

                    std::error_code begin_ec;
                    if (!writer.begin_usb_import(begin_ec))
                        throw std::runtime_error("Cannot prepare USB model write: " +
                                                 begin_ec.message());

                    // Destination: LocalState\models\<name>
                    // CreateFolderAsync with OpenIfExists creates or opens — no try/catch needed.
                    auto local_folder2 =
                        winrt::Windows::Storage::ApplicationData::Current().LocalFolder();
                    auto models_folder = co_await local_folder2.CreateFolderAsync(
                        L"models", CreationCollisionOption::OpenIfExists);
                    StorageFolder dest_folder = co_await models_folder.CreateFolderAsync(
                        model_name, CreationCollisionOption::OpenIfExists);

                    for (const auto& [name, file] : copies) {
                        StorageFolder folder = dest_folder;
                        for (const auto& dir : std::filesystem::path(name).parent_path())
                            folder = co_await folder.CreateFolderAsync(
                                dir.wstring(), CreationCollisionOption::OpenIfExists);
                        if (normalize_model_path(name) == L"adapter.gguf" &&
                            co_await folder.TryGetItemAsync(file.Name()))
                            continue; // preserve the owner's existing adapter
                        co_await file.CopyAsync(folder, file.Name(),
                                                NameCollisionOption::ReplaceExisting);
                    }
                    std::error_code complete_ec;
                    if (!writer.complete(expected_files, complete_ec))
                        throw std::runtime_error("Cannot complete USB model write: " +
                                                 complete_ec.message());
                    copy_ok = true;
                } catch (winrt::hresult_error const& e) {
                    copy_err = std::wstring(e.message().c_str());
                } catch (std::exception const& e) {
                    copy_err = ::xllama::utf8_to_wstring(e.what());
                } catch (...) {
                    copy_err = L"Unknown error during USB copy";
                }

                writer.release();
                self->m_loadingBar.IsIndeterminate(false);
                self->m_loadingBar.Visibility(winrt::Windows::UI::Xaml::Visibility::Collapsed);
                if (!copy_ok) {
                    self->SetStatus(L"USB copy failed: " + copy_err, StatusKind::Error);
                    co_return;
                }
                while (!usb_root.empty() && usb_root.back() == L'\\')
                    usb_root.pop_back();
                ::xllama::write_local_text("usb_model_root.txt",
                                           ::xllama::wstring_to_utf8(usb_root));
                log_output("[xllama] USB model copy complete\n");
                if (set_app_ready) {
                    self->LoadModelName();
                    self->SetStatus(L"Ready", StatusKind::Success);
                    self->m_runButton.IsEnabled(true);
                    self->m_model_ready.store(true);
                    self->EnsureGpuModelIfNeeded();
                    self->PreloadSessionAsync();
                }
                co_return;
            }
        }
    #endif // !XLLAMA_STORE_SKU
        // Neither found: consult the model catalogue (loaded above) — any entry with
        // an hf_base_url can be auto-downloaded; anything else must be provided via
        // USB or Device Portal upload (dev SKU only for USB).
        if (!entry || entry->hf_base_url.empty() || entry->files.empty()) {
            log_output(("[xllama] EnsureModel: '" + ::xllama::wstring_to_utf8(model_name) +
                        "' not in catalogue (no hf_base_url)\n")
                           .c_str());
            co_await resume_foreground(dispatcher);
            if (set_app_ready) {
    #ifdef XLLAMA_STORE_SKU
                self->SetStatus(L"Model '" + model_name +
                                    L"' not found.\n"
                                    L"Choose a catalogue model that can download, "
                                    L"or reinstall and try again.",
                                StatusKind::Error);
    #else
                self->SetStatus(L"Model '" + model_name +
                                    L"' not found.\n"
                                    L"Upload to LocalState\\models\\" +
                                    model_name +
                                    L" via Device Portal or USB "
                                    L"(see docs/model-selection.md).",
                                StatusKind::Error);
    #endif
                self->m_runButton.IsEnabled(false);
            }
            co_return;
        }

        log_output(("[xllama] EnsureModel: downloading '" + ::xllama::wstring_to_utf8(model_name) +
                    "' from catalogue\n")
                       .c_str());

        co_await resume_background();

        // Auto-download from the catalogue entry's Hugging Face repo.
        co_await resume_foreground(dispatcher);
        if (set_app_ready) {
            self->SetStatus(L"Downloading model...", StatusKind::Working);
            self->m_loadingBar.IsIndeterminate(false);
            self->m_loadingBar.Value(0);
            self->m_loadingBar.Visibility(winrt::Windows::UI::Xaml::Visibility::Visible);
            self->m_runButton.IsEnabled(false);
        }

        co_await ModelDownloader::DownloadAsync(
            entry->hf_base_url, local_model_dir, entry->files, dispatcher,
            [self, set_app_ready](uint64_t done, uint64_t total) {
                if (!set_app_ready)
                    return;
                if (total > 0) {
                    double pct = static_cast<double>(done) / static_cast<double>(total) * 100.0;
                    self->m_loadingBar.Value(pct);
                    self->SetStatus(L"Downloading model... " +
                                        std::to_wstring(done / (1024 * 1024)) + L" MB",
                                    StatusKind::Working);
                } else {
                    self->SetStatus(L"Downloading model... " +
                                        std::to_wstring(done / (1024 * 1024)) + L" MB",
                                    StatusKind::Working);
                }
            },
            [self, set_app_ready](bool ok, std::wstring err) {
                if (set_app_ready) {
                    self->m_loadingBar.Visibility(winrt::Windows::UI::Xaml::Visibility::Collapsed);
                    self->m_runButton.IsEnabled(true);
                }
                if (!ok) {
                    log_output(("[xllama] EnsureModel: download failed: " +
                                ::xllama::wstring_to_utf8(err) + "\n")
                                   .c_str());
                    if (set_app_ready)
                        self->SetStatus(L"Download failed: " + err, StatusKind::Error);
                    return;
                }
                log_output("[xllama] EnsureModel: download complete\n");
                if (set_app_ready) {
                    self->SetStatus(L"Model ready", StatusKind::Success);
                    self->LoadModelName();
                    self->m_model_ready.store(true);
                    self->EnsureGpuModelIfNeeded();
                    self->PreloadSessionAsync();
                }
            });
    } catch (winrt::hresult_error const& e) {
        err = std::wstring(e.message().c_str());
        log_output(
            ("[xllama] EnsureModel: exception: " + ::xllama::wstring_to_utf8(*err) + "\n").c_str());
    } catch (const std::exception& e) {
        err = ::xllama::utf8_to_wstring(e.what());
        log_output((std::string("[xllama] EnsureModel: exception: ") + e.what() + "\n").c_str());
    } catch (...) {
        err = L"unexpected exception";
        log_output(
            "[xllama] EnsureModel: unexpected exception; provision aborted, process continues\n");
    }
    if (err) {
        try {
            co_await resume_foreground(dispatcher);
            if (set_app_ready) {
                self->m_loadingBar.Visibility(winrt::Windows::UI::Xaml::Visibility::Collapsed);
                self->m_runButton.IsEnabled(true);
                self->SetStatus(L"Model setup failed: " + *err, StatusKind::Error);
            }
        } catch (...) {
        }
    }
}

// ---------------------------------------------------------------------------
// LoadModelName: read model.txt from LocalFolder
// ---------------------------------------------------------------------------

void MainPageController::LoadModelName() {
    // m_model_filename is already populated by LoadSettings() (called in Init()).
    // This method just refreshes the header TextBlock with the current value.
    if (m_model_filename.empty())
        m_model_filename = DefaultChatModelId();
    m_modelText.Text(m_model_filename);
}

// ---------------------------------------------------------------------------
// EnsureSession: lazy-build the resident Session (hot path is a name check).
// The session lives in xllama::session_hub() — one owner shared with the LAN
// API, so the two surfaces can no longer hold a model each ("never 2× model
// in RAM" now holds process-wide). The CALLER must hold hub.mtx and keep it
// held for as long as it uses m_turn_session.
// ---------------------------------------------------------------------------

void MainPageController::PreloadSessionAsync() {
    const std::string model = ::xllama::wstring_to_utf8(m_model_filename);
    if (model.empty())
        return;
    auto self = shared_from_this();
    std::thread([self, model]() {
        auto& hub = ::xllama::session_hub();
        if (!::xllama::claim_session_preload(hub.preloading))
            return; // another preload owns the startup-wait state
        auto compute_permit = ::xllama::try_acquire_compute_job();
        if (!compute_permit) {
            hub.preloading.store(false, std::memory_order_release);
            return;
        }
        try {
            std::lock_guard<std::mutex> hub_lk(hub.mtx);
            std::string err;
            if (!self->EnsureSession(model, &err))
                ::xllama::log_output(
                    ("[xllama] session preload failed (non-fatal): " + err + "\n").c_str());
        } catch (...) {
        }
        compute_permit.reset();
        hub.preloading.store(false, std::memory_order_release);
    }).detach();
}

bool MainPageController::EnsureSession(const std::string& model, std::string* err_out) {
    // ADR 0007: the same resolver as the LAN API (catalogue n_ctx, backend from
    // the catalogue kind, GPU layers, runtime LoRA), and the hub reuses the
    // resident session only when it was loaded with these parameters.
    // n_threads is 0 (auto) unless LocalState\gguf_n_threads.txt holds a
    // positive count; 0 or invalid stays auto. detect_threads_llama() is
    // capped at 6 on UWP. ORT reads intra_op_num_threads from genai_config.json.
    const xllama::SessionParams sp = ::xllama::CatalogueSessionParams(model);
    // Every GUI turn, preload and model selection ensures its session here, so
    // this is where the console app takes the session over from a LAN client:
    // an Ollama keep_alive lease on it ends (session_residency.h).
    auto& hub = ::xllama::session_hub();
    if (hub.session && hub.model == model && ::xllama::same_load(hub.params, sp)) {
        hub.use_locked(::xllama::SessionOwner::Gui);
        m_turn_session = hub.session.get();
        return true;
    }
    m_turn_session = nullptr;
    m_kv_valid = false; // new session object → no reusable KV carries over
    std::string err;
    ::xllama::Session* s = hub.ensure_locked(model, sp, &err, ::xllama::SessionOwner::Gui);
    if (!s) {
        if (err_out)
            *err_out = err;
        return false;
    }
    m_turn_session = s;
    return true;
}

} // namespace xllama

#endif // XLLAMA_UWP
