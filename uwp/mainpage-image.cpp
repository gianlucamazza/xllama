// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// MainPageController: image dialog and diffusion. Split out of MainPage.cpp (#549); shared
// includes and helpers are in mainpage-internal.h.

#include "mainpage-internal.h"

#ifdef XLLAMA_UWP

namespace xllama {

// ---------------------------------------------------------------------------
// ShowImageDialog — view the last generated image and run SD-Turbo in-process.
// Plain ORT DirectML coexists with the XAML compositor (887A0036 applies only
// to ORT GenAI chat DML, not this pipeline — see docs/uwp-constraints.md §7).
// ---------------------------------------------------------------------------

winrt::fire_and_forget MainPageController::ShowImageDialog() {
    auto self = shared_from_this();
    if (m_is_running.load())
        co_return;

    winrt::Windows::UI::Xaml::Controls::StackPanel panel;
    panel.Orientation(Orientation::Vertical);
    panel.Spacing(12);

    // Last generated image (if any) — loaded from LocalState via a stream so a
    // regenerated file is never masked by URI caching.
    winrt::Windows::UI::Xaml::Controls::TextBlock imgStatus;
    imgStatus.TextWrapping(TextWrapping::Wrap);
    panel.Children().Append(imgStatus);
    try {
        auto local = ApplicationData::Current().LocalFolder();
        auto file = co_await local.GetFileAsync(L"diffuse-out.png");
        auto stream = co_await file.OpenAsync(winrt::Windows::Storage::FileAccessMode::Read);
        winrt::Windows::UI::Xaml::Media::Imaging::BitmapImage bmp;
        co_await bmp.SetSourceAsync(stream);
        winrt::Windows::UI::Xaml::Controls::Image img;
        img.Source(bmp);
        img.MaxHeight(320);
        imgStatus.Text(L"Last generated image (512×512):");
        panel.Children().Append(img);
    } catch (...) {
        imgStatus.Text(L"No image generated yet.");
    }

    winrt::Windows::UI::Xaml::Controls::TextBlock lastSeed;
    lastSeed.Opacity(0.7);
    const std::string previous_seed = ::xllama::read_local_text("diffuse-seed.txt");
    lastSeed.Text(previous_seed.empty()
                      ? L"Last seed: none"
                      : L"Last seed: " + ::xllama::utf8_to_wstring(previous_seed));
    panel.Children().Append(lastSeed);

    winrt::Windows::UI::Xaml::Controls::TextBox promptBox;
    no_keyboard_on_programmatic_focus(promptBox);
    promptBox.Header(winrt::box_value(L"Image prompt"));
    // Show the prompt that produced the image above it, not a fixed suggestion.
    // The dialog already displays the last image and its seed, so a hardcoded
    // string here made the three disagree: the Store listing screenshot showed a
    // pixel-art robot with "a red sports car on a mountain road at sunset"
    // underneath, and pressing Generate would have produced neither.
    {
        const std::string last = ::xllama::read_local_text("prompt.txt");
        promptBox.Text(last.empty() ? L"a red sports car on a mountain road at sunset"
                                    : ::xllama::utf8_to_wstring(last).c_str());
    }
    promptBox.TextWrapping(TextWrapping::Wrap);
    promptBox.AcceptsReturn(false);
    promptBox.FontSize(16);
    panel.Children().Append(promptBox);

    winrt::Windows::UI::Xaml::Controls::Slider stepsSlider;
    stepsSlider.Minimum(::xllama::diffusion::kStepsMin);
    stepsSlider.Maximum(::xllama::diffusion::kStepsInteractiveMax);
    stepsSlider.StepFrequency(1);
    stepsSlider.Value(::xllama::diffusion::kDefaultSteps);
    stepsSlider.Header(winrt::box_value(L"Steps (SD-Turbo: 1 is enough)"));
    panel.Children().Append(stepsSlider);

    winrt::Windows::UI::Xaml::Controls::TextBox seedBox;
    no_keyboard_on_programmatic_focus(seedBox);
    seedBox.Header(winrt::box_value(L"Seed (0 = random)"));
    seedBox.Text(std::to_wstring(self->m_diffuse_seed));
    seedBox.InputScope([] {
        winrt::Windows::UI::Xaml::Input::InputScopeName name;
        name.NameValue(winrt::Windows::UI::Xaml::Input::InputScopeNameValue::Number);
        winrt::Windows::UI::Xaml::Input::InputScope scope;
        scope.Names().Append(name);
        return scope;
    }());
    panel.Children().Append(seedBox);

    winrt::Windows::UI::Xaml::Controls::ToggleSwitch taesdToggle;
    taesdToggle.Header(winrt::box_value(L"TAESD fast VAE (smaller decoder, ~4.5 s total)"));
    taesdToggle.IsOn(self->m_diffuse_taesd);
    panel.Children().Append(taesdToggle);

    winrt::Windows::UI::Xaml::Controls::TextBlock note;
    note.TextWrapping(TextWrapping::Wrap);
    note.Opacity(0.7);
    note.Text(L"Generate runs SD-Turbo on the GPU in-process. Progress appears in the "
              L"status bar; press Cancel to abort between UNet steps.");
    panel.Children().Append(note);

    // The Image dialog currently always generates with sd-turbo-fp16. The
    // helper still keys off licence-then-id so a future non-Stability image
    // model can hide the line without a second UI change. See
    // stability_attribution.h for the mapping.
    constexpr const wchar_t* kDiffusionModel = L"sd-turbo-fp16";
    if (::xllama::show_stability_attribution(
            ::xllama::wstring_to_utf8(std::wstring(kDiffusionModel)))) {
        winrt::Windows::UI::Xaml::Controls::TextBlock credit;
        credit.TextWrapping(TextWrapping::Wrap);
        credit.Opacity(0.7);
        credit.Text(::xllama::utf8_to_wstring(std::string(::xllama::kStabilityAttribution)));
        panel.Children().Append(credit);
    }

    winrt::Windows::UI::Xaml::Controls::ScrollViewer sv;
    sv.Content(panel);
    sv.MaxHeight(480);
    sv.VerticalScrollBarVisibility(ScrollBarVisibility::Auto);

    winrt::Windows::UI::Xaml::Controls::ContentDialog dlg;
    dlg.Title(winrt::box_value(L"Image generation (SD-Turbo on GPU)"));
    dlg.Content(sv);
    dlg.PrimaryButtonText(L"Generate");
    dlg.CloseButtonText(L"Close");
    dlg.XamlRoot(m_root.XamlRoot());
    ApTrackDialog(dlg);

    auto result = co_await dlg.ShowAsync();
    if (result != winrt::Windows::UI::Xaml::Controls::ContentDialogResult::Primary)
        co_return;

    const std::string seed_text = ::xllama::wstring_to_utf8(std::wstring(seedBox.Text().c_str()));
    // Empty = 0 = random. Otherwise a whole number in uint32, the seed domain of
    // the pipeline (diffusion/request.h); "12abc" is rejected, not read as 12.
    const std::optional<long long> requested_seed =
        seed_text.empty() ? std::optional<long long>(0) : ::xllama::parse_knob_i64(seed_text);
    if (!requested_seed) {
        self->SetStatus(L"Invalid image seed", StatusKind::Error);
        co_return;
    }
    if (*requested_seed < 0 || *requested_seed > std::numeric_limits<uint32_t>::max()) {
        self->SetStatus(L"Image seed must fit uint32", StatusKind::Error);
        co_return;
    }
    self->m_diffuse_seed = static_cast<uint32_t>(*requested_seed);
    self->m_diffuse_taesd = taesdToggle.IsOn();
    self->SaveSettings();

    // Ensure the diffusion model is present; download it from the catalogue if
    // missing (kind "diffusion" entries never reach the chat picker).
    constexpr const wchar_t* kTaesdVaeRemote = L"sd-turbo-fp16_taesd_vae_decoder_model.onnx";
    constexpr const wchar_t* kTaesdLicenseRemote = L"sd-turbo-fp16_taesd_LICENSE";
    constexpr const wchar_t* kTaesdLicenseSha256 =
        L"261497cf8e9ed613ed1fa5ddf48af32cdf50545b2ccd9edd7eacc91fb9255d6e";
    {
        auto local = ApplicationData::Current().LocalFolder();
        std::wstring model_dir =
            std::wstring(local.Path().c_str()) + L"\\models\\" + kDiffusionModel;
        std::error_code ec;
        const bool present = std::filesystem::exists(
                                 std::filesystem::path(model_dir) / L"unet" / L"model.onnx", ec) ||
                             ModelDownloader::IsComplete(model_dir);
        if (!present) {
            auto manifest = ::xllama::LoadModelManifest();
            auto* entry = ::xllama::FindManifestEntry(manifest, kDiffusionModel);
            if (!entry || entry->hf_base_url.empty() || entry->files.empty()) {
                self->SetStatus(std::wstring(L"Model '") + kDiffusionModel +
                                    L"' not found. Provision it via Device Portal "
                                    L"(see diffusion/README.md).",
                                StatusKind::Error);
                co_return;
            }
            std::filesystem::create_directories(model_dir, ec);
            if (ec) {
                self->SetStatus(L"Cannot create the diffusion model dir", StatusKind::Error);
                co_return;
            }
            self->SetStatus(L"Downloading image model (~2.4 GB)...", StatusKind::Working);
            self->m_loadingBar.IsIndeterminate(false);
            self->m_loadingBar.Value(0);
            self->m_loadingBar.Visibility(winrt::Windows::UI::Xaml::Visibility::Visible);
            auto dl_ok = std::make_shared<bool>(false);
            auto dl_err = std::make_shared<std::wstring>();
            co_await ModelDownloader::DownloadAsync(
                entry->hf_base_url, model_dir, entry->files, m_root.Dispatcher(),
                [self](uint64_t done, uint64_t total) {
                    if (total > 0)
                        self->m_loadingBar.Value((double)done / (double)total * 100.0);
                    self->SetStatus(L"Downloading image model... " +
                                        std::to_wstring(done / (1024 * 1024)) + L" MB",
                                    StatusKind::Working);
                },
                [dl_ok, dl_err](bool ok2, std::wstring err) {
                    *dl_ok = ok2;
                    *dl_err = std::move(err);
                });
            self->m_loadingBar.Visibility(winrt::Windows::UI::Xaml::Visibility::Collapsed);
            if (!*dl_ok) {
                self->SetStatus(L"Image model download failed: " + *dl_err, StatusKind::Error);
                co_return;
            }
        }

        // TAESD: drop-in tiny VAE (~5 MB) overwrites vae_decoder/model.onnx in the
        // same sd-turbo-fp16 tree (models-v1 assets kTaesdVaeRemote + MIT licence).
        if (self->m_diffuse_taesd) {
            auto manifest = ::xllama::LoadModelManifest();
            auto* entry = ::xllama::FindManifestEntry(manifest, kDiffusionModel);
            if (!entry || entry->hf_base_url.empty()) {
                self->SetStatus(L"TAESD VAE: catalogue entry missing hf_base_url",
                                StatusKind::Error);
                co_return;
            }
            std::vector<::xllama::ModelFile> taesd_vae{
                {L"TAESD_LICENSE", kTaesdLicenseRemote, 1073, kTaesdLicenseSha256, L"licence"},
                {L"vae_decoder/model.onnx", kTaesdVaeRemote, 5'000'000},
            };
            self->SetStatus(L"Downloading TAESD VAE (~5 MB)...", StatusKind::Working);
            self->m_loadingBar.IsIndeterminate(false);
            self->m_loadingBar.Value(0);
            self->m_loadingBar.Visibility(winrt::Windows::UI::Xaml::Visibility::Visible);
            auto vae_ok = std::make_shared<bool>(false);
            auto vae_err = std::make_shared<std::wstring>();
            co_await ModelDownloader::DownloadAsync(
                entry->hf_base_url, model_dir, taesd_vae, m_root.Dispatcher(),
                [self](uint64_t done, uint64_t total) {
                    if (total > 0)
                        self->m_loadingBar.Value((double)done / (double)total * 100.0);
                },
                [vae_ok, vae_err](bool ok2, std::wstring err) {
                    *vae_ok = ok2;
                    *vae_err = std::move(err);
                });
            self->m_loadingBar.Visibility(winrt::Windows::UI::Xaml::Visibility::Collapsed);
            if (!*vae_ok) {
                self->SetStatus(L"TAESD VAE download failed: " + *vae_err, StatusKind::Error);
                co_return;
            }
        }
    }

    std::string prompt_utf8 = ::xllama::wstring_to_utf8(std::wstring(promptBox.Text().c_str()));
    if (prompt_utf8.empty()) {
        // Do not substitute one. Generating something the user did not ask for
        // is the defect class already paid for in the bench (#205): the run
        // succeeds and produces a real-looking artefact nobody requested.
        self->SetStatus(L"Enter an image prompt first", StatusKind::Error);
        co_return;
    }
    const int steps = ::xllama::diffusion::interactive_steps(stepsSlider.Value());
    const uint32_t seed = self->m_diffuse_seed == 0
                              ? static_cast<uint32_t>(GetTickCount64() % 1'000'000'000ULL) + 1
                              : self->m_diffuse_seed;
    self->m_last_diffuse_seed = seed;
    if (self->m_is_running.load() || self->m_train_running.load() ||
        self->m_diffuse_running.load()) {
        self->SetStatus(L"Busy — wait for the current job to finish", StatusKind::Error);
        co_return;
    }
    auto compute_permit = ::xllama::try_acquire_compute_job();
    if (!compute_permit) {
        self->SetStatus(L"Busy — another compute job is active", StatusKind::Error);
        co_return;
    }
    ::xllama::write_local_text("prompt.txt", prompt_utf8);
    ::xllama::write_local_text("diffuse-steps.txt", std::to_string(steps));
    ::xllama::write_local_text("diffuse-seed.txt", std::to_string(seed));
    ::xllama::write_local_text("diffuse-model.txt", "sd-turbo-fp16");

    self->StartDiffusion(std::move(compute_permit));
}

// ---------------------------------------------------------------------------
// StartDiffusion — run SD-Turbo on a background MTA thread (same as the
// diffuse-inproc.flag experiment in App.cpp, but driven from the Image dialog).
// ---------------------------------------------------------------------------

void MainPageController::StartDiffusion(::xllama::ComputeJobAdmission::Handle permit) {
    if (!permit)
        return;
    if (m_is_running.load() || m_train_running.load() || m_diffuse_running.load()) {
        SetStatus(L"Busy — wait for the current job to finish", StatusKind::Error);
        return;
    }

    m_diffuse_running.store(true);
    SetRunning(true);
    m_loadingBar.IsIndeterminate(true);
    SetStatus(L"Generating image (seed " + std::to_wstring(m_last_diffuse_seed) + L")...",
              StatusKind::Working);

    // Reset stale state from a previous run: a leftover cancel flag would abort
    // the new run at the first UNet-step check, and a leftover "done"/"error"
    // progress line would flash in the status bar before the worker's first write.
    _wremove(local_wpath(L"diffuse-cancel.flag").c_str());
    ::xllama::write_local_text("diffuse-progress.txt", "start");

    auto self = shared_from_this();
    auto dispatcher = m_root.Dispatcher();

    if (!m_diffuse_timer) {
        m_diffuse_timer = winrt::Windows::UI::Xaml::DispatcherTimer{};
        m_diffuse_timer.Interval(std::chrono::milliseconds(200));
    } else {
        m_diffuse_timer.Stop();
        m_diffuse_timer.Tick(m_diffuse_tick_token);
    }
    auto weak_self = std::weak_ptr<MainPageController>(self);
    m_diffuse_tick_token =
        m_diffuse_timer.Tick([weak_self](IInspectable const&, IInspectable const&) {
            if (auto s = weak_self.lock())
                s->PollDiffuseProgress();
        });
    m_diffuse_timer.Start();

    std::thread([self, dispatcher, permit = std::move(permit)]() {
        bool published = false;
        try {
            winrt::init_apartment(); // MTA — ApplicationData + ORT DML
            // Free the resident chat model first when the pipeline would not fit
            // next to it (#548); the permit is held, so the hub lock comes second.
            if (::xllama::diffusion::make_room_for_diffusion(::xllama::session_hub(),
                                                             ::xllama::working_set_mb(),
                                                             ::xllama::app_memory_limit_mb()))
                ::xllama::log_output("[xllama] diffuse: released the resident chat model to fit "
                                     "the image pipeline\n");
            published = ::xllama::bridge::run_diffuse() != 0;
        } catch (const std::exception& ex) {
            ::xllama::log_output(std::string("[xllama] diffuse thread: ") + ex.what() + "\n");
        } catch (...) {
            ::xllama::log_output("[xllama] diffuse thread: unknown exception\n");
        }
        dispatcher.RunAsync(CoreDispatcherPriority::Normal,
                            [self, permit, published]() { self->FinishDiffusion(published); });
    }).detach();
}

void MainPageController::PollDiffuseProgress() {
    const std::string stage = ::xllama::read_local_text("diffuse-progress.txt");
    if (stage.empty())
        return;
    SetStatus(format_diffuse_stage(stage),
              stage == "error" ? StatusKind::Error
                               : (stage == "done" ? StatusKind::Success : StatusKind::Working));
}

void MainPageController::FinishDiffusion(bool published) {
    if (m_diffuse_timer) {
        m_diffuse_timer.Stop();
        m_diffuse_timer.Tick(m_diffuse_tick_token);
    }
    m_diffuse_running.store(false);

    const std::string stage = ::xllama::read_local_text("diffuse-progress.txt");
    if (published) {
        SetStatus(L"Image ready — open [*] Image to view", StatusKind::Success);
    } else if (stage == "cancelled") {
        SetStatus(L"Image generation cancelled", StatusKind::Info);
    } else {
        SetStatus(L"Image generation failed — see xllama.log", StatusKind::Error);
    }
    SetRunning(false);
}

} // namespace xllama

#endif // XLLAMA_UWP
