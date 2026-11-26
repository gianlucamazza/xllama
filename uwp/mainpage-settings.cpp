// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// MainPageController: settings load/save and dialog. Split out of MainPage.cpp (#549); shared
// includes and helpers are in mainpage-internal.h.

#include "mainpage-internal.h"

#ifdef XLLAMA_UWP

namespace xllama {

// settings.json is xllama::AppSettings (app_settings.h): one parser, one writer,
// host-tested. These two functions only bind it to the controller's members.
xllama::AppSettings MainPageController::CurrentSettings() const {
    xllama::AppSettings s;
    s.system_prompt = m_system_prompt;
    s.model = ::xllama::wstring_to_utf8(std::wstring(m_model_filename));
    s.kv_reuse = m_kv_reuse;
    s.debug_dashboard = m_debug_dashboard;
    s.routing = m_routing;
    s.gpu_model = m_gpu_model;
    s.diffuse_taesd_vae = m_diffuse_taesd;
    s.diffuse_seed = m_diffuse_seed;
    s.temperature = m_temperature;
    s.top_p = m_top_p;
    s.top_k = m_top_k;
    s.repetition_penalty = m_repetition_penalty;
    s.n_predict = m_n_predict;
    return s;
}

void MainPageController::LoadSettings() {
    if (const auto text = ::xllama::read_local_file(::xllama::kSettingsFileName)) {
        const ::xllama::AppSettingsLoad loaded = ::xllama::parse_app_settings(*text);
        if (!loaded.error.empty())
            log_output("[xllama] settings: " + loaded.error + " — using defaults\n");
        for (const std::string& key : loaded.rejected)
            log_output("[xllama] settings: invalid " + key + " — using its default\n");
        const xllama::AppSettings& s = loaded.settings;
        m_system_prompt = s.system_prompt;
        if (!s.model.empty())
            m_model_filename = ::xllama::utf8_to_wstring(s.model);
        m_kv_reuse = s.kv_reuse;
        m_debug_dashboard = s.debug_dashboard;
        m_routing = s.routing;
        m_gpu_model = s.gpu_model;
        m_diffuse_taesd = s.diffuse_taesd_vae;
        m_diffuse_seed = s.diffuse_seed;
        m_temperature = s.temperature;
        m_top_p = s.top_p;
        m_top_k = s.top_k;
        m_repetition_penalty = s.repetition_penalty;
        m_n_predict = s.n_predict;
    }

    // Back-compat: also read model.txt if model not set via settings.json
    if (m_model_filename.empty()) {
        // UTF-8, read the same way as the headless bench (local_knob.h).
        m_model_filename = ::xllama::utf8_to_wstring(::xllama::read_local_text("model.txt"));
        if (m_model_filename.empty())
            m_model_filename = DefaultChatModelId();
    }

    const std::wstring flag_w =
        ::xllama::utf8_to_wstring(std::string(::xllama::kDebugDashboardFlagName));
    const bool flag_present = std::filesystem::exists(local_wpath(flag_w.c_str()));
    m_debug_dashboard = ::xllama::debug_dashboard_opt_in(m_debug_dashboard, flag_present);
}

void MainPageController::SaveSettings() {
    if (!::xllama::write_local_text(::xllama::kSettingsFileName,
                                    ::xllama::format_app_settings(CurrentSettings())))
        log_output("[xllama] settings: cannot write settings.json\n");
    // Any settings change (system prompt, model, sampling) invalidates the KV
    // cache bound to the old settings — force a fresh generator next turn.
    m_kv_valid = false;
}

winrt::fire_and_forget MainPageController::ShowSettings() {
    auto self = shared_from_this();
    if (m_is_running.load())
        co_return;

    // --- Model selection ComboBox ---
    winrt::Windows::UI::Xaml::Controls::ComboBox modelBox;
    modelBox.Header(winrt::box_value(L"Model"));
    modelBox.FontSize(16);
    modelBox.HorizontalAlignment(HorizontalAlignment::Stretch);
    // Model list comes from the catalogue (models/manifest.json; LocalState
    // override wins). If the active model is not in the catalogue (e.g. a dir
    // uploaded via Device Portal under a custom name), append it so the current
    // selection is always representable.
    auto manifest = ::xllama::LoadModelManifest();
    std::vector<std::wstring> model_keys;
    std::vector<bool> model_is_gguf;             // parallel to model_keys; gates the KV/routing UI
    std::vector<std::wstring> model_lora_status; // read-only adapter status per entry
    int model_sel = 0;
    for (auto const& e : manifest) {
        if (e.kind == L"diffusion")
            continue; // image models belong to the Image dialog, not the chat picker
        if (::xllama::role_is_embedding(::xllama::wstring_to_utf8(e.role)))
            continue; // embedding models are LAN-API only; not for chat
        modelBox.Items().Append(winrt::box_value(winrt::hstring(e.display)));
        if (m_model_filename == e.name)
            model_sel = (int)model_keys.size();
        model_keys.push_back(e.name);
        model_is_gguf.push_back(e.kind == L"gguf");
        if (e.lora.empty()) {
            model_lora_status.push_back(L"Runtime LoRA: none");
        } else {
            model_lora_status.push_back(L"Runtime LoRA: " + e.lora + L" (scale " +
                                        std::to_wstring(e.lora_scale) + L")");
        }
    }
    if (!m_model_filename.empty() && !::xllama::FindManifestEntry(manifest, m_model_filename)) {
        modelBox.Items().Append(winrt::box_value(winrt::hstring(m_model_filename + L" (custom)")));
        model_sel = (int)model_keys.size();
        model_keys.push_back(m_model_filename);
        model_is_gguf.push_back(false);
        model_lora_status.push_back(L"Runtime LoRA: unknown (custom model)");
    }
    modelBox.SelectedIndex(model_sel);

    winrt::Windows::UI::Xaml::Controls::TextBlock loraStatus;
    loraStatus.TextWrapping(TextWrapping::Wrap);
    loraStatus.Opacity(0.7);
    if (model_sel >= 0 && model_sel < static_cast<int>(model_lora_status.size()))
        loraStatus.Text(model_lora_status[static_cast<size_t>(model_sel)]);

    // --- System prompt TextBox ---
    winrt::Windows::UI::Xaml::Controls::TextBox sysPromptBox;
    no_keyboard_on_programmatic_focus(sysPromptBox);
    sysPromptBox.Text(::xllama::utf8_to_wstring(m_system_prompt));
    sysPromptBox.AcceptsReturn(true);
    sysPromptBox.TextWrapping(TextWrapping::Wrap);
    sysPromptBox.MinHeight(100);
    sysPromptBox.FontSize(16);
    sysPromptBox.IsFocusEngagementEnabled(true);
    sysPromptBox.Header(winrt::box_value(L"System prompt"));

    // --- Sampling sliders / number boxes ---
    auto make_slider = [](double val, double lo, double hi, double step,
                          const wchar_t* label) -> winrt::Windows::UI::Xaml::Controls::Slider {
        winrt::Windows::UI::Xaml::Controls::Slider s;
        s.Minimum(lo);
        s.Maximum(hi);
        s.StepFrequency(step);
        s.Value(val);
        s.Header(winrt::box_value(winrt::hstring(label)));
        s.HorizontalAlignment(HorizontalAlignment::Stretch);
        return s;
    };
    namespace lim = ::xllama::settings_limits; // the ranges settings.json enforces
    auto tempSlider = make_slider(m_temperature, lim::kTemperatureMin, lim::kTemperatureMax, 0.05,
                                  L"Temperature (0–2)");
    auto topPSlider = make_slider(m_top_p, lim::kTopPMin, lim::kTopPMax, 0.05, L"Top-p (0–1)");
    auto repSlider = make_slider(m_repetition_penalty, lim::kRepetitionPenaltyMin,
                                 lim::kRepetitionPenaltyMax, 0.05, L"Repetition penalty (1–2)");

    // top_k and n_predict as simple Sliders (NumberBox not available in older SDK targets)
    auto topKSlider = make_slider(m_top_k, lim::kTopKMin, lim::kTopKMax, 1.0, L"Top-k (1–200)");
    auto nPredSlider = make_slider(m_n_predict, lim::kNPredictMin, lim::kNPredictMax, 16.0,
                                   L"Max new tokens (16–2048)");

    // --- KV-cache reuse toggle (continuous decoding) ---
    winrt::Windows::UI::Xaml::Controls::ToggleSwitch kvToggle;
    kvToggle.Header(winrt::box_value(L"KV-cache reuse (faster multi-turn)"));
    kvToggle.OnContent(winrt::box_value(L"On"));
    kvToggle.OffContent(winrt::box_value(L"Off"));
    kvToggle.IsOn(m_kv_reuse);

    winrt::Windows::UI::Xaml::Controls::ToggleSwitch debugToggle;
    debugToggle.Header(winrt::box_value(L"Debug dashboard (resident model, LAN, inference)"));
    debugToggle.OnContent(winrt::box_value(L"On"));
    debugToggle.OffContent(winrt::box_value(L"Off"));
    debugToggle.IsOn(m_debug_dashboard);

    // --- EP routing ComboBox (experimental; needs the DML fp16 model on device) ---
    winrt::Windows::UI::Xaml::Controls::ComboBox routingBox;
    routingBox.Header(winrt::box_value(L"EP routing (per conversation)"));
    routingBox.FontSize(16);
    routingBox.HorizontalAlignment(HorizontalAlignment::Stretch);
    routingBox.Items().Append(winrt::box_value(L"CPU only (default)"));
    routingBox.Items().Append(winrt::box_value(L"GPU only (DML)"));
    routingBox.Items().Append(winrt::box_value(L"Auto (long prompts → GPU)"));
    routingBox.SelectedIndex(m_routing >= 0 && m_routing <= 2 ? m_routing : 0);

    #ifndef XLLAMA_STORE_SKU
    const bool api_enabled = std::filesystem::exists(local_wpath(L"api.flag"));
    // The port the server binds (api-server.h), not the raw file text.
    const int api_port =
        ::xllama::api::configured_api_port(::xllama::read_local_text("api-port.txt"));
    winrt::Windows::UI::Xaml::Controls::ToggleSwitch apiToggle;
    apiToggle.Header(winrt::box_value(L"LAN API (OpenAI-compatible)"));
    apiToggle.OnContent(winrt::box_value(L"On"));
    apiToggle.OffContent(winrt::box_value(L"Off"));
    apiToggle.IsOn(api_enabled);

    winrt::Windows::UI::Xaml::Controls::TextBox apiPortBox;
    no_keyboard_on_programmatic_focus(apiPortBox);
    apiPortBox.Header(winrt::box_value(L"LAN API port (1025–49151, except 11443)"));
    apiPortBox.Text(std::to_wstring(api_port));
    apiPortBox.IsEnabled(api_enabled);
    apiPortBox.InputScope([] {
        winrt::Windows::UI::Xaml::Input::InputScopeName name;
        name.NameValue(winrt::Windows::UI::Xaml::Input::InputScopeNameValue::Number);
        winrt::Windows::UI::Xaml::Input::InputScope scope;
        scope.Names().Append(name);
        return scope;
    }());
    apiToggle.Toggled([apiPortBox](IInspectable const& sender, RoutedEventArgs const&) {
        apiPortBox.IsEnabled(sender.as<ToggleSwitch>().IsOn());
    });

    winrt::Windows::UI::Xaml::Controls::TextBlock apiStatus;
    apiStatus.TextWrapping(TextWrapping::Wrap);
    apiStatus.Opacity(0.7);
    const auto api_snapshot = ::xllama::api::server_status();
    switch (api_snapshot.state) {
    case ::xllama::api::ServerState::Running:
        apiStatus.Text(L"Status: listening at http://" +
                       (m_lan_ip.empty() ? std::wstring(L"<console IP>") : m_lan_ip) + L":" +
                       std::to_wstring(api_snapshot.port) +
                       L"/v1. Trusted LAN only; no authentication.");
        break;
    case ::xllama::api::ServerState::Starting:
        apiStatus.Text(L"Status: starting. Trusted LAN only; no authentication.");
        break;
    case ::xllama::api::ServerState::Error:
        apiStatus.Text(L"Status: error — " + ::xllama::utf8_to_wstring(api_snapshot.message));
        break;
    default:
        apiStatus.Text(L"Status: stopped. Trusted LAN only; no authentication.");
        break;
    }
    #endif // !XLLAMA_STORE_SKU

    // GGUF models run stateless on CPU-only llama.cpp: KV-reuse and EP routing do
    // not apply, so grey them out whenever a GGUF entry is selected (and restore
    // them for ORT entries). Wired live on the model ComboBox.
    auto sync_backend_toggles = [kvToggle, routingBox, loraStatus, model_is_gguf,
                                 model_lora_status](int idx) {
        bool gguf = idx >= 0 && idx < (int)model_is_gguf.size() && model_is_gguf[idx];
        // KV reuse works on GGUF now (persistent llama_context); only EP routing
        // stays ORT-only (the llama.cpp UWP build is CPU-only).
        kvToggle.IsEnabled(true);
        routingBox.IsEnabled(!gguf);
        if (idx >= 0 && idx < static_cast<int>(model_lora_status.size()))
            loraStatus.Text(model_lora_status[static_cast<size_t>(idx)]);
    };
    sync_backend_toggles(model_sel);
    modelBox.SelectionChanged(
        [sync_backend_toggles](
            winrt::Windows::Foundation::IInspectable const& sender,
            winrt::Windows::UI::Xaml::Controls::SelectionChangedEventArgs const&) {
            auto box = sender.as<winrt::Windows::UI::Xaml::Controls::ComboBox>();
            sync_backend_toggles(box.SelectedIndex());
        });

    // --- Phase 11 personalize (#116) ---
    const std::string samples_path =
        ::xllama::wstring_to_utf8(local_wpath(L"training\\samples.jsonl"));
    const int sample_n = ::xllama::count_usable_preference_samples(samples_path);
    const std::string base_hint = ResolvePersonalizeBase();
    winrt::Windows::UI::Xaml::Controls::TextBlock personalizeStatus;
    personalizeStatus.TextWrapping(TextWrapping::Wrap);
    personalizeStatus.Opacity(0.85);
    {
        std::wstring line =
            L"Personalize: " + std::to_wstring(sample_n) + L" usable preference sample(s)";
        if (base_hint.empty())
            line += L". Base GGUF missing — place training\\base-f16.gguf (or pick a "
                    L"SmolLM2 GGUF) before training.";
        else
            line += L". Base: " + ::xllama::utf8_to_wstring(base_hint);
        personalizeStatus.Text(line);
    }

    winrt::Windows::UI::Xaml::Controls::StackPanel panel;
    panel.Orientation(Orientation::Vertical);
    panel.Spacing(12);
    panel.Children().Append(modelBox);
    panel.Children().Append(loraStatus);
    panel.Children().Append(personalizeStatus);
    panel.Children().Append(sysPromptBox);
    panel.Children().Append(tempSlider);
    panel.Children().Append(topPSlider);
    panel.Children().Append(topKSlider);
    panel.Children().Append(repSlider);
    panel.Children().Append(nPredSlider);
    panel.Children().Append(kvToggle);
    panel.Children().Append(debugToggle);
    panel.Children().Append(routingBox);
    #ifndef XLLAMA_STORE_SKU
    panel.Children().Append(apiToggle);
    panel.Children().Append(apiPortBox);
    panel.Children().Append(apiStatus);
    #endif

    // Store policy 11.16: a path to report inappropriate generative-AI output.
    // Opens the GitHub Issues form; internetClient is declared on both SKUs.
    winrt::Windows::UI::Xaml::Controls::HyperlinkButton reportLink;
    reportLink.Content(winrt::box_value(L"Report inappropriate generated content"));
    reportLink.NavigateUri(winrt::Windows::Foundation::Uri(
        L"https://github.com/gianlucamazza/xllama/issues/new?template=catalogue-report.yml"));
    panel.Children().Append(reportLink);

    winrt::Windows::UI::Xaml::Controls::ScrollViewer sv;
    sv.Content(panel);
    sv.MaxHeight(480);
    sv.VerticalScrollBarVisibility(ScrollBarVisibility::Auto);

    winrt::Windows::UI::Xaml::Controls::ContentDialog dlg;
    dlg.Title(winrt::box_value(L"Settings"));
    dlg.Content(sv);
    dlg.PrimaryButtonText(L"Save");
    dlg.SecondaryButtonText(L"Train on my feedback");
    dlg.CloseButtonText(L"Cancel");
    dlg.IsSecondaryButtonEnabled(sample_n > 0 && !base_hint.empty() && !m_train_running.load());
    dlg.XamlRoot(m_root.XamlRoot());
    ApTrackDialog(dlg);

    auto result = co_await dlg.ShowAsync();
    if (result == winrt::Windows::UI::Xaml::Controls::ContentDialogResult::Secondary) {
        // In-process train (#116); does not require Save of other settings.
        self->StartPersonalizeTrain();
        co_return;
    }
    if (result != winrt::Windows::UI::Xaml::Controls::ContentDialogResult::Primary)
        co_return;

    // Read back values. System prompt is whatever the user left in the box —
    // no silent rewrite when the model role changes (that was string-equality
    // magic and fought the Settings field). API empty-system fill still uses
    // role → kCodingSystemPrompt (routing_policy / chat_prompt).
    self->m_system_prompt = ::xllama::wstring_to_utf8(std::wstring(sysPromptBox.Text().c_str()));
    int mi = modelBox.SelectedIndex();
    self->m_temperature = static_cast<float>(tempSlider.Value());
    self->m_top_p = static_cast<float>(topPSlider.Value());
    self->m_top_k = static_cast<int>(topKSlider.Value());
    self->m_repetition_penalty = static_cast<float>(repSlider.Value());
    // Slider first — model switch may override with a catalogue n_predict
    // (thinking tier 1024). The slider still shows the previous model’s value
    // until Settings is reopened, so a stale read must not clobber the knob.
    self->m_n_predict = static_cast<int>(nPredSlider.Value());
    if (mi >= 0 && mi < (int)model_keys.size()) {
        std::wstring new_model = model_keys[mi];
        self->SelectModel(new_model);
    }
    self->m_kv_reuse = kvToggle.IsOn();
    self->ApplyDebugDashboard(debugToggle.IsOn());
    #ifndef XLLAMA_STORE_SKU
    int selected_api_port = api_port;
    if (apiToggle.IsOn()) {
        const std::optional<int> typed = ::xllama::parse_knob_int(
            ::xllama::wstring_to_utf8(std::wstring(apiPortBox.Text().c_str())));
        if (!typed) {
            self->SetStatus(L"Invalid LAN API port", StatusKind::Error);
            co_return;
        }
        selected_api_port = *typed;
        if (!::xllama::api::port_bindable(selected_api_port)) {
            self->SetStatus(L"LAN API port must be 1025–49151 and not 11443", StatusKind::Error);
            co_return;
        }
    }
    #endif
    int ri = routingBox.SelectedIndex();
    self->m_routing = (ri >= ::xllama::settings_limits::kRoutingMin &&
                       ri <= ::xllama::settings_limits::kRoutingMax)
                          ? ri
                          : 0;
    // Routing is per-conversation: a change applies from the next new/loaded chat
    // (m_active_model stays fixed for the conversation in progress).
    self->SaveSettings();
    #ifndef XLLAMA_STORE_SKU
    self->ApplyApiSettings(apiToggle.IsOn(), selected_api_port);
    #endif
}

} // namespace xllama

#endif // XLLAMA_UWP
