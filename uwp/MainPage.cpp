// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#ifdef XLLAMA_UWP

    #include "mainpage-internal.h"

namespace xllama {

// First-launch / empty-settings default. Shipping builds are `unified`
// (XLLAMA_USE_LLAMA): default chat role is `lfm25-350m`. Fastest/lightest
// floor is `lfm25-230m`. Numbers live in `docs/benchmarks.md`. ORT-only
// builds keep the classic SmolLM2 CPU int4.
// Matches bench/configs/settings-modern.json and docs/recommended-config.md.
std::wstring DefaultChatModelId() {
    #if defined(XLLAMA_USE_LLAMA)
    return L"lfm25-350m";
    #else
    return L"smollm2-360m-cpu-int4";
    #endif
}

// ---------------------------------------------------------------------------
// BuildUI — assembles the UI tree programmatically.
// Equivalent to MainPage.xaml without requiring MarkupCompilePass2 or
// IXamlMetadataProvider metadata for xllama types.
// ---------------------------------------------------------------------------

void MainPageController::BuildUI() {
    m_root = Page();

    // ---- outer grid (3 rows: header / body / footer) ----
    Grid outerGrid;
    outerGrid.Margin(ThicknessHelper::FromLengths(48, 27, 48, 27)); // Xbox TV safe area (5%)

    RowDefinition rowAuto1;
    rowAuto1.Height(GridLengthHelper::Auto());
    RowDefinition rowStar;
    rowStar.Height(GridLengthHelper::FromValueAndType(1, GridUnitType::Star));
    RowDefinition rowAuto2;
    rowAuto2.Height(GridLengthHelper::Auto());
    outerGrid.RowDefinitions().Append(rowAuto1);
    outerGrid.RowDefinitions().Append(rowStar);
    outerGrid.RowDefinitions().Append(rowAuto2);

    // ---- Row 0: header (left: chat model + status + progress; right: LAN API) ----
    Grid header;
    Grid::SetRow(header, 0);
    ColumnDefinition headerMain;
    headerMain.Width(GridLengthHelper::FromValueAndType(1, GridUnitType::Star));
    ColumnDefinition headerApi;
    headerApi.Width(GridLengthHelper::Auto());
    header.ColumnDefinitions().Append(headerMain);
    header.ColumnDefinitions().Append(headerApi);
    StackPanel headerLeft;
    Grid::SetColumn(headerLeft, 0);

    m_modelText = TextBlock();
    m_modelText.FontSize(28);
    m_modelText.FontWeight(FontWeights::SemiBold());
    m_modelText.Text(L"xllama");

    // 10-foot: 18pt at full or near-full opacity reads from the couch; the old
    // 14pt / 0.7 status line did not.
    m_statusText = TextBlock();
    m_statusText.FontSize(18);
    m_statusText.Opacity(0.85);
    m_statusText.Margin(ThicknessHelper::FromLengths(0, 4, 0, 0));
    m_statusText.Text(L"Loading model...");

    m_loadingBar = ProgressBar();
    m_loadingBar.IsIndeterminate(true);
    m_loadingBar.Visibility(Visibility::Collapsed);
    m_loadingBar.Margin(ThicknessHelper::FromLengths(0, 8, 0, 0));

    headerLeft.Children().Append(m_modelText);
    headerLeft.Children().Append(m_statusText);
    headerLeft.Children().Append(m_loadingBar);
    header.Children().Append(headerLeft);

    // LAN API panel: hidden while the endpoint is stopped (and in the Store SKU,
    // which has no endpoint). Lines: endpoint URL, current request, counters,
    // last error.
    m_apiPanel = StackPanel();
    Grid::SetColumn(m_apiPanel, 1);
    m_apiPanel.Margin(ThicknessHelper::FromLengths(32, 0, 0, 0));
    // Effective pixels: the Xbox shell lays apps out at 960x540 (2x scale). Cap
    // the panel at half the width so a long error never squeezes the chat model
    // and status out of the left column (seen on-console before this cap).
    m_apiPanel.MaxWidth(480);
    m_apiPanel.Visibility(Visibility::Collapsed);
    auto api_line = [](double size, double opacity) {
        TextBlock line;
        line.FontSize(size);
        line.Opacity(opacity);
        line.HorizontalAlignment(HorizontalAlignment::Right);
        line.TextAlignment(TextAlignment::Right);
        line.TextTrimming(TextTrimming::CharacterEllipsis);
        return line;
    };
    m_apiEndpointText = api_line(20, 1.0);
    m_apiEndpointText.FontWeight(FontWeights::SemiBold());
    m_apiNowText = api_line(18, 1.0);
    m_apiCountsText = api_line(16, 0.85);
    m_apiErrorText = api_line(16, 1.0);
    m_apiErrorText.TextWrapping(TextWrapping::Wrap);
    m_apiErrorText.MaxLines(2);
    m_apiErrorText.Foreground(winrt::Windows::UI::Xaml::Media::SolidColorBrush({255, 240, 80, 70}));
    m_apiErrorText.Visibility(Visibility::Collapsed);
    m_apiPanel.Children().Append(m_apiEndpointText);
    m_apiPanel.Children().Append(m_apiNowText);
    m_apiPanel.Children().Append(m_apiCountsText);
    m_apiPanel.Children().Append(m_apiErrorText);
    header.Children().Append(m_apiPanel);

    // ---- Row 1: scroll area (prompt input + output text) ----
    m_outputScroll = ScrollViewer();
    Grid::SetRow(m_outputScroll, 1);
    m_outputScroll.VerticalScrollBarVisibility(ScrollBarVisibility::Auto);
    m_outputScroll.Margin(ThicknessHelper::FromLengths(0, 24, 0, 0));
    // IsFocusEngagementEnabled intentionally NOT set on ScrollViewer — only set on the
    // inner TextBox to avoid requiring two A-presses to engage text input on Xbox.

    StackPanel bodyStack;

    // Optional debug dashboard: collapsed by default so the chat layout is
    // unchanged until Settings (or debug-dashboard.flag) opts in.
    m_debugDashboardPanel = Border();
    m_debugDashboardPanel.Visibility(Visibility::Collapsed);
    m_debugDashboardPanel.Margin(ThicknessHelper::FromLengths(0, 0, 0, 16));
    m_debugDashboardPanel.Padding(ThicknessHelper::FromLengths(16, 12, 16, 12));
    m_debugDashboardPanel.BorderThickness(ThicknessHelper::FromLengths(1, 1, 1, 1));
    m_debugDashboardPanel.BorderBrush(
        winrt::Windows::UI::Xaml::Media::SolidColorBrush({255, 70, 90, 100}));
    m_debugDashboardPanel.Background(
        winrt::Windows::UI::Xaml::Media::SolidColorBrush({200, 14, 18, 24}));
    m_debugDashboardText = TextBlock();
    m_debugDashboardText.FontFamily(winrt::Windows::UI::Xaml::Media::FontFamily(L"Consolas"));
    m_debugDashboardText.FontSize(16);
    m_debugDashboardText.TextWrapping(TextWrapping::Wrap);
    m_debugDashboardText.Text(L"");
    m_debugDashboardPanel.Child(m_debugDashboardText);
    bodyStack.Children().Append(m_debugDashboardPanel);

    m_promptInput = TextBox();
    m_promptInput.PlaceholderText(L"Type your prompt here (gamepad A → opens keyboard)...");
    m_promptInput.AcceptsReturn(true);
    m_promptInput.TextWrapping(TextWrapping::Wrap);
    m_promptInput.MinHeight(120);
    m_promptInput.IsSpellCheckEnabled(false);
    m_promptInput.FontSize(18);
    // Focus returns here after every generation (SetRunning), which is right for
    // the pad — the next prompt is one A press away. What was not intended is
    // the side effect: on Xbox an editable TextBox receiving focus opens the
    // on-screen keyboard, so the keyboard reappeared over the answer the user
    // had just waited for, and stayed for the rest of the session.
    //
    // Found by looking at demo footage rather than by using the app: the
    // keyboard covers the UI from the first completed answer onward, in half of
    // the captured video and over the image viewer in a Store screenshot.
    // Pressing A still opens it, because that is user interaction, not
    // programmatic focus.
    no_keyboard_on_programmatic_focus(m_promptInput);
    m_promptInput.IsFocusEngagementEnabled(true);
    {
        using namespace winrt::Windows::UI::Xaml::Input;
        InputScopeName sname;
        sname.NameValue(InputScopeNameValue::Chat);
        InputScope scope;
        scope.Names().Append(sname);
        m_promptInput.InputScope(scope);
    }

    // RichTextBlock: O(1) per-token append via Paragraph.Inlines (vs O(n²) TextBlock.Text)
    m_outputBody = RichTextBlock();
    m_outputBody.TextWrapping(TextWrapping::Wrap);
    m_outputBody.FontFamily(winrt::Windows::UI::Xaml::Media::FontFamily(L"Consolas"));
    m_outputBody.FontSize(20);
    m_outputBody.Margin(ThicknessHelper::FromLengths(0, 16, 0, 0));
    m_outputBody.IsTextSelectionEnabled(true);
    m_currentParagraph = winrt::Windows::UI::Xaml::Documents::Paragraph();
    m_outputBody.Blocks().Append(m_currentParagraph);

    bodyStack.Children().Append(m_promptInput);
    bodyStack.Children().Append(m_outputBody);
    m_outputScroll.Content(bodyStack);

    // ---- Row 2: footer (metrics + buttons) ----
    Grid footer;
    Grid::SetRow(footer, 2);
    footer.Margin(ThicknessHelper::FromLengths(0, 16, 0, 0));

    // Two rows: metrics above, buttons below. Sharing one row, the icon buttons
    // left ~135 px for the metrics, which truncated "0.1s to first token" (seen
    // on-console).
    RowDefinition metricsRow;
    metricsRow.Height(GridLengthHelper::Auto());
    RowDefinition buttonsRow;
    buttonsRow.Height(GridLengthHelper::Auto());
    footer.RowDefinitions().Append(metricsRow);
    footer.RowDefinitions().Append(buttonsRow);

    m_metricsText = TextBlock();
    m_metricsText.FontSize(16);
    m_metricsText.Opacity(0.85);
    m_metricsText.Margin(ThicknessHelper::FromLengths(0, 0, 0, 8));
    m_metricsText.TextTrimming(TextTrimming::CharacterEllipsis);
    Grid::SetRow(m_metricsText, 0);

    StackPanel btnPanel;
    btnPanel.Orientation(Orientation::Horizontal);
    btnPanel.HorizontalAlignment(HorizontalAlignment::Right);
    Grid::SetRow(btnPanel, 1);

    // Segoe MDL2 Assets glyphs (system font on Xbox) instead of ASCII stand-ins.
    auto icon_label = [](wchar_t glyph, wchar_t const* label) {
        StackPanel content;
        content.Orientation(Orientation::Horizontal);
        FontIcon icon;
        icon.FontFamily(winrt::Windows::UI::Xaml::Media::FontFamily(L"Segoe MDL2 Assets"));
        icon.Glyph(std::wstring(1, glyph));
        icon.FontSize(18);
        icon.Margin(ThicknessHelper::FromLengths(0, 0, 10, 0));
        TextBlock text;
        text.Text(label);
        text.VerticalAlignment(VerticalAlignment::Center);
        content.Children().Append(icon);
        content.Children().Append(text);
        return content;
    };

    m_settingsButton = Button();
    m_settingsButton.Content(icon_label(0xE713, L"Settings"));
    m_settingsButton.MinWidth(100);
    m_settingsButton.Margin(ThicknessHelper::FromLengths(0, 0, 12, 0));

    m_newChatButton = Button();
    m_newChatButton.Content(icon_label(0xE710, L"New"));
    m_newChatButton.MinWidth(100);
    m_newChatButton.Margin(ThicknessHelper::FromLengths(0, 0, 12, 0));

    m_historyButton = Button();
    m_historyButton.Content(icon_label(0xE81C, L"History"));
    m_historyButton.MinWidth(100);
    m_historyButton.Margin(ThicknessHelper::FromLengths(0, 0, 12, 0));

    m_imageButton = Button();
    m_imageButton.Content(icon_label(0xEB9F, L"Image"));
    m_imageButton.MinWidth(100);
    m_imageButton.Margin(ThicknessHelper::FromLengths(0, 0, 12, 0));

    m_runButton = Button();
    m_runButton.Content(icon_label(0xE768, L"Run"));
    m_runButton.MinWidth(120);
    m_runButton.Margin(ThicknessHelper::FromLengths(0, 0, 12, 0));
    m_runButton.IsEnabled(false); // disabled until EnsureModelAsync confirms model is ready

    m_cancelButton = Button();
    // E711 (Cancel, an X): the Stop square E71A read as an empty checkbox on TV.
    m_cancelButton.Content(icon_label(0xE711, L"Cancel"));
    m_cancelButton.IsEnabled(false);
    m_cancelButton.MinWidth(120);

    btnPanel.Children().Append(m_settingsButton);
    btnPanel.Children().Append(m_newChatButton);
    btnPanel.Children().Append(m_historyButton);
    btnPanel.Children().Append(m_imageButton);
    btnPanel.Children().Append(m_runButton);
    btnPanel.Children().Append(m_cancelButton);

    footer.Children().Append(m_metricsText);
    footer.Children().Append(btnPanel);

    // ---- wire grid ----
    outerGrid.Children().Append(header);
    outerGrid.Children().Append(m_outputScroll);
    outerGrid.Children().Append(footer);

    m_root.Content(outerGrid);

    // Dark theme fallback on desktop (Xbox inherits from Application)
    m_root.RequestedTheme(ElementTheme::Dark);
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

MainPageController::MainPageController() {
    BuildUI();
}

// ---------------------------------------------------------------------------
// Init — must be called once after make_shared (shared_from_this is valid here)
// ---------------------------------------------------------------------------

void MainPageController::Init() {
    auto self = weak_from_this();
    m_runButton.Click([self](IInspectable const&, RoutedEventArgs const&) {
        if (auto s = self.lock()) {
            auto prompt = s->m_promptInput.Text();
            if (prompt.empty()) {
                s->SetStatus(L"Enter a prompt first");
                return;
            }
            s->m_promptInput.Text(L""); // clear immediately so the user sees the send action
            s->StartInference(std::wstring(prompt.c_str()));
        }
    });
    m_cancelButton.Click([self](IInspectable const&, RoutedEventArgs const&) {
        if (auto s = self.lock())
            s->OnCancelClick(nullptr, RoutedEventArgs{});
    });
    m_settingsButton.Click([self](IInspectable const&, RoutedEventArgs const&) {
        if (auto s = self.lock())
            s->ShowSettings();
    });
    m_newChatButton.Click([self](IInspectable const&, RoutedEventArgs const&) {
        if (auto s = self.lock())
            s->NewChat();
    });
    m_historyButton.Click([self](IInspectable const&, RoutedEventArgs const&) {
        if (auto s = self.lock())
            s->ShowHistory();
    });
    m_imageButton.Click([self](IInspectable const&, RoutedEventArgs const&) {
        if (auto s = self.lock())
            s->ShowImageDialog();
    });

    // B button (gamepad Back). On Xbox an unhandled BackRequested is the shell's
    // cue to suspend the app and return to Home, so a B press anywhere outside a
    // ContentDialog used to drop the user out of a running chat with no warning.
    // We therefore mark EVERY BackRequested handled: B cancels a running job and
    // otherwise does nothing. Leaving the app stays on the Xbox (Guide) button,
    // which the shell owns and we cannot intercept.
    // Do not "simplify" this by only handling the running case — that reinstates
    // the accidental exit.
    // Cancelling goes through OnCancelClick and must keep going through it:
    // SetRunning() is called by inference, image generation AND on-device
    // training, so m_is_running says "a job is running", not "text is running".
    // Aborting inline here would set m_abort — which only the text loop reads —
    // and disable the Cancel button while an image or an epoch kept going.
    auto nav = winrt::Windows::UI::Core::SystemNavigationManager::GetForCurrentView();
    nav.BackRequested(
        [self](IInspectable const&, winrt::Windows::UI::Core::BackRequestedEventArgs const& e) {
            e.Handled(true); // suppress the shell exit even if the page is gone
            if (auto s = self.lock())
                s->OnCancelClick(nullptr, RoutedEventArgs{});
        });

    // Gamepad keys: View = clear output, Y = jump to prompt
    m_root.KeyDown(
        [self](IInspectable const&, winrt::Windows::UI::Xaml::Input::KeyRoutedEventArgs const& e) {
            auto s = self.lock();
            if (!s)
                return;
            using VK = winrt::Windows::System::VirtualKey;
            switch (e.Key()) {
            case VK::GamepadView:
                s->m_outputBody.Blocks().Clear();
                s->m_currentParagraph = winrt::Windows::UI::Xaml::Documents::Paragraph();
                s->m_outputBody.Blocks().Append(s->m_currentParagraph);
                s->m_metricsText.Text(L"");
                s->m_last_inference.reset();
                s->m_live_decode_tok_s.reset();
                s->m_live_tokens.reset();
                if (s->m_debug_dashboard)
                    s->RefreshDebugDashboard();
                s->SetStatus(L"Ready");
                e.Handled(true);
                break;
            case VK::GamepadY:
                s->m_promptInput.Focus(FocusState::Programmatic);
                e.Handled(true);
                break;
            default:
                break;
            }
        });

    // Smart autoscroll: track whether the user is at the bottom of the scroll view.
    // Auto-scroll is suppressed while the user has manually scrolled up during streaming.
    m_outputScroll.ViewChanged(
        [self](IInspectable const&,
               winrt::Windows::UI::Xaml::Controls::ScrollViewerViewChangedEventArgs const&) {
            if (auto s = self.lock()) {
                double sv = s->m_outputScroll.ScrollableHeight();
                double vo = s->m_outputScroll.VerticalOffset();
                s->m_at_bottom = (sv - vo < 24.0);
            }
        });

    // Start with focus on Run button
    m_runButton.Focus(FocusState::Programmatic);

    // Init chat history and settings
    {
        auto folder = winrt::Windows::Storage::ApplicationData::Current().LocalFolder();
        std::string chats_dir =
            ::xllama::wstring_to_utf8(std::wstring(folder.Path().c_str()) + L"\\chats");
        m_history.SetDir(chats_dir);
        m_history.LoadIndex();
    }
    LoadSettings();
    ApplyDebugDashboard(m_debug_dashboard, /*persist_flag=*/false);

    EnsureModelAsync();
    ShowDisclaimerIfNeeded();
    #ifndef XLLAMA_STORE_SKU
    StartApiPanel();
    #endif
}

// ---------------------------------------------------------------------------
// UI helpers (must be called on UI thread)
// ---------------------------------------------------------------------------

void MainPageController::AppendOutput(std::wstring const& text) {
    using namespace winrt::Windows::UI::Xaml::Documents;
    // Split on '\n': each segment becomes a Run; newlines become LineBreak inlines so
    // multi-line model output is rendered correctly in RichTextBlock.
    std::wstring seg;
    for (wchar_t c : text) {
        if (c == L'\n') {
            if (!seg.empty()) {
                Run r;
                r.Text(seg);
                m_currentParagraph.Inlines().Append(r);
                seg.clear();
            }
            m_currentParagraph.Inlines().Append(LineBreak());
        } else {
            seg += c;
        }
    }
    if (!seg.empty()) {
        Run r;
        r.Text(seg);
        m_currentParagraph.Inlines().Append(r);
    }
    if (m_at_bottom)
        m_outputScroll.ChangeView(nullptr, m_outputScroll.ScrollableHeight(), nullptr);
}

void MainPageController::FlushTokenBuffer() {
    std::string batch;
    {
        std::lock_guard<std::mutex> lk(m_token_mutex);
        batch = std::move(m_token_buffer);
    }
    if (!batch.empty())
        AppendOutput(::xllama::utf8_to_wstring(batch));

    // Live tok/s counter. Measured from the FIRST TOKEN, not from turn start:
    // m_gen_start precedes model load and prefill, and on a long prompt prefill
    // is several seconds, so dividing by it under-reported the decode rate by
    // more than half (#130). n - 1 tokens have been produced since the stamp.
    int n = m_tokens_received.load();
    if (n > 0 && !m_status_flipped_to_generating) {
        m_status_flipped_to_generating = true;
        SetStatus(L"generating", StatusKind::Working);
    }
    if (n > 1 && m_first_token_seen.load(std::memory_order_acquire)) {
        double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - m_first_token_at)
                .count();
        if (elapsed > 0.2) {
            wchar_t buf[80];
            swprintf_s(buf, L"~%.0f tok/s  ·  %d tok", (n - 1) / elapsed, n);
            m_metricsText.Text(buf);
            m_live_decode_tok_s = (n - 1) / elapsed;
            m_live_tokens = n;
            if (m_debug_dashboard)
                RefreshDebugDashboard();
        }
    }
}

void MainPageController::SetStatus(std::wstring const& msg, StatusKind kind) {
    namespace Media = winrt::Windows::UI::Xaml::Media;
    switch (kind) {
    case StatusKind::Working:
        m_statusText.Foreground(Media::SolidColorBrush({255, 80, 190, 255})); // accent blue
        m_statusText.Opacity(1.0);
        m_statusText.Text(L">> " + msg);
        break;
    case StatusKind::Success:
        m_statusText.Foreground(Media::SolidColorBrush({255, 100, 220, 100})); // green
        m_statusText.Opacity(1.0);
        m_statusText.Text(L"> " + msg);
        break;
    case StatusKind::Error:
        m_statusText.Foreground(Media::SolidColorBrush({255, 240, 80, 70})); // red
        m_statusText.Opacity(1.0);
        m_statusText.Text(L"! " + msg);
        break;
    default: // Info
        m_statusText.ClearValue(TextBlock::ForegroundProperty());
        m_statusText.Opacity(0.85);
        m_statusText.Text(msg);
        break;
    }
}

void MainPageController::SetRunning(bool running) {
    m_is_running.store(running);
    m_runButton.IsEnabled(!running);
    m_cancelButton.IsEnabled(running);
    m_loadingBar.Visibility(running ? Visibility::Visible : Visibility::Collapsed);
    if (!running) {
        m_at_bottom = true; // re-enable autoscroll for next generation
        if (m_flush_timer && m_flush_timer.IsEnabled()) {
            m_flush_timer.Stop();
            FlushTokenBuffer(); // drain any remaining tokens
        }
        m_promptInput.Focus(FocusState::Programmatic); // return focus to prompt after inference
    }
}

// ---------------------------------------------------------------------------
// Multi-turn chat helpers
// ---------------------------------------------------------------------------

void MainPageController::AddUserParagraph(std::wstring const& text) {
    using namespace winrt::Windows::UI::Xaml::Documents;
    namespace Media = winrt::Windows::UI::Xaml::Media;

    // "You:" label in bold
    Paragraph p;
    Run label;
    label.Text(L"You: ");
    label.FontWeight(winrt::Windows::UI::Text::FontWeights::Bold());
    p.Inlines().Append(label);
    Run content;
    content.Text(text);
    p.Inlines().Append(content);
    m_outputBody.Blocks().Append(p);

    // Empty separator paragraph
    m_outputBody.Blocks().Append(Paragraph());

    // "Assistant:" label paragraph (streaming will fill inline content)
    m_currentParagraph = Paragraph();
    Run alabel;
    alabel.Text(L"Assistant: ");
    alabel.FontWeight(winrt::Windows::UI::Text::FontWeights::Bold());
    m_currentParagraph.Inlines().Append(alabel);
    m_outputBody.Blocks().Append(m_currentParagraph);
}

MainPageController::PromptPlan
MainPageController::BuildPromptPlan(const std::string& user_text) const {
    // The whole conversation, oldest first, as (user, assistant) exchanges plus
    // the new user_text. What fits n_ctx is fit_prompt's decision, in the worker,
    // in tokens: a chars-per-token estimate may size the prompt for routing but
    // must not decide what the user receives (AGENTS.md invariants).
    PromptPlan plan;
    plan.n_ctx = ::xllama::kDefaultNCtx;
    std::string role;
    {
        const auto& manifest = CachedManifest();
        if (const auto* e = ::xllama::FindManifestEntry(manifest, m_model_filename)) {
            plan.n_ctx = ::xllama::catalogue_n_ctx(e->n_ctx, e->kind == L"gguf");
            role = ::xllama::wstring_to_utf8(e->role);
        }
    }
    plan.chars_per_token = ::xllama::chars_per_token_for_role(role);
    std::size_t chars = m_system_prompt.size() + user_text.size();
    for (size_t i = 0; i < m_current.messages.size(); ++i) {
        if (m_current.messages[i].role != xllama::ui::MessageRole::User)
            continue;
        std::string assistant;
        if (i + 1 < m_current.messages.size() &&
            m_current.messages[i + 1].role == xllama::ui::MessageRole::Assistant)
            assistant = m_current.messages[i + 1].content;
        chars += m_current.messages[i].content.size() + assistant.size();
        plan.turns.push_back({m_current.messages[i].content, std::move(assistant)});
    }
    plan.estimated_tokens = ::xllama::estimate_tokens_from_chars(chars, plan.chars_per_token);
    plan.prompt = chat_format().render_prompt(m_system_prompt, plan.turns, user_text);
    return plan;
}

xllama::ChatFormat MainPageController::chat_format() const {
    return ::xllama::chat_format_for(::xllama::wstring_to_utf8(m_model_filename));
}

const std::vector<::xllama::ManifestEntry>& MainPageController::CachedManifest() const {
    if (!m_manifest_cached) {
        m_manifest_cache = ::xllama::LoadModelManifest();
        m_manifest_cached = true;
    }
    return m_manifest_cache;
}

std::string MainPageController::BuildDeltaPrompt(const std::string& user_text) const {
    // The persistent KV cache already holds everything through the previous
    // assistant's generated tokens. The chat format closes that turn (only a
    // newline if the model already emitted the stop token; the full turn close
    // otherwise) and appends the new user turn + assistant header. Concatenated
    // onto the KV this reproduces exactly what BuildPrompt would have built.
    return chat_format().render_delta(user_text, m_kv_last_ended_with_stop);
}

// #170b: the KV of the conversation being left is worth ~12 KiB per resident
// token of prefill we would otherwise repeat on return (measured: a 1476-token
// conversation is a 17.6 MB file that loads in 33 ms against a 4.5 s
// re-prefill on console). Written on a detached thread — the UI thread must
// not wait on tens of MB — under the hub lock, and only while the resident
// session is still the one this conversation was built on.
//
// #216: a fire-and-forget save races the next turn. The worker holds hub.mtx
// for the whole generate, so if "Say hello" on a new chat starts before the
// snapshot finishes, save_state can capture the *new* conversation's KV and
// write it to the *old* conversation's path. On return, load_state succeeds
// (log: restored) but the #170a prefix never matches → full re-prefill (ret ≈
// cold). The gate under `all` surfaces this ~1/6; standalone rarely does.
// Fix: one outstanding save future; every generate waits before touching the
// session. Disk I/O still stays off the UI thread.
void MainPageController::WaitKvSnapshotSave() {
    std::future<void> f;
    {
        std::lock_guard<std::mutex> lk(m_kv_save_mutex);
        if (m_kv_save_future.valid())
            f = std::move(m_kv_save_future);
    }
    if (f.valid())
        f.wait();
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

void MainPageController::SaveKvSnapshotAsync() {
    if (!m_kv_reuse || !m_kv_valid || m_current.messages.empty())
        return;
    // Thinking models: the saved history holds the STRIPPED answer while the KV
    // holds the full <think> stream, so the #170a prefix diff always diverges
    // inside the first assistant turn on return — and on LFM's hybrid cache a
    // tail rewind is refused, collapsing to a full re-prefill. The snapshot
    // would cost tens of MB of writes to buy nothing.
    if (chat_format().strip_thinking_content)
        return;
    const std::string routed =
        ::xllama::wstring_to_utf8(m_active_model.empty() ? m_model_filename : m_active_model);
    if (!::xllama::model_uses_llama_backend(routed))
        return; // ORT keeps no serialisable per-sequence state
    CreateDirectoryW(local_wpath(L"kv").c_str(), nullptr);
    const auto store = kv_store();
    const std::string path = store.path_for(m_current.id);
    if (path.empty())
        return;
    const uint64_t epoch = m_kv_epoch.load();
    // Finish any previous save before scheduling another (same path possible).
    WaitKvSnapshotSave();
    {
        std::lock_guard<std::mutex> lk(m_kv_save_mutex);
        m_kv_save_future = std::async(std::launch::async, [store, path, epoch]() {
            auto& hub = ::xllama::session_hub();
            std::lock_guard<std::mutex> hub_lk(hub.mtx);
            if (!hub.session || hub.session->kv_epoch() != epoch)
                return; // the resident KV is not this conversation's any more
                        // (the API, a reload or another model) — nothing to save
            std::string err;
            // A snapshot larger than the whole pool would be pruned right after
            // it was written: refuse it before the state is copied (#570).
            if (hub.session->save_state(path, &err,
                                        static_cast<std::size_t>(::xllama::kKvStoreMaxBytes)))
                store.prune(::xllama::kKvStoreMaxFiles, ::xllama::kKvStoreMaxBytes);
            else
                ::xllama::log_output("[xllama] KV snapshot not saved: " + err + "\n");
        });
    }
}

winrt::fire_and_forget MainPageController::ShowDisclaimerIfNeeded() {
    auto self = shared_from_this();
    if (std::filesystem::exists(local_wpath(L"disclaimer.accepted")))
        co_return;

    // Yield once so XamlRoot is attached after Window.Content/Activate.
    co_await winrt::resume_foreground(m_root.Dispatcher());

    winrt::Windows::UI::Xaml::Controls::TextBlock body;
    body.TextWrapping(winrt::Windows::UI::Xaml::TextWrapping::Wrap);
    body.Text(L"xllama runs language and image models entirely on this device. "
              L"Generated text and images can be inaccurate, biased, or inappropriate. "
              L"Nothing leaves the console except model downloads you start. "
              L"This project is not affiliated with Microsoft. "
              L"Report inappropriate generated content from Settings.");
    body.MaxWidth(520);

    winrt::Windows::UI::Xaml::Controls::ContentDialog dlg;
    dlg.Title(winrt::box_value(L"Before you start"));
    dlg.Content(body);
    dlg.PrimaryButtonText(L"I understand");
    dlg.XamlRoot(m_root.XamlRoot());
    ApTrackDialog(dlg);
    co_await dlg.ShowAsync();
    ::xllama::write_local_text("disclaimer.accepted", "1\n");
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
        const bool skip_early = skip_provision_early_exits(set_app_ready, marker);

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
            if (bundled) {
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
            bool usb_found = false;
            try {
                using winrt::Windows::Storage::KnownFolders;
                using winrt::Windows::Storage::StorageFolder;
                auto removable = KnownFolders::RemovableDevices();
                auto drives = co_await removable.GetFoldersAsync();
                for (auto const& drive : drives) {
                    std::wstring drive_path(drive.Path().c_str());
                    // StorageFolder.Path may include trailing backslash (e.g. "E:\"); strip it.
                    while (!drive_path.empty() && drive_path.back() == L'\\')
                        drive_path.pop_back();
                    log_output(
                        ("[xllama] USB probe: " + ::xllama::wstring_to_utf8(drive_path) + "\n")
                            .c_str());
                    // Use WinRT TryGetItemAsync — GetFileAttributesW is blocked by
                    // AppContainer even with removableStorage capability.
                    try {
                        using winrt::Windows::Storage::IStorageItem;
                        auto sub = winrt::hstring(L"xllama\\models\\") + model_name +
                                   winrt::hstring(L"\\genai_config.json");
                        auto item = co_await drive.TryGetItemAsync(sub);
                        if (item) {
                            log_output(("[xllama] USB model found on " +
                                        ::xllama::wstring_to_utf8(drive_path) + "\n")
                                           .c_str());
                            // Cache USB root (drive_path without trailing \) for
                            // resolve_model_path() sync path.
                            ::xllama::write_local_text("usb_model_root.txt",
                                                       ::xllama::wstring_to_utf8(drive_path));
                            usb_found = true;
                        }
                    } catch (...) {
                        // drive not accessible — skip
                    }
                    if (usb_found)
                        break;
                }
            } catch (...) {
                log_output("[xllama] USB probe: RemovableDevices enumeration failed\n");
            }
            if (usb_found) {
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
                    using winrt::Windows::Storage::KnownFolders;
                    using winrt::Windows::Storage::StorageFolder;

                    // Source: USB xllama\models\<name>
                    auto removable2 = KnownFolders::RemovableDevices();
                    auto drives2 = co_await removable2.GetFoldersAsync();
                    StorageFolder usb_model_folder{nullptr};
                    for (auto const& d : drives2) {
                        auto candidate = co_await d.TryGetItemAsync(
                            winrt::hstring(L"xllama\\models\\") + model_name);
                        if (candidate) {
                            usb_model_folder = candidate.as<StorageFolder>();
                            break;
                        }
                    }
                    if (!usb_model_folder)
                        throw std::runtime_error("USB folder disappeared");

                    std::error_code begin_ec;
                    if (!writer.begin(begin_ec))
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

                    // Copy each file
                    auto files = co_await usb_model_folder.GetFilesAsync();
                    for (auto const& f : files) {
                        if (f.Name() == L".complete")
                            continue; // publish only after all files have been copied
                        log_output(("[xllama] USB copy: " +
                                    ::xllama::wstring_to_utf8(std::wstring(f.Name().c_str())) +
                                    "\n")
                                       .c_str());
                        if (normalize_model_path(std::wstring(f.Name().c_str())) ==
                                L"adapter.gguf" &&
                            co_await dest_folder.TryGetItemAsync(f.Name()))
                            continue; // a local personalization adapter belongs to its owner
                        co_await f.CopyAsync(dest_folder, f.Name(),
                                             NameCollisionOption::ReplaceExisting);
                    }
                    std::error_code complete_ec;
                    if (!writer.complete({}, complete_ec))
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
    auto& hub = ::xllama::session_hub();
    if (hub.session && hub.model == model && ::xllama::same_load(hub.params, sp)) {
        m_turn_session = hub.session.get();
        return true;
    }
    m_turn_session = nullptr;
    m_kv_valid = false; // new session object → no reusable KV carries over
    std::string err;
    ::xllama::Session* s = ::xllama::session_hub().ensure_locked(model, sp, &err);
    if (!s) {
        if (err_out)
            *err_out = err;
        return false;
    }
    m_turn_session = s;
    return true;
}

// ---------------------------------------------------------------------------
// StartInference: called on UI thread; spawns background thread
// ---------------------------------------------------------------------------

bool MainPageController::StartInference(std::wstring const& prompt_w) {
    auto compute_permit = ::xllama::try_acquire_compute_job();
    if (!compute_permit && !::xllama::session_hub().preloading.load(std::memory_order_acquire)) {
        m_promptInput.Text(winrt::hstring(prompt_w.c_str()));
        SetStatus(L"Busy — another compute job is active", StatusKind::Error);
        return false;
    }
    m_abort.store(false);
    SetStatus(L"Loading model...", StatusKind::Working);
    SetRunning(true);

    // Add user message to display + prepare empty assistant paragraph
    AddUserParagraph(prompt_w);
    m_metricsText.Text(L"");
    m_live_decode_tok_s.reset();
    m_live_tokens.reset();

    // Reset streaming counters
    m_tokens_received.store(0);
    {
        std::lock_guard<std::mutex> lk(m_token_mutex);
        m_token_buffer.clear();
    }
    m_gen_start = std::chrono::steady_clock::now();
    m_first_token_seen.store(false);
    m_status_flipped_to_generating = false;

    // Start flush timer (40 ms tick: batches tokens, updates live tok/s)
    auto self = shared_from_this();
    if (!m_flush_timer) {
        m_flush_timer = winrt::Windows::UI::Xaml::DispatcherTimer{};
        m_flush_timer.Interval(std::chrono::milliseconds(40));
    } else {
        m_flush_timer.Stop();
        m_flush_timer.Tick(m_flush_tick_token); // revoke previous handler
    }
    auto weak_self = std::weak_ptr<MainPageController>(self);
    m_flush_tick_token = m_flush_timer.Tick([weak_self](IInspectable const&, IInspectable const&) {
        if (auto s = weak_self.lock())
            s->FlushTokenBuffer();
    });
    m_flush_timer.Start();

    // Build multi-turn ChatML prompt from conversation history.
    // BuildPrompt uses existing m_current.messages (prev turns) and appends user_text.
    std::string user_text = ::xllama::wstring_to_utf8(prompt_w);
    // Two independent conditions, and tying them together cost every
    // conversation its title: NewChat() already assigns an id (it needs one for
    // the KV snapshot path), so `id.empty()` is false by the time the first
    // message is sent and the title was never derived. Only a conversation that
    // reached here with no id at all — the very first one after a cold launch —
    // ever got one. Measured on console: 14 of 14 saved conversations had
    // "title":"", and History rendered them as "(2 msgs) • today 13:20".
    if (m_current.id.empty())
        m_current.id = xllama::ui::ChatHistory::NewId();
    if (m_current.title.empty())
        m_current.title = xllama::ui::ChatHistory::TitleFrom(user_text);
    const PromptPlan plan = BuildPromptPlan(user_text);
    std::string full_prompt = plan.prompt;

    // Is the base model a GGUF (llama.cpp backend)? On Xbox llama.cpp is CPU-only
    // (no ggml GPU backend), so EP routing is meaningless and stays gated off for
    // GGUF (below). KV-cache reuse, however, IS supported: LlamaSession keeps a
    // persistent llama_context, so a reuse turn appends only the delta (turn-2
    // prefill 4.07×). Reuse is gated by kv_reuse_supported_for_model(), not by
    // this flag.
    bool base_is_gguf = false;
    {
        // Cached: this runs on the UI thread on every turn, and the manifest
        // JSON only changes on a personalized-model publish (which invalidates).
        const auto& manifest = CachedManifest();
        const auto* e = ::xllama::FindManifestEntry(manifest, m_model_filename);
        base_is_gguf = e && e->kind == L"gguf";
    }

    // Stage 3: decide EP routing once per conversation (sticky — the KV cache is
    // per-EP). m_active_model is cleared on new/loaded chat, so this fires on the
    // first turn and stays fixed after. Default (m_routing==0) keeps the CPU model.
    if (m_active_model.empty()) {
        // Filesystem probe (LocalState + InstalledPath stats) — only the sticky
        // first-turn decision consumes it, so keep it off the per-turn path (#174).
        // GPU is available only when every weight/data pin classifies Ok on a
        // size-only listing (`weight_pins_verified`), the LocalState `.complete`
        // marker exists (hashes ran at download/promotion), and no model write
        // is in flight. Licence and notice rows do not participate. The loose
        // "any ONNX/GGUF exists" probe used to route a long turn at a
        // still-downloading DML tree (#370).
        const std::wstring gpu_name = ::xllama::utf8_to_wstring(m_gpu_model);
        auto local_folder = winrt::Windows::Storage::ApplicationData::Current().LocalFolder();
        const std::wstring local_gpu_dir =
            std::wstring(local_folder.Path().c_str()) + L"\\models\\" + gpu_name;
        std::vector<PresentFile> gpu_present;
        std::vector<ManifestFilePin> gpu_pins;
        {
            const auto& manifest = CachedManifest();
            if (const auto* e = ::xllama::FindManifestEntry(manifest, gpu_name)) {
                gpu_present.reserve(e->files.size());
                gpu_pins.reserve(e->files.size());
                for (const auto& f : e->files) {
                    uint64_t sz = 0;
                    std::error_code ec;
                    const auto p = std::filesystem::path(local_gpu_dir) / f.filename;
                    if (std::filesystem::is_regular_file(p, ec)) {
                        const auto n = std::filesystem::file_size(p, ec);
                        if (!ec)
                            sz = static_cast<uint64_t>(n);
                    }
                    gpu_present.push_back({f.filename, sz, {}});
                    gpu_pins.push_back(make_file_pin(f.filename, f.approx_bytes,
                                                     ::xllama::wstring_to_utf8(f.sha256),
                                                     ::xllama::wstring_to_utf8(f.role)));
                }
            }
        }
        const bool expected_verified = ::xllama::gpu_tree_ready_for_routing(
            ::xllama::weight_pins_verified(gpu_present, gpu_pins),
            ::xllama::ModelDownloader::IsComplete(local_gpu_dir));
        const bool write_in_progress = ::xllama::ModelDownloader::WriterGate().busy();
        const auto gpu_state = ::xllama::gpu_route_readiness(expected_verified, write_in_progress);
        const bool gpu_provisioned =
            ::xllama::gpu_ready_for_routing(expected_verified, write_in_progress);
        const bool loose_present =
            expected_verified ? true : ::xllama::IsModelProvisioned(gpu_name);
        ::xllama::RoutingSettings rs;
        rs.mode = static_cast<::xllama::RoutingMode>(m_routing);
        rs.cpu_model = ::xllama::wstring_to_utf8(m_model_filename);
        rs.gpu_model = m_gpu_model;

        int n_tok = 0;
        if (m_routing == 2 && !base_is_gguf) {
            // Exact count only when the CPU session is already resident (a
            // name-compare no-op). The previous shape loaded the CPU model
            // just to tokenize — ON THE UI THREAD — and then could tear it
            // down for the DML model (+1.2 s, uwp-constraints §5, 2786 ms vs
            // 1570 ms). With no session resident the role's chars-per-token
            // estimate sizes it. Either way routing sees at most the context
            // ceiling, which is what fit_prompt will prefill (#133).
            // try_lock: if the hub is busy (LAN API mid-turn) fall back to the
            // estimator rather than blocking the UI thread.
            auto& hub = ::xllama::session_hub();
            std::unique_lock<std::mutex> hub_lk(hub.mtx, std::try_to_lock);
            const int counted = (hub_lk.owns_lock() && hub.session && hub.model == rs.cpu_model)
                                    ? hub.session->count_tokens(full_prompt)
                                    : ::xllama::estimate_tokens_from_chars(full_prompt.size(),
                                                                           plan.chars_per_token);
            n_tok = ::xllama::routing_prompt_tokens(counted, plan.n_ctx);
        }

        // #91/#95: for a gpu_model that is not a parity-validated DML text asset,
        // decide_routing resolves every mode to the CPU model and the GPU model is
        // intentionally not provisioned — a missing gpu_model must not block the
        // turn or nag the user. A validated asset that is only partial or mid-write
        // is not "missing": fall through to CPU via gpu_provisioned == false.
        const bool warn_gpu_missing =
            ::xllama::dml_text_model_ok(m_gpu_model) && !loose_present && !write_in_progress;
        if (gpu_state != ::xllama::GpuRouteReadiness::Ready &&
            ::xllama::dml_text_model_ok(m_gpu_model) && !warn_gpu_missing &&
            (m_routing == 1 || (m_routing == 2 && !base_is_gguf && n_tok > rs.token_threshold))) {
            ::xllama::log_output(::xllama::gpu_route_fallback_log(gpu_state));
        }
        if (m_routing == 1 && warn_gpu_missing) {
            SetStatus(L"GPU model '" + ::xllama::utf8_to_wstring(m_gpu_model) +
                          L"' is not on this console.\n"
                          L"Upload it to LocalState\\models\\" +
                          ::xllama::utf8_to_wstring(m_gpu_model) +
                          L" via Device Portal or USB (see docs/model-selection.md).",
                      StatusKind::Error);
            SetRunning(false);
            return true;
        }
        if (m_routing == 2 && n_tok > rs.token_threshold && warn_gpu_missing) {
            SetStatus(L"GPU model '" + ::xllama::utf8_to_wstring(m_gpu_model) +
                          L"' is not on this console — staying on CPU.\n"
                          L"Upload it to LocalState\\models\\" +
                          ::xllama::utf8_to_wstring(m_gpu_model) + L" to enable auto-routing.",
                      StatusKind::Error);
            SetRunning(false);
            return true;
        }

        const auto decision = ::xllama::decide_routing(rs, n_tok, base_is_gguf, gpu_provisioned);
        m_active_model = ::xllama::utf8_to_wstring(decision.active_model);

        if (m_routing == 1) {
            ::xllama::log_output("[xllama] routing: gpu-only\n");
        } else if (m_routing == 2 && !base_is_gguf) {
            char rbuf[160];
            snprintf(rbuf, sizeof(rbuf),
                     "[xllama] routing: auto → %s (%d tok, threshold %d, model=%s)\n",
                     decision.use_gpu ? "gpu" : "cpu", decision.token_count, rs.token_threshold,
                     decision.active_model.c_str());
            ::xllama::log_output(rbuf);
        }
    }

    // KV-cache reuse decision (continuous decoding). Reuse only when enabled, the
    // persistent generator already holds this conversation (m_kv_valid), and the
    // conversation is not expected to outgrow the context this round (RewindTo
    // cannot drop from the head, so an overflowing delta forces a full re-prefill).
    // The estimate only decides whether to ATTEMPT reuse; a wrong guess costs the
    // fail-fast retry below, never the reply's content. A reuse turn appends only the delta;
    // otherwise we (re)prefill the full prompt — which also (re)seeds the persistent generator.
    const std::string routed_model =
        ::xllama::wstring_to_utf8(m_active_model.empty() ? m_model_filename : m_active_model);
    // KV reuse now works for both backends: ORT-GenAI (persistent generator) and
    // GGUF/llama.cpp (persistent llama_context in LlamaSession). ep_kv_ok still
    // excludes the DirectML routing model (continuous decoding unsupported there).
    const bool ep_kv_ok = ::xllama::kv_reuse_supported_for_model(routed_model);
    // #169: on the llama backend a trimmed round no longer forces a full
    // re-prefill — the session evicts the oldest resident tokens (context
    // shift) when the delta would overflow n_ctx, keeping long chats in the
    // reuse regime. If the arch cannot shift, the continuation fail-fasts
    // (#173) and the retry below re-seeds with the trimmed full prompt.
    const bool shift_capable = ::xllama::model_uses_llama_backend(routed_model);
    const bool expect_overflow =
        plan.estimated_tokens > ::xllama::max_prompt_tokens_for_n_ctx(plan.n_ctx);
    bool do_reuse = m_kv_reuse && m_kv_valid && (!expect_overflow || shift_capable) && ep_kv_ok;
    bool kv_reuse = m_kv_reuse && ep_kv_ok;
    std::string delta_prompt = do_reuse ? BuildDeltaPrompt(user_text) : std::string();

    // Record user message in history AFTER building prompt (avoids duplicate)
    {
        xllama::ui::ChatMessage umsg;
        umsg.role = xllama::ui::MessageRole::User;
        umsg.content = user_text;
        umsg.ts_unix = static_cast<int64_t>(std::time(nullptr));
        m_current.messages.push_back(std::move(umsg));
    }

    std::string model =
        ::xllama::wstring_to_utf8(m_active_model.empty() ? m_model_filename : m_active_model);
    auto dispatcher = m_root.Dispatcher();

    // Per-model chat format (stop sequences + output post-processing) — resolved
    // on the UI thread (chat_format() reads m_model_filename) and captured by value.
    xllama::ChatFormat fmt = chat_format();

    // #169: the pinned head a context shift must never evict — the system
    // block render_prompt puts before the first user turn. Rendered here
    // (m_system_prompt is UI-thread state) and tokenized per turn in the
    // worker, where the session is available.
    const std::string sys_prefix = fmt.render_system_prefix(m_system_prompt);

    // #170b: the snapshot to try when this conversation's KV is not resident.
    const std::string kv_path = kv_reuse ? kv_store().path_for(m_current.id) : std::string();

    // Copies for the worker: the surviving turns and the numbers fit_prompt needs.
    // m_system_prompt / m_n_predict are UI-thread state, read here.
    const std::vector<::xllama::ChatTurn> plan_turns = plan.turns;
    const std::string system_prompt = m_system_prompt;
    int plan_n_ctx = plan.n_ctx; // replaced by the loaded session's n_ctx in the worker
    const int plan_n_predict = m_n_predict;

    std::thread([self, full_prompt, delta_prompt, do_reuse, kv_reuse, ep_kv_ok, model, fmt,
                 sys_prefix, kv_path, dispatcher, plan_turns, user_text, system_prompt, plan_n_ctx,
                 plan_n_predict, compute_permit, prompt_w]() mutable {
        try {
            if (!compute_permit) {
                compute_permit = ::xllama::try_acquire_compute_job_while_preloading(
                    ::xllama::session_hub().preloading);
                if (!compute_permit) {
                    dispatcher.RunAsync(CoreDispatcherPriority::Normal, [self, prompt_w]() {
                        self->m_promptInput.Text(winrt::hstring(prompt_w.c_str()));
                        self->SetStatus(L"Busy — another compute job is active", StatusKind::Error);
                        self->SetRunning(false);
                    });
                    return;
                }
            }
            // #216: drain any #170b snapshot save before taking hub.mtx. The save
            // also locks the hub; waiting first keeps the leave-conversation
            // token list on the snapshot path instead of a later turn's KV.
            self->WaitKvSnapshotSave();

            // One hub lock for the whole turn: the resident session cannot be
            // swapped from under us (LAN API requests report busy meanwhile,
            // exactly as they do against each other).
            auto& hub = ::xllama::session_hub();
            std::unique_lock<std::mutex> hub_lk(hub.mtx, std::try_to_lock);
            if (!hub_lk.owns_lock()) {
                // Another job holds the single slot: a LAN request, a pre-load,
                // a KV snapshot save or a pull's model swap. Say which when we
                // can, instead of a silent stall, then wait for it.
                std::wstring waiting = L"Waiting for the model to be free...";
    #ifndef XLLAMA_STORE_SKU
                if (::xllama::api::activity().snapshot().in_flight)
                    waiting = L"Waiting for LAN API request to finish...";
    #endif
                dispatcher.RunAsync(CoreDispatcherPriority::Normal, [self, waiting]() {
                    self->SetStatus(waiting, StatusKind::Working);
                });
                hub_lk.lock();
                dispatcher.RunAsync(CoreDispatcherPriority::Normal, [self]() {
                    self->SetStatus(L"Loading model...", StatusKind::Working);
                });
            }
            std::string load_err;
            if (!self->EnsureSession(model, &load_err)) {
                dispatcher.RunAsync(CoreDispatcherPriority::Normal, [self, load_err]() {
                    self->SetStatus(::xllama::utf8_to_wstring("Load failed: " + load_err),
                                    StatusKind::Error);
                    self->SetRunning(false);
                });
                return;
            }
            // Budget against the context the resident session holds, not the
            // UI-thread manifest cache (a LocalState manifest override can leave
            // it stale) and not the requested n_ctx, which a GGUF session caps
            // at its trained context (#570).
            const int loaded_n_ctx = hub.session ? hub.session->context_length() : 0;
            plan_n_ctx = loaded_n_ctx > 0 ? loaded_n_ctx : hub.params.n_ctx;

            // #170b: this conversation's KV is not resident (switched away and
            // back, or the app was restarted). Restoring the snapshot does not
            // change the turn below — it stays a full-prompt turn, and the
            // #170a prefix diff is what turns the restored cache into a delta
            // prefill. A missing, stale or foreign snapshot just fails here and
            // the turn prefills as it always did.
            if (!do_reuse && !kv_path.empty()) {
                std::string kv_err;
                if (self->m_turn_session->load_state(kv_path, &kv_err))
                    ::xllama::log_output("[xllama] KV snapshot restored (#170b)\n");
            }

            auto on_status = [self, dispatcher](const std::string& s) {
                // The backends emit "generating" BEFORE prefill — nothing is
                // generated yet, and on a long prompt that is several seconds of
                // the user staring at a word that is not true. Call the phase
                // what it is; FlushTokenBuffer flips it on the first real token.
                auto ws = ::xllama::utf8_to_wstring(s == "generating" ? "reading prompt" : s);
                StatusKind k =
                    (s.rfind("error:", 0) == 0) ? StatusKind::Error : StatusKind::Working;
                dispatcher.RunAsync(CoreDispatcherPriority::Normal,
                                    [self, ws, k]() { self->SetStatus(ws, k); });
            };
            // Token accumulation — no per-token RunAsync dispatch (batched by flush timer)
            auto on_token = [self](std::string_view tok) {
                // Stamp prefill-end on the first token — the only place that
                // knows when the model stopped reading and started writing.
                // This runs on the inference thread; FlushTokenBuffer reads the
                // timestamp on the UI thread. Write the value FIRST, then
                // publish with a release store, so an acquire load of the flag
                // guarantees the timestamp is visible. (Relaxed here would be a
                // data race that x64 happens to hide.)
                if (!self->m_first_token_seen.load(std::memory_order_relaxed)) {
                    self->m_first_token_at = std::chrono::steady_clock::now();
                    self->m_first_token_seen.store(true, std::memory_order_release);
                }
                self->m_tokens_received.fetch_add(1, std::memory_order_relaxed);
                std::lock_guard<std::mutex> lk(self->m_token_mutex);
                self->m_token_buffer += tok;
            };

            auto run_turn = [&](const std::string& p, bool reuse, bool reset) {
                xllama::GenerateParams gp;
                gp.prompt = p;
                gp.n_predict = self->m_n_predict;
                gp.temperature = self->m_temperature;
                gp.top_p = self->m_top_p;
                gp.top_k = self->m_top_k;
                gp.repetition_penalty = self->m_repetition_penalty;
                gp.abort_flag = &self->m_abort;
                gp.stop_sequences = fmt.stop_sequences;
                gp.reuse_kv = reuse;
                gp.reset_kv = reset;
                // A continuation is only valid on the KV this conversation left;
                // the session refuses it otherwise and the caller retries in full.
                if (reuse && !reset)
                    gp.expected_kv_epoch = self->m_kv_epoch.load();
                // count_tokens matches generate()'s full-prompt tokenization
                // (BOS included), and the system block ends on a special
                // token, so the boundary is exact. Ignored unless a context
                // shift actually runs (#169).
                gp.n_keep = self->m_turn_session->count_tokens(sys_prefix);
                // A context shift evicts whole turns, up to this terminator.
                gp.turn_close = fmt.turn_close;
                gp.on_status = on_status;
                gp.on_token = on_token;
                return self->m_turn_session->generate(gp);
            };

            // THE budget enforcement point. The estimate upstream chose what routing
            // would see; here the tokenizer of the model that will generate decides
            // what actually fits, so no chars-per-token constant can shorten a reply
            // behind the user's back (xllama::fit_prompt). Lazy: a reuse turn sends
            // only the delta and never needs it — but its retry does.
            bool fitted = false;
            std::string fit_error; // set when even the final message cannot fit
            auto fit_full_prompt = [&]() -> const std::string& {
                if (fitted)
                    return full_prompt;
                fitted = true;
                const ::xllama::PromptFit fit =
                    ::xllama::fit_prompt(fmt, system_prompt, plan_turns, user_text, plan_n_ctx,
                                         plan_n_predict, [self](const std::string& text) {
                                             return self->m_turn_session->count_tokens(text);
                                         });
                char fb[192];
                snprintf(fb, sizeof(fb),
                         "[xllama] prompt budget: %d tok exact, dropped %d old turn(s), "
                         "n_ctx %d, reply %d%s\n",
                         fit.n_tokens, fit.dropped, plan_n_ctx, plan_n_predict,
                         fit.fits ? "" : " — DOES NOT FIT");
                ::xllama::log_output(fb);
                if (!fit.fits) {
                    // Logged as well as shown: the refusal never reaches the
                    // session, so nothing else records it in xllama.log.
                    fit_error = ::xllama::prompt_too_long_message(
                        "final user message", fit.n_tokens, plan_n_predict, plan_n_ctx);
                    ::xllama::log_output("[xllama] chat: " + fit_error + "\n");
                }
                if (fit.dropped > 0) {
                    const int dropped = fit.dropped;
                    dispatcher.RunAsync(CoreDispatcherPriority::Normal, [self, dropped]() {
                        self->SetStatus(L"Context trimmed — " + std::to_wstring(dropped) +
                                        L" old turn(s) dropped");
                    });
                }
                full_prompt = fit.prompt;
                return full_prompt;
            };
            // A full-prompt turn, or the same refusal the LAN API gives (400)
            // when the final message alone plus the reply's room exceeds n_ctx —
            // never a silently shortened reply.
            auto run_full_turn = [&](bool reuse, bool reset) {
                const std::string& prompt = fit_full_prompt();
                if (!fit_error.empty()) {
                    xllama::InferenceResult refused;
                    refused.error_msg = fit_error;
                    return refused;
                }
                return run_turn(prompt, reuse, reset);
            };

            xllama::InferenceResult res;
            if (do_reuse) {
                res = run_turn(delta_prompt, /*reuse=*/true, /*reset=*/false);
                // A continuation that fails before emitting any token (e.g. appending
                // to a finished generator) falls back to a full re-prefill. No tokens
                // were streamed yet, so the UI stays clean.
                if (!res.success && res.n_eval == 0) {
                    ::xllama::log_output("[xllama] KV reuse failed, retrying with full prefill\n");
                    res = run_full_turn(/*reuse=*/kv_reuse, /*reset=*/true);
                }
            } else {
                // First turn / post-reset: seed the persistent generator (reuse+reset)
                // when KV reuse is enabled, else a pure stateless turn.
                res = run_full_turn(/*reuse=*/kv_reuse, /*reset=*/kv_reuse);
            }

            // The KV this conversation now owns (still under hub.mtx); the next
            // continuation must find the session at this epoch.
            self->m_kv_epoch = self->m_turn_session->kv_epoch();

            std::wstring metrics;
            if (res.success) {
                wchar_t buf[256];
                double dt = (res.n_eval > 0 && res.t_eval_ms > 0)
                                ? (double)res.n_eval / (res.t_eval_ms / 1000.0)
                                : 0.0;
                // #130: lead with time-to-first-token. The app streams, so TTFT
                // is the wait the user actually experiences; the decode rate
                // that follows is well above reading speed on both backends and
                // is the less interesting number. res.t_p_eval_ms was measured
                // by both backends all along and simply never shown.
                const double ttft_s = res.t_p_eval_ms / 1000.0;
                if (res.n_p_eval > 0 && res.t_p_eval_ms > 0) {
                    swprintf_s(buf,
                               L"%.1fs to first token (%d tok prompt)  ·  decode %.1f tok/s  ·  "
                               L"%d tok  ·  peak %zu MB",
                               ttft_s, res.n_p_eval, dt, res.n_eval, res.peak_ws_mb);
                } else {
                    swprintf_s(buf, L"decode %.1f tok/s  ·  %d tok  ·  peak %zu MB", dt, res.n_eval,
                               res.peak_ws_mb);
                }
                metrics = buf;
            } else {
                metrics = ::xllama::utf8_to_wstring(res.error_msg.empty() ? "inference failed"
                                                                          : res.error_msg);
            }

            std::string output_text = fmt.postprocess_output(res.output_text);
            bool was_aborted = self->m_abort.load();
            // A thinking model that spent its whole budget reasoning leaves
            // NOTHING after postprocess (strip_thinking_blocks drops an unclosed
            // <think> to EOF). Without a stand-in the turn vanished: no message
            // saved, no FinalizeStreamedTurn, the streamed chain of thought left
            // orphaned on screen and a status of "Done". Say what happened
            // instead — the fix for the cut-off reply is the Max-new-tokens box.
            //
            // Only for a turn that COMPLETED: a cancel or a failure mid-thought
            // empties the output too, and blaming the token budget there would
            // be a lie (and would persist a reply the user stopped).
            const bool thinking_only =
                res.success && !was_aborted && output_text.empty() && !res.output_text.empty();
            if (thinking_only) {
                output_text = "(reasoning only — the answer did not fit; raise \"Max new "
                              "tokens\" in Settings)";
                ::xllama::log_output(
                    "[xllama] postprocess left no answer (truncated reasoning block)\n");
            }
            dispatcher.RunAsync(
                CoreDispatcherPriority::Normal,
                [self, metrics, res, output_text, was_aborted, ep_kv_ok, thinking_only]() {
                    self->m_metricsText.Text(metrics);
                    self->m_last_inference = res;
                    self->m_live_decode_tok_s.reset();
                    self->m_live_tokens.reset();
                    if (self->m_debug_dashboard)
                        self->RefreshDebugDashboard();
                    const wchar_t* status_text = L"Error";
                    StatusKind status_kind = StatusKind::Error;
                    if (was_aborted) {
                        status_text = L"Cancelled";
                        status_kind = StatusKind::Info;
                    } else if (res.success && thinking_only) {
                        // Not a failure and not a normal answer: the model spent the
                        // whole budget reasoning.
                        status_text = L"Reasoning cut off — raise Max new tokens";
                        status_kind = StatusKind::Info;
                    } else if (res.success) {
                        status_text = L"Done";
                        status_kind = StatusKind::Success;
                    }
                    self->SetStatus(status_text, status_kind);
                    self->SetRunning(false); // also stops timer + flushes remaining tokens
                    // KV-reuse bookkeeping: the persistent generator now holds this
                    // turn only if it completed cleanly. On failure or abort, force a
                    // fresh generator (full re-prefill) next turn.
                    if (self->m_kv_reuse && res.success && !was_aborted && ep_kv_ok) {
                        self->m_kv_valid = true;
                        self->m_kv_last_ended_with_stop = res.ended_with_stop;
                    } else {
                        self->m_kv_valid = false;
                    }
                    // Save assistant response (partial-flagged if user aborted)
                    if (!output_text.empty()) {
                        xllama::ui::ChatMessage amsg;
                        amsg.role = xllama::ui::MessageRole::Assistant;
                        amsg.content = output_text;
                        amsg.ts_unix = static_cast<int64_t>(std::time(nullptr));
                        amsg.partial = was_aborted;
                        self->m_current.messages.push_back(std::move(amsg));
                    }
                    self->SaveCurrentConversation(was_aborted);
                    if (!was_aborted && !output_text.empty())
                        self->FinalizeStreamedTurn(output_text);
                });
        } catch (const std::exception& ex) {
            ::xllama::log_output(std::string("[xllama] thread terminated: ") + ex.what() + "\n");
            dispatcher.RunAsync(CoreDispatcherPriority::Normal, [self]() {
                self->SetStatus(L"Fatal error — see xllama.log", StatusKind::Error);
                self->SaveCurrentConversation(/*partial=*/true);
                self->SetRunning(false);
            });
        } catch (...) {
            ::xllama::log_output("[xllama] thread terminated: unknown exception\n");
            dispatcher.RunAsync(CoreDispatcherPriority::Normal, [self]() {
                self->SetStatus(L"Fatal error — see xllama.log", StatusKind::Error);
                self->SaveCurrentConversation(/*partial=*/true);
                self->SetRunning(false);
            });
        }
    }).detach();
    return true;
}

// ---------------------------------------------------------------------------
// Event handlers (wired via weak_ptr lambda in ctor; kept for direct call)
// ---------------------------------------------------------------------------

void MainPageController::OnRunClick(IInspectable const&, RoutedEventArgs const&) {
    auto prompt = m_promptInput.Text();
    if (prompt.empty()) {
        SetStatus(L"Enter a prompt first");
        return;
    }
    StartInference(std::wstring(prompt.c_str()));
}

void MainPageController::OnCancelClick(IInspectable const&, RoutedEventArgs const&) {
    // Which job this targets is policy, and it lives in cancel_policy.h where
    // the host tests can reach it — the combination that shipped broken (image
    // running, generic flag also set) is not reproducible from this file.
    // m_is_running is the generic flag: SetRunning() is called by all three
    // jobs, so it is passed last and the specific flags decide.
    switch (::xllama::cancel_target(m_diffuse_running.load(), m_train_running.load(),
                                    m_is_running.load())) {
    case ::xllama::CancelTarget::Image:
        ::xllama::write_local_text("diffuse-cancel.flag", "cancel");
        SetStatus(L"Cancelling image...");
        m_cancelButton.IsEnabled(false);
        return;
    case ::xllama::CancelTarget::Training:
        m_train_abort.store(true);
        SetStatus(L"Cancelling training (between epochs)...");
        m_cancelButton.IsEnabled(false);
        return;
    case ::xllama::CancelTarget::Text:
        m_abort.store(true);
        SetStatus(L"Cancelling...");
        m_cancelButton.IsEnabled(false);
        return;
    case ::xllama::CancelTarget::None:
        // Reachable from the B button on an idle chat, never from the Cancel
        // button (which is disabled unless a job runs). Doing nothing is the
        // whole point: there is no job to abort.
        return;
    }
}

} // namespace xllama

#endif // XLLAMA_UWP
