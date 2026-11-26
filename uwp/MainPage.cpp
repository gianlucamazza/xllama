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

} // namespace xllama

#endif // XLLAMA_UWP
