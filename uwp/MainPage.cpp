// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

#ifdef XLLAMA_UWP

// clang-format off
#include "pch.h"
#include "MainPage.h"
// clang-format on

    #include "chat-history.h"
    #include "inference-bridge.h"
    #include "xllama/platform.h"
    #include "xllama/utf8_utils.h"

    #include <winrt/Windows.UI.Xaml.Media.Imaging.h>

    #include <cstdio>
    #include <ctime>
    #include <filesystem>
    #include <string>
    #include <thread>

using namespace winrt;
using namespace winrt::Windows::Foundation;
using namespace winrt::Windows::Storage;
using namespace winrt::Windows::UI::Core;
using namespace winrt::Windows::UI::Xaml;
using namespace winrt::Windows::UI::Xaml::Controls;
using namespace winrt::Windows::UI::Text;

namespace xllama {

// Wide LocalFolder path helper.
static std::wstring local_wpath(const wchar_t* filename_w) {
    auto folder = ApplicationData::Current().LocalFolder();
    return std::wstring(folder.Path().c_str()) + L"\\" + filename_w;
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

    // ---- Row 0: header (model name + status + progress bar) ----
    StackPanel header;
    Grid::SetRow(header, 0);

    m_modelText = TextBlock();
    m_modelText.FontSize(28);
    m_modelText.FontWeight(FontWeights::SemiBold());
    m_modelText.Text(L"xllama");

    m_statusText = TextBlock();
    m_statusText.FontSize(14);
    m_statusText.Opacity(0.7);
    m_statusText.Margin(ThicknessHelper::FromLengths(0, 4, 0, 0));
    m_statusText.Text(L"Loading model...");

    m_loadingBar = ProgressBar();
    m_loadingBar.IsIndeterminate(true);
    m_loadingBar.Visibility(Visibility::Collapsed);
    m_loadingBar.Margin(ThicknessHelper::FromLengths(0, 8, 0, 0));

    header.Children().Append(m_modelText);
    header.Children().Append(m_statusText);
    header.Children().Append(m_loadingBar);

    // ---- Row 1: scroll area (prompt input + output text) ----
    m_outputScroll = ScrollViewer();
    Grid::SetRow(m_outputScroll, 1);
    m_outputScroll.VerticalScrollBarVisibility(ScrollBarVisibility::Auto);
    m_outputScroll.Margin(ThicknessHelper::FromLengths(0, 24, 0, 0));
    // IsFocusEngagementEnabled intentionally NOT set on ScrollViewer — only set on the
    // inner TextBox to avoid requiring two A-presses to engage text input on Xbox.

    StackPanel bodyStack;

    m_promptInput = TextBox();
    m_promptInput.PlaceholderText(L"Type your prompt here (gamepad A → opens keyboard)...");
    m_promptInput.AcceptsReturn(true);
    m_promptInput.TextWrapping(TextWrapping::Wrap);
    m_promptInput.MinHeight(120);
    m_promptInput.IsSpellCheckEnabled(false);
    m_promptInput.FontSize(18);
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

    ColumnDefinition colStar;
    colStar.Width(GridLengthHelper::FromValueAndType(1, GridUnitType::Star));
    ColumnDefinition colAuto;
    colAuto.Width(GridLengthHelper::Auto());
    footer.ColumnDefinitions().Append(colStar);
    footer.ColumnDefinitions().Append(colAuto);

    m_metricsText = TextBlock();
    m_metricsText.FontSize(12);
    m_metricsText.Opacity(0.7);
    m_metricsText.VerticalAlignment(VerticalAlignment::Center);
    Grid::SetColumn(m_metricsText, 0);

    StackPanel btnPanel;
    btnPanel.Orientation(Orientation::Horizontal);
    Grid::SetColumn(btnPanel, 1);

    m_settingsButton = Button();
    m_settingsButton.Content(winrt::box_value(L"[S]  Settings"));
    m_settingsButton.MinWidth(100);
    m_settingsButton.Margin(ThicknessHelper::FromLengths(0, 0, 12, 0));

    m_newChatButton = Button();
    m_newChatButton.Content(winrt::box_value(L"+  New"));
    m_newChatButton.MinWidth(100);
    m_newChatButton.Margin(ThicknessHelper::FromLengths(0, 0, 12, 0));

    m_historyButton = Button();
    m_historyButton.Content(winrt::box_value(L"[=]  History"));
    m_historyButton.MinWidth(100);
    m_historyButton.Margin(ThicknessHelper::FromLengths(0, 0, 12, 0));

    m_imageButton = Button();
    m_imageButton.Content(winrt::box_value(L"[*]  Image"));
    m_imageButton.MinWidth(100);
    m_imageButton.Margin(ThicknessHelper::FromLengths(0, 0, 12, 0));

    m_runButton = Button();
    m_runButton.Content(winrt::box_value(L"▶  Run"));
    m_runButton.MinWidth(120);
    m_runButton.Margin(ThicknessHelper::FromLengths(0, 0, 12, 0));
    m_runButton.IsEnabled(false); // disabled until EnsureModelAsync confirms model is ready

    m_cancelButton = Button();
    m_cancelButton.Content(winrt::box_value(L"■  Cancel"));
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
        if (auto s = self.lock()) {
            s->m_abort.store(true);
            s->SetStatus(L"Cancelling...");
            s->m_cancelButton.IsEnabled(false);
        }
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

    // B button: cancel inference if running, otherwise let system exit the app
    auto nav = winrt::Windows::UI::Core::SystemNavigationManager::GetForCurrentView();
    nav.BackRequested(
        [self](IInspectable const&, winrt::Windows::UI::Core::BackRequestedEventArgs const& e) {
            if (auto s = self.lock()) {
                if (s->m_is_running.load()) {
                    s->m_abort.store(true);
                    s->SetStatus(L"Cancelling...");
                    s->m_cancelButton.IsEnabled(false);
                    e.Handled(true);
                }
            }
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

    EnsureModelAsync();
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

    // Live tok/s counter
    int n = m_tokens_received.load();
    if (n > 1) {
        double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - m_gen_start).count();
        if (elapsed > 0.2) {
            wchar_t buf[80];
            swprintf_s(buf, L"~%.0f tok/s  ·  %d tok", n / elapsed, n);
            m_metricsText.Text(buf);
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
        m_statusText.Opacity(0.7);
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

std::string MainPageController::BuildChatMLPrompt(const std::string& user_text,
                                                  int* out_dropped) const {
    // Estimate token count (heuristic: chars/4). Trim oldest turns if over limit.
    // Threshold aligned with n_ctx=2048: 1800 estimated tokens + ~250 generation buffer.
    constexpr int kMaxEstimatedTokens = 1800;
    // Collect turns (skip system which always stays)
    std::vector<size_t> turn_starts; // index of first User message in each turn
    for (size_t i = 0; i < m_current.messages.size(); ++i) {
        if (m_current.messages[i].role == xllama::ui::MessageRole::User)
            turn_starts.push_back(i);
    }

    // Build prompt starting from oldest turn; drop turns if over token budget
    auto calc_size = [&](size_t from_turn) -> int {
        int chars = (int)m_system_prompt.size();
        for (size_t ti = from_turn; ti < turn_starts.size(); ++ti) {
            size_t i = turn_starts[ti];
            chars += (int)m_current.messages[i].content.size(); // user
            if (i + 1 < m_current.messages.size() &&
                m_current.messages[i + 1].role == xllama::ui::MessageRole::Assistant)
                chars += (int)m_current.messages[i + 1].content.size(); // assistant
        }
        chars += (int)user_text.size(); // new user message
        return chars / 4;
    };

    size_t first_turn = 0;
    while (first_turn < turn_starts.size() && calc_size(first_turn) > kMaxEstimatedTokens)
        ++first_turn;
    if (first_turn > 0)
        log_output("[xllama] context trimmed: dropped " + std::to_string(first_turn) +
                   " old turn(s)\n");
    if (out_dropped)
        *out_dropped = static_cast<int>(first_turn);

    std::string prompt = "<|im_start|>system\n" + m_system_prompt + "<|im_end|>\n";

    for (size_t ti = first_turn; ti < turn_starts.size(); ++ti) {
        size_t i = turn_starts[ti];
        prompt += "<|im_start|>user\n" + m_current.messages[i].content + "<|im_end|>\n";
        prompt += "<|im_start|>assistant\n";
        if (i + 1 < m_current.messages.size() &&
            m_current.messages[i + 1].role == xllama::ui::MessageRole::Assistant) {
            prompt += m_current.messages[i + 1].content + "<|im_end|>\n";
        }
    }
    prompt += "<|im_start|>user\n" + user_text + "<|im_end|>\n";
    prompt += "<|im_start|>assistant\n";
    return prompt;
}

std::string MainPageController::BuildDeltaPrompt(const std::string& user_text) const {
    // The persistent KV cache already holds everything through the previous
    // assistant's generated tokens. Close that turn — the model emits <|im_end|>
    // when it stops on the stop sequence, so add it only if it didn't (n_predict
    // cap) — then append the new user turn and the assistant header. Concatenated
    // onto the KV this reproduces exactly what BuildChatMLPrompt would have built.
    std::string d = m_kv_last_ended_with_stop ? "\n" : "<|im_end|>\n";
    d += "<|im_start|>user\n" + user_text + "<|im_end|>\n<|im_start|>assistant\n";
    return d;
}

void MainPageController::SaveCurrentConversation(bool partial) {
    if (m_current.id.empty())
        return;
    // Mark last assistant message as partial if needed
    if (partial && !m_current.messages.empty() &&
        m_current.messages.back().role == xllama::ui::MessageRole::Assistant) {
        m_current.messages.back().partial = true;
    }
    m_history.Save(m_current);
}

void MainPageController::NewChat() {
    if (m_is_running.load())
        return;                // don't allow while running
    SaveCurrentConversation(); // save current (no-op if empty)
    m_kv_valid = false;        // new conversation → discard reused KV
    m_active_model.clear();    // re-decide EP routing for the new conversation
    m_current = xllama::ui::Conversation{};
    m_current.id = xllama::ui::ChatHistory::NewId();
    m_outputBody.Blocks().Clear();
    m_currentParagraph = winrt::Windows::UI::Xaml::Documents::Paragraph();
    m_outputBody.Blocks().Append(m_currentParagraph);
    m_promptInput.Text(L"");
    m_metricsText.Text(L"");
    SetStatus(L"New conversation");
}

void MainPageController::RenderConversation() {
    using namespace winrt::Windows::UI::Xaml::Documents;
    m_outputBody.Blocks().Clear();
    for (const auto& msg : m_current.messages) {
        if (msg.role == xllama::ui::MessageRole::System)
            continue;
        Paragraph p;
        const wchar_t* role_label =
            (msg.role == xllama::ui::MessageRole::User) ? L"You: " : L"Assistant: ";
        Run label;
        label.Text(role_label);
        label.FontWeight(winrt::Windows::UI::Text::FontWeights::Bold());
        p.Inlines().Append(label);
        Run content;
        content.Text(::xllama::utf8_to_wstring(msg.content));
        if (msg.partial)
            content.Text(content.Text() + L" [cancelled]");
        p.Inlines().Append(content);
        m_outputBody.Blocks().Append(p);
        m_outputBody.Blocks().Append(Paragraph()); // spacing
    }
    // Prepare fresh paragraph for next assistant turn
    m_currentParagraph = Paragraph();
    m_outputBody.Blocks().Append(m_currentParagraph);
    m_outputScroll.UpdateLayout();
    m_outputScroll.ChangeView(nullptr, m_outputScroll.ScrollableHeight(), nullptr);
}

void MainPageController::LoadConversation(const std::string& id) {
    SaveCurrentConversation();
    m_kv_valid = false;     // switching conversations → the reused KV no longer applies
    m_active_model.clear(); // re-decide EP routing for the loaded conversation
    m_current = m_history.Load(id);
    if (m_current.id.empty()) {
        m_current.id = id;
    }
    RenderConversation();
    SetStatus(L"Conversation loaded");
}

// Format unix timestamp as relative string ("today HH:MM", "yesterday HH:MM", "DD Mon HH:MM")
static std::wstring FormatRelativeTs(int64_t unix_ts) {
    if (unix_ts <= 0)
        return L"";
    std::time_t now = std::time(nullptr);
    std::time_t ts = static_cast<std::time_t>(unix_ts);
    std::tm now_tm{}, ts_tm{};
    #ifdef _WIN32
    localtime_s(&now_tm, &now);
    localtime_s(&ts_tm, &ts);
    #else
    localtime_r(&now, &now_tm);
    localtime_r(&ts, &ts_tm);
    #endif
    wchar_t buf[64];
    if (now_tm.tm_year == ts_tm.tm_year && now_tm.tm_yday == ts_tm.tm_yday)
        swprintf_s(buf, L"today %02d:%02d", ts_tm.tm_hour, ts_tm.tm_min);
    else if (now_tm.tm_year == ts_tm.tm_year && now_tm.tm_yday - ts_tm.tm_yday == 1)
        swprintf_s(buf, L"yesterday %02d:%02d", ts_tm.tm_hour, ts_tm.tm_min);
    else {
        static const wchar_t* months[] = {L"Jan", L"Feb", L"Mar", L"Apr", L"May", L"Jun",
                                          L"Jul", L"Aug", L"Sep", L"Oct", L"Nov", L"Dec"};
        swprintf_s(buf, L"%d %s %02d:%02d", ts_tm.tm_mday, months[ts_tm.tm_mon], ts_tm.tm_hour,
                   ts_tm.tm_min);
    }
    return buf;
}

winrt::fire_and_forget MainPageController::ShowHistory() {
    auto self = shared_from_this();
    if (m_is_running.load())
        co_return;

    m_history.LoadIndex(); // refresh index before showing
    const auto& index = m_history.Index();

    winrt::Windows::UI::Xaml::Controls::ContentDialog dlg;
    dlg.Title(winrt::box_value(L"Conversation History"));
    dlg.XamlRoot(m_root.XamlRoot());

    if (index.empty()) {
        winrt::Windows::UI::Xaml::Controls::TextBlock empty_tb;
        empty_tb.Text(L"No conversations yet — start chatting to see history here.");
        empty_tb.FontSize(16);
        empty_tb.Opacity(0.6);
        empty_tb.TextWrapping(winrt::Windows::UI::Xaml::TextWrapping::Wrap);
        dlg.Content(empty_tb);
        dlg.CloseButtonText(L"Close");
        co_await dlg.ShowAsync();
        co_return;
    }

    // Build a ListView — each row: [title TextBlock | ✕ Delete button]
    winrt::Windows::UI::Xaml::Controls::ListView lv;
    lv.SelectionMode(winrt::Windows::UI::Xaml::Controls::ListViewSelectionMode::Single);
    lv.Height(400);
    lv.Width(580);
    std::string current_id = m_current.id;

    // Shared state: when a Delete button fires it sets this and hides the dialog
    auto delete_pending = std::make_shared<std::string>();

    for (const auto& meta : index) {
        winrt::Windows::UI::Xaml::Controls::StackPanel row;
        row.Orientation(winrt::Windows::UI::Xaml::Controls::Orientation::Horizontal);

        winrt::Windows::UI::Xaml::Controls::TextBlock tb;
        tb.FontSize(16);
        tb.TextWrapping(winrt::Windows::UI::Xaml::TextWrapping::Wrap);
        tb.MaxWidth(480);
        tb.VerticalAlignment(winrt::Windows::UI::Xaml::VerticalAlignment::Center);
        std::wstring prefix = (meta.id == current_id) ? L"● " : L"   ";
        std::wstring ts = FormatRelativeTs(meta.last_modified);
        wchar_t buf[256];
        swprintf_s(buf, L"%s%s  (%d msgs)  •  %s", prefix.c_str(),
                   ::xllama::utf8_to_wstring(meta.title).c_str(), meta.n_messages, ts.c_str());
        tb.Text(buf);

        winrt::Windows::UI::Xaml::Controls::Button del_btn;
        del_btn.Content(winrt::box_value(L"✕"));
        del_btn.Width(48);
        del_btn.VerticalAlignment(winrt::Windows::UI::Xaml::VerticalAlignment::Center);
        del_btn.Margin(winrt::Windows::UI::Xaml::ThicknessHelper::FromLengths(8, 0, 0, 0));
        auto meta_id = meta.id;
        del_btn.Click([delete_pending, meta_id,
                       dlg](IInspectable const&, winrt::Windows::UI::Xaml::RoutedEventArgs const&) {
            *delete_pending = meta_id;
            dlg.Hide();
        });

        row.Children().Append(tb);
        row.Children().Append(del_btn);
        lv.Items().Append(row);
    }

    dlg.Content(lv);
    dlg.PrimaryButtonText(L"Open");
    dlg.SecondaryButtonText(L"Clear all");
    dlg.CloseButtonText(L"Cancel");

    auto result = co_await dlg.ShowAsync();

    // Per-item delete: a Delete button was pressed, hide closed the dialog
    if (!delete_pending->empty()) {
        std::string id_to_delete = *delete_pending;
        winrt::Windows::UI::Xaml::Controls::ContentDialog confirm;
        confirm.Title(winrt::box_value(L"Delete conversation?"));
        winrt::Windows::UI::Xaml::Controls::TextBlock ctb;
        ctb.Text(L"This will permanently delete this conversation.");
        ctb.TextWrapping(winrt::Windows::UI::Xaml::TextWrapping::Wrap);
        confirm.Content(ctb);
        confirm.PrimaryButtonText(L"Delete");
        confirm.CloseButtonText(L"Cancel");
        confirm.XamlRoot(m_root.XamlRoot());
        auto cr = co_await confirm.ShowAsync();
        if (cr == winrt::Windows::UI::Xaml::Controls::ContentDialogResult::Primary) {
            bool was_current = (id_to_delete == self->m_current.id);
            self->m_history.Delete(id_to_delete);
            if (was_current)
                self->NewChat();
            self->SetStatus(L"Conversation deleted");
        }
        co_return;
    }

    if (result == winrt::Windows::UI::Xaml::Controls::ContentDialogResult::Secondary) {
        // Clear all conversations — confirm first
        winrt::Windows::UI::Xaml::Controls::ContentDialog confirm;
        confirm.Title(winrt::box_value(L"Clear all conversations?"));
        winrt::Windows::UI::Xaml::Controls::TextBlock ctb;
        ctb.Text(L"This will permanently delete all conversation history.");
        ctb.TextWrapping(winrt::Windows::UI::Xaml::TextWrapping::Wrap);
        confirm.Content(ctb);
        confirm.PrimaryButtonText(L"Delete all");
        confirm.CloseButtonText(L"Cancel");
        confirm.XamlRoot(m_root.XamlRoot());
        auto cr = co_await confirm.ShowAsync();
        if (cr == winrt::Windows::UI::Xaml::Controls::ContentDialogResult::Primary) {
            self->m_history.Clear();
            self->NewChat();
            self->SetStatus(L"All conversations cleared");
        }
        co_return;
    }

    if (result != winrt::Windows::UI::Xaml::Controls::ContentDialogResult::Primary)
        co_return;

    int sel = lv.SelectedIndex();
    if (sel < 0 || sel >= static_cast<int>(index.size()))
        co_return;

    self->LoadConversation(index[static_cast<size_t>(sel)].id);
}

// Escape a UTF-8 string for inline JSON (same logic as chat-history.cpp json_escape)
static std::string settings_json_escape(const std::string& s) {
    std::string out;
    for (unsigned char c : s) {
        if (c == '"')
            out += "\\\"";
        else if (c == '\\')
            out += "\\\\";
        else if (c == '\n')
            out += "\\n";
        else if (c == '\r')
            out += "\\r";
        else if (c >= 0x20)
            out += static_cast<char>(c);
    }
    return out;
}

// Read a quoted JSON string starting after the opening '"'. Returns "" on failure.
static std::string settings_read_string(const std::string& json, size_t& pos) {
    std::string out;
    while (pos < json.size()) {
        char c = json[pos++];
        if (c == '"')
            return out;
        if (c == '\\' && pos < json.size()) {
            char e = json[pos++];
            if (e == 'n')
                out += '\n';
            else if (e == 't')
                out += '\t';
            else
                out += e;
        } else
            out += c;
    }
    return out;
}

// Read a JSON number/token (up to next `,`, `}`, `]`, or whitespace). Returns "".
static std::string settings_read_token(const std::string& json, size_t& pos) {
    while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\n'))
        ++pos;
    std::string out;
    while (pos < json.size()) {
        char c = json[pos];
        if (c == ',' || c == '}' || c == ']' || c == ' ' || c == '\t' || c == '\n')
            break;
        out += c;
        ++pos;
    }
    return out;
}

void MainPageController::LoadSettings() {
    auto folder = winrt::Windows::Storage::ApplicationData::Current().LocalFolder();
    std::wstring wpath(folder.Path().c_str());
    wpath += L"\\settings.json";
    FILE* f = _wfopen(wpath.c_str(), L"r");
    if (!f)
        return;
    std::string json;
    char buf[16384];
    while (size_t n = fread(buf, 1, sizeof(buf) - 1, f)) {
        buf[n] = 0;
        json += buf;
    }
    fclose(f);

    // Parse flat keys: "system_prompt", "model", and "sampling" object
    auto read_key = [&](size_t& pos) -> std::string {
        while (pos < json.size() && json[pos] != '"' && json[pos] != '}')
            ++pos;
        if (pos >= json.size() || json[pos] == '}')
            return "";
        ++pos; // skip opening "
        return settings_read_string(json, pos);
    };

    auto seek_colon_and_advance = [&](size_t& pos) {
        while (pos < json.size() && json[pos] != ':')
            ++pos;
        ++pos;
    };

    size_t pos = 0;
    while (pos < json.size()) {
        std::string key = read_key(pos);
        if (key.empty())
            break;
        seek_colon_and_advance(pos);
        while (pos < json.size() && (json[pos] == ' ' || json[pos] == '\t' || json[pos] == '\n'))
            ++pos;

        if (key == "system_prompt") {
            if (pos < json.size() && json[pos] == '"') {
                ++pos;
                m_system_prompt = settings_read_string(json, pos);
            }
        } else if (key == "model") {
            if (pos < json.size() && json[pos] == '"') {
                ++pos;
                std::string m = settings_read_string(json, pos);
                if (!m.empty())
                    m_model_filename = ::xllama::utf8_to_wstring(m);
            }
        } else if (key == "kv_reuse") {
            std::string v = settings_read_token(json, pos);
            m_kv_reuse = (v == "true" || v == "1");
        } else if (key == "routing") {
            std::string v = settings_read_token(json, pos);
            if (!v.empty())
                m_routing = std::stoi(v);
        } else if (key == "gpu_model") {
            if (pos < json.size() && json[pos] == '"') {
                ++pos;
                std::string g = settings_read_string(json, pos);
                if (!g.empty())
                    m_gpu_model = g;
            }
        } else if (key == "sampling") {
            // Parse nested object {"temperature":0.8, ...}
            if (pos < json.size() && json[pos] == '{') {
                ++pos;
                while (pos < json.size()) {
                    while (pos < json.size() && json[pos] != '"' && json[pos] != '}')
                        ++pos;
                    if (pos >= json.size() || json[pos] == '}') {
                        ++pos;
                        break;
                    }
                    ++pos;
                    std::string sk = settings_read_string(json, pos);
                    seek_colon_and_advance(pos);
                    std::string sv = settings_read_token(json, pos);
                    if (sk == "temperature" && !sv.empty())
                        m_temperature = std::stof(sv);
                    else if (sk == "top_p" && !sv.empty())
                        m_top_p = std::stof(sv);
                    else if (sk == "top_k" && !sv.empty())
                        m_top_k = std::stoi(sv);
                    else if (sk == "repetition_penalty" && !sv.empty())
                        m_repetition_penalty = std::stof(sv);
                    else if (sk == "n_predict" && !sv.empty())
                        m_n_predict = std::stoi(sv);
                    // skip comma
                    while (pos < json.size() && (json[pos] == ',' || json[pos] == ' '))
                        ++pos;
                }
            }
        } else {
            // Unknown key: skip value
            settings_read_token(json, pos);
        }
    }

    // Back-compat: also read model.txt if model not set via settings.json
    if (m_model_filename.empty()) {
        auto model_path = local_wpath(L"model.txt");
        FILE* mf = _wfopen(model_path.c_str(), L"r");
        if (mf) {
            wchar_t mbuf[512] = {};
            if (fgetws(mbuf, 511, mf)) {
                size_t len = wcslen(mbuf);
                while (len > 0 &&
                       (mbuf[len - 1] == L'\n' || mbuf[len - 1] == L'\r' || mbuf[len - 1] == L' '))
                    mbuf[--len] = L'\0';
                if (len > 0)
                    m_model_filename = mbuf;
            }
            fclose(mf);
        }
        if (m_model_filename.empty())
            m_model_filename = L"smollm2-360m-cpu-int4";
    }
}

void MainPageController::SaveSettings() {
    auto folder = winrt::Windows::Storage::ApplicationData::Current().LocalFolder();
    std::wstring wpath(folder.Path().c_str());
    wpath += L"\\settings.json";
    FILE* f = _wfopen(wpath.c_str(), L"w");
    if (!f)
        return;
    std::string model_utf8 = ::xllama::wstring_to_utf8(std::wstring(m_model_filename));
    fprintf(f,
            "{\n"
            "  \"system_prompt\": \"%s\",\n"
            "  \"model\": \"%s\",\n"
            "  \"kv_reuse\": %s,\n"
            "  \"routing\": %d,\n"
            "  \"gpu_model\": \"%s\",\n"
            "  \"sampling\": {\n"
            "    \"temperature\": %.2f,\n"
            "    \"top_p\": %.2f,\n"
            "    \"top_k\": %d,\n"
            "    \"repetition_penalty\": %.2f,\n"
            "    \"n_predict\": %d\n"
            "  }\n"
            "}\n",
            settings_json_escape(m_system_prompt).c_str(), settings_json_escape(model_utf8).c_str(),
            m_kv_reuse ? "true" : "false", m_routing, settings_json_escape(m_gpu_model).c_str(),
            static_cast<double>(m_temperature), static_cast<double>(m_top_p), m_top_k,
            static_cast<double>(m_repetition_penalty), m_n_predict);
    fclose(f);
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
    std::vector<bool> model_is_gguf; // parallel to model_keys; gates the KV/routing UI
    int model_sel = 0;
    for (auto const& e : manifest) {
        if (e.kind == L"diffusion")
            continue; // image models belong to the Image dialog, not the chat picker
        modelBox.Items().Append(winrt::box_value(winrt::hstring(e.display)));
        if (m_model_filename == e.name)
            model_sel = (int)model_keys.size();
        model_keys.push_back(e.name);
        model_is_gguf.push_back(e.kind == L"gguf");
    }
    if (!m_model_filename.empty() && !::xllama::FindManifestEntry(manifest, m_model_filename)) {
        modelBox.Items().Append(winrt::box_value(winrt::hstring(m_model_filename + L" (custom)")));
        model_sel = (int)model_keys.size();
        model_keys.push_back(m_model_filename);
        model_is_gguf.push_back(false);
    }
    modelBox.SelectedIndex(model_sel);

    // --- System prompt TextBox ---
    winrt::Windows::UI::Xaml::Controls::TextBox sysPromptBox;
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
    auto tempSlider = make_slider(m_temperature, 0.0, 2.0, 0.05, L"Temperature (0–2)");
    auto topPSlider = make_slider(m_top_p, 0.0, 1.0, 0.05, L"Top-p (0–1)");
    auto repSlider = make_slider(m_repetition_penalty, 1.0, 2.0, 0.05, L"Repetition penalty (1–2)");

    // top_k and n_predict as simple Sliders (NumberBox not available in older SDK targets)
    auto topKSlider = make_slider(m_top_k, 1.0, 200.0, 1.0, L"Top-k (1–200)");
    auto nPredSlider = make_slider(m_n_predict, 16.0, 2048.0, 16.0, L"Max new tokens (16–2048)");

    // --- KV-cache reuse toggle (continuous decoding) ---
    winrt::Windows::UI::Xaml::Controls::ToggleSwitch kvToggle;
    kvToggle.Header(winrt::box_value(L"KV-cache reuse (faster multi-turn)"));
    kvToggle.OnContent(winrt::box_value(L"On"));
    kvToggle.OffContent(winrt::box_value(L"Off"));
    kvToggle.IsOn(m_kv_reuse);

    // --- EP routing ComboBox (experimental; needs the DML fp16 model on device) ---
    winrt::Windows::UI::Xaml::Controls::ComboBox routingBox;
    routingBox.Header(winrt::box_value(L"EP routing (per conversation)"));
    routingBox.FontSize(16);
    routingBox.HorizontalAlignment(HorizontalAlignment::Stretch);
    routingBox.Items().Append(winrt::box_value(L"CPU only (default)"));
    routingBox.Items().Append(winrt::box_value(L"GPU only (DML)"));
    routingBox.Items().Append(winrt::box_value(L"Auto (long prompts → GPU)"));
    routingBox.SelectedIndex(m_routing >= 0 && m_routing <= 2 ? m_routing : 0);

    // GGUF models run stateless on CPU-only llama.cpp: KV-reuse and EP routing do
    // not apply, so grey them out whenever a GGUF entry is selected (and restore
    // them for ORT entries). Wired live on the model ComboBox.
    auto sync_backend_toggles = [kvToggle, routingBox, model_is_gguf](int idx) {
        bool gguf = idx >= 0 && idx < (int)model_is_gguf.size() && model_is_gguf[idx];
        kvToggle.IsEnabled(!gguf);
        routingBox.IsEnabled(!gguf);
    };
    sync_backend_toggles(model_sel);
    modelBox.SelectionChanged(
        [sync_backend_toggles](
            winrt::Windows::Foundation::IInspectable const& sender,
            winrt::Windows::UI::Xaml::Controls::SelectionChangedEventArgs const&) {
            auto box = sender.as<winrt::Windows::UI::Xaml::Controls::ComboBox>();
            sync_backend_toggles(box.SelectedIndex());
        });

    winrt::Windows::UI::Xaml::Controls::StackPanel panel;
    panel.Orientation(Orientation::Vertical);
    panel.Spacing(12);
    panel.Children().Append(modelBox);
    panel.Children().Append(sysPromptBox);
    panel.Children().Append(tempSlider);
    panel.Children().Append(topPSlider);
    panel.Children().Append(topKSlider);
    panel.Children().Append(repSlider);
    panel.Children().Append(nPredSlider);
    panel.Children().Append(kvToggle);
    panel.Children().Append(routingBox);

    winrt::Windows::UI::Xaml::Controls::ScrollViewer sv;
    sv.Content(panel);
    sv.MaxHeight(480);
    sv.VerticalScrollBarVisibility(ScrollBarVisibility::Auto);

    winrt::Windows::UI::Xaml::Controls::ContentDialog dlg;
    dlg.Title(winrt::box_value(L"Settings"));
    dlg.Content(sv);
    dlg.PrimaryButtonText(L"Save");
    dlg.CloseButtonText(L"Cancel");
    dlg.XamlRoot(m_root.XamlRoot());

    auto result = co_await dlg.ShowAsync();
    if (result != winrt::Windows::UI::Xaml::Controls::ContentDialogResult::Primary)
        co_return;

    // Read back values
    int mi = modelBox.SelectedIndex();
    if (mi >= 0 && mi < (int)model_keys.size()) {
        std::wstring new_model = model_keys[mi];
        if (new_model != self->m_model_filename) {
            self->m_model_filename = new_model;
            self->m_modelText.Text(new_model);
            self->SetStatus(L"Model changed — will reload on next message");
        }
    }
    self->m_system_prompt = ::xllama::wstring_to_utf8(std::wstring(sysPromptBox.Text().c_str()));
    self->m_temperature = static_cast<float>(tempSlider.Value());
    self->m_top_p = static_cast<float>(topPSlider.Value());
    self->m_top_k = static_cast<int>(topKSlider.Value());
    self->m_repetition_penalty = static_cast<float>(repSlider.Value());
    self->m_n_predict = static_cast<int>(nPredSlider.Value());
    self->m_kv_reuse = kvToggle.IsOn();
    int ri = routingBox.SelectedIndex();
    self->m_routing = (ri >= 0 && ri <= 2) ? ri : 0;
    // Routing is per-conversation: a change applies from the next new/loaded chat
    // (m_active_model stays fixed for the conversation in progress).
    self->SaveSettings();
    self->SetStatus(L"Settings saved", StatusKind::Success);
}

// ---------------------------------------------------------------------------
// ShowImageDialog — view the last generated image and stage a new headless
// generation. DML cannot initialise inside the XAML process (887A0036), so
// "Generate" writes the diffusion inputs + diffuse.flag and restarts the app:
// the restart boots headless (App.cpp flag dispatch), runs the GPU pipeline,
// and exits; the image appears here on the next launch.
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

    winrt::Windows::UI::Xaml::Controls::TextBox promptBox;
    promptBox.Header(winrt::box_value(L"Image prompt"));
    promptBox.Text(L"a red sports car on a mountain road at sunset");
    promptBox.TextWrapping(TextWrapping::Wrap);
    promptBox.AcceptsReturn(false);
    promptBox.FontSize(16);
    panel.Children().Append(promptBox);

    winrt::Windows::UI::Xaml::Controls::Slider stepsSlider;
    stepsSlider.Minimum(1);
    stepsSlider.Maximum(4);
    stepsSlider.StepFrequency(1);
    stepsSlider.Value(1);
    stepsSlider.Header(winrt::box_value(L"Steps (SD-Turbo: 1 is enough)"));
    panel.Children().Append(stepsSlider);

    winrt::Windows::UI::Xaml::Controls::TextBlock note;
    note.TextWrapping(TextWrapping::Wrap);
    note.Opacity(0.7);
    note.Text(L"Generate restarts the app to run the GPU pipeline (it cannot run inside "
              L"this window). When the console returns to Dev Home, relaunch xllama and "
              L"reopen this dialog to view the result (~15 s).");
    panel.Children().Append(note);

    winrt::Windows::UI::Xaml::Controls::ScrollViewer sv;
    sv.Content(panel);
    sv.MaxHeight(480);
    sv.VerticalScrollBarVisibility(ScrollBarVisibility::Auto);

    winrt::Windows::UI::Xaml::Controls::ContentDialog dlg;
    dlg.Title(winrt::box_value(L"Image generation (SD-Turbo on GPU)"));
    dlg.Content(sv);
    dlg.PrimaryButtonText(L"Generate (restarts app)");
    dlg.CloseButtonText(L"Close");
    dlg.XamlRoot(m_root.XamlRoot());

    auto result = co_await dlg.ShowAsync();
    if (result != winrt::Windows::UI::Xaml::Controls::ContentDialogResult::Primary)
        co_return;

    // Ensure the diffusion model is present; download it from the catalogue if
    // missing (kind "diffusion" entries never reach the chat picker).
    constexpr const wchar_t* kDiffusionModel = L"sd-turbo-fp16";
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
    }

    // Stage the headless generation inputs. The flag is written LAST so a
    // partially staged run can never trigger.
    auto write_local = [](const wchar_t* name, const std::string& bytes) {
        FILE* fp = _wfopen(local_wpath(name).c_str(), L"wb");
        if (!fp)
            return false;
        fwrite(bytes.data(), 1, bytes.size(), fp);
        fclose(fp);
        return true;
    };
    std::string prompt_utf8 = ::xllama::wstring_to_utf8(std::wstring(promptBox.Text().c_str()));
    if (prompt_utf8.empty())
        prompt_utf8 = "a red sports car on a mountain road at sunset";
    const int steps = (int)stepsSlider.Value();
    // Fresh seed per generation so repeated prompts give new images.
    const unsigned seed = (unsigned)(GetTickCount64() % 1'000'000'000ULL);
    bool ok = write_local(L"prompt.txt", prompt_utf8) &&
              write_local(L"diffuse-steps.txt", std::to_string(steps)) &&
              write_local(L"diffuse-seed.txt", std::to_string(seed)) &&
              write_local(L"diffuse-model.txt", "sd-turbo-fp16") &&
              write_local(L"diffuse.flag", "diffuse");
    if (!ok) {
        self->SetStatus(L"Could not stage the generation files", StatusKind::Error);
        co_return;
    }

    self->SetStatus(L"Restarting to generate on GPU...", StatusKind::Working);
    auto reason =
        co_await winrt::Windows::ApplicationModel::Core::CoreApplication::RequestRestartAsync(
            L"diffuse");
    // Only reached if the restart was denied (e.g. not permitted right now):
    // the flag is staged, so a manual relaunch will still run the generation.
    (void)reason;
    self->SetStatus(L"Restart not permitted — quit and relaunch xllama to generate",
                    StatusKind::Error);
}

// ---------------------------------------------------------------------------
// EnsureModelAsync — checks whether the model is available locally;
// downloads from Hugging Face if not found in LocalState or InstalledPath.
// Calls LoadModelName() when ready.
// ---------------------------------------------------------------------------

fire_and_forget MainPageController::EnsureModelAsync() {
    auto self = shared_from_this();
    auto dispatcher = self->m_root.Dispatcher();

    // Model name is already loaded via LoadSettings() in Init() before EnsureModelAsync runs.
    co_await resume_background();
    std::wstring model_name =
        self->m_model_filename.empty() ? L"smollm2-360m-cpu-int4" : self->m_model_filename;

    // Check 1: LocalState model complete?
    auto local_folder = winrt::Windows::Storage::ApplicationData::Current().LocalFolder();
    std::wstring local_models_root = std::wstring(local_folder.Path().c_str()) + L"\\models";
    std::wstring local_model_dir = local_models_root + L"\\" + model_name;

    if (ModelDownloader::IsComplete(local_model_dir)) {
        co_await resume_foreground(dispatcher);
        self->LoadModelName();
        self->SetStatus(L"Ready", StatusKind::Success);
        self->m_runButton.IsEnabled(true);
        co_return;
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
            self->LoadModelName();
            self->SetStatus(L"Ready", StatusKind::Success);
            self->m_runButton.IsEnabled(true);
            co_return;
        }
    }

    // Check 3: USB removable storage via KnownFolders.RemovableDevices
    // (requires <uap:Capability Name="removableStorage" /> in manifest).
    // Enumerates all removable drives; looks for xllama/models/<name>/genai_config.json.
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
                log_output(("[xllama] USB probe: " + ::xllama::wstring_to_utf8(drive_path) + "\n")
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
                        auto cache_path = local_wpath(L"usb_model_root.txt");
                        FILE* fp = _wfopen(cache_path.c_str(), L"w");
                        if (fp) {
                            fputws(drive_path.c_str(), fp);
                            fclose(fp);
                        }
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
                    log_output(("[xllama] USB copy: " +
                                ::xllama::wstring_to_utf8(std::wstring(f.Name().c_str())) + "\n")
                                   .c_str());
                    co_await f.CopyAsync(dest_folder, f.Name(),
                                         NameCollisionOption::ReplaceExisting);
                }
                // Write .complete marker
                auto marker = co_await dest_folder.CreateFileAsync(
                    L".complete", CreationCollisionOption::ReplaceExisting);
                co_await winrt::Windows::Storage::FileIO::WriteTextAsync(marker, L"ok");
                copy_ok = true;
            } catch (winrt::hresult_error const& e) {
                copy_err = std::wstring(e.message().c_str());
            } catch (std::exception const& e) {
                copy_err = ::xllama::utf8_to_wstring(e.what());
            } catch (...) {
                copy_err = L"Unknown error during USB copy";
            }

            self->m_loadingBar.IsIndeterminate(false);
            self->m_loadingBar.Visibility(winrt::Windows::UI::Xaml::Visibility::Collapsed);
            if (!copy_ok) {
                self->SetStatus(L"USB copy failed: " + copy_err, StatusKind::Error);
                co_return;
            }
            log_output("[xllama] USB model copy complete\n");
            self->LoadModelName();
            self->SetStatus(L"Ready", StatusKind::Success);
            self->m_runButton.IsEnabled(true);
            co_return;
        }
    }
    // Neither found: consult the model catalogue (models/manifest.json) — any
    // entry with an hf_base_url can be auto-downloaded; anything else must be
    // provided via USB or Device Portal upload.
    auto manifest = ::xllama::LoadModelManifest();
    const ::xllama::ManifestEntry* entry = ::xllama::FindManifestEntry(manifest, model_name);
    if (!entry || entry->hf_base_url.empty() || entry->files.empty()) {
        self->SetStatus(L"Model '" + model_name +
                            L"' not found.\n"
                            L"Connect USB with xllama/models/" +
                            model_name + L"/",
                        StatusKind::Error);
        self->m_runButton.IsEnabled(false);
        co_return;
    }

    co_await resume_background();

    // Auto-download from the catalogue entry's Hugging Face repo.
    co_await resume_foreground(dispatcher);
    self->SetStatus(L"Downloading model...", StatusKind::Working);
    self->m_loadingBar.IsIndeterminate(false);
    self->m_loadingBar.Value(0);
    self->m_loadingBar.Visibility(winrt::Windows::UI::Xaml::Visibility::Visible);
    self->m_runButton.IsEnabled(false);
    co_await resume_background();

    // Create local model directory.
    {
        std::error_code ec;
        std::filesystem::create_directories(local_model_dir, ec);
        if (ec) {
            co_await resume_foreground(dispatcher);
            self->SetStatus(std::wstring(L"Cannot create model dir: ") +
                                winrt::to_hstring(ec.message()).c_str(),
                            StatusKind::Error);
            co_return;
        }
    }

    co_await resume_foreground(dispatcher);

    co_await ModelDownloader::DownloadAsync(
        entry->hf_base_url, local_model_dir, entry->files, dispatcher,
        [self](uint64_t done, uint64_t total) {
            // progress callback — called on UI thread
            if (total > 0) {
                double pct = static_cast<double>(done) / static_cast<double>(total) * 100.0;
                self->m_loadingBar.Value(pct);
                self->SetStatus(L"Downloading model... " + std::to_wstring(done / (1024 * 1024)) +
                                    L" MB",
                                StatusKind::Working);
            } else {
                self->SetStatus(L"Downloading model... " + std::to_wstring(done / (1024 * 1024)) +
                                    L" MB",
                                StatusKind::Working);
            }
        },
        [self, dispatcher](bool ok, std::wstring err) {
            // done callback — called on UI thread
            self->m_loadingBar.Visibility(winrt::Windows::UI::Xaml::Visibility::Collapsed);
            self->m_runButton.IsEnabled(true);
            if (!ok) {
                self->SetStatus(L"Download failed: " + err, StatusKind::Error);
                return;
            }
            self->SetStatus(L"Model ready", StatusKind::Success);
            self->LoadModelName();
        });
}

// LoadModelName: read model.txt from LocalFolder
// ---------------------------------------------------------------------------

void MainPageController::LoadModelName() {
    // m_model_filename is already populated by LoadSettings() (called in Init()).
    // This method just refreshes the header TextBlock with the current value.
    if (m_model_filename.empty())
        m_model_filename = L"smollm2-360m-cpu-int4";
    m_modelText.Text(m_model_filename);
}

// ---------------------------------------------------------------------------
// EnsureSession: lazy-build persistent Session (hot path is a pointer check)
// ---------------------------------------------------------------------------

bool MainPageController::EnsureSession(const std::string& model, std::string* err_out) {
    if (m_session && m_session_model == model)
        return true;
    m_session.reset();  // free before creating new (avoid 2× model in RAM)
    m_kv_valid = false; // new session object → no reusable KV carries over
    xllama::SessionParams sp;
    sp.model_path = model;
    sp.n_ctx = 2048;
    sp.n_threads = 0; // auto
    // Pick the backend from the catalogue kind: a "gguf" entry runs on llama.cpp,
    // everything else (an ORT GenAI directory) on ORT. On a single-backend build
    // this field is ignored (only one path is compiled). The model name is bare
    // (no extension), so Backend::Auto's suffix sniffing cannot classify it.
    {
        auto manifest = ::xllama::LoadModelManifest();
        const auto* entry = ::xllama::FindManifestEntry(manifest, ::xllama::utf8_to_wstring(model));
        if (entry && entry->kind == L"gguf")
            sp.backend = xllama::Backend::LlamaCpp;
    }
    std::string err;
    auto s = xllama::Session::create(sp, &err);
    if (!s) {
        if (err_out)
            *err_out = err;
        return false;
    }
    m_session = std::move(s);
    m_session_model = model;
    return true;
}

// ---------------------------------------------------------------------------
// StartInference: called on UI thread; spawns background thread
// ---------------------------------------------------------------------------

void MainPageController::StartInference(std::wstring const& prompt_w) {
    m_abort.store(false);
    SetStatus(L"Loading model...", StatusKind::Working);
    SetRunning(true);

    // Add user message to display + prepare empty assistant paragraph
    AddUserParagraph(prompt_w);
    m_metricsText.Text(L"");

    // Reset streaming counters
    m_tokens_received.store(0);
    {
        std::lock_guard<std::mutex> lk(m_token_mutex);
        m_token_buffer.clear();
    }
    m_gen_start = std::chrono::steady_clock::now();

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
    // BuildChatMLPrompt uses existing m_current.messages (prev turns) and appends user_text.
    std::string user_text = ::xllama::wstring_to_utf8(prompt_w);
    if (m_current.id.empty()) {
        m_current.id = xllama::ui::ChatHistory::NewId();
        m_current.title = xllama::ui::ChatHistory::TitleFrom(user_text);
    }
    int n_dropped = 0;
    std::string full_prompt = BuildChatMLPrompt(user_text, &n_dropped);
    if (n_dropped > 0)
        SetStatus(L"Context trimmed — " + std::to_wstring(n_dropped) + L" old turn(s) dropped");

    // Is the base model a GGUF (llama.cpp backend)? On Xbox llama.cpp is CPU-only
    // (no ggml GPU backend) and LlamaSession is stateless, so both EP routing and
    // KV-cache reuse are meaningless — and reuse would be *incorrect* (it sends
    // only the turn delta, which a stateless backend would treat as the whole
    // prompt). Gate both off for GGUF models.
    bool base_is_gguf = false;
    {
        auto manifest = ::xllama::LoadModelManifest();
        const auto* e = ::xllama::FindManifestEntry(manifest, m_model_filename);
        base_is_gguf = e && e->kind == L"gguf";
    }

    // Stage 3: decide EP routing once per conversation (sticky — the KV cache is
    // per-EP). m_active_model is cleared on new/loaded chat, so this fires on the
    // first turn and stays fixed after. Default (m_routing==0) keeps the CPU model.
    if (m_active_model.empty()) {
        constexpr int kRoutingTokThreshold = 500; // ~crossover from the v0.3.6 matrix
        if (base_is_gguf) {
            m_active_model = m_model_filename; // no EP routing for llama.cpp (CPU-only)
        } else if (m_routing == 1) {
            m_active_model = ::xllama::utf8_to_wstring(m_gpu_model);
        } else if (m_routing == 2) {
            int est_tok = static_cast<int>(full_prompt.size() / 4);
            m_active_model = est_tok > kRoutingTokThreshold ? ::xllama::utf8_to_wstring(m_gpu_model)
                                                            : m_model_filename;
        } else {
            m_active_model = m_model_filename;
        }
    }

    // KV-cache reuse decision (continuous decoding). Reuse only when enabled, the
    // persistent generator already holds this conversation (m_kv_valid), and no
    // turn was evicted this round (RewindTo cannot drop from the head, so eviction
    // forces a full re-prefill). A reuse turn appends only the delta; otherwise we
    // (re)prefill the full prompt — which also (re)seeds the persistent generator.
    bool do_reuse = m_kv_reuse && m_kv_valid && n_dropped == 0 && !base_is_gguf;
    bool kv_reuse = m_kv_reuse && !base_is_gguf;
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

    std::thread([self, full_prompt, delta_prompt, do_reuse, kv_reuse, model, dispatcher]() {
        try {
            std::string load_err;
            if (!self->EnsureSession(model, &load_err)) {
                dispatcher.RunAsync(CoreDispatcherPriority::Normal, [self, load_err]() {
                    self->SetStatus(::xllama::utf8_to_wstring("Load failed: " + load_err),
                                    StatusKind::Error);
                    self->SetRunning(false);
                });
                return;
            }

            auto on_status = [self, dispatcher](const std::string& s) {
                auto ws = ::xllama::utf8_to_wstring(s);
                StatusKind k =
                    (s.rfind("error:", 0) == 0) ? StatusKind::Error : StatusKind::Working;
                dispatcher.RunAsync(CoreDispatcherPriority::Normal,
                                    [self, ws, k]() { self->SetStatus(ws, k); });
            };
            // Token accumulation — no per-token RunAsync dispatch (batched by flush timer)
            auto on_token = [self](const std::string& tok) {
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
                gp.stop_sequences.push_back("<|im_end|>");
                gp.reuse_kv = reuse;
                gp.reset_kv = reset;
                gp.on_status = on_status;
                gp.on_token = on_token;
                return self->m_session->generate(gp);
            };

            xllama::InferenceResult res;
            if (do_reuse) {
                res = run_turn(delta_prompt, /*reuse=*/true, /*reset=*/false);
                // A continuation that fails before emitting any token (e.g. appending
                // to a finished generator) falls back to a full re-prefill. No tokens
                // were streamed yet, so the UI stays clean.
                if (!res.success && res.n_eval == 0) {
                    ::xllama::log_output("[xllama] KV reuse failed, retrying with full prefill\n");
                    res = run_turn(full_prompt, /*reuse=*/kv_reuse, /*reset=*/true);
                }
            } else {
                // First turn / post-reset: seed the persistent generator (reuse+reset)
                // when KV reuse is enabled, else a pure stateless turn.
                res = run_turn(full_prompt, /*reuse=*/kv_reuse, /*reset=*/kv_reuse);
            }

            std::wstring metrics;
            if (res.success) {
                wchar_t buf[256];
                double dt = (res.n_eval > 0 && res.t_eval_ms > 0)
                                ? (double)res.n_eval / (res.t_eval_ms / 1000.0)
                                : 0.0;
                swprintf_s(buf, L"decode %.1f tok/s  ·  %d tok  ·  peak %zu MB", dt, res.n_eval,
                           res.peak_ws_mb);
                metrics = buf;
            } else {
                metrics = ::xllama::utf8_to_wstring(res.error_msg.empty() ? "inference failed"
                                                                          : res.error_msg);
            }

            std::string output_text = res.output_text;
            bool was_aborted = self->m_abort.load();
            dispatcher.RunAsync(
                CoreDispatcherPriority::Normal, [self, metrics, res, output_text, was_aborted]() {
                    self->m_metricsText.Text(metrics);
                    self->SetStatus(was_aborted ? L"Cancelled" : (res.success ? L"Done" : L"Error"),
                                    was_aborted
                                        ? StatusKind::Info
                                        : (res.success ? StatusKind::Success : StatusKind::Error));
                    self->SetRunning(false); // also stops timer + flushes remaining tokens
                    // KV-reuse bookkeeping: the persistent generator now holds this
                    // turn only if it completed cleanly. On failure or abort, force a
                    // fresh generator (full re-prefill) next turn.
                    if (self->m_kv_reuse && res.success && !was_aborted) {
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
    m_abort.store(true);
    SetStatus(L"Cancelling...");
    m_cancelButton.IsEnabled(false);
}

} // namespace xllama

#endif // XLLAMA_UWP
