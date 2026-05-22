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

    #include <cstdio>
    #include <ctime>
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
    m_statusText.Text(L"Ready");

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
    m_outputScroll.IsFocusEngagementEnabled(true);

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

    m_runButton = Button();
    m_runButton.Content(winrt::box_value(L"▶  Run"));
    m_runButton.MinWidth(120);
    m_runButton.Margin(ThicknessHelper::FromLengths(0, 0, 12, 0));

    m_cancelButton = Button();
    m_cancelButton.Content(winrt::box_value(L"■  Cancel"));
    m_cancelButton.IsEnabled(false);
    m_cancelButton.MinWidth(120);

    btnPanel.Children().Append(m_settingsButton);
    btnPanel.Children().Append(m_newChatButton);
    btnPanel.Children().Append(m_historyButton);
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
        if (auto s = self.lock()) s->ShowSettings();
    });
    m_newChatButton.Click([self](IInspectable const&, RoutedEventArgs const&) {
        if (auto s = self.lock()) s->NewChat();
    });
    m_historyButton.Click([self](IInspectable const&, RoutedEventArgs const&) {
        if (auto s = self.lock()) s->ShowHistory();
    });

    // B button: cancel inference if running, otherwise let system exit the app
    auto nav = winrt::Windows::UI::Core::SystemNavigationManager::GetForCurrentView();
    nav.BackRequested([self](IInspectable const&,
                             winrt::Windows::UI::Core::BackRequestedEventArgs const& e) {
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
    m_root.KeyDown([self](IInspectable const&,
                          winrt::Windows::UI::Xaml::Input::KeyRoutedEventArgs const& e) {
        auto s = self.lock();
        if (!s) return;
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

    LoadModelName();
    CheckBenchMode();
}

// ---------------------------------------------------------------------------
// UI helpers (must be called on UI thread)
// ---------------------------------------------------------------------------

void MainPageController::AppendOutput(std::wstring const& text) {
    using namespace winrt::Windows::UI::Xaml::Documents;
    Run r;
    r.Text(text);
    m_currentParagraph.Inlines().Append(r);
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
    if (!running && m_flush_timer && m_flush_timer.IsEnabled()) {
        m_flush_timer.Stop();
        FlushTokenBuffer(); // drain any remaining tokens
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

std::string MainPageController::BuildChatMLPrompt(const std::string& user_text) const {
    // Estimate token count (heuristic: chars/4). Trim oldest turns if over limit.
    constexpr int kMaxEstimatedTokens = 3500;
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
        log_output("[xllama] context trimmed: dropped " + std::to_string(first_turn) + " old turn(s)\n");

    std::string prompt =
        "<|im_start|>system\n" + m_system_prompt + "<|im_end|>\n";

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

void MainPageController::SaveCurrentConversation(bool partial) {
    if (m_current.id.empty()) return;
    // Mark last assistant message as partial if needed
    if (partial && !m_current.messages.empty() &&
        m_current.messages.back().role == xllama::ui::MessageRole::Assistant) {
        m_current.messages.back().partial = true;
    }
    m_history.Save(m_current);
}

void MainPageController::NewChat() {
    if (m_is_running.load()) return; // don't allow while running
    SaveCurrentConversation(); // save current (no-op if empty)
    m_current = xllama::ui::Conversation{};
    m_current.id = xllama::ui::ChatHistory::NewId();
    m_outputBody.Blocks().Clear();
    m_currentParagraph = winrt::Windows::UI::Xaml::Documents::Paragraph();
    m_outputBody.Blocks().Append(m_currentParagraph);
    m_metricsText.Text(L"");
    SetStatus(L"New conversation");
}

void MainPageController::RenderConversation() {
    using namespace winrt::Windows::UI::Xaml::Documents;
    m_outputBody.Blocks().Clear();
    for (const auto& msg : m_current.messages) {
        if (msg.role == xllama::ui::MessageRole::System) continue;
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
    m_current = m_history.Load(id);
    if (m_current.id.empty()) { m_current.id = id; }
    RenderConversation();
    SetStatus(L"Conversation loaded");
}

winrt::fire_and_forget MainPageController::ShowHistory() {
    auto self = shared_from_this();
    if (m_is_running.load()) co_return;

    const auto& index = m_history.Index();
    if (index.empty()) {
        SetStatus(L"No history yet");
        co_return;
    }

    // Build a ListView with conversation titles
    winrt::Windows::UI::Xaml::Controls::ListView lv;
    lv.SelectionMode(winrt::Windows::UI::Xaml::Controls::ListViewSelectionMode::Single);
    lv.Height(400);
    for (const auto& meta : index) {
        winrt::Windows::UI::Xaml::Controls::TextBlock tb;
        tb.FontSize(16);
        tb.TextWrapping(winrt::Windows::UI::Xaml::TextWrapping::Wrap);
        wchar_t buf[128];
        swprintf_s(buf, L"%s  (%d msgs)",
                   ::xllama::utf8_to_wstring(meta.title).c_str(),
                   meta.n_messages);
        tb.Text(buf);
        lv.Items().Append(tb);
    }

    winrt::Windows::UI::Xaml::Controls::ContentDialog dlg;
    dlg.Title(winrt::box_value(L"Conversation History"));
    dlg.Content(lv);
    dlg.PrimaryButtonText(L"Open");
    dlg.CloseButtonText(L"Cancel");
    dlg.XamlRoot(m_root.XamlRoot());

    auto result = co_await dlg.ShowAsync();
    if (result != winrt::Windows::UI::Xaml::Controls::ContentDialogResult::Primary)
        co_return;

    int sel = lv.SelectedIndex();
    if (sel < 0 || sel >= static_cast<int>(index.size())) co_return;

    self->LoadConversation(index[static_cast<size_t>(sel)].id);
}

void MainPageController::LoadSettings() {
    // Read LocalFolder/settings.json: {"system_prompt":"..."}
    auto folder = winrt::Windows::Storage::ApplicationData::Current().LocalFolder();
    std::wstring wpath(folder.Path().c_str());
    wpath += L"\\settings.json";
    FILE* f = _wfopen(wpath.c_str(), L"r");
    if (!f) return;
    std::string json;
    char buf[8192];
    while (size_t n = fread(buf, 1, sizeof(buf) - 1, f)) { buf[n] = 0; json += buf; }
    fclose(f);
    // Minimal parse: find "system_prompt":"..."
    size_t key = json.find("\"system_prompt\"");
    if (key == std::string::npos) return;
    size_t colon = json.find(':', key);
    if (colon == std::string::npos) return;
    size_t quote = json.find('"', colon + 1);
    if (quote == std::string::npos) return;
    ++quote;
    std::string sp;
    while (quote < json.size()) {
        char c = json[quote++];
        if (c == '"') break;
        if (c == '\\' && quote < json.size()) {
            char e = json[quote++];
            if (e == 'n') sp += '\n'; else if (e == 't') sp += '\t'; else sp += e;
        } else sp += c;
    }
    if (!sp.empty()) m_system_prompt = sp;
}

void MainPageController::SaveSettings() {
    auto folder = winrt::Windows::Storage::ApplicationData::Current().LocalFolder();
    std::wstring wpath(folder.Path().c_str());
    wpath += L"\\settings.json";
    FILE* f = _wfopen(wpath.c_str(), L"w");
    if (!f) return;
    // Minimal JSON escape for system_prompt
    std::string esc;
    for (unsigned char c : m_system_prompt) {
        if (c == '"')       esc += "\\\"";
        else if (c == '\\') esc += "\\\\";
        else if (c == '\n') esc += "\\n";
        else if (c == '\r') esc += "\\r";
        else                esc += static_cast<char>(c);
    }
    fprintf(f, "{\"system_prompt\":\"%s\"}\n", esc.c_str());
    fclose(f);
}

winrt::fire_and_forget MainPageController::ShowSettings() {
    auto self = shared_from_this();
    if (m_is_running.load()) co_return;

    // System prompt TextBox
    winrt::Windows::UI::Xaml::Controls::TextBox sysPromptBox;
    sysPromptBox.Text(::xllama::utf8_to_wstring(m_system_prompt));
    sysPromptBox.AcceptsReturn(true);
    sysPromptBox.TextWrapping(TextWrapping::Wrap);
    sysPromptBox.MinHeight(120);
    sysPromptBox.FontSize(16);
    sysPromptBox.IsFocusEngagementEnabled(true);
    sysPromptBox.Header(winrt::box_value(L"System prompt"));

    winrt::Windows::UI::Xaml::Controls::ContentDialog dlg;
    dlg.Title(winrt::box_value(L"Settings"));
    dlg.Content(sysPromptBox);
    dlg.PrimaryButtonText(L"Save");
    dlg.CloseButtonText(L"Cancel");
    dlg.XamlRoot(m_root.XamlRoot());

    auto result = co_await dlg.ShowAsync();
    if (result != winrt::Windows::UI::Xaml::Controls::ContentDialogResult::Primary)
        co_return;

    self->m_system_prompt = ::xllama::wstring_to_utf8(std::wstring(sysPromptBox.Text().c_str()));
    self->SaveSettings();
    self->SetStatus(L"Settings saved");
}

// ---------------------------------------------------------------------------
// LoadModelName: read model.txt from LocalFolder
// ---------------------------------------------------------------------------

void MainPageController::LoadModelName() {
    auto path = local_wpath(L"model.txt");
    FILE* f = _wfopen(path.c_str(), L"r");
    if (f) {
        wchar_t buf[512] = {};
        if (fgetws(buf, 511, f)) {
            size_t len = wcslen(buf);
            while (len > 0 &&
                   (buf[len - 1] == L'\n' || buf[len - 1] == L'\r' || buf[len - 1] == L' '))
                buf[--len] = L'\0';
            m_model_filename = buf;
        }
        fclose(f);
    }
    if (m_model_filename.empty())
        m_model_filename = L"smollm2-360m-cpu-int4";

    m_modelText.Text(m_model_filename);
}

// ---------------------------------------------------------------------------
// CheckBenchMode: detect bench.flag and auto-run main_loop if present
// ---------------------------------------------------------------------------

fire_and_forget MainPageController::CheckBenchMode() {
    auto self = shared_from_this();

    co_await resume_background();

    auto flag_path = local_wpath(L"bench.flag");
    FILE* f = _wfopen(flag_path.c_str(), L"r");
    if (!f)
        co_return;
    fclose(f);

    co_await resume_foreground(self->m_root.Dispatcher());
    self->SetStatus(L"Bench mode — running...", StatusKind::Working);
    self->SetRunning(true);
    self->m_runButton.IsEnabled(false);

    co_await resume_background();
    ::xllama::bridge::main_loop();

    co_await resume_foreground(self->m_root.Dispatcher());
    self->SetStatus(L"Bench complete", StatusKind::Success);
    self->SetRunning(false);
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
    { std::lock_guard<std::mutex> lk(m_token_mutex); m_token_buffer.clear(); }
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
        if (auto s = weak_self.lock()) s->FlushTokenBuffer();
    });
    m_flush_timer.Start();

    // Build multi-turn ChatML prompt from conversation history.
    // BuildChatMLPrompt uses existing m_current.messages (prev turns) and appends user_text.
    std::string user_text = ::xllama::wstring_to_utf8(prompt_w);
    if (m_current.id.empty()) {
        m_current.id = xllama::ui::ChatHistory::NewId();
        m_current.title = xllama::ui::ChatHistory::TitleFrom(user_text);
    }
    std::string prompt = BuildChatMLPrompt(user_text);

    // Record user message in history AFTER building prompt (avoids duplicate)
    {
        xllama::ui::ChatMessage umsg;
        umsg.role = xllama::ui::MessageRole::User;
        umsg.content = user_text;
        umsg.ts_unix = static_cast<int64_t>(std::time(nullptr));
        m_current.messages.push_back(std::move(umsg));
    }

    std::string model = ::xllama::wstring_to_utf8(m_model_filename);
    auto dispatcher = m_root.Dispatcher();

    std::thread([self, prompt, model, dispatcher]() {
        try {
            ::xllama::bridge::InferenceParams params;
            params.model_path = model;
            params.prompt = prompt;
            params.n_predict = 512;
            params.abort_flag = &self->m_abort;

            params.on_status = [self, dispatcher](const std::string& s) {
                auto ws = ::xllama::utf8_to_wstring(s);
                StatusKind k = (s.rfind("error:", 0) == 0) ? StatusKind::Error : StatusKind::Working;
                dispatcher.RunAsync(CoreDispatcherPriority::Normal,
                                    [self, ws, k]() { self->SetStatus(ws, k); });
            };

            // Token accumulation — no per-token RunAsync dispatch (batched by flush timer)
            params.on_token = [self](const std::string& tok) {
                self->m_tokens_received.fetch_add(1, std::memory_order_relaxed);
                std::lock_guard<std::mutex> lk(self->m_token_mutex);
                self->m_token_buffer += tok;
            };

            auto res = ::xllama::bridge::run_inference(params);

            std::wstring metrics;
            if (res.success) {
                wchar_t buf[256];
                double dt = (res.n_eval > 0 && res.t_eval_ms > 0)
                                ? (double)res.n_eval / (res.t_eval_ms / 1000.0)
                                : 0.0;
                swprintf_s(buf, L"decode %.1f tok/s  ·  %d tok  ·  peak %zu MB",
                           dt, res.n_eval, res.peak_ws_mb);
                metrics = buf;
            } else {
                metrics = ::xllama::utf8_to_wstring(res.error_msg.empty() ? "inference failed"
                                                                          : res.error_msg);
            }

            std::string output_text = res.output_text;
            dispatcher.RunAsync(CoreDispatcherPriority::Normal, [self, metrics, res, output_text]() {
                self->m_metricsText.Text(metrics);
                self->SetStatus(res.success ? L"Done" : L"Error",
                                res.success ? StatusKind::Success : StatusKind::Error);
                self->SetRunning(false); // also stops timer + flushes remaining tokens
                // Save assistant response to conversation history
                if (res.success && !output_text.empty()) {
                    xllama::ui::ChatMessage amsg;
                    amsg.role = xllama::ui::MessageRole::Assistant;
                    amsg.content = output_text;
                    amsg.ts_unix = static_cast<int64_t>(std::time(nullptr));
                    self->m_current.messages.push_back(std::move(amsg));
                }
                self->SaveCurrentConversation();
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
