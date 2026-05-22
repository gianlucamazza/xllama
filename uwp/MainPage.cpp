// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

#ifdef XLLAMA_UWP

// clang-format off
#include "pch.h"
#include "MainPage.h"
// clang-format on

    #include "inference-bridge.h"
    #include "xllama/platform.h"
    #include "xllama/utf8_utils.h"

    #include <cstdio>
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
    m_outputScroll.XYFocusKeyboardNavigation(XYFocusKeyboardNavigationMode::Enabled);

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

    m_runButton = Button();
    m_runButton.Content(winrt::box_value(L"▶  Run"));
    m_runButton.MinWidth(120);
    m_runButton.Margin(ThicknessHelper::FromLengths(0, 0, 12, 0));

    m_cancelButton = Button();
    m_cancelButton.Content(winrt::box_value(L"■  Cancel"));
    m_cancelButton.IsEnabled(false);
    m_cancelButton.MinWidth(120);

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
            m_statusText.Text(L"⏳ " + msg);
            break;
        case StatusKind::Success:
            m_statusText.Foreground(Media::SolidColorBrush({255, 100, 220, 100})); // green
            m_statusText.Opacity(1.0);
            m_statusText.Text(L"✓ " + msg);
            break;
        case StatusKind::Error:
            m_statusText.Foreground(Media::SolidColorBrush({255, 240, 80, 70})); // red
            m_statusText.Opacity(1.0);
            m_statusText.Text(L"⚠ " + msg);
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

    // Reset output body for new inference
    m_outputBody.Blocks().Clear();
    m_currentParagraph = winrt::Windows::UI::Xaml::Documents::Paragraph();
    m_outputBody.Blocks().Append(m_currentParagraph);
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

    // Wrap user text in ChatML template (required for SmolLM2-360M-Instruct).
    // Bare prompts cause the model to generate EOS immediately.
    std::string user_text = ::xllama::wstring_to_utf8(prompt_w);
    std::string prompt =
        "<|im_start|>system\nYou are a helpful AI assistant.<|im_end|>\n"
        "<|im_start|>user\n" + user_text + "<|im_end|>\n"
        "<|im_start|>assistant\n";
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

            dispatcher.RunAsync(CoreDispatcherPriority::Normal, [self, metrics, res]() {
                self->m_metricsText.Text(metrics);
                self->SetStatus(res.success ? L"Done" : L"Error",
                                res.success ? StatusKind::Success : StatusKind::Error);
                self->SetRunning(false); // also stops timer + flushes remaining tokens
            });
        } catch (const std::exception& ex) {
            ::xllama::log_output(std::string("[xllama] thread terminated: ") + ex.what() + "\n");
            dispatcher.RunAsync(CoreDispatcherPriority::Normal, [self]() {
                self->SetStatus(L"Fatal error — see xllama.log", StatusKind::Error);
                self->SetRunning(false);
            });
        } catch (...) {
            ::xllama::log_output("[xllama] thread terminated: unknown exception\n");
            dispatcher.RunAsync(CoreDispatcherPriority::Normal, [self]() {
                self->SetStatus(L"Fatal error — see xllama.log", StatusKind::Error);
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
