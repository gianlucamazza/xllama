// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

#ifdef XLLAMA_UWP

// clang-format off
#include "pch.h"
#include "MainPage.h"
// clang-format on

    #include "inference-bridge.h"
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
    outerGrid.Margin(ThicknessHelper::FromUniformLength(48)); // Xbox TV safe area

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

    StackPanel bodyStack;

    m_promptInput = TextBox();
    m_promptInput.PlaceholderText(L"Type your prompt here (gamepad A → opens keyboard)...");
    m_promptInput.AcceptsReturn(true);
    m_promptInput.TextWrapping(TextWrapping::Wrap);
    m_promptInput.MinHeight(120);
    m_promptInput.IsSpellCheckEnabled(false);
    m_promptInput.FontSize(18);

    m_outputText = TextBlock();
    m_outputText.TextWrapping(TextWrapping::Wrap);
    m_outputText.FontFamily(winrt::Windows::UI::Xaml::Media::FontFamily(L"Consolas"));
    m_outputText.FontSize(20);
    m_outputText.Margin(ThicknessHelper::FromLengths(0, 16, 0, 0));
    m_outputText.IsTextSelectionEnabled(true);

    bodyStack.Children().Append(m_promptInput);
    bodyStack.Children().Append(m_outputText);
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

    LoadModelName();
    CheckBenchMode();
}

// ---------------------------------------------------------------------------
// UI helpers (must be called on UI thread)
// ---------------------------------------------------------------------------

void MainPageController::AppendOutput(std::wstring const& text) {
    m_outputText.Text(m_outputText.Text() + text);
    m_outputScroll.UpdateLayout();
    m_outputScroll.ChangeView(nullptr, m_outputScroll.ScrollableHeight(), nullptr);
}

void MainPageController::SetStatus(std::wstring const& status) {
    m_statusText.Text(status);
}

void MainPageController::SetRunning(bool running) {
    m_runButton.IsEnabled(!running);
    m_cancelButton.IsEnabled(running);
    m_loadingBar.Visibility(running ? Visibility::Visible : Visibility::Collapsed);
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
        m_model_filename = L"Phi-3.5-mini-instruct-onnx-directml";

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
    self->SetStatus(L"Bench mode — running...");
    self->SetRunning(true);
    self->m_runButton.IsEnabled(false);

    co_await resume_background();
    ::xllama::bridge::main_loop();

    co_await resume_foreground(self->m_root.Dispatcher());
    self->SetStatus(L"Bench complete");
    self->SetRunning(false);
}

// ---------------------------------------------------------------------------
// StartInference: called on UI thread; spawns background thread
// ---------------------------------------------------------------------------

void MainPageController::StartInference(std::wstring const& prompt_w) {
    m_abort.store(false);
    SetStatus(L"Loading model...");
    SetRunning(true);
    m_outputText.Text(L"");
    m_metricsText.Text(L"");

    auto self = shared_from_this();
    std::string prompt = ::xllama::wstring_to_utf8(prompt_w);
    std::string model = ::xllama::wstring_to_utf8(m_model_filename);
    auto dispatcher = m_root.Dispatcher();

    std::thread([self, prompt, model, dispatcher]() {
        ::xllama::bridge::InferenceParams params;
        params.model_path = model;
        params.prompt = prompt;
        params.n_predict = 512;
        params.abort_flag = &self->m_abort;

        params.on_status = [self, dispatcher](const std::string& s) {
            auto ws = ::xllama::utf8_to_wstring(s);
            dispatcher.RunAsync(CoreDispatcherPriority::Normal,
                                [self, ws]() { self->SetStatus(ws); });
        };

        params.on_token = [self, dispatcher](const std::string& tok) {
            auto wtok = ::xllama::utf8_to_wstring(tok);
            dispatcher.RunAsync(CoreDispatcherPriority::Normal,
                                [self, wtok]() { self->AppendOutput(wtok); });
        };

        auto res = ::xllama::bridge::run_inference(params);

        std::wstring metrics;
        if (res.success) {
            wchar_t buf[256];
            double pt = (res.n_p_eval > 0 && res.t_p_eval_ms > 0)
                            ? (double)res.n_p_eval / (res.t_p_eval_ms / 1000.0)
                            : 0.0;
            double dt = (res.n_eval > 0 && res.t_eval_ms > 0)
                            ? (double)res.n_eval / (res.t_eval_ms / 1000.0)
                            : 0.0;
            swprintf_s(buf, L"prompt %.1f tok/s  ·  decode %.1f tok/s  ·  peak %zu MB", pt, dt,
                       res.peak_ws_mb);
            metrics = buf;
        } else {
            metrics = ::xllama::utf8_to_wstring(res.error_msg.empty() ? "inference failed"
                                                                      : res.error_msg);
        }

        dispatcher.RunAsync(CoreDispatcherPriority::Normal, [self, metrics, res]() {
            self->m_metricsText.Text(metrics);
            self->SetStatus(res.success ? L"Done" : L"Error");
            self->SetRunning(false);
        });
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
