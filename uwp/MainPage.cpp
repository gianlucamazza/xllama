#ifdef XLLAMA_UWP

#include "pch.h"
#include <winrt/Microsoft.UI.Xaml.h>
#include <winrt/Microsoft.UI.Xaml.Controls.h>
#include <winrt/Microsoft.UI.Xaml.XamlTypeInfo.h>
#include "MainPage.h"
#include "MainPage.g.cpp"

#include <cstdio>
#include <string>
#include <thread>

using namespace winrt;
using namespace winrt::Windows::Foundation;
using namespace winrt::Windows::Storage;
using namespace winrt::Windows::UI::Core;
using namespace winrt::Windows::UI::Xaml;
using namespace winrt::Windows::UI::Xaml::Controls;

namespace winrt::xllama::implementation {

// ---------------------------------------------------------------------------
// Helper: resolve a filename relative to ApplicationData::LocalFolder
// ---------------------------------------------------------------------------
static std::wstring local_path(const wchar_t* filename) {
    auto folder = ApplicationData::Current().LocalFolder();
    std::wstring path(folder.Path().c_str());
    path += L"\\";
    path += filename;
    return path;
}

static std::string wstr_to_utf8(std::wstring const& w) {
    if (w.empty()) return {};
    int sz = WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, nullptr, 0, nullptr, nullptr);
    std::string r(sz, '\0');
    WideCharToMultiByte(CP_UTF8, 0, w.c_str(), -1, r.data(), sz, nullptr, nullptr);
    if (!r.empty() && r.back() == '\0') r.pop_back();
    return r;
}

static std::wstring utf8_to_wstr(std::string const& s) {
    if (s.empty()) return {};
    int sz = MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, nullptr, 0);
    std::wstring r(sz, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, s.c_str(), -1, r.data(), sz);
    if (!r.empty() && r.back() == L'\0') r.pop_back();
    return r;
}

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

MainPage::MainPage() {
    InitializeComponent();
    LoadModelName();
    CheckBenchMode();
}

// ---------------------------------------------------------------------------
// UI helpers (must be called on UI thread)
// ---------------------------------------------------------------------------

void MainPage::AppendOutput(std::wstring const& text) {
    OutputText().Text(OutputText().Text() + text);
    // Auto-scroll to bottom
    OutputScroll().UpdateLayout();
    OutputScroll().ChangeView(nullptr, OutputScroll().ScrollableHeight(), nullptr);
}

void MainPage::SetStatus(std::wstring const& status) {
    StatusText().Text(status);
}

void MainPage::SetRunning(bool running) {
    RunButton().IsEnabled(!running);
    CancelButton().IsEnabled(running);
    LoadingBar().Visibility(running ? Visibility::Visible : Visibility::Collapsed);
}

// ---------------------------------------------------------------------------
// LoadModelName: read model.txt from LocalFolder
// ---------------------------------------------------------------------------

void MainPage::LoadModelName() {
    auto path = local_path(L"model.txt");
    FILE* f = _wfopen(path.c_str(), L"r");
    if (f) {
        wchar_t buf[512] = {};
        if (fgetws(buf, 511, f)) {
            // trim trailing whitespace
            size_t len = wcslen(buf);
            while (len > 0 && (buf[len-1] == L'\n' || buf[len-1] == L'\r' ||
                               buf[len-1] == L' '))
                buf[--len] = L'\0';
            m_model_filename = buf;
        }
        fclose(f);
    }
    if (m_model_filename.empty())
        m_model_filename = L"Qwen_Qwen3-1.7B-Q4_K_M.gguf";

    ModelText().Text(m_model_filename);
}

// ---------------------------------------------------------------------------
// CheckBenchMode: detect bench.flag and auto-run main_loop if present
// ---------------------------------------------------------------------------

fire_and_forget MainPage::CheckBenchMode() {
    auto self = get_strong();

    co_await resume_background();

    auto flag_path = local_path(L"bench.flag");
    FILE* f = _wfopen(flag_path.c_str(), L"r");
    if (!f) co_return;
    fclose(f);

    // Bench mode detected — update UI and run main_loop
    co_await resume_foreground(self->Dispatcher());
    self->SetStatus(L"Bench mode — running...");
    self->SetRunning(true);
    self->RunButton().IsEnabled(false);

    co_await resume_background();
    xllama::bridge::main_loop();

    co_await resume_foreground(self->Dispatcher());
    self->SetStatus(L"Bench complete");
    self->SetRunning(false);
}

// ---------------------------------------------------------------------------
// StartInference: called on UI thread; spawns background thread
// ---------------------------------------------------------------------------

void MainPage::StartInference(std::wstring const& prompt_w) {
    m_abort.store(false);
    SetStatus(L"Loading model...");
    SetRunning(true);
    OutputText().Text(L"");
    MetricsText().Text(L"");

    auto self = get_strong();
    std::string prompt = wstr_to_utf8(prompt_w);
    std::string model  = wstr_to_utf8(m_model_filename);

    std::thread([self, prompt, model]() {
        xllama::bridge::InferenceParams params;
        params.model_path = model;
        params.prompt     = prompt;
        params.n_predict  = 512;
        params.abort_flag = &self->m_abort;

        auto dispatcher = self->Dispatcher();

        params.on_status = [self, dispatcher](const std::string& s) {
            auto ws = utf8_to_wstr(s);
            dispatcher.RunAsync(CoreDispatcherPriority::Normal,
                [self, ws]() { self->SetStatus(ws); });
        };

        params.on_token = [self, dispatcher](const std::string& tok) {
            auto wtok = utf8_to_wstr(tok);
            dispatcher.RunAsync(CoreDispatcherPriority::Normal,
                [self, wtok]() { self->AppendOutput(wtok); });
        };

        auto res = xllama::bridge::run_inference(params);

        // Format metrics
        std::wstring metrics;
        if (res.success) {
            wchar_t buf[256];
            double pt = (res.n_p_eval > 0 && res.t_p_eval_ms > 0)
                ? (double)res.n_p_eval / (res.t_p_eval_ms / 1000.0) : 0.0;
            double dt = (res.n_eval > 0 && res.t_eval_ms > 0)
                ? (double)res.n_eval / (res.t_eval_ms / 1000.0) : 0.0;
            swprintf_s(buf, L"prompt %.1f tok/s  ·  decode %.1f tok/s  ·  peak %zu MB",
                       pt, dt, res.peak_ws_mb);
            metrics = buf;
        } else {
            metrics = utf8_to_wstr(res.error_msg.empty() ? "inference failed" : res.error_msg);
        }

        dispatcher.RunAsync(CoreDispatcherPriority::Normal, [self, metrics, res]() {
            self->MetricsText().Text(metrics);
            self->SetStatus(res.success ? L"Done" : L"Error");
            self->SetRunning(false);
        });
    }).detach();
}

// ---------------------------------------------------------------------------
// Event handlers
// ---------------------------------------------------------------------------

void MainPage::OnRunClick(IInspectable const&, RoutedEventArgs const&) {
    auto prompt = PromptInput().Text();
    if (prompt.empty()) {
        SetStatus(L"Enter a prompt first");
        return;
    }
    StartInference(std::wstring(prompt.c_str()));
}

void MainPage::OnCancelClick(IInspectable const&, RoutedEventArgs const&) {
    m_abort.store(true);
    SetStatus(L"Cancelling...");
    CancelButton().IsEnabled(false);
}

} // namespace winrt::xllama::implementation

#endif // XLLAMA_UWP
