#ifdef XLLAMA_UWP

#include "App.h"
#include "llama-bridge.h"

#include <thread>
#include <string>
#include <cstdio>

using namespace winrt;
using namespace winrt::Windows::ApplicationModel::Core;
using namespace winrt::Windows::Foundation;
using namespace winrt::Windows::Storage;
using namespace winrt::Windows::UI::Core;

namespace xllama {

// ---------------------------------------------------------------------------
// File logger: writes to LocalFolder/xllama.log + OutputDebugString
// ---------------------------------------------------------------------------
static FILE * g_log_fp = nullptr;

static void log_init() {
    try {
        auto folder = ApplicationData::Current().LocalFolder();
        auto path   = folder.Path();
        // Build a narrow path for fopen
        std::wstring wpath(path.c_str());
        wpath += L"\\xllama.log";
        g_log_fp = _wfopen(wpath.c_str(), L"a");
    } catch (...) {}
}

static void log_write(const char * msg) {
    OutputDebugStringA(msg);
    if (g_log_fp) {
        fputs(msg, g_log_fp);
        fflush(g_log_fp);
    }
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

IFrameworkView App::CreateView() {
    return *this;
}

void App::Initialize(CoreApplicationView const&) {
    log_init();
    log_write("[xllama] Initialize\n");
}

void App::SetWindow(CoreWindow const&) {
    log_write("[xllama] SetWindow\n");
}

void App::Load(hstring const&) {
    log_write("[xllama] Load\n");
}

void App::Run() {
    log_write("[xllama] Run — starting inference thread\n");

    // Run inference on a background thread.
    // winrt apartment is already initialised by wWinMain below.
    std::thread inference_thread([]() {
        try {
            xllama::bridge::main_loop();
        } catch (const std::exception & e) {
            log_write(("[xllama] exception in inference thread: " +
                       std::string(e.what()) + "\n").c_str());
        } catch (...) {
            log_write("[xllama] unknown exception in inference thread\n");
        }
        log_write("[xllama] inference thread finished\n");
    });

    CoreWindow window = CoreWindow::GetForCurrentThread();
    window.Activate();

    CoreDispatcher dispatcher = window.Dispatcher();
    dispatcher.ProcessEvents(CoreProcessEventsOption::ProcessUntilQuit);

    if (inference_thread.joinable()) inference_thread.join();

    log_write("[xllama] Run exiting\n");
}

void App::Uninitialize() {
    log_write("[xllama] Uninitialize\n");
    if (g_log_fp) { fclose(g_log_fp); g_log_fp = nullptr; }
}

} // namespace xllama

int __stdcall wWinMain(HINSTANCE, HINSTANCE, PWSTR, int) {
    winrt::init_apartment();
    CoreApplication::Run(winrt::make<xllama::App>());
}

#endif // XLLAMA_UWP
