#ifdef XLLAMA_UWP

#include "App.h"
#include "llama-bridge.h"

#include <thread>
#include <string>

using namespace winrt;
using namespace winrt::Windows::ApplicationModel::Core;
using namespace winrt::Windows::Foundation;
using namespace winrt::Windows::UI::Core;

namespace xllama {

IFrameworkView App::CreateView() {
    return *this;
}

void App::Initialize(CoreApplicationView const&) {
    OutputDebugStringA("[xllama] Initialize\n");
}

void App::SetWindow(CoreWindow const&) {
    OutputDebugStringA("[xllama] SetWindow\n");
}

void App::Load(hstring const&) {
    OutputDebugStringA("[xllama] Load\n");
}

void App::Run() {
    OutputDebugStringA("[xllama] Run — starting inference thread\n");

    // Run inference on a background thread; keep the message pump alive.
    std::thread inference_thread([]() {
        xllama::bridge::main_loop();
    });

    CoreWindow window = CoreWindow::GetForCurrentThread();
    window.Activate();

    CoreDispatcher dispatcher = window.Dispatcher();
    dispatcher.ProcessEvents(CoreProcessEventsOption::ProcessUntilQuit);

    if (inference_thread.joinable()) {
        inference_thread.join();
    }
}

void App::Uninitialize() {
    OutputDebugStringA("[xllama] Uninitialize\n");
}

} // namespace xllama

int __stdcall wWinMain(HINSTANCE, HINSTANCE, PWSTR, int) {
    winrt::init_apartment();
    CoreApplication::Run(winrt::make<xllama::App>());
}

#endif // XLLAMA_UWP
