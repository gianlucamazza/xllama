// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

#ifdef XLLAMA_UWP

    #include "App.h"
    #include "MainPage.h"
    #include "pch.h"

using namespace winrt;
using namespace winrt::Windows::ApplicationModel::Activation;
using namespace winrt::Windows::UI::Xaml;
using namespace winrt::Windows::UI::Xaml::Controls;
using namespace winrt::Windows::Storage;

namespace winrt::xllama::implementation {

// ---------------------------------------------------------------------------
// File logger: writes to LocalFolder/xllama.log + OutputDebugString
// ---------------------------------------------------------------------------
static FILE* g_log_fp = nullptr;

static void log_init() {
    try {
        auto folder = ApplicationData::Current().LocalFolder();
        std::wstring wpath(folder.Path().c_str());
        wpath += L"\\xllama.log";
        g_log_fp = _wfopen(wpath.c_str(), L"a");
    } catch (...) {
    }
}

static void log_write(const char* msg) {
    OutputDebugStringA(msg);
    if (g_log_fp) {
        fputs(msg, g_log_fp);
        fflush(g_log_fp);
    }
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

void App::InitializeComponent() {
    // App.xaml is minimal (no resources). No-op: prevents AppT<App>::InitializeComponent
    // from attempting to load App.xbf which is not a valid XBF in our package.
}

App::App() {
    log_init();
    log_write("[xllama] App::App()\n");
    InitializeComponent();
}

void App::OnLaunched(LaunchActivatedEventArgs const&) {
    log_write("[xllama] App::OnLaunched\n");

    auto rootFrame = Window::Current().Content().try_as<Frame>();
    if (!rootFrame) {
        rootFrame = Frame();
        Window::Current().Content(rootFrame);
    }

    if (rootFrame.Content() == nullptr) {
        rootFrame.Navigate(xaml_typename<xllama::MainPage>());
    }

    Window::Current().Activate();
    log_write("[xllama] Window activated\n");
}

} // namespace winrt::xllama::implementation

// ---------------------------------------------------------------------------
// Entry point — Application::Start replaces CoreApplication::Run
// ---------------------------------------------------------------------------
int __stdcall wWinMain(HINSTANCE, HINSTANCE, PWSTR, int) {
    winrt::init_apartment();
    winrt::Windows::UI::Xaml::Application::Start(
        [](auto&&) { winrt::make<winrt::xllama::implementation::App>(); });
}

#endif // XLLAMA_UWP
