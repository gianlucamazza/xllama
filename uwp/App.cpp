// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

#ifdef XLLAMA_UWP

// clang-format off
// pch.h must be first: it includes <unknwn.h> before WinRT headers (COM IUnknown guard).
#include "pch.h"
#include "App.h"
#include "MainPage.h"
// clang-format on

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

// Bootstrap log: write to package LocalState via GetCurrentPackagePath.
// GetTempPath is inaccessible from UWP; LocalState is writable by the package.
static void boot_log(const char* msg) {
    OutputDebugStringA(msg);
    UINT32 len = 0;
    // First call to get required length
    GetCurrentPackagePath(&len, nullptr);
    if (len == 0) return;
    std::wstring pkg(len, L'\0');
    if (GetCurrentPackagePath(&len, pkg.data()) != ERROR_SUCCESS) return;
    // Package root: trim trailing NUL, append \LocalState\xllama-boot.log
    pkg.resize(wcslen(pkg.c_str()));
    std::wstring path = pkg + L"\\LocalState\\xllama-boot.log";
    if (FILE* fp = _wfopen(path.c_str(), L"a")) {
        fputs(msg, fp);
        fclose(fp);
    }
}

int __stdcall wWinMain(HINSTANCE, HINSTANCE, PWSTR, int) {
    boot_log("[boot] wWinMain\n");
    try {
        winrt::Windows::UI::Xaml::Application::Start([](auto&&) {
            boot_log("[boot] callback: making App\n");
            winrt::make<winrt::xllama::implementation::App>();
            boot_log("[boot] App created\n");
        });
        boot_log("[boot] Application::Start returned\n");
    } catch (winrt::hresult_error const& e) {
        char buf[256];
        snprintf(buf, sizeof(buf), "[boot] hresult 0x%08X\n",
                 static_cast<unsigned>(e.code().value));
        boot_log(buf);
    } catch (...) {
        boot_log("[boot] unknown exception\n");
    }
    boot_log("[boot] exit\n");
    return 0;
}

#endif // XLLAMA_UWP
