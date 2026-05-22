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
using namespace winrt::Windows::Storage;

namespace winrt::xllama::implementation {

// ---------------------------------------------------------------------------
// File logger: writes to LocalFolder/xllama.log + OutputDebugString
// ---------------------------------------------------------------------------
FILE* g_log_fp = nullptr;

static void log_init() {
    try {
        auto folder = ApplicationData::Current().LocalFolder();
        std::wstring wpath(folder.Path().c_str());
        wpath += L"\\xllama.log";
        g_log_fp = _wfopen(wpath.c_str(), L"a");
    } catch (...) {
    }
}

// Prefix every entry with HH:MM:SS.mmm for post-mortem correlation.
static void log_write(const char* msg) {
    SYSTEMTIME st = {};
    GetSystemTime(&st);
    char ts[20];
    snprintf(ts, sizeof(ts), "%02u:%02u:%02u.%03u ", st.wHour, st.wMinute, st.wSecond,
             st.wMilliseconds);
    OutputDebugStringA(ts);
    OutputDebugStringA(msg);
    if (g_log_fp) {
        fputs(ts, g_log_fp);
        fputs(msg, g_log_fp);
        fflush(g_log_fp);
    }
}

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------

App::App() {
    log_init();
    log_write("[xllama] App::App()\n");

    // Xbox UX: disable mouse cursor mode, force dark theme
    RequiresPointerMode(ApplicationRequiresPointerMode::WhenRequested);
    RequestedTheme(ApplicationTheme::Dark);

    // Reveal focus highlight (Xbox only)
    if (winrt::Windows::System::Profile::AnalyticsInfo::VersionInfo().DeviceFamily() ==
        L"Windows.Xbox") {
        FocusVisualKind(winrt::Windows::UI::Xaml::FocusVisualKind::Reveal);
    }

    // Sound effects (controller focus/click cues)
    ElementSoundPlayer::State(ElementSoundPlayerState::On);

    // Catch unhandled exceptions in the XAML framework
    UnhandledException([](winrt::Windows::Foundation::IInspectable const&,
                          winrt::Windows::UI::Xaml::UnhandledExceptionEventArgs const& e) {
        char buf[256];
        snprintf(buf, sizeof(buf), "[xllama] UnhandledException: 0x%08X\n",
                 static_cast<unsigned>(e.Exception().value));
        if (g_log_fp) {
            fputs(buf, g_log_fp);
            fflush(g_log_fp);
        }
        OutputDebugStringA(buf);
        e.Handled(false); // let it propagate
    });

    log_write("[xllama] App::App() complete\n");
}

void App::OnLaunched(LaunchActivatedEventArgs const&) {
    log_write("[xllama] App::OnLaunched\n");

    try {
        if (!m_controller) {
            log_write("[xllama] building MainPageController\n");
            m_controller = std::make_shared<::xllama::MainPageController>();
            log_write("[xllama] MainPageController built\n");
            m_controller->Init();
            log_write("[xllama] MainPageController init done\n");
        }
        Window::Current().Content(m_controller->Root());
        log_write("[xllama] Window.Content set\n");
        Window::Current().Activate();
        log_write("[xllama] Window activated\n");
    } catch (winrt::hresult_error const& e) {
        char buf[512];
        snprintf(buf, sizeof(buf), "[xllama] OnLaunched hresult 0x%08X: %ls\n",
                 static_cast<unsigned>(e.code().value), e.message().c_str());
        log_write(buf);
        throw;
    } catch (std::exception const& e) {
        char buf[512];
        snprintf(buf, sizeof(buf), "[xllama] OnLaunched std::exception: %s\n", e.what());
        log_write(buf);
        throw;
    } catch (...) {
        log_write("[xllama] OnLaunched unknown exception\n");
        throw;
    }
}

} // namespace winrt::xllama::implementation

// ---------------------------------------------------------------------------
// Entry point — Application::Start replaces CoreApplication::Run
// ---------------------------------------------------------------------------
int __stdcall wWinMain(HINSTANCE, HINSTANCE, PWSTR, int) {
    try {
        winrt::Windows::UI::Xaml::Application::Start(
            [](auto&&) { winrt::make<winrt::xllama::implementation::App>(); });
    } catch (winrt::hresult_error const& e) {
        char buf[256];
        snprintf(buf, sizeof(buf), "[xllama] wWinMain exception: 0x%08X\n",
                 static_cast<unsigned>(e.code().value));
        OutputDebugStringA(buf);
        if (winrt::xllama::implementation::g_log_fp) {
            fputs(buf, winrt::xllama::implementation::g_log_fp);
            fflush(winrt::xllama::implementation::g_log_fp);
        }
    } catch (std::exception const& e) {
        char buf[512];
        snprintf(buf, sizeof(buf), "[xllama] wWinMain std::exception: %s\n", e.what());
        OutputDebugStringA(buf);
        if (winrt::xllama::implementation::g_log_fp) {
            fputs(buf, winrt::xllama::implementation::g_log_fp);
            fflush(winrt::xllama::implementation::g_log_fp);
        }
    } catch (...) {
        OutputDebugStringA("[xllama] wWinMain: unknown exception\n");
    }
    return 0;
}

#endif // XLLAMA_UWP
