// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

#ifdef XLLAMA_UWP

// clang-format off
// pch.h must be first: it includes <unknwn.h> before WinRT headers (COM IUnknown guard).
#include "pch.h"
#include "App.h"
#include "MainPage.h"
#include "api-server.h"
#include "inference-bridge.h"
#include "xllama/platform.h"
// clang-format on

    #include <thread>

// Defined in the headless section below; also used by the in-process
// diffusion experiment in App::OnLaunched.
static std::wstring flag_path_if_present(const wchar_t* name);

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

        // §7 experiment: the 887A0036 device conflict was measured with ORT
        // GenAI's Agility-factory device; diffuse.cpp uses plain ORT DML, whose
        // device may coexist with the compositor device the line above just
        // created. diffuse-inproc.flag runs the diffusion pipeline on a
        // background MTA thread INSIDE the XAML process to test exactly that.
        // Consumed before the run, same semantics as the headless flags.
        std::wstring inproc = flag_path_if_present(L"diffuse-inproc.flag");
        if (!inproc.empty()) {
            _wremove(inproc.c_str());
            log_write("[xllama] diffuse-inproc.flag detected -> in-process diffusion "
                      "(compositor alive)\n");
            std::thread([] {
                winrt::init_apartment(); // MTA: run_diffuse uses ApplicationData
                ::xllama::bridge::run_diffuse();
                ::xllama::log_output("[xllama] diffuse-inproc: run returned (result/error above; "
                                     "XAML window still up)\n");
            }).detach();
        }

        // Autopilot: scripted validation of the live XAML UI (no-op unless
        // LocalState\autopilot.flag exists). Consumes the flag and drives the
        // controller from a background thread — see MainPage.cpp.
        if (m_controller)
            m_controller->StartAutopilotIfRequested();

        // LAN HTTP endpoint (OpenAI-compat), opt-in and default OFF: started
        // only when LocalState\api.flag exists. The flag is NOT consumed — the
        // server is persistent and coexists with the live XAML chat UI, unlike
        // the headless bench/diffuse flags. Runs on a detached MTA thread; the
        // StreamSocketListener stays bound for the app lifetime. See
        // api-server.cpp.
        if (!flag_path_if_present(L"api.flag").empty()) {
            log_write("[xllama] api.flag detected -> starting LAN HTTP endpoint\n");
            std::thread([] {
                winrt::init_apartment(); // MTA: WinRT sockets + ApplicationData
                ::xllama::api::run_server();
            }).detach();
        }
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
// Headless bench mode — no XAML, no compositor D3D12 device
//
// The XAML compositor (D3D11on12 on Xbox) creates a process-wide D3D12 device
// at Window.Activate(), before any OgaCreateModel. ORT GenAI's DML EP then
// creates its own device through the Agility SDK device factory
// (dml_helpers.cpp: CreateDeviceFactory(614) + factory->CreateDevice), which
// collides with the compositor's in-box-runtime device and throws 887A0036
// DXGI_ERROR_ALREADY_EXISTS. Running the bench without XAML leaves the process
// D3D12-clean so the DML EP can initialise.
// ---------------------------------------------------------------------------

// Returns the wide path of LocalFolder\<name> if it exists, empty otherwise.
// Requires an initialised COM apartment (ApplicationData). Any exception =>
// empty => normal interactive path.
static std::wstring flag_path_if_present(const wchar_t* name) {
    try {
        auto folder = winrt::Windows::Storage::ApplicationData::Current().LocalFolder();
        std::wstring flag = std::wstring(folder.Path().c_str()) + L"\\" + name;
        DWORD attr = GetFileAttributesW(flag.c_str());
        if (attr != INVALID_FILE_ATTRIBUTES && !(attr & FILE_ATTRIBUTE_DIRECTORY))
            return flag;
    } catch (...) {
    }
    return {};
}

// Minimal IFrameworkView: activates the CoreWindow (satisfies the PLM
// activation watchdog, dismisses the splash) without a swapchain/compositor,
// so no D3D12 device exists in-process. Same pattern as the UWP DX12 template.
// Runs `m_entry` (bench main_loop or run_diffuse) on an MTA thread, then
// exits the process — a D3D12-clean host for any DirectML workload.
struct HeadlessView
    : winrt::implements<HeadlessView, winrt::Windows::ApplicationModel::Core::IFrameworkViewSource,
                        winrt::Windows::ApplicationModel::Core::IFrameworkView> {
    void (*m_entry)() = nullptr;
    const char* m_label = "headless";
    explicit HeadlessView(void (*entry)(), const char* label) : m_entry(entry), m_label(label) {}

    winrt::Windows::ApplicationModel::Core::IFrameworkView CreateView() {
        return *this;
    }
    void Initialize(winrt::Windows::ApplicationModel::Core::CoreApplicationView const&) {}
    void Load(winrt::hstring const&) {}
    void Uninitialize() {}
    void SetWindow(winrt::Windows::UI::Core::CoreWindow const&) {}
    void Run() {
        auto window = winrt::Windows::UI::Core::CoreWindow::GetForCurrentThread();
        window.Activate();
        auto dispatcher = window.Dispatcher();
        void (*entry)() = m_entry;
        const char* label = m_label;
        std::thread([dispatcher, entry, label]() {
            winrt::init_apartment(); // MTA: entry uses ApplicationData (WinRT)
            ::xllama::log_output(std::string("[xllama] headless ") + label + ": starting\n");
            entry();
            ::xllama::log_output(std::string("[xllama] headless ") + label + ": done, exiting\n");
            dispatcher.RunAsync(winrt::Windows::UI::Core::CoreDispatcherPriority::Normal, [] {
                winrt::Windows::ApplicationModel::Core::CoreApplication::Exit();
            });
        }).detach();
        dispatcher.ProcessEvents(
            winrt::Windows::UI::Core::CoreProcessEventsOption::ProcessUntilQuit);
    }
};

// ---------------------------------------------------------------------------
// Entry point — Application::Start replaces CoreApplication::Run
// ---------------------------------------------------------------------------
int __stdcall wWinMain(HINSTANCE, HINSTANCE, PWSTR, int) {
    try {
        winrt::init_apartment(); // MTA: needed for ApplicationData in the detection
        // Consume the flag BEFORE the run:
        // a later start without the flag goes back to interactive.
        std::wstring diffuse_flag = flag_path_if_present(L"diffuse.flag");
        if (!diffuse_flag.empty()) {
            _wremove(diffuse_flag.c_str());
            ::xllama::log_output("[xllama] diffuse.flag detected -> headless diffusion mode\n");
            winrt::Windows::ApplicationModel::Core::CoreApplication::Run(
                winrt::make<HeadlessView>(&::xllama::bridge::run_diffuse, "diffuse"));
            return 0; // not reached: CoreApplication::Exit terminates the process
        }
        std::wstring bench_flag = flag_path_if_present(L"bench.flag");
        if (!bench_flag.empty()) {
            _wremove(bench_flag.c_str());
            ::xllama::log_output("[xllama] bench.flag detected -> headless bench mode\n");
            winrt::Windows::ApplicationModel::Core::CoreApplication::Run(
                winrt::make<HeadlessView>(&::xllama::bridge::main_loop, "bench"));
            return 0; // not reached: CoreApplication::Exit terminates the process
        }
        std::wstring membw_flag = flag_path_if_present(L"membw.flag");
        if (!membw_flag.empty()) {
            _wremove(membw_flag.c_str());
            ::xllama::log_output("[xllama] membw.flag detected -> headless membw mode\n");
            winrt::Windows::ApplicationModel::Core::CoreApplication::Run(
                winrt::make<HeadlessView>(&::xllama::bridge::run_membw, "membw"));
            return 0; // not reached: CoreApplication::Exit terminates the process
        }
        std::wstring logits_flag = flag_path_if_present(L"logits.flag");
        if (!logits_flag.empty()) {
            _wremove(logits_flag.c_str());
            ::xllama::log_output("[xllama] logits.flag detected -> headless logit-parity dump\n");
            winrt::Windows::ApplicationModel::Core::CoreApplication::Run(
                winrt::make<HeadlessView>(&::xllama::bridge::run_logits, "logits"));
            return 0; // not reached: CoreApplication::Exit terminates the process
        }
        winrt::uninit_apartment(); // restore pre-existing thread state for XAML
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
