// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#ifdef XLLAMA_UWP

// clang-format off
// pch.h must be first: it includes <unknwn.h> before WinRT headers (COM IUnknown guard).
#include "pch.h"
#include "App.h"
#include "MainPage.h"
#include "inference-bridge.h"
#include "xllama/platform.h"
#ifndef XLLAMA_STORE_SKU
#include "api-server.h"
#include <winrt/Windows.Foundation.Metadata.h>
#include <winrt/Windows.Media.AppRecording.h>
#endif
// clang-format on

    #include <thread>

    #ifndef XLLAMA_STORE_SKU
// Defined in the headless section below; also used by the in-process
// diffusion experiment in App::OnLaunched.
static std::wstring flag_path_if_present(const wchar_t* name);
    #endif

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

    #ifndef XLLAMA_STORE_SKU
// ---------------------------------------------------------------------------
// Can the app record its own output?
//
// The demo pipeline reconstructs a video from Device Portal screenshots at 1 Hz
// (scripts/capture-demo-video.sh) — a slideshow, not a recording.
// AppRecordingManager is the alternative, and the argument for it is not
// smoothness: it hands frames to the SoC video encoder, whereas any software
// encoder inside this process would steal CPU from the very inference the demo
// exists to show — on a machine with ~6 usable cores and a livelock at 7-8
// (docs/uwp-constraints.md).
//
// Whether it works here is genuinely unknown and is NOT assumed in either
// direction. There are two separate unknowns and both get answered:
//
//   1. is the type registered on this console at all? AppRecordingManager is in
//      the Desktop extension contract, and declaring the Windows.Desktop target
//      device family in the manifest does not make that contract present.
//      IsTypePresent answers it, and is also the mandatory guard: a projection
//      existing at compile time says nothing about runtime, and touching an
//      absent extension type throws.
//   2. if it is, does this environment allow recording? Dev Mode may disable the
//      capture policy. GetStatus() answers with a reason rather than a bare
//      false — the difference between "no" and "no, because X".
//
// Reading (2) is why uwp/xllama.vcxproj carries an SDKReference to the Desktop
// extension: the CppWinRT NuGet only projects namespaces it has metadata for.
// Substituting the Windows SDK's own AppRecording header is not an alternative;
// the comment on that SDKReference records what happens if you try.
//
// GraphicsCaptureSession is reported alongside because it is the other
// self-capture route and lives in the Universal contract, so knowing whether it
// exists here costs one string comparison.
static void log_app_recording_probe() {
    using winrt::Windows::Foundation::Metadata::ApiInformation;
    namespace rec = winrt::Windows::Media::AppRecording;

    char buf[640];
    try {
        const bool gfx_present =
            ApiInformation::IsTypePresent(L"Windows.Graphics.Capture.GraphicsCaptureSession");
        if (!ApiInformation::IsTypePresent(L"Windows.Media.AppRecording.AppRecordingManager")) {
            snprintf(buf, sizeof(buf),
                     "[caprec] AppRecordingManager absent (Desktop contract not on this device)"
                     " GraphicsCaptureSession present=%d\n",
                     gfx_present ? 1 : 0);
            log_write(buf);
            return;
        }
        auto mgr = rec::AppRecordingManager::GetDefault();
        if (!mgr) {
            snprintf(
                buf, sizeof(buf),
                "[caprec] type present but GetDefault=null GraphicsCaptureSession present=%d\n",
                gfx_present ? 1 : 0);
            log_write(buf);
            return;
        }
        auto status = mgr.GetStatus();
        auto d = status.Details();
        snprintf(buf, sizeof(buf),
                 "[caprec] CanRecord=%d CanRecordTimeSpan=%d GraphicsCaptureSession present=%d"
                 " | disabledByUser=%d disabledBySystem=%d blockedForApp=%d"
                 " captureResourceUnavailable=%d gpuConstrained=%d appInactive=%d"
                 " anyAppBroadcasting=%d gameStreamInProgress=%d timeSpanDisabled=%d\n",
                 status.CanRecord() ? 1 : 0, status.CanRecordTimeSpan() ? 1 : 0,
                 gfx_present ? 1 : 0, d.IsDisabledByUser() ? 1 : 0, d.IsDisabledBySystem() ? 1 : 0,
                 d.IsBlockedForApp() ? 1 : 0, d.IsCaptureResourceUnavailable() ? 1 : 0,
                 d.IsGpuConstrained() ? 1 : 0, d.IsAppInactive() ? 1 : 0,
                 d.IsAnyAppBroadcasting() ? 1 : 0, d.IsGameStreamInProgress() ? 1 : 0,
                 d.IsTimeSpanRecordingDisabled() ? 1 : 0);
        log_write(buf);
    } catch (winrt::hresult_error const& e) {
        snprintf(buf, sizeof(buf), "[caprec] hresult 0x%08X: %ls\n",
                 static_cast<unsigned>(e.code().value), e.message().c_str());
        log_write(buf);
    } catch (...) {
        log_write("[caprec] unknown exception\n");
    }
}
    #endif

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

    #ifndef XLLAMA_STORE_SKU
        // Answered on every interactive launch, after Activate(): recording
        // needs a live window, and IsAppInactive would read true before it.
        log_app_recording_probe();

        // §7 experiment: the 887A0036 device conflict was measured with ORT
        // GenAI's Agility-factory device; diffuse.cpp uses plain ORT DML, whose
        // device may coexist with the compositor device the line above just
        // created. diffuse-inproc.flag runs the diffusion pipeline on a
        // background MTA thread INSIDE the XAML process to test exactly that.
        // Consumed before the run, same semantics as the headless flags.
        // Store SKU: research flags omitted (docs/store-readiness.md).
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

        // LAN HTTP endpoint (OpenAI-compat), opt-in and default OFF: started
        // only when LocalState\api.flag exists. The flag is NOT consumed — the
        // server is persistent and coexists with the live XAML chat UI, unlike
        // the headless bench/diffuse flags. Runs on a detached MTA thread; the
        // Settings UI may later stop or rebind the listener. See api-server.cpp.
        const bool start_persisted_api = !flag_path_if_present(L"api.flag").empty();
        if (start_persisted_api) {
            log_write("[xllama] api.flag detected -> starting LAN HTTP endpoint\n");
            auto controller = m_controller;
            std::thread([controller] {
                winrt::init_apartment(); // MTA: WinRT sockets + ApplicationData
                ::xllama::api::run_server();
                // Preserve lifecycle ordering: an autopilot set_api action must
                // run after the persisted listener has finished binding.
                if (controller)
                    controller->StartAutopilotIfRequested();
            }).detach();
        } else if (m_controller) {
            m_controller->StartAutopilotIfRequested();
        }
    #else
        if (m_controller)
            m_controller->StartAutopilotIfRequested();
    #endif
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

    #ifndef XLLAMA_STORE_SKU
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
//
// Entire headless path is Dev Mode / research only (not compiled into Store SKU).
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
// Runs `m_entry` (bench main_loop, run_diffuse, run_membw, run_logits or
// run_train) on an MTA thread, then
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
    #endif // !XLLAMA_STORE_SKU

// ---------------------------------------------------------------------------
// Entry point — Application::Start replaces CoreApplication::Run
// ---------------------------------------------------------------------------
int __stdcall wWinMain(HINSTANCE, HINSTANCE, PWSTR, int) {
    try {
        winrt::init_apartment(); // MTA: needed for ApplicationData in the detection
    #ifndef XLLAMA_STORE_SKU
        // Headless operator modes (Device Portal flags). Omitted from the Store
        // SKU — retail builds always take the interactive XAML path.
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
        std::wstring ramceil_flag = flag_path_if_present(L"ramceil.flag");
        if (!ramceil_flag.empty()) {
            _wremove(ramceil_flag.c_str());
            ::xllama::log_output("[xllama] ramceil.flag detected -> headless heap-ceiling probe\n");
            winrt::Windows::ApplicationModel::Core::CoreApplication::Run(
                winrt::make<HeadlessView>(&::xllama::bridge::run_ramceil, "ramceil"));
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
        std::wstring oprepro_flag = flag_path_if_present(L"oprepro.flag");
        if (!oprepro_flag.empty()) {
            _wremove(oprepro_flag.c_str());
            ::xllama::log_output("[xllama] oprepro.flag detected -> single-op CPU-vs-DML repro\n");
            winrt::Windows::ApplicationModel::Core::CoreApplication::Run(
                winrt::make<HeadlessView>(&::xllama::bridge::run_oprepro, "oprepro"));
            return 0; // not reached: CoreApplication::Exit terminates the process
        }
        std::wstring train_flag = flag_path_if_present(L"train.flag");
        if (!train_flag.empty()) {
            _wremove(train_flag.c_str());
            ::xllama::log_output(
                "[xllama] train.flag detected -> headless device training (Lane B)\n");
            winrt::Windows::ApplicationModel::Core::CoreApplication::Run(
                winrt::make<HeadlessView>(&::xllama::bridge::run_train, "train"));
            return 0; // not reached: CoreApplication::Exit terminates the process
        }
    #endif                         // !XLLAMA_STORE_SKU
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
