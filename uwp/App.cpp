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
#include "xllama/capture_probe.h"
#include <roapi.h> // RoInitialize — WinRT MTA (see wWinMain)
#ifndef XLLAMA_STORE_SKU
#include "api-server.h"
#include <winrt/Windows.Foundation.Metadata.h>
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
// MEASURED, 2026-07-30, console on 1.5.2.x Dev Mode:
//
//   [caprec] AppRecordingManager absent (Desktop contract not on this device)
//            GraphicsCaptureSession present=1
//
// So AppRecordingManager is not an option here at all, and the reason is the
// one the guard exists for: it lives in the Desktop extension contract, and
// AppxManifest.xml declaring the Windows.Desktop device family does not make
// that contract present on a given device. Reading CanRecord and its reason
// flags would need a Desktop Extension SDK reference in the vcxproj; one was
// added and then removed, because a projection for a type that is absent buys
// nothing. See the comment where that reference used to be.
//
// GraphicsCaptureSession IS present, which keeps self-capture open by a
// different route — one that would encode in-process rather than on the SoC
// video encoder, so the CPU-cost argument above applies to it and it needs its
// own measurement before anyone builds on it (#214).
//
// The probe stays because the answer is a property of the OS, not of this
// build: a GameOS update could make either line flip, and a demo pipeline
// planned around a stills slideshow should find that out from a log line rather
// than from someone guessing again.
//
// A structured runtime-presence probe lives in capture_probe.cpp
// (include/xllama/capture_probe.h). It does not activate a session or request
// consent; capture viability still requires a measured frame-pool and encoder.
// See docs/uwp-constraints.md §10b.1.
// ---------------------------------------------------------------------------
static void log_app_recording_probe() {
    using winrt::Windows::Foundation::Metadata::ApiInformation;

    char buf[256];
    try {
        snprintf(
            buf, sizeof(buf),
            "[caprec] AppRecordingManager present=%d"
            " GraphicsCaptureSession present=%d\n",
            ApiInformation::IsTypePresent(L"Windows.Media.AppRecording.AppRecordingManager") ? 1
                                                                                             : 0,
            ApiInformation::IsTypePresent(L"Windows.Graphics.Capture.GraphicsCaptureSession") ? 1
                                                                                              : 0);
        log_write(buf);
    } catch (winrt::hresult_error const& e) {
        snprintf(buf, sizeof(buf), "[caprec] hresult 0x%08X: %ls\n",
                 static_cast<unsigned>(e.code().value), e.message().c_str());
        log_write(buf);
    } catch (...) {
        log_write("[caprec] unknown exception\n");
    }
}

// ---------------------------------------------------------------------------
// Is the audio-capture API surface even here? (Phase 16 WS-F, card H16.6)
//
// Separate from [caprec] on purpose. That probe answers "can the app record its
// own output"; this one answers "can the app hear the room", and they have
// different answers, different manifest costs and different consumers. Sharing
// a tag would make both log lines ambiguous to whoever greps for one of them.
//
// This is free — two IsTypePresent calls — and it is deliberately NOT the
// answer to WS-F. AudioGraph and MediaCapture are Universal contract types, so
// they are expected present even where capture is refused; §10b already records
// the same trap for GraphicsCaptureSession, which is present and unusable. The
// verdict comes from run_mic_probe's real 3-second capture. This line exists so
// that a future GameOS update removing the surface entirely is visible in an
// ordinary boot log rather than only in a probe nobody reran.
// ---------------------------------------------------------------------------
static void log_microphone_probe() {
    using winrt::Windows::Foundation::Metadata::ApiInformation;

    char buf[256];
    try {
        snprintf(buf, sizeof(buf),
                 "[mic] AudioGraph present=%d MediaCapture present=%d"
                 " (presence is not permission — see uwp-constraints.md)\n",
                 ApiInformation::IsTypePresent(L"Windows.Media.Audio.AudioGraph") ? 1 : 0,
                 ApiInformation::IsTypePresent(L"Windows.Media.Capture.MediaCapture") ? 1 : 0);
        log_write(buf);
    } catch (winrt::hresult_error const& e) {
        snprintf(buf, sizeof(buf), "[mic] hresult 0x%08X: %ls\n",
                 static_cast<unsigned>(e.code().value), e.message().c_str());
        log_write(buf);
    } catch (...) {
        log_write("[mic] unknown exception\n");
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
        // After Activate(), so that if the Desktop contract ever appears the
        // status flags this could then read are answered against a live window
        // rather than an inactive one.
        log_app_recording_probe();
        log_microphone_probe();
        // GraphicsCapture runtime-presence probe — writes [gcprobe] line.
        // See include/xllama/capture_probe.h and docs/uwp-constraints.md §10b.1.
        ::xllama::probe_graphics_capture();

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
        // WinRT apartment first (RoInitialize). C++/WinRT's init_apartment on
        // current headers only calls CoInitializeEx; MSVC UWP images still
        // import RoInitialize (CRT / other TUs). Xbox XAML activation needs a
        // real WinRT MTA — without it Start fails as 0x8027025b (gotcha 17).
        winrt::check_hresult(RoInitialize(RO_INIT_MULTITHREADED));
        winrt::init_apartment(); // CoInitializeEx MTA; pairs with RoInitialize
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
        std::wstring diskbw_flag = flag_path_if_present(L"diskbw.flag");
        if (!diskbw_flag.empty()) {
            _wremove(diskbw_flag.c_str());
            ::xllama::log_output("[xllama] diskbw.flag detected -> headless diskbw mode\n");
            winrt::Windows::ApplicationModel::Core::CoreApplication::Run(
                winrt::make<HeadlessView>(&::xllama::bridge::run_diskbw, "diskbw"));
            return 0; // not reached: CoreApplication::Exit terminates the process
        }
        std::wstring gpubw_flag = flag_path_if_present(L"gpubw.flag");
        if (!gpubw_flag.empty()) {
            _wremove(gpubw_flag.c_str());
            ::xllama::log_output("[xllama] gpubw.flag detected -> headless gpubw mode (#211)\n");
            winrt::Windows::ApplicationModel::Core::CoreApplication::Run(
                winrt::make<HeadlessView>(&::xllama::bridge::run_gpubw, "gpubw"));
            return 0; // not reached: CoreApplication::Exit terminates the process
        }
        std::wstring gpugemv_flag = flag_path_if_present(L"gpugemv.flag");
        if (!gpugemv_flag.empty()) {
            _wremove(gpugemv_flag.c_str());
            ::xllama::log_output(
                "[xllama] gpugemv.flag detected -> headless Q4_K GEMV mode (#228)\n");
            winrt::Windows::ApplicationModel::Core::CoreApplication::Run(
                winrt::make<HeadlessView>(&::xllama::bridge::run_gpugemv, "gpugemv"));
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
        std::wstring mic_flag = flag_path_if_present(L"mic.flag");
        if (!mic_flag.empty()) {
            _wremove(mic_flag.c_str());
            ::xllama::log_output("[xllama] mic.flag detected -> headless mic capture probe\n");
            winrt::Windows::ApplicationModel::Core::CoreApplication::Run(
                winrt::make<HeadlessView>(&::xllama::bridge::run_mic_probe, "mic"));
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
    #endif // !XLLAMA_STORE_SKU
        // Keep the MTA from init_apartment() through Application::Start.
        // XAML requires the *first* Application access to come from the MTA
        // (uwp-crossbuild gotcha 17 / hello-uwp). Do NOT uninit_apartment()
        // here: MSVC's UWP CRT often holds a residual RoInitialize MTA so a
        // mistaken uninit still left the thread apartmented; clang + store
        // CRT /MT has no such residual, uninit leaves no apartment, and
        // Start dies as winrt::hresult_wrong_thread → 0x8027025b on Xbox.
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
