// Bounded native GraphicsCapture frame/timestamp probe.
#ifdef XLLAMA_UWP

    #include "native_capture.h"
    #include "pch.h"

    #include "xllama/platform.h"

    #include <Windows.Graphics.DirectX.Direct3D11.interop.h>
    #include <d3d11.h>
    #include <winrt/Windows.Graphics.Capture.h>
    #include <winrt/Windows.Graphics.DirectX.Direct3D11.h>
    #include <winrt/Windows.Graphics.DirectX.h>
    #include <winrt/Windows.UI.Composition.h>
    #include <winrt/Windows.UI.Xaml.Hosting.h>

    #include <atomic>
    #include <chrono>
    #include <cstdio>
    #include <fstream>
    #include <thread>

namespace xllama {

namespace {

using winrt::Windows::Graphics::Capture::Direct3D11CaptureFramePool;
using winrt::Windows::Graphics::Capture::GraphicsCaptureItem;
using winrt::Windows::Graphics::DirectX::DirectXPixelFormat;
using winrt::Windows::Graphics::DirectX::Direct3D11::IDirect3DDevice;

struct CaptureState {
    std::atomic<std::uint64_t> frames{0};
    std::atomic<std::uint64_t> first_timestamp_100ns{0};
    std::atomic<std::uint64_t> last_timestamp_100ns{0};
};

IDirect3DDevice create_capture_device() {
    winrt::com_ptr<ID3D11Device> native_device;
    winrt::com_ptr<ID3D11DeviceContext> context;
    D3D_FEATURE_LEVEL level{};
    const D3D_FEATURE_LEVEL levels[] = {D3D_FEATURE_LEVEL_11_1, D3D_FEATURE_LEVEL_11_0};
    winrt::check_hresult(D3D11CreateDevice(
        nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, D3D11_CREATE_DEVICE_BGRA_SUPPORT, levels,
        ARRAYSIZE(levels), D3D11_SDK_VERSION, native_device.put(), &level, context.put()));

    auto dxgi_device = native_device.as<IDXGIDevice>();
    IDirect3DDevice device{nullptr};
    winrt::check_hresult(CreateDirect3D11DeviceFromDXGIDevice(
        dxgi_device.get(), reinterpret_cast<::IInspectable**>(winrt::put_abi(device))));
    return device;
}

void write_result(std::uint32_t duration_seconds, CaptureState const& state,
                  std::uint64_t elapsed_ms, char const* error = nullptr) noexcept {
    try {
        auto folder = winrt::Windows::Storage::ApplicationData::Current().LocalFolder();
        std::wstring path = std::wstring(folder.Path().c_str()) + L"\\native-capture-result.json";
        std::ofstream out(path);
        if (!out)
            return;
        const auto frames = state.frames.load();
        const auto first = state.first_timestamp_100ns.load();
        const auto last = state.last_timestamp_100ns.load();
        const double timestamp_seconds = last > first ? (last - first) / 10'000'000.0 : 0.0;
        const double fps = timestamp_seconds > 0.0 ? (frames - 1) / timestamp_seconds : 0.0;
        out << "{\n"
            << "  \"duration_requested_s\": " << duration_seconds << ",\n"
            << "  \"elapsed_ms\": " << elapsed_ms << ",\n"
            << "  \"frames\": " << frames << ",\n"
            << "  \"timestamp_span_s\": " << timestamp_seconds << ",\n"
            << "  \"fps_from_timestamps\": " << fps << ",\n"
            << "  \"error\": ";
        if (error)
            out << '"' << error << '"';
        else
            out << "null";
        out << "\n}\n";
    } catch (...) {
        // Diagnostics must never terminate the app.
    }
}

} // namespace

void start_native_capture(winrt::Windows::UI::Xaml::Controls::Page const& page,
                          std::uint32_t duration_seconds) noexcept {
    try {
        auto visual =
            winrt::Windows::UI::Xaml::Hosting::ElementCompositionPreview::GetElementVisual(page);
        auto item = GraphicsCaptureItem::CreateFromVisual(visual);
        auto device = create_capture_device();
        std::thread([item, device, duration_seconds] {
            CaptureState state;
            try {
                auto pool = Direct3D11CaptureFramePool::CreateFreeThreaded(
                    device, DirectXPixelFormat::B8G8R8A8UIntNormalized, 2, item.Size());
                auto session = pool.CreateCaptureSession(item);
                pool.FrameArrived([&state](auto const& sender, auto const&) {
                    auto frame = sender.TryGetNextFrame();
                    if (!frame)
                        return;
                    const auto timestamp =
                        static_cast<std::uint64_t>(frame.SystemRelativeTime().count());
                    const auto number = state.frames.fetch_add(1) + 1;
                    if (number == 1)
                        state.first_timestamp_100ns.store(timestamp);
                    state.last_timestamp_100ns.store(timestamp);
                });
                const auto start = std::chrono::steady_clock::now();
                session.StartCapture();
                std::this_thread::sleep_for(std::chrono::seconds(duration_seconds));
                session.Close();
                pool.Close();
                const auto elapsed_ms = static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - start)
                        .count());
                write_result(duration_seconds, state, elapsed_ms);
                char line[256];
                std::snprintf(line, sizeof(line), "[native-capture] frames=%llu elapsed_ms=%llu\n",
                              static_cast<unsigned long long>(state.frames.load()),
                              static_cast<unsigned long long>(elapsed_ms));
                log_output(line);
            } catch (winrt::hresult_error const& error) {
                write_result(duration_seconds, state, 0, "winrt_error");
                log_output(std::string("[native-capture] hresult=0x") +
                           std::to_string(static_cast<unsigned>(error.code().value)) + "\n");
            } catch (...) {
                write_result(duration_seconds, state, 0, "unknown_error");
                log_output("[native-capture] unknown error\n");
            }
        }).detach();
    } catch (winrt::hresult_error const& error) {
        log_output(std::string("[native-capture] setup hresult=0x") +
                   std::to_string(static_cast<unsigned>(error.code().value)) + "\n");
    } catch (...) {
        log_output("[native-capture] setup unknown error\n");
    }
}

} // namespace xllama

#endif // XLLAMA_UWP
