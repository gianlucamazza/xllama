// Runtime-only GraphicsCapture probe. It deliberately does not activate a
// session: presence is not proof that Xbox can capture or encode a video.
#include "xllama/capture_probe.h"

#include "xllama/platform.h"

#include <chrono>
#include <cstdio>

#ifdef XLLAMA_UWP
    #include <winrt/Windows.Foundation.Metadata.h>
#endif

namespace xllama {

capture_metadata probe_graphics_capture() noexcept {
    capture_metadata meta{};
    const auto start = std::chrono::steady_clock::now();

#ifdef XLLAMA_UWP
    try {
        using winrt::Windows::Foundation::Metadata::ApiInformation;
        meta.graphics_capture_session_present =
            ApiInformation::IsTypePresent(L"Windows.Graphics.Capture.GraphicsCaptureSession");
        meta.graphics_capture_item_present =
            ApiInformation::IsTypePresent(L"Windows.Graphics.Capture.GraphicsCaptureItem");
    } catch (winrt::hresult_error const& error) {
        char diagnostic[128];
        std::snprintf(diagnostic, sizeof(diagnostic), "[gcprobe] presence_error=0x%08X\n",
                      static_cast<unsigned>(error.code().value));
        meta.presence_error = error.code().value;
        meta.diagnostic = diagnostic;
    } catch (...) {
        meta.presence_error = -1;
        meta.diagnostic = "[gcprobe] presence_error=unknown\n";
    }
#endif

    meta.capture_path_viable = false;
    meta.encoding_seam_exists = false;
    meta.probe_duration_us =
        static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::microseconds>(
                                       std::chrono::steady_clock::now() - start)
                                       .count());

    char line[256];
    std::snprintf(line, sizeof(line),
                  "[gcprobe] session_present=%d item_present=%d viable=0 "
                  "encoder=0 presence_error=%d duration_us=%llu\n",
                  meta.graphics_capture_session_present ? 1 : 0,
                  meta.graphics_capture_item_present ? 1 : 0, meta.presence_error,
                  static_cast<unsigned long long>(meta.probe_duration_us));
    meta.diagnostic = line;
    log_output(line);

    return meta;
}

bool is_graphics_capture_ready() noexcept {
    return false;
}

} // namespace xllama
