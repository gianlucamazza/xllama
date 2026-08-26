// UWP-native GraphicsCapture availability probe.
#pragma once

#include <cstdint>
#include <string>

namespace xllama {

// Runtime metadata only. This probe never starts a capture session or encoder.
struct capture_metadata {
    bool graphics_capture_session_present = false;
    bool graphics_capture_item_present = false;
    bool capture_path_viable = false;
    bool encoding_seam_exists = false;
    std::int32_t presence_error = 0;
    std::uint64_t probe_duration_us = 0;
    std::string diagnostic;
};

// Safe, non-throwing runtime presence probe. On UWP it logs [gcprobe].
capture_metadata probe_graphics_capture() noexcept;

// True only after both a measured frame-pool path and encoder exist.
bool is_graphics_capture_ready() noexcept;

} // namespace xllama
