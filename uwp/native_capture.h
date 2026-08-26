// Opt-in native GraphicsCapture frame-rate probe.
#pragma once

#include "pch.h"

#include <cstdint>

namespace xllama {

// Starts a bounded, non-blocking capture probe for the application's XAML
// visual. The result is written to LocalState/native-capture-result.json.
// Call only after Window::Current().Activate().
void start_native_capture(winrt::Windows::UI::Xaml::Controls::Page const& page,
                          std::uint32_t duration_seconds = 5) noexcept;

} // namespace xllama
