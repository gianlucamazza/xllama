// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// The image-generation request contract shared by the Image dialog, the LAN API
// (/v1/images/generations), autopilot generate_image and the in-process
// pipeline that reads them back from LocalState (uwp/diffuse.cpp).
//
// The surfaces used to disagree: the pipeline accepted seeds up to 1e9 and
// silently replaced a larger one with 42, while the dialog accepted any uint32
// and the API up to INT_MAX, so a seed the user typed did not reproduce. And
// autopilot cast JSON numbers to int unclamped. The pipeline's own domain is
// the one that counts: std::mt19937 takes a uint32 seed.
#pragma once

#include "xllama/local_knob.h"

#include <cstdint>
#include <optional>
#include <string_view>

namespace xllama::diffusion {

inline constexpr int kStepsMin = 1;
// SD-Turbo is a 1-4 step model; the interactive surfaces clamp to this.
inline constexpr int kStepsInteractiveMax = 4;
// Headless diffuse-steps.txt may ask for more, up to the LCM scheduler's
// original_inference_steps (lcm_scheduler.h).
inline constexpr int kStepsMax = 50;
inline constexpr int kDefaultSteps = 1;
inline constexpr std::uint32_t kDefaultSeed = 42;

// Steps from a JSON number (API, autopilot): clamped, never rejected. NaN and
// anything below the range take kStepsMin; the clamp happens before the
// conversion, so an out-of-range double cannot reach an undefined cast.
inline int interactive_steps(double v) {
    if (!(v >= kStepsMin))
        return kStepsMin;
    if (v > kStepsInteractiveMax)
        return kStepsInteractiveMax;
    return static_cast<int>(v);
}

// Seed from a JSON number: a negative seed (llama.cpp's "-1 = random"
// convention) or NaN keeps kDefaultSeed; above uint32 clamps to its max.
inline std::uint32_t request_seed(double v) {
    if (!(v >= 0))
        return kDefaultSeed;
    if (v >= 4294967295.0)
        return 4294967295u;
    return static_cast<std::uint32_t>(v);
}

// diffuse-steps.txt as the pipeline reads it.
inline int steps_from_knob(std::string_view text) {
    const std::optional<int> v = parse_knob_int(text);
    return (v && *v >= kStepsMin && *v <= kStepsMax) ? *v : kDefaultSteps;
}

// diffuse-seed.txt as the pipeline reads it: any uint32.
inline std::uint32_t seed_from_knob(std::string_view text) {
    const std::optional<long long> v = parse_knob_i64(text);
    return (v && *v >= 0 && *v <= 4294967295LL) ? static_cast<std::uint32_t>(*v) : kDefaultSeed;
}

} // namespace xllama::diffusion
