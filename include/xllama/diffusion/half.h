// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT
//
// IEEE 754 half <-> float conversion as raw uint16_t bit patterns. The console
// diffusion pipeline (uwp/diffuse.cpp) feeds fp16 tensors to ORT DirectML but
// computes the scheduler math in float; it works in uint16_t half-bits and builds
// fp16 tensors with the untyped Ort::Value::CreateTensor(..., FLOAT16) overload,
// avoiding any dependency on the version-specific Ort::Float16_t constructor.
// Pure + header-only so the conversion is unit-tested on the host.
#pragma once

#include <cstdint>
#include <cstring>
#include <vector>

namespace xllama::diffusion {

// float -> half bits, round-to-nearest-even, with inf/NaN and subnormal handling.
inline uint16_t float_to_half(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof(x));
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int32_t exp = (int32_t)((x >> 23) & 0xFF) - 127 + 15;
    const uint32_t mant = x & 0x7FFFFFu;

    if (((x >> 23) & 0xFF) == 0xFF) // inf / NaN
        return (uint16_t)(sign | 0x7C00u | (mant ? 0x200u : 0u));
    if (exp >= 0x1F) // overflow -> inf
        return (uint16_t)(sign | 0x7C00u);
    if (exp <= 0) { // subnormal or zero
        if (exp < -10)
            return (uint16_t)sign;
        const uint32_t m = mant | 0x800000u;
        const int shift = 14 - exp;
        uint32_t half = m >> shift;
        // round to nearest even
        const uint32_t rem = m & ((1u << shift) - 1);
        const uint32_t halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (half & 1)))
            ++half;
        return (uint16_t)(sign | half);
    }
    uint16_t half = (uint16_t)((exp << 10) | (mant >> 13));
    const uint32_t rem = mant & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (half & 1)))
        ++half; // round-to-even
    return (uint16_t)(sign | half);
}

// half bits -> float.
inline float half_to_float(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1F;
    const uint32_t mant = h & 0x3FFu;
    uint32_t x;
    if (exp == 0) {
        if (mant == 0) {
            x = sign; // +/- 0
        } else {
            // subnormal -> normalize
            int e = -1;
            uint32_t m = mant;
            do {
                m <<= 1;
                ++e;
            } while (!(m & 0x400u));
            m &= 0x3FFu;
            x = sign | ((uint32_t)(127 - 15 - e) << 23) | (m << 13);
        }
    } else if (exp == 0x1F) {
        x = sign | 0x7F800000u | (mant << 13); // inf / NaN
    } else {
        x = sign | ((exp - 15 + 127) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &x, sizeof(f));
    return f;
}

// Vector helpers used by the pipeline (uwp/diffuse.cpp).
inline std::vector<uint16_t> to_half(const std::vector<float>& in) {
    std::vector<uint16_t> out(in.size());
    for (size_t i = 0; i < in.size(); ++i)
        out[i] = float_to_half(in[i]);
    return out;
}
inline std::vector<float> from_half(const uint16_t* p, size_t n) {
    std::vector<float> out(n);
    for (size_t i = 0; i < n; ++i)
        out[i] = half_to_float(p[i]);
    return out;
}

} // namespace xllama::diffusion
