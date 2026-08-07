// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Phase 15 W3 (#211) — GPU STREAM bandwidth probe (own compute shader).
//
// Kill criterion (predeclared, SSOT docs/phase15-re-opt.md): under 100 GB/s
// STREAM read on ~1 GB VRAM, H6 is "Do not reopen". Checksum of the buffer
// contents is mandatory evidence (no PIX in Dev Mode UWP).
//
// Pure helpers (pattern fill, checksum, CSV) are host-testable on Linux.
// measure_gpubw() runs D3D12 on Windows/UWP; elsewhere it returns d3d12_ran=false.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>

namespace xllama {

// Predeclared kill threshold (decimal GB/s, 1e9 bytes). Do not change without
// updating docs/phase15-re-opt.md and issue #211.
inline constexpr double kGpubwKillReadGbs = 100.0;

// Default working set: 1 GiB of uint32 words (issue #211 ~1 GB VRAM).
inline constexpr std::size_t kGpubwDefaultBufferBytes = static_cast<std::size_t>(1) << 30;

struct GpubwResult {
    std::size_t buffer_bytes = 0;
    int iterations = 0;
    double read_gbs = 0.0; // best-of STREAM-style GPU read (1× buffer moved)
    std::uint32_t checksum = 0;
    std::uint32_t expected_checksum = 0;
    bool checksum_ok = false;
    bool d3d12_ran = false;
    std::string error_msg; // empty on full success
};

// Deterministic pattern: word[i] = mix(i). Shared by CPU expected checksum and
// the GPU upload path so a wrong read fails checksum_ok.
inline std::uint32_t gpubw_pattern_word(std::uint32_t index) {
    std::uint32_t x = index * 0x9e3779b9u + 0x85ebca6bu;
    x ^= x >> 16;
    x *= 0x7feb352du;
    x ^= x >> 15;
    x *= 0x846ca68bu;
    x ^= x >> 16;
    return x;
}

void gpubw_fill_pattern(std::uint32_t* data, std::size_t n_words);

// Folded XOR of all words (associative — matches the CS parallel reduction).
std::uint32_t gpubw_checksum_words(const std::uint32_t* data, std::size_t n_words);

// Windows/UWP: D3D12 system device (never Agility), timed CS STREAM read +
// checksum. Non-Windows: d3d12_ran=false, error_msg set, no throw.
GpubwResult measure_gpubw(std::size_t buffer_bytes = kGpubwDefaultBufferBytes, int iterations = 3);

// Verdict helper for SSOT / scripts (does not invent numbers).
inline bool gpubw_passes_kill_gate(const GpubwResult& r) {
    return r.d3d12_ran && r.checksum_ok && r.read_gbs >= kGpubwKillReadGbs;
}

const char* gpubw_csv_header();
// buffer_mb,iterations,read_gbs,checksum_ok,d3d12_ran,checksum,expected,host,date,error
std::string format_gpubw_row(const GpubwResult& r, const char* host_label);

} // namespace xllama
