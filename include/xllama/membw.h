// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT
//
// CPU memory-bandwidth micro-bench — pins the denominator behind decode.
//
// LLM decode is a bandwidth-bound M=1 GEMV: every token streams the whole weight
// matrix from DRAM once, so the achievable decode rate is (weight bytes / effective
// read bandwidth). xllama's docs quote "~13 GB/s effective from CPU int4 GEMV" as a
// *deduced* figure; this measures the sustained bandwidth ceiling directly (a
// STREAM-style read / copy / triad over a buffer larger than the LLC), so the
// decode number can be stated as a fraction of a measured ceiling rather than a
// guess. Portable (no platform deps) so it runs on the Linux host and, via
// `membw.flag`, on the Xbox in Game mode.
#pragma once

#include <cstddef>
#include <string>

namespace xllama {

struct MembwResult {
    std::size_t buffer_bytes = 0;
    int iterations = 0;
    int threads = 0;
    double read_gbs = 0.0;  // streaming read (sum reduction), 1×buffer moved
    double copy_gbs = 0.0;  // memcpy, 2×buffer moved (STREAM copy convention)
    double triad_gbs = 0.0; // a=b+s*c, 3×buffer moved (STREAM triad convention)
};

// Measure sustained bandwidth with `threads` workers (0 = hardware_concurrency)
// over a `buffer_bytes` working set, reporting the best (min-time) of `iterations`
// passes. GB = 1e9 bytes (decimal, matching the "224 GB/s bus" spec convention).
MembwResult measure_membw(std::size_t buffer_bytes = static_cast<std::size_t>(256) << 20,
                          int iterations = 5, int threads = 0);

// CSV serialization (schema is self-contained; not the model-bench schema).
const char* membw_csv_header(); // "buffer_mb,threads,read_gbs,copy_gbs,triad_gbs,host,date\n"
std::string format_membw_row(const MembwResult& r, const char* host_label);

} // namespace xllama
