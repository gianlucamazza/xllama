// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Disk (NVMe) read-bandwidth micro-bench — pins the denominator behind any
// SSD-streamed inference scenario.
//
// Weight streaming from disk (LLM-in-a-flash / llama.cpp PR #25294 style MoE
// expert streaming) is bounded by sustained file-read bandwidth, exactly as
// decode is bounded by DRAM bandwidth (membw.h). The Series S NVMe is PCIe 4.0
// x2 (~2.4 GB/s raw), but the effective rate through the UWP/GameOS sandboxed
// file APIs is unknown — this measures it. Two access patterns: sequential
// (bulk load) and random blocks (MoE expert-fetch shape). Portable: POSIX
// pread (+O_DIRECT when available) on the host, CreateFile2/ReadFile
// (+FILE_FLAG_NO_BUFFERING when the AppContainer accepts it) on UWP.
#pragma once

#include <cstddef>
#include <string>

namespace xllama {

struct DiskbwResult {
    std::size_t file_bytes = 0;
    std::size_t block_bytes = 0;
    int threads = 0;
    int iterations = 0;
    bool random = false;         // access pattern: false = sequential
    bool unbuffered = false;     // O_DIRECT / FILE_FLAG_NO_BUFFERING actually in effect
    double read_gbs_first = 0.0; // pass 1 (cold-ish; honest lower bound)
    double read_gbs_best = 0.0;  // best pass (page-cache-assisted upper bound)
    std::string error_msg;       // non-empty on failure (read_gbs stay 0)
};

// Create (or reuse, if already the right size) the incompressible test file.
// Written in chunks from a xorshift stream so filesystem/transparent
// compression cannot inflate the measured rate. Returns false + *err on failure.
bool ensure_diskbw_file(const std::string& path, std::size_t file_bytes, std::string* err);

// Read the whole file `iterations` times with `threads` workers (each on its
// own handle/fd, partitioned offsets; `random` shuffles block order with a
// fixed seed so runs are comparable). `try_unbuffered` attempts O_DIRECT /
// FILE_FLAG_NO_BUFFERING and falls back to buffered — the result records which
// mode actually ran. block_bytes is rounded up to 1 MiB alignment (covers any
// sector size the unbuffered path requires).
DiskbwResult measure_diskbw(const std::string& path, std::size_t file_bytes,
                            std::size_t block_bytes, bool random, int threads, int iterations,
                            bool try_unbuffered);

// Default geometry: 4 GiB file (above the 3801 MB console RAM budget, so a
// buffered pass cannot be served entirely from page cache), 8 MiB sequential /
// 2 MiB random blocks (expert-tensor scale).
inline constexpr std::size_t kDiskbwDefaultFileBytes = static_cast<std::size_t>(4) << 30;
inline constexpr std::size_t kDiskbwSeqBlockBytes = static_cast<std::size_t>(8) << 20;
inline constexpr std::size_t kDiskbwRndBlockBytes = static_cast<std::size_t>(2) << 20;

// CSV serialization (schema is self-contained; not the model-bench schema).
// "file_mb,block_kb,pattern,threads,unbuffered,read_gbs_first,read_gbs_best,host,date\n"
const char* diskbw_csv_header();
std::string format_diskbw_row(const DiskbwResult& r, const char* host_label);

} // namespace xllama
