// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// ggml backend "d3d12" — GGUF GPU decode D2 (docs/gguf-gpu-decode.md).
//
// A GPU-type ggml device, registered at runtime, that runs only MUL_MAT with
// Q4_0 / Q4_K / Q6_K weights and f32 activations on our D3D12 compute shaders.
// Two buffer types:
//   D3D12_Weights  DEFAULT heap, holds matmul weights (exposed as an extra buft)
//   D3D12_Host     CUSTOM WRITE_BACK heap, is_host — the device default buft, so
//                  the scheduler's activations are CPU-visible: CPU<->GPU copies
//                  are memcpy and the CPU reads results in place.
// supports_op accepts a MUL_MAT only when its weight already lives in
// D3D12_Weights, which steers llama.cpp's per-weight buft probe past the host
// buft; every other op and weight falls back to the CPU.
//
// This header is WinRT- and D3D12-free: the pure rules below are host-tested on
// Linux, where ggml_d3d12_register() returns false.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "ggml.h"

namespace xllama {

// Rows per thread group and threads per group of every mmv kernel
// (shaders/ggml_d3d12_mmv_*.hlsl).
inline constexpr int kD3d12MmvRows = 4;
inline constexpr int kD3d12MmvThreads = 128;
// Chunks in flight per group: 16 threads share one 256-element chunk.
inline constexpr int kD3d12InFlight = kD3d12MmvThreads / 16;
// Every kernel walks K in 256-element chunks (one Q4_K/Q6_K super-block, eight
// Q4_0 blocks).
inline constexpr int kD3d12Chunk = 256;
inline constexpr std::uint32_t kD3d12MaxGroups = 65535;

// Weight types the backend runs.
bool d3d12_weight_type_supported(ggml_type t);

// What supports_op sees, as plain data (no ggml_tensor needed in tests).
struct D3d12MatmulDesc {
    ggml_type src0_type = GGML_TYPE_F32;
    std::int64_t ne00 = 0, ne01 = 0, ne02 = 1, ne03 = 1; // weight [K, N]
    std::int64_t ne10 = 0, ne11 = 0, ne12 = 1, ne13 = 1; // activations [K, M]
    ggml_type src1_type = GGML_TYPE_F32;
    ggml_type dst_type = GGML_TYPE_F32;
    bool src0_contiguous = true;
    bool src1_contiguous = true;
    bool src0_in_weight_buffer = false; // weight lives in a D3D12_Weights buffer
};

// The MUL_MAT rules of supports_op.
bool d3d12_mm_supported(const D3d12MatmulDesc& d);

struct D3d12Dispatch {
    std::uint32_t groups_x = 0; // ceil(N / kD3d12MmvRows)
    std::uint32_t groups_y = 0; // one per activation column
    bool ok = false;            // within the 65535 group limit
};

D3d12Dispatch d3d12_mm_dispatch(std::int64_t n, std::int64_t ncols);

// Host emulation of the shaders' per-thread lane mapping, unaligned loads and
// algebra: y[c*y_stride + r] = sum_k W[r,k] * x[c*x_stride + k]. `w` holds N
// rows of `w_row_bytes` packed ggml blocks. Tests compare it with ggml's own
// dequantizers; the console selftest compares the GPU with the same reference.
void d3d12_mmv_emulate(ggml_type t, const std::uint8_t* w, std::size_t w_row_bytes, const float* x,
                       std::size_t x_stride, float* y, std::size_t y_stride, int n, int k,
                       int ncols);

// Register the backend with ggml (idempotent). False when D3D12 is unavailable
// (always on non-Windows). The device appears as a GPU device named "D3D12".
bool ggml_d3d12_register();

// --- Console selftest (d3d12be.flag) ---

struct D3d12SelftestRow {
    std::string type; // q4_0 | q4_k | q6_k
    int n = 0, k = 0, ncols = 0;
    double rel_err = 0.0; // max |gpu - ref| / max |ref|
    double gpu_ms = 0.0;  // GPU timestamp time of the last compute, ms
    double packed_gbs = 0.0;
    bool ok = false;
    bool d3d12_ran = false;
    std::string error;
};

inline constexpr double kD3d12SelftestRelTol = 1e-2;

const char* d3d12_selftest_csv_header();
std::string format_d3d12_selftest_row(const D3d12SelftestRow& r, const char* host_label);

// Windows: run every type x shape through the backend and append rows.
// Non-Windows: one row with d3d12_ran=false.
void run_d3d12_selftest(std::vector<D3d12SelftestRow>* out);

} // namespace xllama
