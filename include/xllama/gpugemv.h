// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Phase 15 H6.1 (#228) — Q4_K GEMV bandwidth probe (own compute shader).
//
// Still a *measurement*, not a GGUF GPU backend. Goal: packed weight bandwidth
// of a dequant-in-register Q4_K GEMV vs STREAM (gpubw, ~119 GB/s) and vs DirectML.
//
// Soft density gate (predeclared, SSOT docs/phase15-re-opt.md):
//   G1 correctness: max_abs_err vs CPU ref below threshold + checksum match
//   G2 density:     packed_gbs >= kGpugemvSoftPackedGbs (40 GB/s) when G1 holds
//
// Pure helpers (pattern, CPU ref GEMV, CSV) are host-testable on Linux.
// measure_gpugemv() runs D3D12 on Windows/UWP; elsewhere d3d12_ran=false.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>

namespace xllama {

// Soft density gate (decimal GB/s of *packed* Q4_K bytes read). Not a product
// kill — documents whether 4-bit density beats ~34 GB/s DML-class BW.
inline constexpr double kGpugemvSoftPackedGbs = 40.0;

// ggml Q4_K: QK_K = 256, K_SCALE_SIZE = 12, block = 144 bytes.
inline constexpr int kGpugemvQK = 256;
inline constexpr int kGpugemvScaleBytes = 12;
inline constexpr int kGpugemvQsBytes = kGpugemvQK / 2; // 128
inline constexpr std::size_t kGpugemvBlockBytes =
    2 * sizeof(std::uint16_t) + kGpugemvScaleBytes + kGpugemvQsBytes; // 144

// Default tile: multiples of 256. ~36 MiB packed weights (N*K/256*144).
inline constexpr int kGpugemvDefaultN = 8192;
inline constexpr int kGpugemvDefaultK = 8192;

// Max abs error allowed for G1 (fp32 ref vs GPU).
inline constexpr float kGpugemvMaxAbsErrTol = 1e-2f;

// Byte-compatible with ggml block_q4_K (d, dmin as IEEE fp16 bits).
struct alignas(2) GpugemvQ4KBlock {
    std::uint16_t d;    // super-block scale (fp16)
    std::uint16_t dmin; // super-block min scale (fp16)
    std::uint8_t scales[kGpugemvScaleBytes];
    std::uint8_t qs[kGpugemvQsBytes];
};
static_assert(sizeof(GpugemvQ4KBlock) == 144, "must match ggml block_q4_K");

struct GpugemvResult {
    int n = 0;
    int k = 0;
    int iterations = 0;
    std::size_t packed_bytes = 0;
    double packed_gbs = 0.0; // best-of: packed weight bytes / sec
    float max_abs_err = 0.f;
    std::uint32_t y_checksum = 0;
    std::uint32_t expected_y_checksum = 0;
    bool checksum_ok = false;
    bool d3d12_ran = false;
    std::string error_msg;
};

// IEEE fp16 helpers (host + shared with fill).
std::uint16_t gpugemv_float_to_half(float f);
float gpugemv_half_to_float(std::uint16_t h);

// ggml get_scale_min_k4 — pure, host-testable.
void gpugemv_get_scale_min_k4(int j, const std::uint8_t* scales, std::uint8_t* d, std::uint8_t* m);

// Dequant one Q4_K block into 256 floats (matches dequantize_row_q4_K).
void gpugemv_dequant_block(const GpugemvQ4KBlock& b, float* y256);

// Deterministic fill of weight blocks and activation vector.
void gpugemv_fill_weights(GpugemvQ4KBlock* blocks, int n, int k);
void gpugemv_fill_x(float* x, int k);

// CPU reference: y[n] = sum_k W[n,k]*x[k] with Q4_K weights stored row-major
// as (n * (k/QK)) blocks.
void gpugemv_cpu_ref(const GpugemvQ4KBlock* blocks, const float* x, float* y, int n, int k);

// Folded checksum of float bits (XOR of uint32 bit patterns).
std::uint32_t gpugemv_checksum_floats(const float* data, std::size_t n);

// Max |a[i]-b[i]|.
float gpugemv_max_abs_err(const float* a, const float* b, std::size_t n);

// Soft gates.
inline bool gpugemv_passes_g1(const GpugemvResult& r) {
    return r.d3d12_ran && r.checksum_ok && r.max_abs_err <= kGpugemvMaxAbsErrTol;
}
inline bool gpugemv_passes_g2(const GpugemvResult& r) {
    return gpugemv_passes_g1(r) && r.packed_gbs >= kGpugemvSoftPackedGbs;
}

// Windows/UWP: D3D12 system device GEMV. Non-Windows: d3d12_ran=false.
GpugemvResult measure_gpugemv(int n = kGpugemvDefaultN, int k = kGpugemvDefaultK,
                              int iterations = 3);

const char* gpugemv_csv_header();
// n,k,iterations,packed_mb,packed_gbs,max_abs_err,checksum_ok,d3d12_ran,checksum,expected,host,date,error
std::string format_gpugemv_row(const GpugemvResult& r, const char* host_label);

inline std::size_t gpugemv_packed_bytes(int n, int k) {
    if (n <= 0 || k <= 0 || (k % kGpugemvQK) != 0)
        return 0;
    return static_cast<std::size_t>(n) * static_cast<std::size_t>(k / kGpugemvQK) *
           kGpugemvBlockBytes;
}

} // namespace xllama
