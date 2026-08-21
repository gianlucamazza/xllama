// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Phase 15 H6.2 (#228) — Q4_K GEMV density probe (own compute shader).
//
// Still a *measurement*, not a GGUF GPU backend. H6.1 naive is A/B only.
// Gated candidate is wave32 (one wave = one row, LDS-red).
//
// Soft density gate (predeclared, SSOT docs/phase15-re-opt.md):
//   G1 correctness: max_abs_err vs CPU ref below threshold + checksum match
//   G2 density:     packed_gbs >= kGpugemvSoftPackedGbs (40 GB/s) when G1 holds
//   K1 kill:        G1 on all 3 recorded runs and median packed_gbs < 8
//
// Pure helpers (pattern, CPU ref GEMV, CSV, dispatch, LDS index, ladder)
// are host-testable on Linux. measure_gpugemv() runs D3D12 on Windows/UWP;
// elsewhere d3d12_ran=false.
#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace xllama {

// Soft density gate (decimal GB/s of *packed* Q4_K bytes read). Not a product
// kill — documents whether 4-bit density beats ~34 GB/s DML-class BW. G2 = 40.
inline constexpr double kGpugemvSoftPackedGbs = 40.0;
// K1: G1 PASS on all 3 recorded runs and median packed_gbs < 8 at N=K=8192.
inline constexpr double kGpugemvKillPackedGbs = 8.0;

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

enum class GpugemvKernel : int {
    Naive = 0,
    Wave32 = 1,
};

inline const char* gpugemv_kernel_name(GpugemvKernel k) {
    switch (k) {
    case GpugemvKernel::Naive:
        return "naive";
    case GpugemvKernel::Wave32:
        return "wave32";
    }
    return "wave32";
}

struct GpugemvResult {
    int n = 0;
    int k = 0;
    int iterations = 0; // recorded runs (not counting warmup)
    int run_index = 0;  // 1..iterations for CSV rows; 0 = summary
    GpugemvKernel kernel = GpugemvKernel::Wave32;
    std::size_t packed_bytes = 0;
    double packed_gbs = 0.0;     // gated: GPU dt of list A if gpu_timestamp else CPU of A
    double packed_gbs_cpu = 0.0; // CPU Execute+fence of list A (one Dispatch, no copy)
    double packed_gbs_h61 = 0.0; // optional, non-gated: CPU of A+B (copy included)
    float max_abs_err = 0.f;
    std::uint32_t y_checksum = 0;
    std::uint32_t expected_y_checksum = 0;
    bool checksum_ok = false;
    bool d3d12_ran = false;
    bool gpu_timestamp = false;
    bool wave_ops = false;     // blob select: PR 1 LDS-red only; always false
    bool wave_ops_cap = false; // D3D12_OPTIONS1.WaveOps (observability; not a gate)
    std::uint32_t wave_lane_min = 0;
    std::uint32_t wave_lane_max = 0;
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

// Soft gates. Signatures unchanged; G2 stays 40.
inline bool gpugemv_passes_g1(const GpugemvResult& r) {
    return r.d3d12_ran && r.checksum_ok && r.max_abs_err <= kGpugemvMaxAbsErrTol;
}
inline bool gpugemv_passes_g2(const GpugemvResult& r) {
    return gpugemv_passes_g1(r) && r.packed_gbs >= kGpugemvSoftPackedGbs;
}

enum class GpugemvLadder { NotAVerdict, K1Kill, K2Park, K3Open };

inline const char* gpugemv_ladder_name(GpugemvLadder l) {
    switch (l) {
    case GpugemvLadder::K1Kill:
        return "K1";
    case GpugemvLadder::K2Park:
        return "K2";
    case GpugemvLadder::K3Open:
        return "K3";
    case GpugemvLadder::NotAVerdict:
        return "NotAVerdict";
    }
    return "NotAVerdict";
}

// Per kernel. g1_all3 = G1 PASS on every recorded run of that kernel.
// Inclusive: 8.0 is K2, 40.0 is K3. G1 FAIL is not a density verdict.
inline GpugemvLadder gpugemv_ladder(double median_packed_gbs, bool g1_all3) {
    if (!g1_all3)
        return GpugemvLadder::NotAVerdict;
    if (median_packed_gbs < kGpugemvKillPackedGbs)
        return GpugemvLadder::K1Kill;
    if (median_packed_gbs < kGpugemvSoftPackedGbs)
        return GpugemvLadder::K2Park;
    return GpugemvLadder::K3Open;
}

struct GpugemvKernelSummary {
    GpugemvKernel kernel = GpugemvKernel::Wave32;
    double median_packed_gbs = 0.0;
    bool g1_all3 = false;
    GpugemvLadder ladder = GpugemvLadder::NotAVerdict;
};

// Denser kernels only (not naive). Best G1-passing median wins.
// If none G1-pass: NotAVerdict (do not close H6 as K1 on a broken shader).
inline GpugemvLadder gpugemv_campaign_verdict(const GpugemvKernelSummary* denser, std::size_t n) {
    if (!denser || n == 0)
        return GpugemvLadder::NotAVerdict;
    const GpugemvKernelSummary* best = nullptr;
    for (std::size_t i = 0; i < n; ++i) {
        if (denser[i].kernel == GpugemvKernel::Naive)
            continue;
        if (!denser[i].g1_all3)
            continue;
        if (!best || denser[i].median_packed_gbs > best->median_packed_gbs)
            best = &denser[i];
    }
    if (!best)
        return GpugemvLadder::NotAVerdict;
    return gpugemv_ladder(best->median_packed_gbs, best->g1_all3);
}

// Dispatch planner (host-testable). naive: ceil(N/64) groups of 64 rows.
// wave32: N groups of 1 row. Does not clamp the 65535 cap.
struct GpugemvDispatch {
    std::uint32_t threads_per_group = 0;
    std::uint32_t rows_per_group = 0;
    std::uint32_t groups_x = 0;
};

inline GpugemvDispatch gpugemv_plan_dispatch(int n, GpugemvKernel kernel) {
    GpugemvDispatch d{};
    if (n < 0)
        n = 0;
    if (kernel == GpugemvKernel::Naive) {
        d.threads_per_group = 64;
        d.rows_per_group = 64;
        d.groups_x = n == 0 ? 0u : static_cast<std::uint32_t>((n + 63) / 64);
    } else {
        d.threads_per_group = 32;
        d.rows_per_group = 1;
        d.groups_x = static_cast<std::uint32_t>(n);
    }
    return d;
}

// LDS transpose index helpers — must match shaders/gpugemv_q4k_wave32.hlsl.
inline std::uint32_t gpugemv_lds_write_index(std::uint32_t pass, std::uint32_t nload,
                                             std::uint32_t lane) {
    return pass * nload + lane;
}
inline std::uint32_t gpugemv_lds_read_index(std::uint32_t lane, std::uint32_t q) {
    return lane * 9u + q;
}
inline std::uint32_t gpugemv_lds_load_byte(std::uint32_t chunk_byte, std::uint32_t pass,
                                           std::uint32_t nload, std::uint32_t lane) {
    return chunk_byte + pass * (nload * 16u) + lane * 16u;
}

// Windows/UWP: D3D12 system device GEMV. Non-Windows: d3d12_ran=false.
// Default kernel is wave32 (H6.2 candidate). Pass Naive explicitly for A/B.
GpugemvResult measure_gpugemv(int n = kGpugemvDefaultN, int k = kGpugemvDefaultK,
                              int iterations = 3, GpugemvKernel kernel = GpugemvKernel::Wave32);

// Per-recorded-run rows without rebuilding the device between runs.
// Linux: out->size()==1, d3d12_ran=false, packed_gbs=0, kernel echoed.
// Windows: out->size()==iterations on success.
void measure_gpugemv_each(int n, int k, int iterations, GpugemvKernel kernel,
                          std::vector<GpugemvResult>* out);

const char* gpugemv_csv_header();
// kernel,n,k,iterations,run_index,packed_mb,packed_gbs,packed_gbs_cpu,packed_gbs_h61,
// max_abs_err,checksum_ok,d3d12_ran,gpu_timestamp,wave_ops,checksum,expected,host,date,error
std::string format_gpugemv_row(const GpugemvResult& r, const char* host_label);

inline std::size_t gpugemv_packed_bytes(int n, int k) {
    if (n <= 0 || k <= 0 || (k % kGpugemvQK) != 0)
        return 0;
    return static_cast<std::size_t>(n) * static_cast<std::size_t>(k / kGpugemvQK) *
           kGpugemvBlockBytes;
}

} // namespace xllama
