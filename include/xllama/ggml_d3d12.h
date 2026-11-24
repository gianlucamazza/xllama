// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// ggml backend "d3d12" — GGUF GPU decode D2 (docs/gguf-gpu-decode.md).
//
// A GPU-type ggml device, registered at runtime, that runs MUL_MAT with
// Q4_0 / Q4_K / Q6_K / Q5_K / IQ4_XS / IQ3_XXS weights, GET_ROWS from a Q6_K
// weight and the F32 bias ADD
// after a projection on our D3D12 compute shaders; matmul inputs are quantized
// to q8 on the CPU first, exactly as the CPU backend does.
// Two buffer types:
//   D3D12_Weights  DEFAULT heap, holds matmul weights (exposed as an extra buft)
//   D3D12_Host     CUSTOM WRITE_BACK heap, is_host — the device default buft, so
//                  the scheduler's activations are CPU-visible: CPU<->GPU copies
//                  are memcpy and the CPU reads results in place.
// supports_op accepts an op only when its weight already lives in
// D3D12_Weights, which steers llama.cpp's per-weight buft probe past the host
// buft; every other op and weight falls back to the CPU.
//
// This header is WinRT- and D3D12-free: the pure rules below are host-tested on
// Linux, where ggml_d3d12_register() returns false.
#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "ggml-backend.h"
#include "ggml.h"

namespace xllama {

// Rows per thread group and threads per group of every mmv kernel
// (shaders/ggml_d3d12_mmv_*.hlsl).
inline constexpr int kD3d12MmvRows = 4;
// Threads per group: 64 or 128, chosen per matmul by d3d12_mm_threads(). 16
// threads share one 256-element chunk, so threads/16 chunks are in flight.
inline constexpr int kD3d12MmvThreadsShort = 64;
inline constexpr int kD3d12MmvThreadsLong = 128;
inline constexpr int kD3d12LongKChunks = 16; // K >= 4096 → 128 threads
// Every kernel walks K in 256-element chunks (one Q4_K/Q6_K super-block, eight
// Q4_0 blocks).
inline constexpr int kD3d12Chunk = 256;
inline constexpr std::uint32_t kD3d12MaxGroups = 65535;

// Weight types the backend runs when their pipelines exist.
bool d3d12_weight_type_supported(ggml_type t);

// PSO slots 0–2 (Q4_0, Q4_K, Q6_K) are required at backend init. Slots 3–5
// (Q5_K, IQ4_XS, IQ3_XXS) are optional: one failed CreateComputePipelineState
// leaves that type on the CPU and keeps the rest of the backend.
inline constexpr unsigned kD3d12OptionalMmvQ5K = 1u << 0;
inline constexpr unsigned kD3d12OptionalMmvIq4Xs = 1u << 1;
inline constexpr unsigned kD3d12OptionalMmvIq3Xxs = 1u << 2;
inline constexpr unsigned kD3d12OptionalMmvAll =
    kD3d12OptionalMmvQ5K | kD3d12OptionalMmvIq4Xs | kD3d12OptionalMmvIq3Xxs;

inline unsigned d3d12_optional_mmv_bit(ggml_type t) {
    switch (t) {
    case GGML_TYPE_Q5_K:
        return kD3d12OptionalMmvQ5K;
    case GGML_TYPE_IQ4_XS:
        return kD3d12OptionalMmvIq4Xs;
    case GGML_TYPE_IQ3_XXS:
        return kD3d12OptionalMmvIq3Xxs;
    default:
        return 0;
    }
}

inline bool d3d12_optional_mmv_ready(ggml_type t, unsigned ready) {
    const unsigned bit = d3d12_optional_mmv_bit(t);
    return bit == 0 || (ready & bit) != 0;
}

// 3×256 IQ3_XXS is a 98-byte unaligned-row correctness case. Packed GB/s is
// launch-overhead dominated, so the decode speed gate skips it.
inline bool d3d12_selftest_speed_gated(std::string_view type, int n, int k, int ncols) {
    if (ncols != 1)
        return false;
    return !(type == "iq3_xxs" && n == 3 && k == 256);
}

// Packed decode GB/s floors (scripts/bench-d3d12-selftest.sh, same numbers).
// Q4 / Q5 / Q6 / IQ4_XS stay at the STREAM-class 100 GB/s kill. IQ3_XXS
// decodes a 256-entry grid per nibble, so it is compute-bound: the 55 GB/s
// floor is measured, not a target. It is derived from the committed Series S
// receipt bench/results/d3d12-selftest-151804d-20261124.csv (IQ3_XXS 2048×2048
// = 61.89 GB/s packed at 151804d), with about 11% margin for run-to-run
// variance (two pre-optimisation runs differed by about 1–2%) and for
// thermal/clock noise. A kernel regression below 55 GB/s fails. These floors
// are a gate, not a product speed claim.
inline constexpr double kD3d12SelftestDecodeMinGbs = 100.0;
inline constexpr double kD3d12SelftestIq3XxsDecodeMinGbs = 55.0;

inline double d3d12_selftest_decode_min_gbs(std::string_view type) {
    if (type == "iq3_xxs")
        return kD3d12SelftestIq3XxsDecodeMinGbs;
    return kD3d12SelftestDecodeMinGbs;
}

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

// The MUL_MAT rules of supports_op. `optional_ready` is the mask of optional
// MMV pipelines that created (kD3d12OptionalMmvAll when every shader loaded).
bool d3d12_mm_supported(const D3d12MatmulDesc& d, unsigned optional_ready = kD3d12OptionalMmvAll);

struct D3d12Dispatch {
    std::uint32_t groups_x = 0; // ceil(N / kD3d12MmvRows)
    std::uint32_t groups_y = 0; // one per activation column (0: nothing to run)
    bool ok = false;            // within the 65535 group limit
};

D3d12Dispatch d3d12_mm_dispatch(std::int64_t n, std::int64_t ncols);

// Kernel width for a K: 128 threads (8 chunks in flight) once K has at least
// kD3d12LongKChunks chunks, else 64 — measured both ways in D2a runs 1 and 2.
int d3d12_mm_threads(std::int64_t k);

// Activations go to the GPU the way the CPU backend feeds its own vec_dot:
// quantized by ggml's from_float of the weight's vec_dot type (Q4_0 → q8_0,
// Q4_K / Q5_K / Q6_K / IQ4_XS / IQ3_XXS → q8_K), one row of ggml blocks per
// column. The kernels then
// sum integers exactly, so GPU and CPU differ only in float summation order
// (#312: f32 activations changed greedy text).
ggml_type d3d12_activation_type(ggml_type weight);
void d3d12_quantize_activations(ggml_type weight, const float* x, std::size_t x_stride, int k,
                                int ncols, std::uint8_t* out);

// Host emulation of the shaders' per-thread lane mapping, unaligned loads and
// algebra, on activations quantized as above: y[c*y_stride + r] ≈ W[r,:] · x_c.
// `w` holds N rows of `w_row_bytes` packed ggml blocks. Tests and the console
// selftest compare against the CPU backend's vec_dot.
void d3d12_mmv_emulate(ggml_type t, const std::uint8_t* w, std::size_t w_row_bytes, const float* x,
                       std::size_t x_stride, float* y, std::size_t y_stride, int n, int k,
                       int ncols);

// --- GET_ROWS on a weight in D3D12_Weights (#309) ---
// A tied model reuses token_embd as the lm_head; llama.cpp then keeps the
// input copy on the CPU and duplicates the tensor into D3D12_Weights. With
// GET_ROWS on the GPU the embedding can live in D3D12_Weights once.

// Embedding types the GET_ROWS kernel dequantizes (the catalogue's tied
// embeddings are Q6_K: Coder-3B, LFM2.5-1.2B, LFM2.5-350M).
bool d3d12_get_rows_type_supported(ggml_type t);

struct D3d12GetRowsDesc {
    ggml_type src0_type = GGML_TYPE_F32;
    std::int64_t ne00 = 0, ne01 = 0, ne02 = 1, ne03 = 1; // weight [K, rows]
    ggml_type src1_type = GGML_TYPE_I32;
    std::int64_t ne10 = 0, ne11 = 1, ne12 = 1; // row ids
    ggml_type dst_type = GGML_TYPE_F32;
    bool src0_contiguous = true;
    bool src1_contiguous = true; // ids are staged with one memcpy
    bool src0_in_weight_buffer = false;
};

bool d3d12_get_rows_supported(const D3d12GetRowsDesc& d);

// Host emulation of shaders/ggml_d3d12_get_rows_q6_k.hlsl: y[i*y_stride + j] =
// dequantized W[ids[i], j]. Tests compare it with ggml's to_float bit for bit.
void d3d12_get_rows_emulate(ggml_type t, const std::uint8_t* w, std::size_t w_row_bytes,
                            const std::int32_t* ids, int n_ids, int k, float* y,
                            std::size_t y_stride);

// Whether llama_gpu.h places a model's token_embd in D3D12_Weights: only a
// tied embedding (no output.weight) of a type GET_ROWS supports, so the one
// copy serves both the input lookup and the lm_head.
bool d3d12_place_tied_embedding(ggml_type embd_type, bool has_output_weight);

// --- Bias ADD on a weight in D3D12_Weights ---
// build_qkv and build_ffn add an F32 bias right after a projection. On the CPU
// each such ADD cuts the scheduler's split, so a model with q/k/v biases
// (Qwen2.5) pays one submission per projection instead of one for all three.
// Only the bias form runs here: src1 a 1-D F32 weight in D3D12_Weights added to
// every column of a 2-D F32 activation. Residual adds (two activations) stay on
// the CPU. fp32 add is correctly rounded on D3D12 (D3D11.3 spec §3.1.3.1) and
// the shader preserves denormals, so the GPU result matches the CPU bit for bit.

inline constexpr int kD3d12AddThreads = 256; // threads per group of the ADD kernel

struct D3d12AddDesc {
    ggml_type src0_type = GGML_TYPE_F32;
    std::int64_t ne00 = 0, ne01 = 0, ne02 = 1, ne03 = 1; // activations [N, M]
    ggml_type src1_type = GGML_TYPE_F32;
    std::int64_t ne10 = 0, ne11 = 1, ne12 = 1, ne13 = 1; // bias [N]
    ggml_type dst_type = GGML_TYPE_F32;
    bool src0_rows_contiguous = true; // nb00 == sizeof(float)
    bool src1_contiguous = true;
    bool src1_in_weight_buffer = false; // bias lives in a D3D12_Weights buffer
};

bool d3d12_add_supported(const D3d12AddDesc& d);

// The D3D12_Weights buffer type, for llama's tensor_buft_overrides. Null
// until ggml_d3d12_register() succeeds (always on non-Windows).
ggml_backend_buffer_type_t ggml_d3d12_weights_buft();

// --- Attributable request counters (#421) ---

// Backend activity between two ggml_d3d12_take_stats() calls. The caller
// brackets a phase (prompt prefill, then decode) by taking once before and
// once after it. Times are measured on the CPU around each step, except
// gpu_ms, which sums GPU timestamp pairs (start/end of each submission) and is
// only meaningful when every submission had a valid pair: gpu_time_valid().
// No field derives CPU compute time from wall minus GPU time.
// --- Per-dispatch GPU timing (#421), opt-in ---
// With LocalState\d3d12-dispatch-timing.txt = "1" when the backend starts,
// every dispatch is followed by a UAV barrier and a GPU timestamp, so each
// dispatch's GPU time can be attributed to its kind below and to a layer
// derived from the weight or graph-node name. Nodes with no layer go in the
// explicit `unattributed` bucket. The barriers serialise dispatches that
// could otherwise overlap: that changes what is measured, so read absolute
// times only next to a run without the knob. Off, the backend path allocates
// and logs nothing extra.

// "1" (surrounding whitespace allowed) enables a LocalState flag file.
bool d3d12_flag_enabled(std::string_view config);

// Column classes of a dispatch: "1" (decode), "2-63", "64-511", "512+".
const char* d3d12_cols_class(std::int64_t ncols);

struct D3d12DispatchKind {
    std::string op;            // "mul_mat" | "get_rows" | "add"
    std::string type;          // weight type name, e.g. "q4_0"
    int threads = 0;           // 64 | 128 for mul_mat, kD3d12AddThreads for add, 0 for get_rows
    int columns_per_group = 1; // kernel variant (the single-column kernel is 1)
    std::string cols;          // d3d12_cols_class(ncols)
    bool operator==(const D3d12DispatchKind& o) const {
        return op == o.op && type == o.type && threads == o.threads &&
               columns_per_group == o.columns_per_group && cols == o.cols;
    }
};

struct D3d12DispatchRow {
    D3d12DispatchKind kind;
    std::uint64_t count = 0;
    double gpu_ms = 0.0;
    double weight_bytes = 0.0; // packed weight bytes the dispatches address
    std::uint64_t columns = 0; // activation columns (rows fetched for GET_ROWS)
};

class D3d12DispatchProfile {
  public:
    void add(const D3d12DispatchKind& kind, double gpu_ms, double weight_bytes, std::int64_t ncols);
    // A dispatch whose timestamp pair was not increasing: its time is unknown.
    void note_missing() {
        ++missing_;
    }
    std::uint64_t missing() const {
        return missing_;
    }
    bool empty() const {
        return rows_.empty() && missing_ == 0;
    }
    const std::vector<D3d12DispatchRow>& rows() const {
        return rows_;
    }
    // Rows by descending GPU time, ties in insertion order.
    std::vector<D3d12DispatchRow> sorted() const;

  private:
    std::vector<D3d12DispatchRow> rows_;
    std::uint64_t missing_ = 0;
};

// One "[xllama] d3d12-dispatch ..." line per kind, by descending GPU time, and
// a closing summary line with the number of untimed dispatches. Empty when
// the profile is empty (the knob is off or nothing ran).
std::string format_d3d12_dispatch_profile(const char* phase, const D3d12DispatchProfile& p);

// Label written for a missing layer index: nodes with no `blk.N` / `*-N`.
inline constexpr const char* kD3d12UnattributedLayer = "unattributed";

// Layer index from a GGUF tensor name or a llama.cpp graph node. Strips a
// scheduler wrap (`D3D12#blk.0.attn_q.weight#0`). Prefers `blk.<n>`; otherwise
// a trailing `-<n>` (`Qcur-0`, `l_out-15`). nullopt → unattributed. Never
// invents an index for `token_embd.weight`, `output.weight`, or similar.
std::optional<int> d3d12_layer_from_name(std::string_view name);
// Weight name first, then the graph node. nullopt if neither has a layer.
std::optional<int> d3d12_layer_from_names(std::string_view tensor_name, std::string_view node_name);
// "unattributed" or the decimal index. Host-tested so the log line stays stable.
std::string d3d12_layer_label(std::optional<int> layer);

// Aggregation key: one transformer layer (or unattributed) × one op.
struct D3d12LayerKey {
    std::optional<int> layer; // nullopt = unattributed
    std::string op;           // "mul_mat" | "get_rows" | "add"
    bool operator==(const D3d12LayerKey& o) const {
        return layer == o.layer && op == o.op;
    }
};

struct D3d12LayerRow {
    D3d12LayerKey key;
    std::uint64_t count = 0;        // dispatches in this bucket
    std::uint64_t gpu_timed = 0;    // dispatches with an increasing timestamp pair
    std::uint64_t gpu_missing = 0;  // pair missing or not increasing
    std::uint64_t wait_timed = 0;   // dispatches whose submission had exec_wait and GPU timestamps
    std::uint64_t wait_missing = 0; // wait not reported: that submission's GPU timestamps missing
    double gpu_ms = 0.0;            // sum of valid per-dispatch GPU times; ignore if gpu_timed==0
    double wait_ms = 0.0; // sum of (exec_wait_ms − gpu_ms) shares; ignore if wait_timed==0
};

// Per phase × layer × op. gpu_ms is never filled in from wall time; wait_ms is
// never filled in when the submission's GPU timestamps are missing.
class D3d12LayerProfile {
  public:
    void add(const D3d12LayerKey& key, const std::optional<double>& gpu_ms,
             const std::optional<double>& wait_ms);
    bool empty() const {
        return rows_.empty();
    }
    const std::vector<D3d12LayerRow>& rows() const {
        return rows_;
    }
    // Layer index ascending, unattributed last; op name for ties.
    std::vector<D3d12LayerRow> sorted() const;

  private:
    std::vector<D3d12LayerRow> rows_;
};

// One "[xllama] d3d12-layer ..." line per (layer, op), then a summary. gpu_ms
// and wait_ms print as "missing" when that bucket has no valid timestamps —
// never 0. Empty when the profile is empty (knob off or nothing ran).
std::string format_d3d12_layer_profile(const char* phase, const D3d12LayerProfile& p);

struct D3d12PhaseStats {
    std::uint64_t calls = 0;       // graph_compute calls that ran at least one op
    std::uint64_t submissions = 0; // ExecuteCommandLists + fence wait
    std::uint64_t dispatches = 0;  // MUL_MAT + GET_ROWS
    std::uint64_t ts_valid = 0;    // submissions with an increasing timestamp pair
    std::uint64_t ts_missing = 0;  // unsupported, unmapped or non-increasing pair
    double wall_ms = 0.0;          // whole graph_compute calls
    double quant_ms = 0.0;         // CPU q8 activation quantization
    double record_ms = 0.0;        // command-list reset/record/close
    double exec_wait_ms = 0.0;     // ExecuteCommandLists through fence completion
    double gpu_ms = 0.0;           // GPU timestamps; see gpu_time_valid()
    D3d12DispatchProfile dispatch; // per-dispatch timing, empty unless enabled
    D3d12LayerProfile layer;       // phase × layer × op; empty unless enabled
    bool gpu_time_valid() const {
        return submissions > 0 && ts_missing == 0;
    }
};

// One machine-readable log line, e.g.
// "[xllama] d3d12-phase phase=prefill calls=2 submissions=2 ... gpu_ms=12.345"
// gpu_ms is printed as "unavailable" unless gpu_time_valid().
std::string format_d3d12_phase(const char* phase, const D3d12PhaseStats& s);

// Return the counters accumulated since the previous call and reset them.
// False (and zeroed stats) when the backend is not available on this
// platform, so a missing measurement is never reported as zero activity.
bool ggml_d3d12_take_stats(D3d12PhaseStats* out);

// Register the backend with ggml (idempotent). False when D3D12 is unavailable
// (always on non-Windows). The device appears as a GPU device named "D3D12".
bool ggml_d3d12_register();

// --- Console selftest (d3d12be.flag) ---

struct D3d12SelftestRow {
    std::string type; // q4_0 | q4_k | q6_k | q5_k | iq4_xs | iq3_xxs
    int n = 0, k = 0, ncols = 0;
    double rel_err = 0.0; // max |gpu - ref| / max |ref|
    double gpu_ms = 0.0;  // GPU timestamp time, median of 5 timed computes (ms)
    double packed_gbs = 0.0;
    bool ok = false;
    bool d3d12_ran = false;
    std::string error;
};

// Against the CPU backend (q8 activations + ggml vec_dot): only float
// summation order may differ (#312, predeclared gate A).
inline constexpr double kD3d12SelftestRelTol = 1e-5;

const char* d3d12_selftest_csv_header();
std::string format_d3d12_selftest_row(const D3d12SelftestRow& r, const char* host_label);

// Windows: run every type x shape through the backend and append rows.
// Non-Windows: one row with d3d12_ran=false.
void run_d3d12_selftest(std::vector<D3d12SelftestRow>* out);

} // namespace xllama
