// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// GGUF GPU decode — probe D1 (docs/gguf-gpu-decode.md). Measures the three
// unknowns that decide the D2 architecture before any backend exists:
//   rt    CPU<->GPU round trip per scheduler split (upload X, dispatch,
//         fence, readback Y)
//   step  one simulated decode token of a real model shape: every split's
//         matmuls on the GPU, CPU sync per split (and a no-sync lower bound)
//   heap  GEMV bandwidth with weights in DEFAULT vs CPU-visible heaps (UMA)
// plus the in-XAML run (D1d). Matmuls use the H6.3 `rows` Q4_K kernel.
//
// Pure helpers (shape tables, split list, bytes, cost model, CSV, gates,
// ladder) are host-testable on Linux. measure_gpustep() runs D3D12 on
// Windows/UWP; elsewhere it returns one row with d3d12_ran=false.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace xllama {

// --- Predeclared D1 gates (SSOT docs/gguf-gpu-decode.md §Gates) ---

// D1a: median round trip, best wait mode.
inline constexpr double kGpustepRtGateUs = 100.0;
// D1b: simulated qwen25-coder-3b token with a CPU sync per split. 40 ms of
// matmul+sync plus <= 11 ms of CPU-side ops is 51 ms = 1.4x the CPU t6
// decode of 14.0 tok/s (bench/results/phase14-console.csv).
inline constexpr double kGpustepStepGateMs = 40.0;
// D1c: a CPU-visible heap must keep this fraction of DEFAULT-heap GEMV BW.
inline constexpr double kGpustepHostHeapMinRatio = 0.9;
// Cost-model bandwidth: H6.3 `rows` console median, packed GB/s
// (bench/results/phase15-gpugemv-h63.csv).
inline constexpr double kGpustepRowsBwGbs = 143.06;

// --- Model shapes ---

enum class GpustepQuant : int {
    Q4K = 0, // 144 B / 256 (Q4_0 has the same density: 18 B / 32)
    Q6K = 1, // 210 B / 256
};

struct GpustepMatmul {
    std::string name; // e.g. "blk.3.ffn_up"
    int n = 0;        // output rows
    int k = 0;        // input width (multiple of 256)
    GpustepQuant quant = GpustepQuant::Q4K;
};

// Adjacent matmuls that share one input: the scheduler runs them in one split
// (one CPU<->GPU round trip). D2 verifies this with ggml_backend_sched_get_n_splits.
struct GpustepSplit {
    std::vector<GpustepMatmul> mm;
};

enum class GpustepModelId : int {
    Coder3B = 0, // qwen25-coder-3b Q4_K_M (Qwen2.5-3B config, tied lm_head)
    Lfm12B = 1,  // lfm25-1.2b-instruct QAD Q4_0 (GGUF header)
    Lfm350M = 2, // lfm25-350m QAD Q4_0 (GGUF header)
};

struct GpustepModel {
    GpustepModelId id = GpustepModelId::Coder3B;
    std::string name; // catalogue-style id, used in the CSV variant
    std::vector<GpustepSplit> splits;
};

const char* gpustep_model_name(GpustepModelId id);
GpustepModel gpustep_model(GpustepModelId id);

// llama.cpp Q4_K_M rule for attn_v / ffn_down promotion to Q6_K
// (llama-quant.cpp use_more_bits).
bool gpustep_q4km_more_bits(int layer, int n_layer);

std::size_t gpustep_matmul_bytes(const GpustepMatmul& m);
std::size_t gpustep_split_bytes(const GpustepSplit& s);
std::size_t gpustep_model_bytes(const GpustepModel& m);
std::size_t gpustep_model_matmuls(const GpustepModel& m);

// Rows the Q4_K `rows` kernel runs for this matmul: Q6_K is simulated as the
// bytes-equivalent number of Q4_K rows (x210/144), rounded up to 4.
int gpustep_sim_rows(const GpustepMatmul& m);

// Cost model: t = sum_splits(t_rt + bytes/BW). Milliseconds.
double gpustep_projected_step_ms(const GpustepModel& m, double t_rt_us, double bw_gbs);
// Tokens/s from matmul+sync time plus CPU-side work (both ms). 0 if <= 0.
double gpustep_projected_toks(double t_step_ms, double t_cpu_rest_ms);

// --- Measurement rows ---

struct GpustepRow {
    std::string kind;    // caps | rt | step | heap
    std::string variant; // rt: event|spin ; step: <model>/sync|nosync ; heap: default|upload|custom
    std::string process; // headless | inproc | host
    int iterations = 0;
    double median = 0.0;
    double p90 = 0.0;
    std::string unit; // us | ms | gbs | -
    std::size_t bytes = 0;
    bool ok = false; // G1 correctness (rt, heap) / measured (step) / queried (caps)
    bool d3d12_ran = false;
    bool uma = false;
    bool cc_uma = false;
    std::string error;
};

const char* gpustep_csv_header();
// kind,variant,process,iterations,median,p90,unit,bytes,ok,d3d12_ran,uma,cc_uma,host,date,error
std::string format_gpustep_row(const GpustepRow& r, const char* host_label);
// Inverse of format_gpustep_row (host/date ignored). False on a malformed line
// or the header.
bool parse_gpustep_row(const std::string& line, GpustepRow* out);

// --- Verdict ---

enum class GpustepLadder {
    NotAVerdict,  // a required row is missing or G1 failed headless
    D2MatmulOnly, // D1a + D1b + D1d PASS
    D2Fused,      // D1b FAIL, but the no-sync step fits the gate: sync dominates
    Park3B,       // even the no-sync step misses the gate
    ParkGui,      // in-XAML run failed (D1d)
};

const char* gpustep_ladder_name(GpustepLadder l);

struct GpustepVerdict {
    bool g1_headless = false;
    bool have_inproc = false;
    bool g1_inproc = false;
    double rt_best_us = 0.0;     // best headless rt median over wait modes
    double step_sync_ms = 0.0;   // headless qwen25-coder-3b/sync median
    double step_nosync_ms = 0.0; // headless qwen25-coder-3b/nosync median
    bool d1a = false;
    bool d1b = false;
    bool d1d = false;
    std::string d1c; // host-visible | default+copy | n/a
    GpustepLadder ladder = GpustepLadder::NotAVerdict;
};

GpustepVerdict gpustep_evaluate(const std::vector<GpustepRow>& rows);

// Windows/UWP: run caps, heap, rt and step for all models in this process and
// append rows. Non-Windows: one caps row with d3d12_ran=false.
void measure_gpustep(const char* process_label, std::vector<GpustepRow>* out);

} // namespace xllama
