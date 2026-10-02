// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "xllama/ggml_d3d12.h"

#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

using namespace xllama;

namespace {

D3d12MatmulDesc decode_desc(ggml_type t, std::int64_t n, std::int64_t k) {
    D3d12MatmulDesc d;
    d.src0_type = t;
    d.ne00 = k;
    d.ne01 = n;
    d.ne10 = k;
    d.ne11 = 1;
    d.src0_in_weight_buffer = true;
    return d;
}

// Max |emulated - reference| / max |reference| for one type and shape, with
// padded activation/output strides. Reference: the CPU backend's own path —
// activations quantized by the vec_dot type's from_float (q8_0 / q8_K), then
// ggml's vec_dot. The kernels must match it, not exact f32 (#312).
double emulate_rel_err(ggml_type t, int n, int k, int ncols, std::size_t x_pad, std::size_t y_pad) {
    std::mt19937 rng(42u + static_cast<unsigned>(n * 7 + k + ncols));
    std::uniform_real_distribution<float> uni(-1.f, 1.f);
    std::vector<float> wf(static_cast<std::size_t>(n) * k);
    for (float& v : wf)
        v = uni(rng);
    const std::size_t row_bytes = ggml_row_size(t, k);
    std::vector<std::uint8_t> q(row_bytes * n + 8); // slack: 2-byte-aligned dword reads
    ggml_quantize_chunk(t, wf.data(), q.data(), 0, n, k, nullptr);

    const std::size_t x_stride = static_cast<std::size_t>(k) + x_pad;
    const std::size_t y_stride = static_cast<std::size_t>(n) + y_pad;
    std::vector<float> x(x_stride * ncols, 1e30f); // padding must never be read
    for (int c = 0; c < ncols; ++c)
        for (int i = 0; i < k; ++i)
            x[c * x_stride + i] = uni(rng);
    std::vector<float> y(y_stride * ncols, -7.f);
    d3d12_mmv_emulate(t, q.data(), row_bytes, x.data(), x_stride, y.data(), y_stride, n, k, ncols);

    ggml_cpu_init(); // fp16 tables used by the CPU vec_dot kernels
    const auto* wt = ggml_get_type_traits_cpu(t);
    const auto* xt = ggml_get_type_traits_cpu(wt->vec_dot_type);
    std::vector<std::uint8_t> xq(ggml_row_size(wt->vec_dot_type, k));
    double max_ref = 0.0, max_diff = 0.0;
    for (int c = 0; c < ncols; ++c) {
        xt->from_float(x.data() + c * x_stride, xq.data(), k);
        for (int r = 0; r < n; ++r) {
            float ref = 0.f;
            wt->vec_dot(k, &ref, 0, q.data() + r * row_bytes, 0, xq.data(), 0, 1);
            max_ref = std::max(max_ref, std::fabs(static_cast<double>(ref)));
            max_diff =
                std::max(max_diff, std::fabs(static_cast<double>(ref) - y[c * y_stride + r]));
        }
    }
    // Output padding untouched.
    for (int c = 0; c < ncols; ++c)
        for (std::size_t p = n; p < y_stride; ++p)
            if (y[c * y_stride + p] != -7.f)
                return 1e9;
    return max_ref > 0.0 ? max_diff / max_ref : max_diff;
}

} // namespace

TEST_CASE("ggml_d3d12: weight types") {
    CHECK(d3d12_weight_type_supported(GGML_TYPE_Q4_0));
    CHECK(d3d12_weight_type_supported(GGML_TYPE_Q4_K));
    CHECK(d3d12_weight_type_supported(GGML_TYPE_Q6_K));
    CHECK_FALSE(d3d12_weight_type_supported(GGML_TYPE_Q5_K));
    CHECK_FALSE(d3d12_weight_type_supported(GGML_TYPE_Q8_0));
    CHECK_FALSE(d3d12_weight_type_supported(GGML_TYPE_F16));
    CHECK_FALSE(d3d12_weight_type_supported(GGML_TYPE_F32));
}

TEST_CASE("ggml_d3d12: supports_op rules for MUL_MAT") {
    D3d12MatmulDesc d = decode_desc(GGML_TYPE_Q4_K, 2048, 2048);
    CHECK(d3d12_mm_supported(d));

    // The buft probe: a weight still in D3D12_Host is refused, so llama.cpp
    // moves on to D3D12_Weights.
    D3d12MatmulDesc host = d;
    host.src0_in_weight_buffer = false;
    CHECK_FALSE(d3d12_mm_supported(host));

    D3d12MatmulDesc prefill = d; // the loader probes with 512 columns
    prefill.ne11 = 512;
    CHECK(d3d12_mm_supported(prefill));

    D3d12MatmulDesc t = d;
    t.src0_type = GGML_TYPE_Q8_0;
    CHECK_FALSE(d3d12_mm_supported(t));
    t = d;
    t.src1_type = GGML_TYPE_F16;
    CHECK_FALSE(d3d12_mm_supported(t));
    t = d;
    t.dst_type = GGML_TYPE_F16;
    CHECK_FALSE(d3d12_mm_supported(t));
    t = d;
    t.ne00 = t.ne10 = 2080; // not a multiple of 256
    CHECK_FALSE(d3d12_mm_supported(t));
    t = d;
    t.ne10 = 1024; // K mismatch
    CHECK_FALSE(d3d12_mm_supported(t));
    t = d;
    t.ne12 = 4; // batched attention-style matmul
    CHECK_FALSE(d3d12_mm_supported(t));
    t = d;
    t.ne02 = 2; // 3-D weight
    CHECK_FALSE(d3d12_mm_supported(t));
    t = d;
    t.src1_contiguous = false;
    CHECK_FALSE(d3d12_mm_supported(t));
    t = d;
    t.src0_contiguous = false;
    CHECK_FALSE(d3d12_mm_supported(t));

    // Coder-3B tied lm_head: 151936 rows → 37984 groups, inside the limit.
    CHECK(d3d12_mm_supported(decode_desc(GGML_TYPE_Q6_K, 151936, 2048)));
    CHECK_FALSE(d3d12_mm_supported(decode_desc(GGML_TYPE_Q6_K, 4 * 65536, 2048)));
}

TEST_CASE("ggml_d3d12: dispatch planner") {
    D3d12Dispatch d = d3d12_mm_dispatch(2048, 1);
    CHECK(d.ok);
    CHECK(d.groups_x == 512);
    CHECK(d.groups_y == 1);
    d = d3d12_mm_dispatch(6, 512);
    CHECK(d.groups_x == 2);
    CHECK(d.groups_y == 512);
    CHECK(d3d12_mm_dispatch(4 * 65535, 1).ok);
    CHECK_FALSE(d3d12_mm_dispatch(4 * 65535 + 1, 1).ok);
    CHECK_FALSE(d3d12_mm_dispatch(256, 65536).ok);
    CHECK_FALSE(d3d12_mm_dispatch(0, 1).ok);
}

TEST_CASE("ggml_d3d12: kernel width follows K (D2a runs 1 and 2)") {
    CHECK(d3d12_mm_threads(1024) == 64); // lm_head of LFM2.5-350M
    CHECK(d3d12_mm_threads(2048) == 64); // most projections
    CHECK(d3d12_mm_threads(3840) == 64);
    CHECK(d3d12_mm_threads(4096) == 128);  // first width with 16 chunks
    CHECK(d3d12_mm_threads(8192) == 128);  // LFM2.5-1.2B ffn_down
    CHECK(d3d12_mm_threads(11008) == 128); // Coder-3B ffn_down
}

TEST_CASE("ggml_d3d12: kernel emulation matches the CPU backend (q8 activations)") {
    struct Case {
        ggml_type t;
        int n, k;
    };
    const Case cases[] = {
        {GGML_TYPE_Q4_0, 13, 1024},
        {GGML_TYPE_Q4_K, 13, 1024},
        {GGML_TYPE_Q6_K, 13, 1024},
        // 9 super-blocks: 1890-byte rows, odd rows start 2-byte aligned.
        {GGML_TYPE_Q6_K, 6, 2304},
        // Coder-3B ffn_down width: 43 super-blocks, 9030-byte rows.
        {GGML_TYPE_Q6_K, 3, 11008},
        {GGML_TYPE_Q4_0, 5, 2304},
    };
    for (const auto& c : cases) {
        CAPTURE(ggml_type_name(c.t));
        CAPTURE(c.k);
        CHECK(emulate_rel_err(c.t, c.n, c.k, 1, 0, 0) <= kD3d12SelftestRelTol);
        CHECK(emulate_rel_err(c.t, c.n, c.k, 7, 12, 3) <= kD3d12SelftestRelTol);
    }
    CHECK(ggml_row_size(GGML_TYPE_Q6_K, 11008) % 4 == 2); // the case the ld32 trick covers
}

TEST_CASE("ggml_d3d12: selftest CSV and non-Windows behaviour") {
    D3d12SelftestRow r;
    r.type = "q6_k";
    r.n = 2048;
    r.k = 11008;
    r.ncols = 1;
    r.error = "a,b";
    const std::string line = format_d3d12_selftest_row(r, "host");
    int commas = 0, hcommas = 0;
    for (char ch : line)
        commas += ch == ',' ? 1 : 0;
    for (const char* p = d3d12_selftest_csv_header(); *p; ++p)
        hcommas += *p == ',' ? 1 : 0;
    CHECK(commas == hcommas);
#if !defined(_WIN32)
    CHECK_FALSE(ggml_d3d12_register());
    std::vector<D3d12SelftestRow> rows;
    run_d3d12_selftest(&rows);
    REQUIRE(rows.size() == 1);
    CHECK_FALSE(rows[0].d3d12_ran);
    CHECK_FALSE(rows[0].error.empty());
#endif
}

// The one place llama params get the GPU-layer request (src/bridge/llama_gpu.h).
#include "../src/bridge/llama_gpu.h"

TEST_CASE("ggml_d3d12: GPU-layer request maps to llama params") {
    llama_model_params mp = llama_model_default_params();
    CHECK(apply_gguf_gpu_layers(0, mp) == 0);
    CHECK(mp.n_gpu_layers == 0);
    CHECK_FALSE(mp.no_host); // a CPU load keeps llama's default weight bufts
    REQUIRE(mp.devices != nullptr);
    CHECK(mp.devices[0] == nullptr); // explicit empty device list: CPU only

    llama_context_params cp = llama_context_default_params();
    const bool kqv_default = cp.offload_kqv;
    apply_gguf_gpu_context(0, cp);
    CHECK(cp.offload_kqv == kqv_default);
    apply_gguf_gpu_context(28, cp);
    CHECK_FALSE(cp.offload_kqv); // KV and attention stay on the CPU

#if !defined(_WIN32)
    // No D3D12 on Linux: a request falls back to the CPU load unchanged.
    mp = llama_model_default_params();
    CHECK(apply_gguf_gpu_layers(99, mp) == 0);
    CHECK(mp.n_gpu_layers == 0);
    CHECK(mp.devices[0] == nullptr);
#endif
}
