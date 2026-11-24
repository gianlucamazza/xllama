// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "xllama/ggml_d3d12.h"

#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <optional>
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
    CHECK(d3d12_weight_type_supported(GGML_TYPE_Q5_K));
    CHECK(d3d12_weight_type_supported(GGML_TYPE_IQ4_XS));
    CHECK(d3d12_weight_type_supported(GGML_TYPE_IQ3_XXS));
    CHECK_FALSE(d3d12_weight_type_supported(GGML_TYPE_Q8_0));
    CHECK_FALSE(d3d12_weight_type_supported(GGML_TYPE_IQ2_XXS));
    CHECK_FALSE(d3d12_weight_type_supported(GGML_TYPE_F16));
    CHECK_FALSE(d3d12_weight_type_supported(GGML_TYPE_F32));
}

TEST_CASE("ggml_d3d12: supports_op rules for MUL_MAT") {
    D3d12MatmulDesc d = decode_desc(GGML_TYPE_Q4_K, 2048, 2048);
    CHECK(d3d12_mm_supported(d));
    CHECK(d3d12_mm_supported(decode_desc(GGML_TYPE_Q5_K, 2048, 2048)));
    CHECK(d3d12_mm_supported(decode_desc(GGML_TYPE_IQ4_XS, 2048, 2048)));
    CHECK(d3d12_mm_supported(decode_desc(GGML_TYPE_IQ3_XXS, 2048, 2048)));

    // The buft probe: a weight still in D3D12_Host is refused, so llama.cpp
    // moves on to D3D12_Weights.
    D3d12MatmulDesc host = d;
    host.src0_in_weight_buffer = false;
    CHECK_FALSE(d3d12_mm_supported(host));

    D3d12MatmulDesc prefill = d; // the loader probes with 512 columns
    prefill.ne11 = 512;
    CHECK(d3d12_mm_supported(prefill));
    D3d12MatmulDesc empty = d; // zero output rows in this ubatch
    empty.ne11 = 0;
    CHECK(d3d12_mm_supported(empty));

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

TEST_CASE("ggml_d3d12: optional MMV pipelines fall back per type") {
    D3d12MatmulDesc q40 = decode_desc(GGML_TYPE_Q4_0, 2048, 2048);
    D3d12MatmulDesc q5 = decode_desc(GGML_TYPE_Q5_K, 2048, 2048);
    D3d12MatmulDesc iq4 = decode_desc(GGML_TYPE_IQ4_XS, 2048, 2048);
    D3d12MatmulDesc iq3 = decode_desc(GGML_TYPE_IQ3_XXS, 2048, 2048);
    CHECK(d3d12_optional_mmv_bit(GGML_TYPE_Q4_0) == 0);
    CHECK(d3d12_optional_mmv_bit(GGML_TYPE_Q4_K) == 0);
    CHECK(d3d12_optional_mmv_bit(GGML_TYPE_Q6_K) == 0);
    // A failed Q5_K PSO must not refuse Q4_0 / Q4_K / Q6_K.
    CHECK(d3d12_mm_supported(q40, 0));
    CHECK(d3d12_mm_supported(decode_desc(GGML_TYPE_Q4_K, 2048, 2048), 0));
    CHECK(d3d12_mm_supported(decode_desc(GGML_TYPE_Q6_K, 2048, 2048), 0));
    CHECK_FALSE(d3d12_mm_supported(q5, 0));
    CHECK_FALSE(d3d12_mm_supported(iq4, 0));
    CHECK_FALSE(d3d12_mm_supported(iq3, 0));
    CHECK(d3d12_mm_supported(q5, kD3d12OptionalMmvQ5K));
    CHECK_FALSE(d3d12_mm_supported(iq3, kD3d12OptionalMmvQ5K));
    CHECK(d3d12_mm_supported(iq4, kD3d12OptionalMmvIq4Xs));
    CHECK(d3d12_mm_supported(iq3, kD3d12OptionalMmvIq3Xxs));
    CHECK(d3d12_mm_supported(q5, kD3d12OptionalMmvAll));
    CHECK(d3d12_mm_supported(iq3)); // default: every optional pipeline present
}

TEST_CASE("ggml_d3d12: 3x256 IQ3_XXS is correctness-only for the speed gate") {
    CHECK_FALSE(d3d12_selftest_speed_gated("iq3_xxs", 3, 256, 1));
    CHECK(d3d12_selftest_speed_gated("iq3_xxs", 2048, 2048, 1));
    CHECK_FALSE(d3d12_selftest_speed_gated("q4_0", 2048, 2048, 512)); // prefill
    CHECK(d3d12_selftest_speed_gated("q4_0", 2048, 2048, 1));
}

TEST_CASE("ggml_d3d12: IQ decode floors are below 100; other types stay at 100") {
    CHECK(d3d12_selftest_decode_min_gbs("q4_0") == kD3d12SelftestDecodeMinGbs);
    CHECK(d3d12_selftest_decode_min_gbs("q4_k") == 100.0);
    CHECK(d3d12_selftest_decode_min_gbs("q6_k") == 100.0);
    CHECK(d3d12_selftest_decode_min_gbs("q5_k") == 100.0);
    CHECK(d3d12_selftest_decode_min_gbs("iq4_xs") == kD3d12SelftestIq4XsDecodeMinGbs);
    CHECK(d3d12_selftest_decode_min_gbs("iq3_xxs") == kD3d12SelftestIq3XxsDecodeMinGbs);
    CHECK(kD3d12SelftestIq4XsDecodeMinGbs < 100.0);
    CHECK(kD3d12SelftestIq3XxsDecodeMinGbs < kD3d12SelftestIq4XsDecodeMinGbs);
    // Pre-opt console rows (80.88 / 51.27) sit below the provisional floors.
    CHECK(80.88 < kD3d12SelftestIq4XsDecodeMinGbs);
    CHECK(51.27 < kD3d12SelftestIq3XxsDecodeMinGbs);
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
    // A ubatch with no output rows leaves the last layer's FFN with zero
    // columns (llama.cpp gathers the output rows first). Refusing it sent the
    // matmul to the CPU, which cannot read D3D12_Weights: the scheduler copied
    // the weights instead (validate genroom timed out, CI 1.6.0.1155).
    d = d3d12_mm_dispatch(2048, 0);
    CHECK(d.ok);
    CHECK(d.groups_y == 0); // nothing to dispatch
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
        {GGML_TYPE_Q5_K, 13, 1024},
        {GGML_TYPE_IQ4_XS, 13, 1024},
        {GGML_TYPE_IQ3_XXS, 13, 1024},
        // 9 super-blocks: 1890-byte rows, odd rows start 2-byte aligned.
        {GGML_TYPE_Q6_K, 6, 2304},
        // Coder-3B ffn_down width: 43 super-blocks, 9030-byte rows.
        {GGML_TYPE_Q6_K, 3, 11008},
        {GGML_TYPE_Q4_0, 5, 2304},
        {GGML_TYPE_Q5_K, 5, 2304},
        // One IQ3_XXS super-block: 98-byte rows, 2-byte aligned.
        {GGML_TYPE_IQ3_XXS, 6, 256},
    };
    for (const auto& c : cases) {
        CAPTURE(ggml_type_name(c.t));
        CAPTURE(c.k);
        CHECK(emulate_rel_err(c.t, c.n, c.k, 1, 0, 0) <= kD3d12SelftestRelTol);
        CHECK(emulate_rel_err(c.t, c.n, c.k, 7, 12, 3) <= kD3d12SelftestRelTol);
    }
    CHECK(ggml_row_size(GGML_TYPE_Q6_K, 11008) % 4 == 2); // the case the ld32 trick covers
    CHECK(ggml_row_size(GGML_TYPE_IQ3_XXS, 256) % 4 == 2);
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
    CHECK(apply_gguf_gpu_layers(0, mp, "") == 0);
    CHECK(mp.n_gpu_layers == 0);
    CHECK_FALSE(mp.no_host); // a CPU load keeps llama's default weight bufts
    REQUIRE(mp.devices != nullptr);
    CHECK(mp.devices[0] == nullptr); // explicit empty device list: CPU only

    llama_context_params cp = llama_context_default_params();
    const bool kqv_default = cp.offload_kqv;
    REQUIRE(cp.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO);
    apply_gguf_gpu_context(0, cp, gguf_gpu_outputs_max(false, false));
    CHECK(cp.offload_kqv == kqv_default);
    CHECK(cp.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_AUTO);
    const uint32_t outputs_default = cp.n_outputs_max;
    apply_gguf_gpu_context(0, cp, gguf_gpu_outputs_max(false, false));
    CHECK(cp.n_outputs_max == outputs_default);
    apply_gguf_gpu_context(28, cp, gguf_gpu_outputs_max(false, false));
    CHECK_FALSE(cp.offload_kqv);   // KV and attention stay on the CPU
    CHECK(cp.n_outputs_max == 1u); // compute reserve sized for one logits row (#309)
    // Same attention as the CPU-only path, where AUTO resolves to enabled.
    CHECK(cp.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_ENABLED);
    llama_context_params no_fa = llama_context_default_params();
    no_fa.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_DISABLED;
    apply_gguf_gpu_context(28, no_fa, gguf_gpu_outputs_max(false, false));
    CHECK(no_fa.flash_attn_type == LLAMA_FLASH_ATTN_TYPE_DISABLED); // a caller's choice stands

    // llama.cpp asserts when a batch asks for more outputs than n_outputs_max,
    // so the cap follows what the context will request: a prompt-lookup verify
    // batch reads 1 + draft rows, an embedding context every token.
    CHECK(gguf_gpu_outputs_max(false, true) == 1u + kSpecDraftKDefault);
    CHECK(gguf_gpu_outputs_max(true, false) == 0u);
    llama_context_params emb = llama_context_default_params();
    apply_gguf_gpu_context(28, emb, gguf_gpu_outputs_max(true, true));
    CHECK(emb.n_outputs_max == outputs_default); // left to llama.cpp

#if !defined(_WIN32)
    // No D3D12 on Linux: a request falls back to the CPU load unchanged.
    mp = llama_model_default_params();
    CHECK(apply_gguf_gpu_layers(99, mp, "") == 0);
    CHECK(mp.n_gpu_layers == 0);
    CHECK(mp.devices[0] == nullptr);
#endif
}

TEST_CASE("ggml_d3d12: GET_ROWS emulation equals ggml's Q6_K dequantizer bit for bit (#309)") {
    const int k = 2304, rows = 9; // 1890-byte rows: odd rows start 2-byte aligned
    std::mt19937 rng(7u);
    std::uniform_real_distribution<float> uni(-1.f, 1.f);
    std::vector<float> wf(static_cast<std::size_t>(rows) * k);
    for (float& v : wf)
        v = uni(rng);
    const std::size_t row_bytes = ggml_row_size(GGML_TYPE_Q6_K, k);
    std::vector<std::uint8_t> q(row_bytes * rows + 8);
    ggml_quantize_chunk(GGML_TYPE_Q6_K, wf.data(), q.data(), 0, rows, k, nullptr);
    const std::int32_t ids[] = {3, 0, 8, 3, 5};
    const std::size_t y_stride = static_cast<std::size_t>(k) + 5;
    std::vector<float> y(y_stride * 5, -7.f);
    d3d12_get_rows_emulate(GGML_TYPE_Q6_K, q.data(), row_bytes, ids, 5, k, y.data(), y_stride);
    std::vector<float> ref(static_cast<std::size_t>(k));
    for (int i = 0; i < 5; ++i) {
        ggml_get_type_traits(GGML_TYPE_Q6_K)
            ->to_float(q.data() + ids[i] * row_bytes, ref.data(), k);
        int mismatches = 0;
        for (int j = 0; j < k; ++j)
            mismatches += y[i * y_stride + j] != ref[j] ? 1 : 0;
        CAPTURE(i);
        CHECK(mismatches == 0);
        for (std::size_t p = k; p < y_stride; ++p)
            CHECK(y[i * y_stride + p] == -7.f);
    }
}

TEST_CASE("ggml_d3d12: GET_ROWS rules and tied-embedding placement") {
    D3d12GetRowsDesc d;
    d.src0_type = GGML_TYPE_Q6_K;
    d.ne00 = 2048;
    d.ne01 = 151936;
    d.ne10 = 512;
    d.src0_in_weight_buffer = true;
    CHECK(d3d12_get_rows_supported(d));
    D3d12GetRowsDesc t = d;
    t.src0_in_weight_buffer = false;
    CHECK_FALSE(d3d12_get_rows_supported(t));
    t = d;
    t.src0_type = GGML_TYPE_Q8_0;
    CHECK_FALSE(d3d12_get_rows_supported(t));
    t = d;
    t.src1_type = GGML_TYPE_I64;
    CHECK_FALSE(d3d12_get_rows_supported(t));
    t = d;
    t.src1_contiguous = false; // ids are staged with one memcpy
    CHECK_FALSE(d3d12_get_rows_supported(t));
    t = d;
    t.ne11 = 2; // batched ids
    CHECK_FALSE(d3d12_get_rows_supported(t));
    t = d;
    t.ne00 = 2080;
    CHECK_FALSE(d3d12_get_rows_supported(t));
    t = d;
    t.ne10 = 0; // no rows to gather: a no-op the backend still owns
    CHECK(d3d12_get_rows_supported(t));

    CHECK(d3d12_place_tied_embedding(GGML_TYPE_Q6_K, /*has_output_weight=*/false));
    CHECK_FALSE(d3d12_place_tied_embedding(GGML_TYPE_Q6_K, /*has_output_weight=*/true));
    CHECK_FALSE(d3d12_place_tied_embedding(GGML_TYPE_Q8_0, false));
#if !defined(_WIN32)
    CHECK(ggml_d3d12_weights_buft() == nullptr);
#endif
}

TEST_CASE("ggml_d3d12: bias ADD rules") {
    // Qcur = ggml_add(Qcur, wq_b) in build_qkv: Qwen2.5-Coder-3B q_proj, decode.
    D3d12AddDesc d;
    d.ne00 = 2048;
    d.ne01 = 1;
    d.ne10 = 2048;
    d.src1_in_weight_buffer = true;
    CHECK(d3d12_add_supported(d));

    // llama.cpp's per-weight buft probe builds ggml_add(a, w) with `a` shaped
    // like the 1-D bias (llama-model-loader.cpp weight_buft_supported).
    D3d12AddDesc t = d;
    t.ne01 = 1;
    CHECK(d3d12_add_supported(t));
    t = d;
    t.ne01 = 512; // prefill ubatch
    CHECK(d3d12_add_supported(t));
    t = d;
    t.ne01 = 0; // a ubatch without output rows: still ours, a no-op
    CHECK(d3d12_add_supported(t));
    t = d;
    t.ne00 = t.ne10 = 18944; // Qwen2.5 ffn width, not a multiple of 256 groups
    CHECK(d3d12_add_supported(t));
    t.ne00 = t.ne10 = 129; // a partial last group
    CHECK(d3d12_add_supported(t));

    // A residual add has no weight in D3D12_Weights: stays on the CPU.
    t = d;
    t.src1_in_weight_buffer = false;
    CHECK_FALSE(d3d12_add_supported(t));
    t = d;
    t.ne11 = 4; // src1 a 2-D tensor, not a bias vector
    CHECK_FALSE(d3d12_add_supported(t));
    t = d;
    t.ne10 = 1; // broadcast scalar
    CHECK_FALSE(d3d12_add_supported(t));
    t = d;
    t.ne02 = 2; // 3-D activations
    CHECK_FALSE(d3d12_add_supported(t));
    t = d;
    t.src1_type = GGML_TYPE_F16;
    CHECK_FALSE(d3d12_add_supported(t));
    t = d;
    t.src0_type = GGML_TYPE_F16;
    CHECK_FALSE(d3d12_add_supported(t));
    t = d;
    t.dst_type = GGML_TYPE_F16;
    CHECK_FALSE(d3d12_add_supported(t));
    t = d;
    t.src0_rows_contiguous = false;
    CHECK_FALSE(d3d12_add_supported(t));
    t = d;
    t.src1_contiguous = false;
    CHECK_FALSE(d3d12_add_supported(t));
    t = d;
    t.ne01 = std::int64_t(kD3d12MaxGroups) + 1; // one group row per column
    CHECK_FALSE(d3d12_add_supported(t));
    t = d;
    t.ne00 = t.ne10 = std::int64_t(kD3d12MaxGroups) * kD3d12AddThreads + 1;
    CHECK_FALSE(d3d12_add_supported(t));
}

TEST_CASE("llama_gpu: token_embd type and tie read from the GGUF header") {
    auto write = [](const char* path, bool tied) {
        ggml_init_params ip = {};
        ip.mem_size = 4 * ggml_tensor_overhead();
        ip.no_alloc = true;
        ggml_context* ctx = ggml_init(ip);
        gguf_context* g = gguf_init_empty();
        ggml_tensor* e = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, 256, 4);
        ggml_set_name(e, "token_embd.weight");
        gguf_add_tensor(g, e);
        if (!tied) {
            ggml_tensor* o = ggml_new_tensor_2d(ctx, GGML_TYPE_Q6_K, 256, 4);
            ggml_set_name(o, "output.weight");
            gguf_add_tensor(g, o);
        }
        const bool ok = gguf_write_to_file(g, path, /*only_meta=*/true);
        gguf_free(g);
        ggml_free(ctx);
        return ok;
    };
    const std::string tied = "test_llama_gpu_tied.gguf", untied = "test_llama_gpu_untied.gguf";
    REQUIRE(write(tied.c_str(), true));
    REQUIRE(write(untied.c_str(), false));
    ggml_type t = GGML_TYPE_COUNT;
    bool has_output = true;
    REQUIRE(gguf_embedding_info(tied, &t, &has_output));
    CHECK(t == GGML_TYPE_Q6_K);
    CHECK_FALSE(has_output);
    REQUIRE(gguf_embedding_info(untied, &t, &has_output));
    CHECK(has_output);
    CHECK_FALSE(gguf_embedding_info("no-such-file.gguf", &t, &has_output));
    std::remove(tied.c_str());
    std::remove(untied.c_str());
}

// Opt-in: XLLAMA_TEST_MODEL=/path/to/model.gguf. The GPU-path context params
// must accept a prompt-lookup verify batch (1 + draft logits): with
// n_outputs_max = 1 llama.cpp aborts in output_reserve. Runs on the CPU — the
// cap is a llama.cpp context rule, not a d3d12 one.
TEST_CASE("llama_gpu: GPU-path context accepts a prompt-lookup verify batch (opt-in)") {
    const char* model_env = std::getenv("XLLAMA_TEST_MODEL");
    if (!model_env) {
        MESSAGE("XLLAMA_TEST_MODEL not set — skipping");
        return;
    }
    llama_model* model = llama_model_load_from_file(model_env, llama_model_default_params());
    REQUIRE(model);
    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = 512;
    apply_gguf_gpu_context(1, cp, gguf_gpu_outputs_max(false, true));
    llama_context* ctx = llama_init_from_model(model, cp);
    REQUIRE(ctx);
    const llama_vocab* vocab = llama_model_get_vocab(model);
    std::vector<llama_token> toks(16);
    const int n = llama_tokenize(vocab, "The quick brown fox", 19, toks.data(), 16, true, false);
    REQUIRE(n > 0);
    CHECK(llama_decode(ctx, llama_batch_get_one(toks.data(), n)) == 0);
    const int n_verify = 1 + kSpecDraftKDefault;
    llama_batch b = llama_batch_init(n_verify, 0, 1);
    for (int i = 0; i < n_verify; ++i) {
        b.token[i] = toks[static_cast<std::size_t>(i % n)];
        b.pos[i] = n + i;
        b.n_seq_id[i] = 1;
        b.seq_id[i][0] = 0;
        b.logits[i] = 1;
    }
    b.n_tokens = n_verify;
    CHECK(llama_decode(ctx, b) == 0);
    llama_batch_free(b);
    llama_free(ctx);
    llama_model_free(model);
}

TEST_CASE("d3d12: phase counters never report missing GPU time as zero") {
    D3d12PhaseStats s;
    CHECK_FALSE(s.gpu_time_valid());
    CHECK(format_d3d12_phase("prefill", s).find("gpu_ms=unavailable") != std::string::npos);
    s.calls = 2;
    s.submissions = 3;
    s.dispatches = 252;
    s.ts_valid = 3;
    s.gpu_ms = 12.3456;
    s.wall_ms = 20.0;
    CHECK(s.gpu_time_valid());
    const std::string line = format_d3d12_phase("decode", s);
    CHECK(line.rfind("[xllama] d3d12-phase phase=decode calls=2 submissions=3 dispatches=252 ",
                     0) == 0);
    CHECK(line.find("gpu_ms=12.346 ts_valid=3 ts_missing=0\n") != std::string::npos);
    s.ts_missing = 1; // one submission without a valid pair invalidates the sum
    CHECK_FALSE(s.gpu_time_valid());
    CHECK(format_d3d12_phase("decode", s).find("gpu_ms=unavailable") != std::string::npos);
}

TEST_CASE("d3d12: counters are unavailable off Windows") {
    D3d12PhaseStats s;
    s.calls = 7;
    CHECK_FALSE(ggml_d3d12_take_stats(&s));
    CHECK(s.calls == 0);
    CHECK_FALSE(ggml_d3d12_take_stats(nullptr));
}

TEST_CASE("d3d12: dispatch timing flag and column classes") {
    CHECK(d3d12_flag_enabled("1"));
    CHECK(d3d12_flag_enabled(" 1\r\n"));
    CHECK_FALSE(d3d12_flag_enabled(""));
    CHECK_FALSE(d3d12_flag_enabled("0"));
    CHECK_FALSE(d3d12_flag_enabled("11"));
    CHECK_FALSE(d3d12_flag_enabled("1 x"));
    CHECK(std::string(d3d12_cols_class(0)) == "1");
    CHECK(std::string(d3d12_cols_class(1)) == "1");
    CHECK(std::string(d3d12_cols_class(2)) == "2-63");
    CHECK(std::string(d3d12_cols_class(63)) == "2-63");
    CHECK(std::string(d3d12_cols_class(64)) == "64-511");
    CHECK(std::string(d3d12_cols_class(511)) == "64-511");
    CHECK(std::string(d3d12_cols_class(512)) == "512+");
}

TEST_CASE("d3d12: layer index from tensor name or graph node") {
    CHECK(d3d12_layer_from_name("blk.0.attn_q.weight") == 0);
    CHECK(d3d12_layer_from_name("blk.3.ffn_up.weight") == 3);
    CHECK(d3d12_layer_from_name("blk.12.ffn_down_exps.weight") == 12);
    CHECK(d3d12_layer_from_name("D3D12#blk.7.attn_output.weight#0") == 7);
    CHECK(d3d12_layer_from_name("CPU#blk.1.ffn_gate.weight#3") == 1);
    CHECK(d3d12_layer_from_name("Qcur-0") == 0);
    CHECK(d3d12_layer_from_name("l_out-15") == 15);
    CHECK(d3d12_layer_from_name("ffn_out-3") == 3);
    CHECK(d3d12_layer_from_name("CPU#Qcur-9#0") == 9);
    CHECK_FALSE(d3d12_layer_from_name("token_embd.weight").has_value());
    CHECK_FALSE(d3d12_layer_from_name("output.weight").has_value());
    CHECK_FALSE(d3d12_layer_from_name("output_norm.weight").has_value());
    CHECK_FALSE(d3d12_layer_from_name("result_output").has_value());
    CHECK_FALSE(d3d12_layer_from_name("").has_value());
    CHECK_FALSE(d3d12_layer_from_name("blk.x.ffn_up.weight").has_value());
    CHECK_FALSE(d3d12_layer_from_name("notblk.2.weight").has_value());
    CHECK_FALSE(d3d12_layer_from_name("blk.2a.weight").has_value());
    // Weight name wins over a graph node that names a different layer.
    CHECK(d3d12_layer_from_names("blk.5.attn_q.weight", "Qcur-0") == 5);
    CHECK(d3d12_layer_from_names("token_embd.weight", "GET_ROWS") == std::nullopt);
    CHECK(d3d12_layer_from_names("", "l_out-4") == 4);
    CHECK(d3d12_layer_from_names("D3D12#blk.0.attn_k.weight#0", "Kcur-0") == 0);
    CHECK(std::string(kD3d12UnattributedLayer) == "unattributed");
    CHECK(d3d12_layer_label(3) == "3");
    CHECK(d3d12_layer_label(0) == "0");
    CHECK(d3d12_layer_label(std::nullopt) == "unattributed");
}

TEST_CASE("d3d12: layer profile aggregates by phase × layer × op") {
    D3d12LayerProfile p;
    CHECK(p.empty());
    CHECK(format_d3d12_layer_profile("prefill", p).empty());
    D3d12LayerKey mat_l3;
    mat_l3.layer = 3;
    mat_l3.op = "mul_mat";
    D3d12LayerKey mat_l0;
    mat_l0.layer = 0;
    mat_l0.op = "mul_mat";
    D3d12LayerKey rows;
    rows.op = "get_rows";
    p.add(mat_l3, 1.0, 0.04);
    p.add(mat_l0, 2.5, 0.01);
    p.add(mat_l3, 2.0, 0.06);
    p.add(rows, 0.5, 0.02);
    REQUIRE(p.rows().size() == 3);
    const auto s = p.sorted();
    REQUIRE(s.size() == 3);
    CHECK(s[0].key.layer == 0); // layer index ascending
    CHECK(s[1].key.layer == 3);
    CHECK(s[1].count == 2);
    CHECK(s[1].gpu_ms == doctest::Approx(3.0));
    CHECK(s[1].wait_ms == doctest::Approx(0.10));
    CHECK_FALSE(s[2].key.layer.has_value()); // unattributed last
    CHECK(s[2].key.op == "get_rows");
    CHECK(d3d12_layer_label(s[2].key.layer) == kD3d12UnattributedLayer);
    CHECK(format_d3d12_layer_profile("prefill", p).find("layer=unattributed op=get_rows") !=
          std::string::npos);
}

TEST_CASE("d3d12: unattributed bucket for names with no layer") {
    CHECK_FALSE(d3d12_layer_from_name("token_embd.weight").has_value());
    CHECK_FALSE(d3d12_layer_from_name("output.weight").has_value());
    CHECK(d3d12_layer_from_names("token_embd.weight", "") == std::nullopt);
    D3d12LayerProfile p;
    D3d12LayerKey embd;
    embd.op = "get_rows";
    embd.layer = d3d12_layer_from_name("token_embd.weight");
    p.add(embd, 0.25, 0.01);
    D3d12LayerKey head;
    head.op = "mul_mat";
    head.layer = d3d12_layer_from_name("output.weight");
    p.add(head, 0.75, 0.02);
    REQUIRE(p.rows().size() == 2); // two ops, one unattributed bucket each
    CHECK(d3d12_layer_label(p.rows()[0].key.layer) == "unattributed");
    CHECK(d3d12_layer_label(p.rows()[1].key.layer) == "unattributed");
    const std::string out = format_d3d12_layer_profile("decode", p);
    CHECK(out.find("layer=unattributed op=get_rows count=1 gpu_ms=0.250") != std::string::npos);
    CHECK(out.find("layer=unattributed op=mul_mat count=1 gpu_ms=0.750") != std::string::npos);
    CHECK(out.find("layer=0") == std::string::npos);
}

TEST_CASE("d3d12: layer profile keeps missing timestamps as missing") {
    D3d12LayerProfile p;
    D3d12LayerKey k;
    k.layer = 1;
    k.op = "mul_mat";
    p.add(k, std::nullopt, std::nullopt);
    REQUIRE(p.rows().size() == 1);
    CHECK(p.rows()[0].count == 1);
    CHECK(p.rows()[0].gpu_timed == 0);
    CHECK(p.rows()[0].gpu_missing == 1);
    CHECK(p.rows()[0].wait_timed == 0);
    CHECK(p.rows()[0].wait_missing == 1);
    CHECK(p.rows()[0].gpu_ms == 0.0); // unused when gpu_timed==0; the line prints missing
    const std::string out = format_d3d12_layer_profile("decode", p);
    CHECK(out.find("gpu_ms=missing") != std::string::npos);
    CHECK(out.find("wait_ms=missing") != std::string::npos);
    CHECK(out.find("gpu_ms=0.000") == std::string::npos);
    CHECK(out.find("wait_ms=0.000") == std::string::npos);
    // A later valid sample does not invent times for the missing one.
    p.add(k, 1.5, 0.05);
    CHECK(p.rows()[0].gpu_timed == 1);
    CHECK(p.rows()[0].gpu_missing == 1);
    CHECK(p.rows()[0].gpu_ms == doctest::Approx(1.5));
    const std::string mixed = format_d3d12_layer_profile("decode", p);
    CHECK(mixed.find("gpu_ms=1.500") != std::string::npos);
    CHECK(mixed.find("gpu_missing=1") != std::string::npos);
    CHECK(mixed.find("wait_ms=0.050") != std::string::npos);
    CHECK(mixed.find("wait_missing=1") != std::string::npos);
    // Valid zero wait is reported as 0.000, not missing.
    D3d12LayerProfile zero;
    D3d12LayerKey z;
    z.layer = 2;
    z.op = "mul_mat";
    zero.add(z, 0.001, 0.0);
    const std::string zline = format_d3d12_layer_profile("prefill", zero);
    CHECK(zline.find("wait_ms=0.000") != std::string::npos);
    CHECK(zline.find("wait_ms=missing") == std::string::npos);
}

TEST_CASE("d3d12: layer profile line format") {
    D3d12LayerProfile p;
    D3d12LayerKey k;
    k.layer = 3;
    k.op = "mul_mat";
    p.add(k, 1.0, 0.04);
    p.add(k, 2.0, 0.06);
    D3d12LayerKey u;
    u.op = "get_rows";
    p.add(u, 0.5, std::nullopt);
    const std::string out = format_d3d12_layer_profile("prefill", p);
    CHECK(out.rfind("[xllama] d3d12-layer phase=prefill layer=3 op=mul_mat count=2 gpu_ms=3.000 "
                    "wait_ms=0.100 gpu_missing=0 wait_missing=0\n",
                    0) == 0);
    CHECK(out.find("[xllama] d3d12-layer phase=prefill layer=unattributed op=get_rows count=1 "
                   "gpu_ms=0.500 wait_ms=missing gpu_missing=0 wait_missing=1\n") !=
          std::string::npos);
    CHECK(out.find("[xllama] d3d12-layer phase=prefill summary rows=2 timed=3 missing=0 "
                   "gpu_ms=3.500 wait_ms=0.100 wait_missing=1\n") != std::string::npos);
}

TEST_CASE("d3d12: dispatch profile groups by kind and reports untimed dispatches") {
    D3d12DispatchProfile p;
    CHECK(p.empty());
    CHECK(format_d3d12_dispatch_profile("prefill", p).empty());
    D3d12DispatchKind q4{"mul_mat", "q4_0", 128, 1, "512+"};
    D3d12DispatchKind q6{"mul_mat", "q6_K", 64, 1, "512+"};
    p.add(q4, 1.0, 1048576.0, 512);
    p.add(q6, 5.0, 2097152.0, 512);
    p.add(q4, 2.0, 1048576.0, 512);
    p.note_missing();
    REQUIRE(p.rows().size() == 2);
    CHECK(p.rows()[0].count == 2);
    CHECK(p.rows()[0].gpu_ms == doctest::Approx(3.0));
    CHECK(p.rows()[0].columns == 1024);
    const auto s = p.sorted();
    CHECK(s[0].kind == q6); // 5 ms before 3 ms
    const std::string out = format_d3d12_dispatch_profile("prefill", p);
    CHECK(out.rfind("[xllama] d3d12-dispatch phase=prefill op=mul_mat type=q6_K cpg=1 "
                    "threads=64 cols=512+ count=1 gpu_ms=5.000",
                    0) == 0);
    CHECK(out.find("type=q4_0 cpg=1 threads=128 cols=512+ count=2 gpu_ms=3.000 ms_per=1.5000 "
                   "weight_mib=2.0 columns=1024\n") != std::string::npos);
    CHECK(out.find("summary kinds=2 timed=3 missing=1 gpu_ms=8.000\n") != std::string::npos);
    D3d12DispatchProfile only_missing;
    only_missing.note_missing();
    CHECK_FALSE(only_missing.empty());
    CHECK(format_d3d12_dispatch_profile("decode", only_missing).find("missing=1") !=
          std::string::npos);
}
