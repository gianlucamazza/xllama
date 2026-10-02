// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "xllama/ggml_d3d12.h"

#include "ggml-cpu.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
