// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "xllama/inference_params.h" // kDefaultNCtx (#171)
#include "xllama/routing_policy.h"

#include <algorithm>

using namespace xllama;

TEST_CASE("routing: cpu-only") {
    RoutingSettings s;
    s.mode = RoutingMode::CpuOnly;
    s.cpu_model = "smollm2-360m-cpu-int4";
    s.gpu_model = "smollm2-360m-dml-fp16";
    auto d = decide_routing(s, 2000, false, true);
    CHECK(d.active_model == "smollm2-360m-cpu-int4");
    CHECK_FALSE(d.use_gpu);
}

// #91 postmortem: GPU text routing is allowed only for parity-validated DML
// assets (dml_text_model_ok — the broken-RMSNorm ones must stay on CPU). A
// non-validated gpu_model forces the CPU model in every mode.
TEST_CASE("routing: #91 gate forces cpu for non-validated gpu_model") {
    RoutingSettings s;
    s.cpu_model = "cpu";
    s.gpu_model = "smollm2-360m-dml-fp16"; // pre-fix asset: never validated
    for (auto mode : {RoutingMode::GpuOnly, RoutingMode::Auto}) {
        s.mode = mode;
        auto d = decide_routing(s, 2000, false, true);
        CHECK(d.active_model == "cpu");
        CHECK_FALSE(d.use_gpu);
    }
}

TEST_CASE("dml_text_model_ok allowlist") {
    CHECK(dml_text_model_ok("smollm2-360m-dml-fp16-v2"));
    CHECK_FALSE(dml_text_model_ok("smollm2-360m-dml-fp16"));
    CHECK_FALSE(dml_text_model_ok(""));
}

TEST_CASE("routing: gpu-only with gpu available") {
    RoutingSettings s;
    s.mode = RoutingMode::GpuOnly;
    s.cpu_model = "cpu";
    s.gpu_model = "smollm2-360m-dml-fp16-v2";
    auto d = decide_routing(s, 10, false, true);
    CHECK(d.active_model == "smollm2-360m-dml-fp16-v2");
    CHECK(d.use_gpu);
}

TEST_CASE("routing: gpu-only without gpu falls back to cpu model name") {
    RoutingSettings s;
    s.mode = RoutingMode::GpuOnly;
    s.cpu_model = "cpu";
    s.gpu_model = "smollm2-360m-dml-fp16-v2";
    auto d = decide_routing(s, 10, false, false);
    CHECK(d.active_model == "cpu");
    CHECK_FALSE(d.use_gpu);
}

TEST_CASE("routing: auto short prompt stays cpu") {
    RoutingSettings s;
    s.mode = RoutingMode::Auto;
    s.token_threshold = 600;
    s.cpu_model = "cpu";
    s.gpu_model = "smollm2-360m-dml-fp16-v2";
    auto d = decide_routing(s, 100, false, true);
    CHECK(d.active_model == "cpu");
    CHECK_FALSE(d.use_gpu);
}

TEST_CASE("routing: auto long prompt uses gpu when available") {
    RoutingSettings s;
    s.mode = RoutingMode::Auto;
    s.token_threshold = 600;
    s.cpu_model = "cpu";
    s.gpu_model = "smollm2-360m-dml-fp16-v2";
    auto d = decide_routing(s, 800, false, true);
    CHECK(d.active_model == "smollm2-360m-dml-fp16-v2");
    CHECK(d.use_gpu);
}

TEST_CASE("routing: auto long prompt without gpu stays cpu") {
    RoutingSettings s;
    s.mode = RoutingMode::Auto;
    s.token_threshold = 600;
    s.cpu_model = "cpu";
    s.gpu_model = "smollm2-360m-dml-fp16-v2";
    auto d = decide_routing(s, 800, false, false);
    CHECK(d.active_model == "cpu");
    CHECK_FALSE(d.use_gpu);
}

TEST_CASE("routing: auto boundary is strictly above the threshold") {
    // The other auto cases probe 100 and 800 against a threshold of 600, so the
    // constant could be changed to anything in (100, 800] without failing a test
    // — including by accident while retuning it. Pin the boundary itself, and the
    // strict '>' in decide_routing (exactly threshold tokens must stay on CPU).
    RoutingSettings s;
    s.mode = RoutingMode::Auto;
    s.cpu_model = "cpu";
    s.gpu_model = "smollm2-360m-dml-fp16-v2";

    // Default threshold, asserted explicitly so a retune is a deliberate edit here.
    CHECK(RoutingSettings{}.token_threshold == 1550);

    s.token_threshold = 600;
    CHECK_FALSE(decide_routing(s, 599, false, true).use_gpu);
    CHECK_FALSE(decide_routing(s, 600, false, true).use_gpu); // strictly greater
    CHECK(decide_routing(s, 601, false, true).use_gpu);

    // The boundary tracks the setting, not a hardcoded 600.
    s.token_threshold = 1200;
    CHECK_FALSE(decide_routing(s, 1200, false, true).use_gpu);
    CHECK(decide_routing(s, 1201, false, true).use_gpu);
}

TEST_CASE("routing: the threshold stays reachable under the context trimmer") {
    // #133. BuildPrompt trims oldest turns while its estimate exceeds
    // kMaxPromptTokens, and it runs BEFORE routing — so a turn longer than that
    // budget never reaches decide_routing() at its full length. If the threshold
    // is at or above the budget, auto GPU routing is unreachable for every input
    // and the feature is dead without any test failing. That is exactly what the
    // 600 -> 1550 retune in #129 did, and it stayed invisible until a console run.
    //
    // The margin below is not decoration: the trimmer measures an ESTIMATE while
    // routing measures REAL tokens, so they disagree by however much the
    // chars-per-token constant is off. Leave room for that disagreement.
    CHECK(RoutingSettings{}.token_threshold < kMaxPromptTokens);
    CHECK(kMaxPromptTokens - RoutingSettings{}.token_threshold >= 100);

    // The estimator underestimates tokens per char on purpose (see the header):
    // 5000 chars of prose is ~940 real tokens, and estimating 1000 trims early
    // rather than overflowing n_ctx.
    CHECK(estimate_tokens_from_chars(5000) == 1000);
    CHECK(estimate_tokens_from_chars(0) == 0);

    // The measured console datapoint that found the bug: 7100 chars tokenized to
    // 1329 real tokens. The estimate must not sit so far above that number that
    // the trimmer cuts a turn routing would have sent to the GPU.
    CHECK(estimate_tokens_from_chars(7100) == 1420);
    CHECK(estimate_tokens_from_chars(7100) < kMaxPromptTokens);
}

TEST_CASE("routing: the trimmer budget fits the shipping context") {
    // #171. kMaxPromptTokens is sized against kDefaultNCtx "with ~250 tokens
    // left for generation" — but until the context size had one home, nothing
    // related the two numbers (the same silent-divergence shape as #133). A
    // trimmed-full prompt must fit the context with real room to generate;
    // if either constant moves, this states the invariant the other must keep.
    CHECK(kMaxPromptTokens < kDefaultNCtx);
    CHECK(kDefaultNCtx - kMaxPromptTokens >= 200);
}

TEST_CASE("resolve_n_ctx clamps and defaults") {
    CHECK(resolve_n_ctx(0) == kDefaultNCtx);
    CHECK(resolve_n_ctx(-1) == kDefaultNCtx);
    CHECK(resolve_n_ctx(4096) == 4096);
    CHECK(resolve_n_ctx(kMinSessionNCtx - 1) == kMinSessionNCtx);
    CHECK(resolve_n_ctx(kMaxSessionNCtx + 1) == kMaxSessionNCtx);
}

TEST_CASE("max_prompt_tokens_for_n_ctx tracks session size") {
    // Shipping default keeps the historical constant (not n-250 arithmetic).
    CHECK(max_prompt_tokens_for_n_ctx(0) == kMaxPromptTokens);
    CHECK(max_prompt_tokens_for_n_ctx(kDefaultNCtx) == kMaxPromptTokens);
    // Coding catalogue n_ctx=4096 leaves ~250 for generation.
    CHECK(max_prompt_tokens_for_n_ctx(4096) == 4096 - kReservedGenerationTokens);
    CHECK(max_prompt_tokens_for_n_ctx(4096) > kMaxPromptTokens);
    CHECK(max_prompt_tokens_for_n_ctx(4096) < 4096);
}

TEST_CASE("routing: the reply's reserve must not shrink the estimate ceiling") {
    // The one that bit. Charging fit_prompt's reply reserve to this ceiling put it
    // at 1536 for the shipping n_predict of 512 — under token_threshold (1550) —
    // so auto GPU routing was unreachable on every default install, and the
    // console gate did not see it because it pinned a lower n_predict. The ceiling
    // is a CONTEXT bound; the reply's room is fit_prompt's, applied exactly and
    // later. Nothing about n_predict may appear in this signature.
    CHECK(max_prompt_tokens_for_n_ctx(kDefaultNCtx) > RoutingSettings{}.token_threshold);
    // Real tokens, not estimated ones, are what decide_routing compares: 5.34
    // measured chars/token against the 5.0 estimator means the ceiling maps to
    // ~1685 real tokens. Keep the margin above the threshold.
    const double real_tokens =
        max_prompt_tokens_for_n_ctx(kDefaultNCtx) * kEstimatedCharsPerToken / 5.34;
    CHECK(real_tokens > RoutingSettings{}.token_threshold);

    // And the coding context keeps its own headroom.
    CHECK(max_prompt_tokens_for_n_ctx(4096) > RoutingSettings{}.token_threshold);
}

TEST_CASE("role_is_coding and denser token estimate") {
    CHECK(role_is_coding("coding"));
    CHECK(role_is_coding("Coding"));
    CHECK(role_is_coding("CODING"));
    CHECK_FALSE(role_is_coding(""));
    CHECK_FALSE(role_is_coding("chat"));
    CHECK_FALSE(role_is_coding("code"));

    CHECK(chars_per_token_for_role("coding") == doctest::Approx(kEstimatedCharsPerTokenCoding));
    CHECK(chars_per_token_for_role("") == doctest::Approx(kEstimatedCharsPerToken));

    // Same char count → more estimated tokens for coding (trims earlier).
    CHECK(estimate_tokens_from_chars(3500, kEstimatedCharsPerTokenCoding) == 1000);
    CHECK(estimate_tokens_from_chars(3500) == 700);
    CHECK(estimate_tokens_from_chars(3500, kEstimatedCharsPerTokenCoding) >
          estimate_tokens_from_chars(3500));
}

TEST_CASE("routing: gguf disables routing") {
    RoutingSettings s;
    s.mode = RoutingMode::Auto;
    s.cpu_model = "lfm25-350m";
    s.gpu_model = "smollm2-360m-dml-fp16-v2";
    auto d = decide_routing(s, 2000, true, true);
    CHECK(d.active_model == "lfm25-350m");
    CHECK_FALSE(d.use_gpu);
}

TEST_CASE("capability gates") {
    CHECK(routing_allowed_for_kind(L"ort-genai"));
    CHECK_FALSE(routing_allowed_for_kind(L"gguf"));
    CHECK(kv_reuse_allowed_for_kind(L"ort-genai"));
    // GGUF now supports KV reuse (persistent llama_context in LlamaSession).
    CHECK(kv_reuse_allowed_for_kind(L"gguf"));
}

TEST_CASE("kv reuse: dml ep disabled") {
    CHECK(kv_reuse_supported_for_model("smollm2-360m-cpu-int4"));
    CHECK_FALSE(kv_reuse_supported_for_model("smollm2-360m-dml-fp16"));
    CHECK_FALSE(kv_reuse_supported_for_model("smollm2-360m-dml-fp16-v2"));
}

TEST_CASE("model_is_dml: gates the load warm-up (#130)") {
    CHECK(model_is_dml("smollm2-360m-dml-fp16-v2"));
    CHECK_FALSE(model_is_dml("smollm2-360m-cpu-int4"));
    CHECK_FALSE(model_is_dml("lfm25-350m"));
    // Single home for the substring check: must agree with KV-reuse gating.
    CHECK(model_is_dml("smollm2-360m-dml-fp16") !=
          kv_reuse_supported_for_model("smollm2-360m-dml-fp16"));
}