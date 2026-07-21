// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT
// Per-workload EP routing policy (ORT GenAI only). GGUF models disable routing
// and KV reuse at the UI layer; this header encodes the token threshold only.
#pragma once

#include <cstddef>
#include <string>
#include <string_view>

namespace xllama {

// ---------------------------------------------------------------------------
// Prompt budget
// ---------------------------------------------------------------------------
// The context trimmer (MainPageController::BuildPrompt) and auto routing both
// bound the same quantity — how many tokens a turn may carry — but the trimmer
// runs first, so its ceiling is the hard limit. If token_threshold ever rises
// above it, decide_routing() can never see a count above the threshold and auto
// GPU routing becomes unreachable. That is #133: the 600 -> 1550 retune in #129
// crossed the ceiling and silently disabled the feature, because nothing here
// related the two numbers. assert in tests/test_routing_policy.cpp now does.
//
// The trimmer has no tokenizer available at the point it runs, so it estimates
// from character count. Measured 2026-07-21 on the console with the SmolLM2
// tokenizer: 7100 chars of English prose -> 1329 real tokens = 5.34 chars/token
// (bench/prompts/standard-512.txt gives 4.84 with its ChatML markup counted).
// The previous divisor of 4 overestimated tokens by ~30%, so the trimmer cut at
// ~1348 real tokens while its comment claimed 1800 — that gap is what put the
// ceiling underneath the threshold.
//
// 5.0 is deliberately below the measured 5.34: underestimating chars-per-token
// means overestimating tokens, which trims early rather than late. Overflow is
// caught downstream by the max_length clamp in run_inference, so erring this way
// costs context, not correctness. Non-Latin scripts tokenize far denser and this
// estimate does not model them.
inline constexpr double kEstimatedCharsPerToken = 5.0;

// Token budget for the prompt itself, sized against n_ctx 2048 with ~250 tokens
// left for generation.
inline constexpr int kMaxPromptTokens = 1800;

inline constexpr int estimate_tokens_from_chars(std::size_t chars) {
    return static_cast<int>(static_cast<double>(chars) / kEstimatedCharsPerToken);
}

enum class RoutingMode {
    CpuOnly = 0,
    GpuOnly = 1,
    Auto = 2,
};

struct RoutingSettings {
    RoutingMode mode = RoutingMode::CpuOnly;
    // Measured 2026-07-21 on the shipping -v2 asset, Series S, n_ctx 2048
    // (bench/results/phase12-dml-crossover.csv, scripts/bench-prompt-sweep.sh).
    // The previous 600 was a midpoint interpolated between two sample points
    // (285 and ~1050 tok) taken on the pre-#91 asset that dml_text_model_ok()
    // now excludes — it assumed a smooth crossover that does not exist.
    //
    // What the sweep found, every point reproduced:
    //   <=  500 tok  CPU wins outright.
    //   557-1098     GPU wins only for answers shorter than ~120-290 tokens;
    //                the app generates up to m_n_predict (512), so CPU usually wins.
    //   1100-1500    GPU is PATHOLOGICALLY slow — prefill 3.8-10.8 s against the
    //                CPU's monotone 5.2-8.0 s (1289 tok: 10.4 s, reproduced twice).
    //                Mechanism unexplained; needs scripts/profile-dml-run.sh.
    //   >= ~1550     GPU always wins: break-even is ~975 generated tokens while
    //                the context leaves room for at most 474, so the prefill win
    //                cannot be amortised away.
    //
    // 1550 keeps GPU routing to the only band where it is unconditionally better
    // and steers clear of the pathological region — but it must also stay below
    // kMaxPromptTokens, or the trimmer cuts the turn down before routing sees it
    // (#133). Re-measure when the asset, the model, or n_ctx changes; this number
    // is not a general law.
    //
    // #130 caveat: the "1100-1500" band above is stated in prompt length, which
    // is not the variable the code controls. Recast against max_length =
    // min(n_ctx, n_prompt + n_predict), every fast point is one where that clamp
    // saturates and every slow one is where it does not. The band is in
    // max_length, not prompt tokens, so this whole calibration is provisional.
    int token_threshold = 1550;
    std::string cpu_model;
    std::string gpu_model;
};

struct RoutingDecision {
    std::string active_model;
    bool use_gpu = false;
    int token_count = 0;
};

// #91 postmortem (2026-07-19, docs/dml-rmsnorm-fix-runbook.md): the DML
// (Skip)SimplifiedLayerNormalization kernel computes wrong results on the
// Series S driver, so any text asset carrying the fused RMSNorm contrib nodes
// produces garbage logits on the GPU (NMSE ~1 — this is what #91/#94 chased as
// an "attention" bug). Assets with those nodes decomposed into primitives
// (decompose_attention.py --skip-attention --also-skipln) pass the on-console
// parity gate (scripts/validate-logit-parity.sh) with the shipping DLLs, so
// GPU text routing is allowed for them only. Old broken copies of
// "smollm2-360m-dml-fp16" may survive in LocalState from ≤1.1.x installs —
// that name must never come back to this allowlist; the fixed asset ships as
// "-v2". Diffusion (plain ORT) was never affected.
inline constexpr bool dml_text_model_ok(std::string_view gpu_model) {
    return gpu_model == "smollm2-360m-dml-fp16-v2";
}

// Decide which model directory to load for the first turn of a conversation.
// |gpu_available| must reflect IsModelProvisioned(gpu_model) — callers gate UX.
inline RoutingDecision decide_routing(const RoutingSettings& s, int n_tok, bool base_is_gguf,
                                      bool gpu_available) {
    RoutingDecision d;
    d.token_count = n_tok;
    if (base_is_gguf || s.mode == RoutingMode::CpuOnly || !dml_text_model_ok(s.gpu_model)) {
        d.active_model = s.cpu_model;
        d.use_gpu = false;
        return d;
    }
    if (s.mode == RoutingMode::GpuOnly) {
        d.active_model = gpu_available ? s.gpu_model : s.cpu_model;
        d.use_gpu = gpu_available;
        return d;
    }
    // Auto
    const bool use_gpu = gpu_available && n_tok > s.token_threshold;
    d.use_gpu = use_gpu;
    d.active_model = use_gpu ? s.gpu_model : s.cpu_model;
    return d;
}

// Feature gates by catalogue kind (mirrors MainPage capability matrix).
inline bool routing_allowed_for_kind(const std::wstring& kind) {
    return kind != L"gguf";
}

inline bool kv_reuse_allowed_for_kind(const std::wstring& kind) {
    // GGUF (llama.cpp) now supports KV reuse via a persistent llama_context
    // (LlamaSession), so it is allowed alongside ORT-GenAI. (Routing stays
    // ORT-only — the llama.cpp UWP build is CPU-only, no GPU model to route to.)
    (void)kind;
    return true;
}

// ORT GenAI continuous decoding (KV reuse) is CPU-only today; DirectML rejects
// AppendTokenSequences on a persistent generator ("Continuous decoding is not
// supported on the selected device type (DirectML)").
inline bool kv_reuse_supported_for_model(const std::string& active_model) {
    return active_model.find("dml") == std::string::npos;
}

} // namespace xllama