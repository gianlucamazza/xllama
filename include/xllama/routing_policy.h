// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// Per-workload EP routing policy (ORT GenAI only). GGUF models disable routing
// and KV reuse at the UI layer; this header encodes the token threshold only.
// Also owns the prompt-budget helpers used by catalogue roles (e.g. coding).
//
// ## SDK configurability (Phase 3)
//
// All routing gates are now callable through a `RoutingPolicy` object whose
// callbacks can be overridden. The free functions below remain the default
// wrappers and require zero changes from the app.
//
// SDK usage:
//   xllama::RoutingPolicy policy;
//   policy.allow_kind = [](const std::wstring& k) { return k != L"gguf"; };
//   policy.reuse_kv_for_model = [](const std::string& m) { return m.find("dml") == std::string::npos; };
//   auto decision = policy.decide(s, n_tok, base_is_gguf, gpu_available);
#pragma once

#include "xllama/inference_params.h" // kDefaultNCtx

#include <cstddef>
#include <functional>
#include <memory>
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

// Coding workloads (source, diffs, stack traces) tokenize denser / less
// predictably than prose. A lower chars-per-token overestimates tokens and
// trims earlier — same safety direction as the prose constant.
inline constexpr double kEstimatedCharsPerTokenCoding = 3.5;

// Token budget for the prompt itself, sized against kDefaultNCtx
// (inference_params.h) with ~250 tokens left for generation —
// tests/test_routing_policy.cpp pins the relation (#171).
inline constexpr int kMaxPromptTokens = 1800;

// Room reserved for the model reply. Owned by xllama::fit_prompt
// (prompt_budget.h), which enforces it EXACTLY with the model's tokenizer; this
// header only publishes the floor so both agree on one number.
//
// It deliberately does NOT enter the estimate ceiling below. Those are two
// different constraints with two different owners, and conflating them killed a
// feature: charging the reply's reserve to the routing ceiling put the ceiling
// (1536 at the shipping n_predict of 512) UNDER token_threshold (1550), so auto
// GPU routing became unreachable for every default install — #133 all over again.
inline constexpr int kReservedGenerationTokens = 250;

// Catalogue n_ctx bounds. 0 / omit → kDefaultNCtx. Above the max is clamped so
// a bad override cannot OOM the console KV pool.
inline constexpr int kMinSessionNCtx = 512;
inline constexpr int kMaxSessionNCtx = 8192;

// Resolve a catalogue (or CLI) n_ctx request: 0 means shipping default.
inline constexpr int resolve_n_ctx(int requested) {
    if (requested <= 0)
        return kDefaultNCtx;
    if (requested < kMinSessionNCtx)
        return kMinSessionNCtx;
    if (requested > kMaxSessionNCtx)
        return kMaxSessionNCtx;
    return requested;
}

// Estimate ceiling for a session opened at |n_ctx|: the bound the CONTEXT puts on
// a prompt, which is what has to stay coherent with token_threshold (#133). Not
// the reply's budget — that is fit_prompt's, applied exactly and later, and it
// must not shrink this number (see kReservedGenerationTokens). The shipping
// default keeps the historical kMaxPromptTokens constant (not n-250) so the #171
// pin cannot drift by arithmetic alone when kDefaultNCtx is retuned with it.
inline constexpr int max_prompt_tokens_for_n_ctx(int n_ctx) {
    const int n = resolve_n_ctx(n_ctx);
    if (n == kDefaultNCtx)
        return kMaxPromptTokens;
    const int budget = n - kReservedGenerationTokens;
    return budget < 256 ? 256 : budget;
}

// Catalogue `role` field (ASCII, case-insensitive). Only "coding" is special
// today: denser token estimate in the UI trimmer, and the LAN API empty-system
// fill (kCodingSystemPrompt). The Settings system-prompt box is never rewritten
// from this flag — that would be string-magic debt.
inline bool role_is_coding(std::string_view role) {
    if (role.size() != 6)
        return false;
    constexpr char kCoding[] = "coding";
    for (size_t i = 0; i < 6; ++i) {
        char c = role[i];
        if (c >= 'A' && c <= 'Z')
            c = static_cast<char>(c - 'A' + 'a');
        if (c != kCoding[i])
            return false;
    }
    return true;
}

inline double chars_per_token_for_role(std::string_view role) {
    return role_is_coding(role) ? kEstimatedCharsPerTokenCoding : kEstimatedCharsPerToken;
}

inline constexpr int estimate_tokens_from_chars(std::size_t chars, double chars_per_token) {
    if (chars_per_token <= 0.0)
        chars_per_token = kEstimatedCharsPerToken;
    return static_cast<int>(static_cast<double>(chars) / chars_per_token);
}

inline constexpr int estimate_tokens_from_chars(std::size_t chars) {
    return estimate_tokens_from_chars(chars, kEstimatedCharsPerToken);
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

// ---------------------------------------------------------------------------
// Internal implementations (called by RoutingPolicy defaults, NOT by wrappers)
// ---------------------------------------------------------------------------
// These must be defined BEFORE RoutingPolicy so lambdas can see them.

// DirectML-routed models are identified by catalogue naming convention
// ("-dml-" in the model id). Single home for the substring check: KV-reuse
// gating and the DML load warm-up both key on it.
inline constexpr bool model_is_dml(std::string_view active_model) {
    return active_model.find("dml") != std::string_view::npos;
}

inline bool routing_allowed_for_kind_impl(const std::wstring& kind) {
    return kind != L"gguf";
}

inline bool kv_reuse_allowed_for_kind_impl(const std::wstring& kind) {
    // GGUF (llama.cpp) now supports KV reuse via a persistent llama_context
    // (LlamaSession), so it is allowed alongside ORT-GenAI. (Routing stays
    // ORT-only — the llama.cpp UWP build is CPU-only, no GPU model to route to.)
    (void)kind;
    return true;
}

// ---------------------------------------------------------------------------
// RoutingPolicy — SDK-configurable policy object (Phase 3)
// ---------------------------------------------------------------------------
// A `RoutingPolicy` bundles every routing gate behind `std::function` callbacks
// so an SDK user can override any gate without patching this header. The free
// functions below (`dml_text_model_ok`, `decide_routing`, …) remain the default
// wrappers and require zero changes from the app or the UWP MainPage.
//
// Default construction uses the current shipping defaults (the free functions).
// SDK users create an instance, replace callbacks, and call the member variants.
struct RoutingPolicy {
    // GPU allowlist — #91 postmortem: only parity-validated DML assets may
    // route to GPU. (Inlined to avoid circular init through default_policy().)
    std::function<bool(std::string_view gpu_model)> dml_text_model_ok_fn =
        [](std::string_view m) { return m == "smollm2-360m-dml-fp16-v2"; };

    // Decide which model to load for the first turn. (Inlined to avoid circular
    // init through default_policy().)
    std::function<RoutingDecision(const RoutingSettings&, int, bool, bool)> decide_fn =
        [](const RoutingSettings& s, int n_tok, bool base_is_gguf, bool gpu_available) {
            RoutingDecision d;
            d.token_count = n_tok;
            // Inline dml_text_model_ok check to avoid circular init.
            const bool gpu_ok = s.gpu_model == "smollm2-360m-dml-fp16-v2";
            if (base_is_gguf || s.mode == RoutingMode::CpuOnly || !gpu_ok) {
                d.active_model = s.cpu_model;
                d.use_gpu = false;
                return d;
            }
            if (s.mode == RoutingMode::GpuOnly) {
                d.active_model = gpu_available ? s.gpu_model : s.cpu_model;
                d.use_gpu = gpu_available;
                return d;
            }
            const bool use_gpu = gpu_available && n_tok > s.token_threshold;
            d.use_gpu = use_gpu;
            d.active_model = use_gpu ? s.gpu_model : s.cpu_model;
            return d;
        };

    // Feature gates by catalogue kind.
    std::function<bool(const std::wstring& kind)> allow_kind_fn =
        [](const std::wstring& k) { return routing_allowed_for_kind_impl(k); };

    // KV-reuse gate by catalogue kind.
    std::function<bool(const std::wstring& kind)> reuse_kv_kind_fn =
        [](const std::wstring& k) { return kv_reuse_allowed_for_kind_impl(k); };

    // KV-reuse gate by model name (DML models reject continuous decoding).
    std::function<bool(std::string_view model)> reuse_kv_model_fn =
        [](std::string_view m) { return !model_is_dml(m); };

    // Convenience wrappers that dispatch through the callbacks.
    inline bool dml_text_model_ok(std::string_view gpu_model) const {
        return dml_text_model_ok_fn(gpu_model);
    }
    inline RoutingDecision decide(const RoutingSettings& s, int n_tok, bool base_is_gguf,
                                  bool gpu_available) const {
        return decide_fn(s, n_tok, base_is_gguf, gpu_available);
    }
    inline bool routing_allowed_for_kind(const std::wstring& kind) const {
        return allow_kind_fn(kind);
    }
    inline bool kv_reuse_allowed_for_kind(const std::wstring& kind) const {
        return reuse_kv_kind_fn(kind);
    }
    inline bool kv_reuse_supported_for_model(std::string_view model) const {
        return reuse_kv_model_fn(model);
    }
};

// Default global policy instance — the free functions below are thin wrappers
// around this same instance, so overriding the callbacks via the default
// instance affects every free-function call (same single-instance pattern as
// SessionHub). To avoid global-initialization order issues the instance is
// created lazily inside this function.
inline const RoutingPolicy& default_policy() {
    static const RoutingPolicy p;
    return p;
}

// -- Free-function wrappers around default_policy() --
// These remain backward-compatible. To customise, replace callbacks on the
// instance returned by `default_policy()` or construct a local `RoutingPolicy`.

inline bool dml_text_model_ok(std::string_view gpu_model) {
    return default_policy().dml_text_model_ok(gpu_model);
}

inline RoutingDecision decide_routing(const RoutingSettings& s, int n_tok, bool base_is_gguf,
                                      bool gpu_available) {
    return default_policy().decide(s, n_tok, base_is_gguf, gpu_available);
}

inline bool routing_allowed_for_kind(const std::wstring& kind) {
    return default_policy().routing_allowed_for_kind(kind);
}

inline bool kv_reuse_allowed_for_kind(const std::wstring& kind) {
    return default_policy().kv_reuse_allowed_for_kind(kind);
}

// KV-reuse gate by model name (DML models reject continuous decoding).
//
// SDK override: replace `reuse_kv_model_fn` on your `RoutingPolicy` instance.
inline bool kv_reuse_supported_for_model(const std::string& active_model) {
    return !model_is_dml(active_model);
}

} // namespace xllama