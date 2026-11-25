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
//   policy.reuse_kv_for_model = [](const std::string& m) { return m.find("dml") ==
//   std::string::npos; }; auto decision = policy.decide(s, n_tok, base_is_gguf, gpu_available);
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
// Auto routing sees the prompt bounded by the context ceiling
// (routing_prompt_tokens), because fit_prompt drops the oldest turns past it
// before prefill; that ceiling is the hard limit. If token_threshold ever rises
// above it, decide_routing() can never see a count above the threshold and auto
// GPU routing becomes unreachable. That is #133: the 600 -> 1550 retune in #129
// crossed the ceiling and silently disabled the feature, because nothing here
// related the two numbers. assert in tests/test_routing_policy.cpp now does.
//
// Routing has no tokenizer when no CPU session is resident, so it estimates
// from character count (routing_prompt_tokens). Nothing that decides what the
// model sees uses this: fit_prompt counts tokens. Measured 2026-07-21 on the console with the
// SmolLM2 tokenizer: 7100 chars of English prose -> 1329 real tokens = 5.34 chars/token
// (bench/prompts/standard-512.txt gives 4.84 with its ChatML markup counted).
// The previous divisor of 4 overestimated tokens by ~30%, so the estimate trimmer
// this replaced cut at ~1348 real tokens while its comment claimed 1800 — that
// gap is what put the ceiling underneath the threshold.
//
// 5.0 is deliberately below the measured 5.34: underestimating chars-per-token
// means overestimating tokens, so routing and the KV-reuse overflow guess err
// toward the long-prompt path. Erring this way costs a routing choice or a
// retry, not correctness. Non-Latin scripts tokenize far denser and this
// estimate does not model them.
inline constexpr double kEstimatedCharsPerToken = 5.0;

// Coding workloads (source, diffs, stack traces) tokenize denser / less
// predictably than prose. A lower chars-per-token overestimates tokens — same
// direction as the prose constant.
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

// Catalogue n_ctx: 0 / omit means kDefaultNCtx, and nothing opens below
// kMinSessionNCtx. There is no fixed maximum here (#570). Each backend applies
// its own bound when the session opens:
//   - GGUF: bound_gguf_n_ctx caps the request at the model's trained context.
//     The memory admission (memory_admission.h) then sizes the real KV and
//     recurrent state at that n_ctx and refuses a load that would not fit.
//   - ORT GenAI: bound_ort_n_ctx keeps kMaxOrtSessionNCtx, since ORT sessions
//     have no memory admission and this cap is all that keeps a bad override
//     from exhausting the console KV pool.
inline constexpr int kMinSessionNCtx = 512;
inline constexpr int kMaxOrtSessionNCtx = 8192;

// Resolve a catalogue (or CLI) n_ctx request: 0 means shipping default.
inline constexpr int resolve_n_ctx(int requested) {
    if (requested <= 0)
        return kDefaultNCtx;
    if (requested < kMinSessionNCtx)
        return kMinSessionNCtx;
    return requested;
}

// A GGUF session's context: the resolved request, capped at the model's
// trained context (GGUF `<arch>.context_length`). Above it, llama.cpp only
// warns and the model runs past what it was trained on. If n_ctx_train is
// unknown (<= 0) the request stands, because the admission still bounds memory.
// A trained context below kMinSessionNCtx does not lower the session floor.
inline constexpr int bound_gguf_n_ctx(int requested, int n_ctx_train) {
    const int n = resolve_n_ctx(requested);
    if (n_ctx_train <= 0 || n <= n_ctx_train)
        return n;
    return n_ctx_train < kMinSessionNCtx ? kMinSessionNCtx : n_ctx_train;
}

// An ORT GenAI session's context (max_length).
inline constexpr int bound_ort_n_ctx(int requested) {
    const int n = resolve_n_ctx(requested);
    return n > kMaxOrtSessionNCtx ? kMaxOrtSessionNCtx : n;
}

// The n_ctx a catalogue entry asks its session for, before the model file is
// read. Used by the resolver and by the surfaces that plan a prompt before
// loading. A GGUF session can still open smaller (bound_gguf_n_ctx); after the
// load, Session::context_length() is the context that holds.
inline constexpr int catalogue_n_ctx(int requested, bool gguf) {
    return gguf ? resolve_n_ctx(requested) : bound_ort_n_ctx(requested);
}

// GGUF layers on the d3d12 backend for a session (docs/gguf-gpu-decode.md, D3).
// The catalogue entry carries the per-model default (`gpu_layers`, 0 = CPU),
// set only for models whose D2-r2 gate passed at their catalogue n_ctx. The
// operator knob LocalState\gguf_gpu_layers.txt, when present, always wins — 0
// included, so the CPU can be forced without a catalogue change.
inline constexpr int kGgufGpuLayersKnobAbsent = -1;

inline constexpr int resolve_gguf_gpu_layers(int catalogue_layers, int knob) {
    if (knob >= 0)
        return knob;
    return catalogue_layers > 0 ? catalogue_layers : 0;
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

// Catalogue role "embedding" marks models that must not appear in the chat
// picker and must not be sent to Session::generate (POST /v1/chat/completions
// returns 400). Exact match, case-insensitive.
inline bool role_is_embedding(std::string_view role) {
    if (role.size() != 9)
        return false;
    constexpr char kEmbedding[] = "embedding";
    for (size_t i = 0; i < 9; ++i) {
        char c = role[i];
        if (c >= 'A' && c <= 'Z')
            c = static_cast<char>(c - 'A' + 'a');
        if (c != kEmbedding[i])
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

// Prompt size routing compares with token_threshold. |counted| is the whole
// history's size (an exact count when the CPU session is resident, else the
// estimate); the context ceiling bounds it because fit_prompt will drop the
// oldest turns past that point, and routing must see what will be prefilled,
// not what was typed (#133). Routing only: what the user receives is decided
// by fit_prompt, in tokens.
inline constexpr int routing_prompt_tokens(int counted, int n_ctx) {
    const int ceiling = max_prompt_tokens_for_n_ctx(n_ctx);
    return counted < ceiling ? counted : ceiling;
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
    std::function<bool(std::string_view gpu_model)> dml_text_model_ok_fn = [](std::string_view m) {
        return m == "smollm2-360m-dml-fp16-v2";
    };

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
    std::function<bool(const std::wstring& kind)> allow_kind_fn = [](const std::wstring& k) {
        return routing_allowed_for_kind_impl(k);
    };

    // KV-reuse gate by catalogue kind.
    std::function<bool(const std::wstring& kind)> reuse_kv_kind_fn = [](const std::wstring& k) {
        return kv_reuse_allowed_for_kind_impl(k);
    };

    // KV-reuse gate by model name (DML models reject continuous decoding).
    std::function<bool(std::string_view model)> reuse_kv_model_fn = [](std::string_view m) {
        return !model_is_dml(m);
    };

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

// GPU availability for decide_routing. The caller must not pass the loose
// "any genai_config.json / *.gguf exists" probe: a partial weight tree or an
// in-flight model write is not a verified DML asset. `expected_files_verified`
// is a size-only `weight_pins_verified` of the weight and data pins (licence
// and notice rows do not participate) AND the LocalState `.complete` marker,
// which attests every pin was hashed at download/promotion. The process-wide
// writer must be idle (`busy()` is any write in the process, not just this
// model).
enum class GpuRouteReadiness {
    Ready = 0,
    PartialOrUnverified = 1,
    WriteInProgress = 2,
};

inline GpuRouteReadiness gpu_route_readiness(bool expected_files_verified, bool write_in_progress) {
    if (write_in_progress)
        return GpuRouteReadiness::WriteInProgress;
    if (!expected_files_verified)
        return GpuRouteReadiness::PartialOrUnverified;
    return GpuRouteReadiness::Ready;
}

inline bool gpu_ready_for_routing(bool expected_files_verified, bool write_in_progress) {
    return gpu_route_readiness(expected_files_verified, write_in_progress) ==
           GpuRouteReadiness::Ready;
}

inline const char* gpu_route_fallback_log(GpuRouteReadiness r) {
    switch (r) {
    case GpuRouteReadiness::WriteInProgress:
        return "[xllama] routing: gpu model write in progress; falling back to CPU\n";
    case GpuRouteReadiness::PartialOrUnverified:
        return "[xllama] routing: gpu model not fully downloaded/verified; falling back to CPU\n";
    case GpuRouteReadiness::Ready:
        return "";
    }
    return "[xllama] routing: gpu unavailable; falling back to CPU\n";
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