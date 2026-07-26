// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// xllama::Session — persistent model session for multi-turn applications.
// Keeps the model and tokenizer loaded between generate() calls,
// eliminating the ~1-2s per-call reload overhead of run_inference().
#pragma once

#include "xllama/inference_params.h"

#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace xllama {

// Which inference backend a session uses. Auto inspects the model identifier
// (suffix or resolved on-disk layout: *.gguf -> LlamaCpp, else OrtGenAI).
// Explicit values are honored when a build links both backends; single-backend
// builds ignore this field.
enum class Backend { Auto, OrtGenAI, LlamaCpp };

struct SessionParams {
    std::string model_path;   // same semantics as InferenceParams::model_path
    int n_ctx = kDefaultNCtx; // one home, inference_params.h (#171)
    int n_threads = 0;        // 0 = auto
    int n_batch = 0;          // llama.cpp only; 0 = default (2048). Logical prefill batch.
    int n_ubatch = 0;         // llama.cpp only; 0 = default (512). Physical prefill chunk.
    Backend backend = Backend::Auto;
    int n_gpu_layers = 0; // llama.cpp only; 0 = CPU (Xbox has no ggml GPU backend)

    // Optional GGUF LoRA adapter (llama.cpp only). Empty = base model only.
    // Loaded via llama_adapter_lora_init + llama_set_adapters_lora (inference-time;
    // not training). Scale defaults to 1.0. ORT GenAI sessions ignore this field.
    std::string lora_path;
    float lora_scale = 1.0f;

    // #171: quantize the llama.cpp KV cache to q8_0 (halves its footprint) and
    // force flash attention, which quantized V requires — the pin throws at
    // context creation otherwise. On any context-creation failure the session
    // retries with default cache types, so an architecture without flash
    // attention support degrades to F16 KV instead of failing to load.
    // Default off until the on-console measurement decides (#171).
    bool kv_q8 = false;

    // #130: run a throwaway generate at load for DirectML models, paying the
    // one-time per-process cost (§5e: cold→warm prefill 1.64-1.72×) inside the
    // "loading model" phase instead of on the user's first turn. Ignored for
    // CPU models (§5e control: 1.00×) and by the llama.cpp backend.
    bool dml_warmup = true;
};

struct GenerateParams {
    std::string prompt;
    int n_predict = 96;
    // Shared with InferenceParams via xllama/sampling.h — see #125. The two
    // surfaces ran different samplers until these were made one source.
    float temperature = sampling_defaults::kTemperature;
    float top_p = sampling_defaults::kTopP;
    int top_k = sampling_defaults::kTopK;
    float repetition_penalty = sampling_defaults::kRepetitionPenalty;
    uint32_t seed = sampling_defaults::kSeed;

    SamplingConfig sampling() const {
        return SamplingConfig{temperature, top_p, top_k, repetition_penalty, seed, false};
    }

    // Stop sequences: checked as substrings of accumulated output.
    // Generation stops and the matching sequence is stripped.
    std::vector<std::string> stop_sequences;

    // Continuous decoding / KV-cache reuse. Honored by BOTH backends now: the
    // ORT path (persistent generator) and the llama.cpp path (persistent
    // llama_context, KV cache retained across turns — turn-2 prefill 4.07×).
    //   reuse_kv = false           → legacy stateless turn: a fresh generator is
    //                                created and destroyed; `prompt` is the full
    //                                context. Proven default.
    //   reuse_kv = true,  reset_kv = false → continuation turn: `prompt` is only
    //                                the NEW turn's tokens; they are appended to
    //                                the generator kept alive from the previous
    //                                turn, so the per-turn prefill covers just the
    //                                delta (the multi-turn TTFT win).
    //   reuse_kv = true,  reset_kv = true  → start/refresh the persistent
    //                                generator (new conversation, context
    //                                eviction, or sampling change); `prompt` is
    //                                the full context. Subsequent turns pass
    //                                reset_kv = false to reuse it.
    // On any failure of a continuation turn the session drops its chat state and
    // reports !success so the caller can retry with reset_kv + the full prompt.
    // Exception (#169): when the session supports context shift
    // (can_context_shift()), a continuation turn that would overflow n_ctx
    // evicts the oldest tokens past n_keep from the resident KV instead of
    // failing, so long chats stay in the reuse regime past the token budget.
    bool reuse_kv = false;
    bool reset_kv = false;

    // #169: number of leading prompt tokens pinned across context shifts —
    // the system prompt (plus BOS), which must survive any eviction. 0 = no
    // pinned head. Callers compute it once per conversation via
    // count_tokens() on the system-only render. Ignored by sessions that
    // cannot shift (ORT, and any arch where llama_memory_can_shift is
    // false — those keep the fail-fast path). Hybrid caches accept the
    // front-drop erase (unlike #170a's tail rewind): the recurrent state
    // holds no cells in the evicted range, only its absorbed history —
    // an approximation the quality gates measure.
    int n_keep = 0;

    // on_token receives a view into a per-iteration buffer: copy it before
    // the callback returns.
    std::function<void(std::string_view)> on_token;
    std::function<void(const std::string&)> on_status;
    std::atomic<bool>* abort_flag = nullptr;
};

// Session owns the loaded model and tokenizer.
// Thread safety: generate() must not be called concurrently.
struct Session {
    // Create a session by loading the model once.
    // On failure returns nullptr and sets *err.
    static std::unique_ptr<Session> create(const SessionParams& params, std::string* err = nullptr);

    virtual InferenceResult generate(const GenerateParams& params) = 0;

    // Token count for routing/heuristics (encode-only; no generation).
    virtual int count_tokens(const std::string& prompt) = 0;

    // #169: whether a continuation turn that would overflow n_ctx evicts the
    // oldest tokens (RoPE shift) instead of failing. False for ORT and for
    // llama archs whose cache cannot shift (llama_memory_can_shift). Callers
    // use it to keep the KV-reuse path for long chats instead of falling back
    // to trimmed full prefills.
    virtual bool can_context_shift() const {
        return false;
    }

    virtual ~Session() = default;
};

} // namespace xllama
