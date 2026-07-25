// Copyright (c) 2024 Venere Labs
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
    std::string model_path; // same semantics as InferenceParams::model_path
    int n_ctx = 2048;
    int n_threads = 0; // 0 = auto
    int n_batch = 0;   // llama.cpp only; 0 = default (2048). Logical prefill batch.
    int n_ubatch = 0;  // llama.cpp only; 0 = default (512). Physical prefill chunk.
    Backend backend = Backend::Auto;
    int n_gpu_layers = 0; // llama.cpp only; 0 = CPU (Xbox has no ggml GPU backend)

    // Optional GGUF LoRA adapter (llama.cpp only). Empty = base model only.
    // Loaded via llama_adapter_lora_init + llama_set_adapters_lora (inference-time;
    // not training). Scale defaults to 1.0. ORT GenAI sessions ignore this field.
    std::string lora_path;
    float lora_scale = 1.0f;
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
    bool reuse_kv = false;
    bool reset_kv = false;

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

    virtual ~Session() = default;
};

} // namespace xllama
