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
    Backend backend = Backend::Auto;
    int n_gpu_layers = 0; // llama.cpp only; 0 = CPU (Xbox has no ggml GPU backend)
};

struct GenerateParams {
    std::string prompt;
    int n_predict = 96;
    float temperature = 0.8f;
    float top_p = 0.9f;
    int top_k = 40;
    float repetition_penalty = 1.1f;
    uint32_t seed = 0xFFFFFFFF;

    // Stop sequences: checked as substrings of accumulated output.
    // Generation stops and the matching sequence is stripped.
    std::vector<std::string> stop_sequences;

    // Continuous decoding / KV-cache reuse (ORT path only; the llama.cpp path
    // ignores these and always runs stateless).
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

    std::function<void(const std::string&)> on_token;
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
    virtual ~Session() = default;
};

} // namespace xllama
