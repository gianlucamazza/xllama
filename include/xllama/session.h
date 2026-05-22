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

struct SessionParams {
    std::string model_path; // same semantics as InferenceParams::model_path
    int n_ctx = 2048;
    int n_threads = 0; // 0 = auto
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
