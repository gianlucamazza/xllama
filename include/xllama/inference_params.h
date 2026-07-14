// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

#pragma once

#include <atomic>
#include <functional>
#include <string>
#include <vector>

namespace xllama {

// ---------------------------------------------------------------------------
// Inference configuration
// ---------------------------------------------------------------------------
struct InferenceParams {
    std::string model_path; // Linux: absolute path; UWP: filename in LocalFolder
    std::string prompt;
    int n_predict = 128;
    int n_ctx = 2048;
    int n_threads = 0; // 0 = auto-detect
    // llama.cpp prefill batching (GGUF path only; 0 = llama.cpp default, i.e.
    // n_batch 2048 / n_ubatch 512). n_ubatch is the physical compute chunk that
    // sets prefill throughput on the Zen 2 CPU — the sweep knob for TTFT tuning.
    int n_batch = 0;
    int n_ubatch = 0;
    float temperature = 0.8f;
    uint32_t seed = 0xFFFFFFFF; // 0xFFFFFFFF = LLAMA_DEFAULT_SEED

    // Stop strings: generation ends (and the match is trimmed from output_text)
    // when the accumulated output ends with any of these. Empty = stop on EOG /
    // n_predict only. Used by the CLI --chat mode and the GGUF bench.
    std::vector<std::string> stop_sequences;

    // CLI --chat: wrap `prompt` with the model's chat template before inference.
    bool chat_template = false;

    // UI callbacks (optional). Called from the inference thread — must marshal
    // to the UI thread before touching XAML controls.
    std::function<void(const std::string&)> on_status; // e.g. "loading model"
    std::function<void(const std::string&)> on_token;  // per-token text piece

    // Set to true from the UI thread to request early termination.
    std::atomic<bool>* abort_flag = nullptr;
};

struct InferenceResult {
    bool success = false;
    double t_load_ms = 0.0;
    double t_p_eval_ms = 0.0;
    double t_eval_ms = 0.0;
    int n_p_eval = 0;
    int n_eval = 0;
    bool ended_with_stop = false; // true = stopped on a stop sequence; false = n_predict/EOS cap.
                                  // Lets a KV-reuse caller know whether the closing <|im_end|>
                                  // is already in the KV cache when building the next turn's delta.
    size_t peak_ws_mb = 0;
    size_t gpu_mem_mb = 0;    // per-process GPU CurrentUsage after model load (0 = N/A)
    size_t gpu_budget_mb = 0; // OS-granted GPU budget for this process (0 = N/A)
    std::string output_text;
    std::string error_msg;
};

} // namespace xllama
