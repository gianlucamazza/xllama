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

    // Deterministic decode: pick argmax instead of sampling. Prerequisite for
    // cross-backend logit parity (llama.cpp vs ORT must agree token-for-token).
    bool greedy = false;

    // Logit-parity harness: when non-empty, dump the last prefill-token logit
    // vector (float32, vocab_size values) to this path plus a "<path>.json"
    // sidecar with metadata. Empty = disabled (normal inference).
    std::string dump_logits_path;

    // Stop strings: generation ends (and the match is trimmed from output_text)
    // when the accumulated output ends with any of these. Empty = stop on EOG /
    // n_predict only. Used by the CLI --chat mode and the GGUF bench.
    std::vector<std::string> stop_sequences;

    // CLI --chat: wrap `prompt` with the model's chat template before inference.
    bool chat_template = false;

    // Optional GGUF LoRA adapter path (llama.cpp path only). Empty = off.
    std::string lora_path;
    float lora_scale = 1.0f;

    // CLI --membw: run the CPU memory-bandwidth micro-bench and exit (no model
    // load). Model/prompt are not required in this mode.
    bool run_membw = false;

    // CLI --validate-train-job <path.json>: parse + validate a TrainingJob and
    // exit (no model). Part of the training pillar (exploration).
    bool run_validate_train_job = false;
    // CLI --train-job <path.json>: shell out to the host training runner.
    bool run_train_job = false;
    // CLI --training-capabilities: print RE-backed capability matrix and exit.
    bool run_training_capabilities = false;
    std::string train_job_path;

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
    // #130: the max_length actually requested of the engine. On DirectML this is
    // the variable that controls prefill throughput — a 1289-token prompt runs at
    // 130 tok/s at max_length 1801 and 611 tok/s at 2048, unchanged otherwise —
    // so a bench row that omits it cannot be interpreted, the same way a row
    // without n_prompt_tok could not (#128). Derived, not independently settable:
    // min(n_ctx, n_prompt_tok + n_predict) on the stateless path. 0 = N/A (GGUF).
    int max_length = 0;
    size_t peak_ws_mb = 0;
    size_t gpu_mem_mb = 0;    // per-process GPU CurrentUsage after model load (0 = N/A)
    size_t gpu_budget_mb = 0; // OS-granted GPU budget for this process (0 = N/A)
    std::string output_text;
    std::string error_msg;
};

} // namespace xllama
