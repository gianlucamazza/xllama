// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#pragma once

#include "xllama/sampling.h"

#include <algorithm>
#include <atomic>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace xllama {

/// The shipping context size.
///
/// Every surface that opens a context (chat UI, LAN API, Session default, CLI
/// default) reads this constant. The trimmer budget `kMaxPromptTokens`
/// (routing_policy.h) is sized against it.
inline constexpr int kDefaultNCtx = 2048;

// ---------------------------------------------------------------------------
// Inference configuration
// ---------------------------------------------------------------------------

/// Configuration for a single inference call.
///
/// Used by the CLI, benchmarks, and the `run_inference()` function. Most fields
/// have defaults so callers only set what they want to change.
struct InferenceParams {
    /// Path to model. Linux: absolute path; UWP: filename in LocalFolder.
    std::string model_path;

    /// Input prompt text.
    std::string prompt;

    /// Max tokens to generate (default: 128).
    int n_predict = 128;

    /// Context size (default: kDefaultNCtx = 2048).
    int n_ctx = kDefaultNCtx;

    /// Override for the engine's max_length.
    ///   0  = derive as min(n_ctx, n_prompt + n_predict) (bench default)
    ///  -1  = saturate to n_ctx (what Session::generate ships)
    ///  >0  = explicit, clamped to (n_prompt+1, n_ctx]
    int max_length_override = 0;

    /// Thread count (0 = auto-detect).
    int n_threads = 0;

    /// Which repetition of a repeated bench this run is (0 = not a bench).
    /// Written verbatim into the bench CSV's run_index column.
    int run_index = 0;

    /// llama.cpp prefill batching (GGUF path only; 0 = default).
    int n_batch = 0;
    int n_ubatch = 0;

    /// Quantize KV cache to q8_0 (GGUF path only).
    bool kv_q8 = false;

    /// Sampling defaults from `xllama/sampling.h`.
    float temperature = sampling_defaults::kTemperature;
    float top_p = sampling_defaults::kTopP;
    int top_k = sampling_defaults::kTopK;
    float repetition_penalty = sampling_defaults::kRepetitionPenalty;
    uint32_t seed = sampling_defaults::kSeed;

    /// Deterministic argmax decode (prerequisite for cross-backend logit parity).
    bool greedy = false;

    /// Build a `SamplingConfig` from the sampling fields.
    SamplingConfig sampling() const {
        return SamplingConfig{temperature, top_p, top_k, repetition_penalty, seed, greedy};
    }

    /// System message for chat template (empty = built-in default).
    std::string system_prompt;

    /// Dump last prefill-token logit vector to this path (+ `.json` sidecar).
    std::string dump_logits_path;

    /// Stop strings; generation ends when output ends with any of these.
    std::vector<std::string> stop_sequences;

    /// Wrap `prompt` with the model's chat template before inference.
    bool chat_template = false;

    /// GGUF LoRA adapter path (llama.cpp path only; empty = off).
    std::string lora_path;
    float lora_scale = 1.0f;

    /// Draft-free prompt-lookup speculative decoding (GGUF path only; default OFF).
    bool prompt_lookup = false;

    /// Probe CPU memory bandwidth and exit (no model load).
    bool run_membw = false;

    /// Probe disk (NVMe) read bandwidth and exit (no model load).
    bool run_diskbw = false;

    /// Probe GPU STREAM bandwidth and exit (D3D12; non-Windows reports unavailable).
    bool run_gpubw = false;

    /// Probe Q4_K GEMV density and exit (D3D12 CS; not a backend).
    bool run_gpugemv = false;

    /// Probe heap ceiling and exit (no model load).
    bool run_ramceil = false;

    /// Validate a TrainingJob JSON and exit (no model).
    bool run_validate_train_job = false;

    /// Shell out to the host training runner (from a TrainingJob JSON).
    bool run_train_job = false;

    /// Print training capability matrix and exit.
    bool run_training_capabilities = false;

    /// Path to TrainingJob JSON (for --train-job / --validate-train-job).
    std::string train_job_path;

    /// UI callback: status changes (e.g. "loading model"). Called from inference thread.
    std::function<void(const std::string&)> on_status;

    /// UI callback: per-token text piece. Called from inference thread; copy before return.
    std::function<void(std::string_view)> on_token;

    /// Stream generated pieces to stdout (interactive CLI; off by default).
    bool echo_stdout = false;

    /// Atomic flag set from UI thread to request early termination.
    std::atomic<bool>* abort_flag = nullptr;
};

/// Resolve the max_length for inference.
///
/// @param n_ctx        Session context size.
/// @param n_prompt     Number of prompt tokens.
/// @param n_predict    Max tokens to generate.
/// @param override_v   Override value (<0=saturate to n_ctx, 0=derive, >0=explicit).
/// @return             The effective max_length.
inline int resolve_max_length(int n_ctx, int n_prompt, int n_predict, int override_v) {
    if (override_v < 0)
        return n_ctx;
    if (override_v > 0)
        return std::clamp(override_v, n_prompt + 1, n_ctx);
    return std::min(n_ctx, n_prompt + n_predict);
}

/// Result of an inference call.
struct InferenceResult {
    /// Whether generation succeeded.
    bool success = false;

    /// Model load time (ms).
    double t_load_ms = 0.0;

    /// Prefill time (ms).
    double t_p_eval_ms = 0.0;

    /// Decode time (ms).
    double t_eval_ms = 0.0;

    /// Number of prefill tokens evaluated.
    int n_p_eval = 0;

    /// Number of decode tokens generated.
    int n_eval = 0;

    /// True if stopped on a stop sequence; false if capped by n_predict/EOS.
    bool ended_with_stop = false;

    /// Max length actually requested of the engine.
    /// On DirectML this controls prefill throughput. 0 = N/A (GGUF).
    int max_length = 0;

    /// Peak working set size (MB).
    size_t peak_ws_mb = 0;

    /// Per-process GPU memory usage after model load (0 = N/A).
    size_t gpu_mem_mb = 0;

    /// OS-granted GPU budget for this process (0 = N/A).
    size_t gpu_budget_mb = 0;

    /// Speculative drafts generated (0 when prompt_lookup is off).
    int n_drafted = 0;

    /// Speculative drafts accepted.
    int n_spec_accepted = 0;

    /// Generated text output.
    std::string output_text;

    /// Error description on failure.
    std::string error_msg;
};

} // namespace xllama