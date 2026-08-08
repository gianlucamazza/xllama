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

// The one home for the shipping context size (#171). Every surface that opens a
// context (chat UI, LAN API, Session default, CLI default) reads this constant;
// the trimmer budget kMaxPromptTokens (routing_policy.h) is sized against it —
// tests/test_routing_policy.cpp pins the relation.
inline constexpr int kDefaultNCtx = 2048;

// ---------------------------------------------------------------------------
// Inference configuration
// ---------------------------------------------------------------------------
struct InferenceParams {
    std::string model_path; // Linux: absolute path; UWP: filename in LocalFolder
    std::string prompt;
    int n_predict = 128;
    int n_ctx = kDefaultNCtx;

    // #130 bench knob (ORT path). max_length is what governs DirectML prefill
    // throughput, and it is normally DERIVED as min(n_ctx, n_prompt + n_predict)
    // — which chains it to n_predict and makes the two axes of the break-even
    // criterion (prompt length, answer length) impossible to vary separately.
    // This override breaks that chain so the bench can measure at the
    // max_length the shipping app actually uses.
    //   0  = derive as before (default; every historical row was taken this way)
    //  -1  = saturate to n_ctx — what Session::generate ships since #135
    //  >0  = explicit, clamped into (n_prompt_tok, n_ctx]
    // Whatever is chosen is echoed back in InferenceResult::max_length, which is
    // a CSV column, so a knob an older build ignores shows up in the data.
    int max_length_override = 0;
    int n_threads = 0; // 0 = auto-detect
    // W1.1 bench knob (#F1): which repetition of a repeated bench this run is.
    // Written verbatim into the bench CSV's run_index column so repeats are
    // individually recoverable instead of being pre-averaged into one row at the
    // source (a median in the writer destroys the spread needed to judge it).
    // 0 = not a repeated bench / unknown (the legacy single-run case); the
    // summary generator treats a 0/absent run_index as a single measurement.
    int run_index = 0;
    // llama.cpp prefill batching (GGUF path only; 0 = llama.cpp default, i.e.
    // n_batch 2048 / n_ubatch 512). n_ubatch is the physical compute chunk that
    // sets prefill throughput on the Zen 2 CPU — the sweep knob for TTFT tuning.
    int n_batch = 0;
    int n_ubatch = 0;
    // #171: q8_0 KV cache + forced flash attention (GGUF path only; quantized V
    // requires FA). Same semantics and fallback as SessionParams::kv_q8.
    bool kv_q8 = false;
    // Sampling. Defaults come from xllama/sampling.h so the CLI/bench surface
    // cannot drift from the GUI/API surface (GenerateParams) the way it did
    // before #125, when this path had no top_p / top_k / repetition_penalty at
    // all and silently ran a different sampler.
    float temperature = sampling_defaults::kTemperature;
    float top_p = sampling_defaults::kTopP;
    int top_k = sampling_defaults::kTopK;
    float repetition_penalty = sampling_defaults::kRepetitionPenalty;
    uint32_t seed = sampling_defaults::kSeed;

    // Deterministic decode: pick argmax instead of sampling. Prerequisite for
    // cross-backend logit parity (llama.cpp vs ORT must agree token-for-token).
    bool greedy = false;

    SamplingConfig sampling() const {
        return SamplingConfig{temperature, top_p, top_k, repetition_penalty, seed, greedy};
    }

    // CLI --chat / --system: system message for the chat template. Empty = the
    // built-in default. The GUI and the LAN API have always made this
    // configurable; the CLI hardcoded it (#125).
    std::string system_prompt;

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

    // Phase 15 W2 (#210): draft-free prompt-lookup on the GGUF path. Default
    // OFF (CLI --prompt-lookup). Host timings are not product claims — use for
    // acceptance-rate A/B only until console M3.
    bool prompt_lookup = false;

    // CLI --membw: run the CPU memory-bandwidth micro-bench and exit (no model
    // load). Model/prompt are not required in this mode.
    bool run_membw = false;

    // CLI --gpubw: Phase 15 W3 (#211) GPU STREAM probe. On non-Windows hosts
    // reports d3d12 unavailable (no invented GB/s). Model/prompt not required.
    bool run_gpubw = false;
    // Phase 15 H6.1 (#228): Q4_K GEMV density probe (D3D12 CS; not a backend).
    bool run_gpugemv = false;

    // CLI --ramceil: probe how much heap this process can commit, and exit (no
    // model load). Measures the ceiling a GGUF actually spends against, which
    // decides model admissibility. Model/prompt are not required in this mode.
    bool run_ramceil = false;

    // CLI --validate-train-job <path.json>: parse + validate a TrainingJob and
    // exit (no model). Part of the training pillar (exploration).
    bool run_validate_train_job = false;
    // CLI --train-job <path.json>: shell out to the host training runner.
    bool run_train_job = false;
    // CLI --training-capabilities: print RE-backed capability matrix and exit.
    bool run_training_capabilities = false;
    std::string train_job_path;

    // UI callbacks (optional). Called from the inference thread — must marshal
    // to the UI thread before touching XAML controls. on_token receives a view
    // into a per-iteration buffer: copy it before the callback returns.
    std::function<void(const std::string&)> on_status; // e.g. "loading model"
    std::function<void(std::string_view)> on_token;    // per-token text piece

    // Stream generated pieces to stdout (interactive CLI). Off by default so
    // the UWP unified build's bench path pays no per-token write+flush.
    bool echo_stdout = false;

    // Set to true from the UI thread to request early termination.
    std::atomic<bool>* abort_flag = nullptr;
};

// #130: single home for the max_length ladder — the ROADMAP hardening item on
// the saturation being "expressed in two files that agree by comment".
//   override_v < 0  → saturate to n_ctx. What Session ALWAYS does (session.cpp
//                     passes -1): on DirectML max_length is the variable that
//                     controls prefill throughput and the interior band
//                     ~1400..n_ctx is a measured valley (uwp-constraints §5c).
//   override_v == 0 → derive min(n_ctx, n_prompt + n_predict). Bench default;
//                     keeps every historical row's meaning.
//   override_v > 0  → explicit, clamped to (n_prompt, n_ctx].
inline int resolve_max_length(int n_ctx, int n_prompt, int n_predict, int override_v) {
    if (override_v < 0)
        return n_ctx;
    if (override_v > 0)
        return std::clamp(override_v, n_prompt + 1, n_ctx);
    return std::min(n_ctx, n_prompt + n_predict);
}

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
    // without n_prompt_tok could not (#128). Normally min(n_ctx, n_prompt_tok +
    // n_predict) on the stateless path, but the bench can override it via
    // max_length_override (#139) — whatever was actually requested is recorded
    // here. 0 = N/A (GGUF, which does not set it).
    int max_length = 0;
    size_t peak_ws_mb = 0;
    size_t gpu_mem_mb = 0;    // per-process GPU CurrentUsage after model load (0 = N/A)
    size_t gpu_budget_mb = 0; // OS-granted GPU budget for this process (0 = N/A)
    // Phase 15 W2 (#210): speculative counters (0 when prompt_lookup off).
    int n_drafted = 0;
    int n_spec_accepted = 0;
    std::string output_text;
    std::string error_msg;
};

} // namespace xllama
