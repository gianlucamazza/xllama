// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// Phase 11 — pure helpers for the in-app personalization arc (#116).
// Job construction, last-block param filters, sample counting, and the
// LocalState manifest override JSON for serving merged.gguf. No WinRT.
#pragma once

#include "xllama/training_params.h"

#include <cstdint>
#include <string>
#include <vector>

namespace xllama {

// Catalogue / published model id for the personalized GGUF.
inline const char* kPersonalizedModelId = "personalized";
inline const char* kPersonalizedDisplay = "Personalized (from your feedback)";
inline const char* kPersonalizeDefaultDataset = "training/samples.jsonl";
inline const char* kPersonalizeDefaultOutDir = "training/out/personalized";
inline const char* kPersonalizeDefaultBase = "training/base-f16.gguf";

// Inputs for build_personalize_job. Paths may be LocalState-relative.
struct PersonalizeSpec {
    std::string base_model; // required: GGUF file or directory
    std::string dataset_path = kPersonalizeDefaultDataset;
    std::string out_dir = kPersonalizeDefaultOutDir;
    std::string name = "personalized";
    // Last transformer block index (n_layer - 1). -1 = guess from base_model id.
    int last_block = -1;
    int epochs = 8;
    float learning_rate = 2e-4f;
    int n_ctx_train = 256;
    // Optional marker eval (empty = skip).
    std::string eval_prompt;
    std::string eval_expect_contains;
};

// Last-block partial_ft filter matching device_train_unsupported_reason rules:
// last block attn_q/attn_output/ffn_* + output_norm (no attn_k/attn_v).
std::vector<std::string> last_block_param_filter(int last_block);

// Guess last_block from a model path or catalogue id. Returns -1 if unknown.
// Known: smollm2-360* / smollm2*360* → 31; smollm2-1.7* → 23.
int guess_last_block_from_model_id(const std::string& model_id_or_path);

// Build a validated PartialFt + Device TrainingJob. On failure sets *err.
bool build_personalize_job(const PersonalizeSpec& spec, TrainingJob& out,
                           std::string* err = nullptr);

// Serialize a PartialFt job to a minimal JSON document (for training/job.json).
std::string format_personalize_job_json(const TrainingJob& job);

// LocalState manifest.json override body: models:[{name,display,kind:gguf,files}].
// approx_bytes may be 0.
std::string personalized_manifest_override_json(const std::string& model_id,
                                                const std::string& display, uint64_t approx_bytes);

// Count non-dislike preference lines in a samples.jsonl file (0 if missing/unreadable).
// Dislike rows are skipped by the engine; they do not count toward preflight.
int count_usable_preference_samples(const std::string& samples_path);

// Progress snapshot as one JSON object (for training/progress.json).
std::string format_train_progress_json(const std::string& stage, int epoch, int epochs,
                                       int64_t ibatch, int64_t ibatch_max, double loss);

// Parse training/result.done content ("ok" / "fail" with optional trailing junk).
// Returns "ok", "fail", or "" if empty/unknown.
std::string parse_train_result_done(const std::string& content);

} // namespace xllama
