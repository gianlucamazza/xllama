// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT
// Lane B — in-process partial fine-tune engine (ggml-opt / llama_opt).
// Runs the whole pipeline on the device that hosts the process (Xbox UWP or
// Linux dev host): prepare (selective f32 upcast + dataset) → train
// (llama_opt_epoch over a name-filtered trainable subset) → export (merged
// GGUF via llama_model_save_to_file) → evaluate (marker A/B in-process).
// The engine is active only when XLLAMA_DEVICE_TRAIN=1; other builds expose a
// clean unsupported-result stub. Contract in training_params.h.
#pragma once

#include "xllama/training_params.h"

#include <atomic>
#include <functional>
#include <string>
#include <vector>

namespace xllama {

// Progress snapshot delivered from the train loop (per optimizer batch and
// per stage transition).
struct DeviceTrainProgress {
    TrainStage stage = TrainStage::Prepare;
    int epoch = 0; // 1-based once training starts
    int epochs = 0;
    int64_t ibatch = 0;
    int64_t ibatch_max = 0;
    double loss = 0.0; // mean train loss of the current result window
};

struct DeviceTrainCallbacks {
    std::function<void(const DeviceTrainProgress&)> on_progress; // optional
    std::function<void(const std::string&)> on_status;           // optional log line sink
    // Best-effort cooperative abort, checked between epochs (llama_opt_epoch
    // is not interruptible mid-epoch). Aborting still writes result.json.
    std::atomic<bool>* abort_flag = nullptr;
};

// Run the full in-process pipeline for a validated PartialFt job.
// - job.base_model: path to a GGUF file (or a directory containing one).
// - job.dataset_path: preference-samples JSONL ({"label","messages":[...]})
//   or plain text; "dislike" samples are skipped.
// - job.out_dir: receives train-base-f32.gguf, merged.gguf, optional
//   checkpoint-epochN.gguf and result.json.
// Not concurrency-safe: one run per process at a time (static progress
// trampoline — ggml_opt_epoch_callback has no userdata).
TrainingResult run_device_train_job(const TrainingJob& job, const DeviceTrainCallbacks& cb = {});

// Substring match of a tensor name against job.param_filter patterns.
// Exposed for unit tests.
bool device_train_tensor_matches(const std::string& tensor_name,
                                 const std::vector<std::string>& patterns);

// Pin limitation (llama.cpp b10025): the KV-cache write is a ggml_set_rows
// node without backward support, so gradients cannot flow through any cache
// write. A tensor is trainable only if nothing downstream of it writes K/V:
// tensors of the LAST transformer block (except attn_k/attn_v, which feed
// that block's own cache write) plus post-block tensors (output_norm,
// output). Embeddings and rope frequencies are not optimizer parameters.
// Returns an empty string when the tensor is trainable-safe, else
// a human-readable reason. last_block = n_layer - 1. Exposed for unit tests.
std::string device_train_unsupported_reason(const std::string& tensor_name, int last_block);

// Build the training corpus text from a dataset file (JSONL chat samples
// rendered with the base model's chat template, or plain text passthrough).
// model_id selects the chat template family. Exposed for unit tests.
std::string device_train_build_corpus(const std::string& dataset_path, const std::string& model_id,
                                      std::string* err = nullptr);

} // namespace xllama
