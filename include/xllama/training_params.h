// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT
// Training pillar contracts (exploration). Parallel to InferenceParams —
// never mixed into Session / GenerateParams.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace xllama {

// How weights are updated. Only LoraPeft is implemented (host PEFT).
enum class TrainMethod {
    LoraPeft,       // PEFT LoRA (host Python; first exploration backend)
    FullFtReserved, // reserved — full fine-tune not supported in-process
};

// Where the train loop runs.
enum class TrainDevice {
    Host,   // Linux/desktop PEFT pipeline — implemented
    Device, // Xbox/UWP — exploration reserved (unsupported in validate)
};

// Ordered pipeline stages shared by host runner + docs.
enum class TrainStage {
    Prepare,
    Train,
    ExportAdapter,
    Merge,
    Evaluate,
    Publish,
};

// On-disk adapter / merge product.
enum class AdapterFormat {
    PeftDir,    // adapter_config.json + adapter_model.safetensors
    GgufLora,   // convert_lora_to_gguf output
    MergedGguf, // llama-export-lora product (loadable by Session)
};

struct AdapterArtifact {
    std::string path;
    AdapterFormat format = AdapterFormat::PeftDir;
    std::string base_model_id;
};

// Declarative train job (job JSON maps 1:1 on the host runner).
struct TrainingJob {
    int schema_version = 1;
    std::string name;
    TrainMethod method = TrainMethod::LoraPeft;
    TrainDevice device = TrainDevice::Host;

    std::string base_model;   // HF id or local snapshot path
    std::string dataset_path; // JSONL chat rows
    std::string out_dir;      // working dir for adapter / gguf / result.json

    // LoRA hyperparams (method == LoraPeft)
    int lora_rank = 8;
    int lora_alpha = 16;
    int steps = 120;
    int seed = 42;
    float learning_rate = 2e-4f;

    // Evaluate stage (optional marker A/B)
    std::string eval_prompt;
    std::string eval_expect_contains;

    bool do_merge = true;
    bool do_quantize = false;
};

struct TrainingResult {
    bool success = false;
    std::vector<TrainStage> stages_completed;
    std::string adapter_path;
    std::string merged_gguf_path;
    std::string error_msg;
    // Optional metrics (host runner may fill; C++ validate leaves empty)
    double last_loss = 0.0;
    double wall_seconds = 0.0;
};

} // namespace xllama
