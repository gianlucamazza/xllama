// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT
// Training pillar contracts (exploration). Parallel to InferenceParams —
// never mixed into Session / GenerateParams.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace xllama {

// How weights are updated. LoraPeft = host PEFT pipeline; PartialFt = the
// in-process ggml-opt engine (Lane B) training a name-filtered tensor subset.
enum class TrainMethod {
    LoraPeft,       // PEFT LoRA (host Python; first exploration backend)
    PartialFt,      // in-process ggml-opt partial fine-tune (device lane engine)
    FullFtReserved, // reserved — full fine-tune not supported in-process
};

// Where the train loop runs.
enum class TrainDevice {
    Host,   // Linux/desktop — PEFT pipeline, or the in-process engine for dev
    Device, // Xbox/UWP — in-process ggml-opt engine (XLLAMA_DEVICE_TRAIN builds)
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

// RE-backed capability matrix (see docs/training-architecture.md).
// available=true only when implementable in the current product path.
enum class TrainingCapability {
    HostPeftLora,               // training/host PEFT — available
    HostMergeGguf,              // llama-export-lora — available
    HostEvaluateMarker,         // xllama-cli A/B — available
    RuntimeLoraLoadLlama,       // llama_set_adapters_lora — available
    RuntimeAdapterLoadOrtGenAI, // OgaLoadAdapter — designed; DML blocked on pin
    DeviceOrtOnDeviceTraining,  // ORT ODT package — research
    DeviceLlamaFinetune,        // llama-finetune full FT — rejected (RAM class)
    DeviceGgmlPartialFt,        // in-process ggml-opt partial FT — available (Lane B gates PASS)
    DevicePreferenceCapture,    // LocalState JSONL — available
};

// One row of the static capability table.
struct TrainingCapabilityInfo {
    TrainingCapability id = TrainingCapability::HostPeftLora;
    bool available = false;  // true => product can use it today
    const char* status = ""; // available | experimental | designed | research | rejected
    const char* name = "";   // stable id string
    const char* reason = ""; // short RE-backed note
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

    // Device lane (method == PartialFt, in-process ggml-opt engine).
    // Tensor-name substring patterns selecting the trainable subset. The current
    // llama.cpp pin supports the last block plus output/output_norm, excluding
    // K/V projections; see device_train_unsupported_reason(). Non-f32 matches
    // are upcast to f32 (llama_opt trains f32 params only).
    std::vector<std::string> param_filter;
    int n_ctx_train = 256;    // train context (= n_batch; activations scale with it)
    int epochs = 1;           // full passes over the dataset
    int checkpoint_every = 0; // save a checkpoint GGUF every N epochs (0 = off)
};

struct TrainingResult {
    bool success = false;
    std::vector<TrainStage> stages_completed;
    std::string adapter_path;
    std::string merged_gguf_path;
    std::string error_msg;
    // Optional metrics (the host runner and the Lane B engine fill these;
    // pure validation leaves them empty)
    double last_loss = 0.0;
    double wall_seconds = 0.0;
    std::size_t peak_ws_mb = 0;
};

} // namespace xllama
