# Training API

## `xllama::TrainingJob`

Configuration for on-device fine-tuning. Defined in `training_params.h`.

```cpp
struct TrainingJob {
    int schema_version = 1;
    std::string name;
    TrainMethod method = TrainMethod::LoraPeft;
    TrainDevice device = TrainDevice::Host;
    std::string base_model;
    std::string dataset_path;
    std::string out_dir;
    int lora_rank = 8;
    int lora_alpha = 16;
    int steps = 120;
    int seed = 42;
    float learning_rate = 2e-4f;
    std::string eval_prompt;
    std::string eval_expect_contains;
    bool do_merge = true;
    bool do_quantize = false;
    std::vector<std::string> param_filter;
    int n_ctx_train = 256;
    int epochs = 1;
    int checkpoint_every = 0;
};
```

## `xllama::TrainingResult`

```cpp
struct TrainingResult {
    bool success = false;
    std::vector<TrainStage> stages_completed;
    std::string adapter_path;
    std::string merged_gguf_path;
    std::string error_msg;
    double last_loss = 0.0;
    double wall_seconds = 0.0;
    std::size_t peak_ws_mb = 0;
};
```

## `xllama::DeviceTrainProgress`

Progress snapshot delivered from the training loop.

```cpp
struct DeviceTrainProgress {
    TrainStage stage = TrainStage::Prepare;
    int epoch = 0;
    int64_t ibatch = 0;
    int64_t ibatch_max = 0;
    double loss = 0.0;
};
```

## Training Functions

```cpp
// Validate a TrainingJob.
bool validate_training_job(const TrainingJob& job, std::string* err = nullptr);

// Parse a TrainingJob from JSON.
bool parse_training_job_json(const std::string& json, TrainingJob& out,
                             std::string* err = nullptr);

// Load a TrainingJob from disk.
bool load_training_job_file(const std::string& path, TrainingJob& out,
                            std::string* err = nullptr);

// Run the in-process pipeline (PartialFt method only).
TrainingResult run_device_train_job(const TrainingJob& job,
                                    const DeviceTrainCallbacks& cb = {});

// Capability matrix.
size_t training_capabilities(const TrainingCapabilityInfo** out);
bool training_capability_available(TrainingCapability c);
const TrainingCapabilityInfo* training_capability_info(TrainingCapability c);
```

## Training Methods

| Method | Description |
|--------|-------------|
| `LoraPeft` | PEFT LoRA via host Python pipeline |
| `PartialFt` | In-process ggml-opt partial fine-tune (Lane B) |
| `FullFtReserved` | Reserved — full fine-tune not supported in-process |

## Training Devices

| Device | Description |
|--------|-------------|
| `Host` | Linux/desktop — PEFT pipeline or in-process engine |
| `Device` | Xbox/UWP — in-process ggml-opt engine (XLLAMA_DEVICE_TRAIN builds) |

## Pipeline Stages

`Prepare` → `Train` → `ExportAdapter` → `Merge` → `Evaluate` → `Publish`