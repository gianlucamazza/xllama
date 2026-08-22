# xllama SDK

xllama is an SDK for running large language models and diffusion models on Xbox Series S|X, Linux, and Windows. It provides a unified C++ interface for inference, training, and preference capture across backends.

## Architecture Overview

```
┌─────────────────────────────────────────────────────┐
│                    xllama SDK                        │
│                                                      │
│  SessionHub ── Session ── Backend (ORT / llama.cpp)  │
│       │              │                               │
│       │          RoutingPolicy                      │
│       │              │                               │
│       │          KvStore / PromptFit                │
│       │                                              │
│  ┌────┴────┬────────┴───────┬────────┬────────┐    │
│  │Inference│   Training     │Personal│Pref.   │    │
│  │  Params │     Job        │  ize   │ Capture│    │
│  └────┬────┴────────┬───────┴────────┴────────┘    │
│       │             │                               │
│  InferenceResult  DeviceTrainProgress               │
└─────────────────────────────────────────────────────┘
```

The SDK is header-only for most of its public API. The only compiled library is `xllama` (a static archive), which links against either:

- **ORT GenAI** — ONNX Runtime + Generative AI extension (Windows/UWP, DirectML/CPU)
- **llama.cpp** — GGUF/CPU (Linux, and UWP CPU fallback)

## Quick Start

```cpp
#include "xllama/session.h"
#include "xllama/session_hub.h"
#include "xllama/routing_policy.h"

int main() {
    // 1. Create a session hub (process-wide model owner)
    xllama::SessionHub hub;

    // 2. Configure the session
    xllama::SessionParams sp;
    sp.model_path = "models/lfm25-350m";  // path to GGUF or ONNX model
    sp.n_ctx = 2048;
    sp.n_threads = 0;  // 0 = auto-detect

    // 3. Load the model
    std::string err;
    auto* session = hub.ensure_locked("lfm25-350m", sp, &err);
    if (!session) {
        std::fprintf(stderr, "Failed to load model: %s\n", err.c_str());
        return 1;
    }

    // 4. Generate a response
    xllama::GenerateParams gp;
    gp.prompt = "What is the capital of France?";
    gp.n_predict = 64;
    gp.reuse_kv = true;  // enable KV-cache reuse for multi-turn

    auto result = session->generate(gp);
    if (result.success) {
        std::printf("%s\n", result.output_text.c_str());
    }
}
```

## Core Types

### `xllama::Session`

The `Session` class owns a loaded model and tokenizer. It supports multi-turn generation via KV-cache reuse.

```cpp
struct Session {
    // Load a model. Returns nullptr on failure.
    static std::unique_ptr<Session> create(const SessionParams& params,
                                           std::string* err = nullptr);

    // Generate a response. Thread-unsafe: do not call concurrently.
    virtual InferenceResult generate(const GenerateParams& params) = 0;

    // Count tokens without generating (for routing/heuristics).
    virtual int count_tokens(const std::string& prompt) = 0;

    // Context shift: evict oldest tokens instead of failing on overflow.
    virtual bool can_context_shift() const;

    // Persist/restore KV cache to disk.
    virtual bool save_state(const std::string& path, std::string* err = nullptr);
    virtual bool load_state(const std::string& path, std::string* err = nullptr);
};
```

### `xllama::SessionHub`

The `SessionHub` is the single process-wide owner of the resident model. Both the GUI and the LAN API lock it for the duration of a turn.

```cpp
struct SessionHub {
    std::mutex mtx;

    // Under mtx: return the resident session for `model_id`, creating it
    // (and destroying any other model's session first) if needed.
    Session* ensure_locked(const std::string& model_id, const SessionParams& sp,
                           std::string* err = nullptr);

    // Under mtx: drop the resident session.
    void reset_locked();
};
```

**SDK usage** — create your own instance instead of using the global:

```cpp
xllama::SessionHub hub;
auto* session = hub.ensure_locked("my-model", params, &err);
```

**Global instance** (backward compatible):

```cpp
auto& hub = xllama::session_hub();
```

### `xllama::RoutingPolicy` (SDK-configurable)

The `RoutingPolicy` struct bundles every routing gate behind `std::function` callbacks so an SDK user can override any gate without patching the header.

```cpp
struct RoutingPolicy {
    // GPU allowlist — #91 postmortem: only parity-validated DML assets may
    // route to GPU.
    std::function<bool(std::string_view gpu_model)> dml_text_model_ok_fn;

    // Decide which model to load for the first turn.
    std::function<RoutingDecision(const RoutingSettings&, int, bool, bool)> decide_fn;

    // Feature gates by catalogue kind.
    std::function<bool(const std::wstring& kind)> allow_kind_fn;

    // KV-reuse gate by catalogue kind.
    std::function<bool(const std::wstring& kind)> reuse_kv_kind_fn;

    // KV-reuse gate by model name (DML models reject continuous decoding).
    std::function<bool(std::string_view model)> reuse_kv_model_fn;

    // Convenience wrappers that dispatch through the callbacks.
    bool dml_text_model_ok(std::string_view gpu_model) const;
    RoutingDecision decide(const RoutingSettings& s, int n_tok, bool base_is_gguf,
                           bool gpu_available) const;
    bool routing_allowed_for_kind(const std::wstring& kind) const;
    bool kv_reuse_allowed_for_kind(const std::wstring& kind) const;
    bool kv_reuse_supported_for_model(std::string_view model) const;
};
```

**Custom routing example:**

```cpp
xllama::RoutingPolicy policy;
// Allow any model on GPU (override the #91 allowlist)
policy.dml_text_model_ok_fn = [](std::string_view m) { return true; };
// Always use GPU
policy.decide_fn = [](const xllama::RoutingSettings& s, int n, bool gguf, bool gpu) {
    xllama::RoutingDecision d;
    d.use_gpu = gpu;
    d.active_model = gpu ? s.gpu_model : s.cpu_model;
    return d;
};
```

**Free-function wrappers** (default behavior, backward compatible):

```cpp
bool ok = xllama::dml_text_model_ok("smollm2-360m-dml-fp16-v2");
auto decision = xllama::decide_routing(settings, n_tok, base_is_gguf, gpu_available);
bool allowed = xllama::routing_allowed_for_kind(L"ort-genai");
bool reuse = xllama::kv_reuse_supported_for_model("smollm2-360m-cpu-int4");
```

## Inference

### `xllama::InferenceParams`

Configuration for a single inference call. Used by the CLI, benchmarks, and the `run_inference()` function.

| Field | Default | Description |
|-------|---------|-------------|
| `model_path` | — | Path to model (Linux: absolute; UWP: filename in LocalFolder) |
| `prompt` | — | Input text |
| `n_predict` | 128 | Max tokens to generate |
| `n_ctx` | 2048 | Context size |
| `max_length_override` | 0 | Override max_length (0=derive, -1=saturate to n_ctx, >0=explicit) |
| `n_threads` | 0 | Thread count (0=auto) |
| `temperature` | 0.8 | Sampling temperature |
| `top_p` | 0.9 | Top-p sampling |
| `top_k` | 40 | Top-k sampling |
| `repetition_penalty` | 1.1 | Repetition penalty |
| `seed` | 0xFFFFFFFF | Random seed |
| `greedy` | false | Deterministic argmax decode |
| `system_prompt` | "" | System message (empty = built-in default) |
| `stop_sequences` | [] | Stop strings |
| `chat_template` | false | Wrap prompt with chat template |
| `lora_path` | "" | GGUF LoRA adapter path |
| `prompt_lookup` | false | Draft-free prompt-lookup speculative decoding |
| `dump_logits_path` | "" | Dump logit vector for parity harness |

### `xllama::InferenceResult`

Result of an inference call.

| Field | Type | Description |
|-------|------|-------------|
| `success` | bool | Whether generation succeeded |
| `t_load_ms` | double | Model load time (ms) |
| `t_p_eval_ms` | double | Prefill time (ms) |
| `t_eval_ms` | double | Decode time (ms) |
| `n_p_eval` | int | Prefill tokens |
| `n_eval` | int | Decode tokens |
| `ended_with_stop` | bool | Stopped on a stop sequence |
| `max_length` | int | Max length actually requested |
| `peak_ws_mb` | size_t | Peak working set (MB) |
| `gpu_mem_mb` | size_t | Per-process GPU memory (CurrentUsage) |
| `gpu_budget_mb` | size_t | OS-granted GPU budget |
| `n_drafted` | int | Speculative drafts (0 if prompt_lookup off) |
| `n_spec_accepted` | int | Speculative accepted |
| `output_text` | std::string | Generated text |
| `error_msg` | std::string | Error description on failure |

### `xllama::run_inference()`

Synchronous inference function. Returns `InferenceResult` with metrics.

```cpp
InferenceResult run_inference(const InferenceParams& params);
```

On Linux: reads `model_path` as absolute filesystem path.
On UWP: `model_path` is relative to `ApplicationData::LocalFolder\models`.

## Session API

### `xllama::SessionParams`

Configuration for creating a persistent session.

| Field | Default | Description |
|-------|---------|-------------|
| `model_path` | — | Path to model |
| `n_ctx` | 2048 | Context size |
| `n_threads` | 0 | Thread count (0=auto) |
| `n_batch` | 0 | llama.cpp logical prefill batch (0=default 2048) |
| `n_ubatch` | 0 | llama.cpp physical prefill chunk (0=default 512) |
| `backend` | Auto | ORTGenAI, LlamaCpp, or Auto (inspects model) |
| `n_gpu_layers` | 0 | llama.cpp GPU layers (0=CPU) |
| `lora_path` | "" | GGUF LoRA adapter |
| `lora_scale` | 1.0f | LoRA scale |
| `kv_q8` | false | Quantize KV cache to q8_0 |
| `dml_warmup` | true | Warm-up DML models at load time |
| `prompt_lookup` | false | Draft-free prompt-lookup speculative decoding |

### `xllama::GenerateParams`

Parameters for a single generation turn.

| Field | Default | Description |
|-------|---------|-------------|
| `prompt` | — | Input text |
| `n_predict` | 96 | Max tokens to generate |
| `temperature` | 0.8 | Sampling temperature |
| `top_p` | 0.9 | Top-p sampling |
| `top_k` | 40 | Top-k sampling |
| `repetition_penalty` | 1.1 | Repetition penalty |
| `seed` | 0xFFFFFFFF | Random seed |
| `stop_sequences` | [] | Stop strings |
| `reuse_kv` | false | Enable KV-cache reuse (continuous decoding) |
| `reset_kv` | false | Reset KV cache for this turn |
| `n_keep` | 0 | Tokens pinned across context shifts |
| `on_token` | — | Callback per generated token |
| `on_status` | — | Callback for status changes |
| `abort_flag` | — | Atomic flag for early termination |

### KV-Cache Reuse Modes

| `reuse_kv` | `reset_kv` | Behavior |
|------------|------------|----------|
| `false` | — | Stateless turn: full context prefill |
| `true` | `false` | Continuation: delta-only prefill (multi-turn TTFT win) |
| `true` | `true` | Reset: full context prefill, subsequent turns reuse |

## Training

### `xllama::TrainingJob`

Configuration for on-device fine-tuning. Defined in `training_params.h`.

```cpp
struct TrainingJob {
    std::string base_model;       // GGUF file or directory
    std::string dataset_path;     // JSONL training data
    std::string output_path;      // Output GGUF path
    TrainStage stage = TrainStage::Train;
    TrainMethod method = TrainMethod::LoRA;
    TrainDevice device = TrainDevice::Host;
    int epochs = 1;
    int batch_size = 1;
    float learning_rate = 1e-5;
    // ... additional fields
};
```

### `xllama::DeviceTrainProgress`

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

## Routing Policy

### `xllama::RoutingSettings`

Configuration for EP (execution provider) routing.

| Field | Default | Description |
|-------|---------|-------------|
| `mode` | CpuOnly | CpuOnly, GpuOnly, or Auto |
| `token_threshold` | 1550 | Token count threshold for GPU routing |
| `cpu_model` | "" | CPU model path |
| `gpu_model` | "" | GPU model path |

### `xllama::RoutingDecision`

Result of a routing decision.

| Field | Type | Description |
|-------|------|-------------|
| `active_model` | std::string | Selected model path |
| `use_gpu` | bool | Whether GPU is active |
| `token_count` | int | Token count at decision time |

## KV Store

### `xllama::KvStore`

On-disk KV snapshot pool for conversation persistence.

```cpp
struct KvStore {
    std::string dir;

    // Validate conversation ID (plain identifier only).
    static bool valid_id(const std::string& id);

    // Resolve path for a conversation ID.
    std::string path_for(const std::string& id) const;

    // Erase a snapshot.
    void erase(const std::string& id) const;
};
```

A snapshot is ~12 KiB per resident token. The pool is capped by file count and total bytes.

## Prompt Budget

### `xllama::fit_prompt`

Exact token-budget trimmer. Enforces the context budget where the tokenizer is available (in `Session::count_tokens`), not from character estimates.

```cpp
struct PromptFit {
    std::string prompt;     // rendered, ready for Session::generate
    int n_tokens = -1;      // exact token count (-1 when no counter given)
    int dropped = 0;        // oldest turns dropped to make it fit
};

PromptFit fit_prompt(const std::string& base_prompt,
                     const std::vector<std::pair<std::string, std::string>>& history,
                     int max_tokens, TokenCounter counter);
```

## Sampling

### `xllama::SamplingConfig`

Shared sampling configuration. Both `InferenceParams` and `GenerateParams` expose these values.

```cpp
struct SamplingConfig {
    float temperature = 0.8f;
    float top_p = 0.9f;
    int top_k = 40;
    float repetition_penalty = 1.1f;
    uint32_t seed = 0xFFFFFFFF;
    bool greedy = false;

    bool is_greedy() const;
};

// Check if two configs would assemble the same sampler chain.
bool same_chain(const SamplingConfig& a, const SamplingConfig& b);
```

## Chat Prompt

### `xllama::ChatFormat`

Chat template format detection and rendering.

```cpp
enum class ChatFormat { ChatML, Gemma, Llama, Unknown };

struct ChatFormat {
    ChatFormat format;
    std::string eos_token;
    std::string bos_token;
};

ChatFormat chat_format_for(const std::string& model_id);
std::string render_prompt(const ChatFormat& fmt, const std::string& system,
                          const std::vector<std::pair<std::string, std::string>>& history,
                          const std::string& prompt);
```

## Path Utilities

```cpp
// Resolve a model filename: UWP -> LocalFolder\models\<filename>
std::string resolve_model_path(const std::string& filename);

// Resolve a generic filename: UWP -> LocalFolder\<filename>
std::string resolve_local_path(const std::string& filename);

// Check if a model uses the llama.cpp backend.
bool model_uses_llama_backend(const std::string& model_id);
```

## Platform Helpers

```cpp
// Number of hardware threads (fallback: 4).
int detect_threads() noexcept;

// Default thread count for llama.cpp (UWP capped at 6).
int detect_threads_llama() noexcept;

// Emit a log line (UWP: OutputDebugStringA, otherwise: stderr).
void log_output(const char* msg) noexcept;

// Peak working set in MB (0 if unavailable).
std::size_t peak_working_set_mb() noexcept;
```

## Build Configuration

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `XLLAMA_BUILD_PROBES` | `ON` | Build benchmarking probes (membw, diskbw, gpubw, gpugemv, ramceil) |
| `XLLAMA_DEVICE_TRAIN` | `OFF` | Enable on-device training engine |

### Build Presets

```bash
# Linux Release
cmake --preset linux-release
cmake --build build/linux-release -j$(nproc)

# Linux with tests
cmake --preset linux-test
cmake --build build/linux-test -j$(nproc)
ctest --test-dir build/linux-test --output-on-failure
```

### Probe Builds (Optional)

Probe targets are controlled by `XLLAMA_BUILD_PROBES`. When `OFF`, the CLI still builds but probe commands (`--membw`, `--diskbw`, `--gpubw`, `--gpugemv`, `--ramceil`) are unavailable.

```bash
cmake --preset linux-release -DXLLAMA_BUILD_PROBES=OFF
cmake --build build/linux-release -j$(nproc)
```

## Model Support

### Supported Backends

| Backend | Format | Platforms |
|---------|--------|-----------|
| ORT GenAI | ONNX | Windows/UWP (CPU + DirectML) |
| llama.cpp | GGUF | Linux, Windows/UWP (CPU) |

### Supported Models

| Tier | Model | Format | Notes |
|------|-------|--------|-------|
| Floor | lfm25-350m | GGUF | Minimum viable model |
| Default | smollm2-360m-cpu-int4 | GGUF | Shipping default |
| Quality | qwen35-0.8b | GGUF | Best quality |
| Coding | coding-specific | GGUF | Code-specialized |
| ORT-only | smollm2-360m-dml-fp16-v2 | ONNX | GPU routing (DML) |

### Model Provisioning

Models are stored in `LocalFolder\models\`. The SDK checks the catalogue manifest to determine whether a model directory is provisioned with the correct files.

```cpp
// Check if a model directory matches the manifest's expected files.
bool dir_satisfies_expected_files(const std::string& dir,
                                  const std::vector<std::wstring>& expected);
```

## SDK Migration Guide

### From Application to SDK

If you're migrating from the application to the SDK:

1. **Replace `session_hub()` with your own `SessionHub` instance** — the global singleton is kept for backward compatibility but SDK users should own their instance.

2. **Configure routing policy** — the `RoutingPolicy` struct allows custom routing decisions without patching the header.

3. **Disable probes if not needed** — set `XLLAMA_BUILD_PROBES=OFF` to reduce build size.

4. **Use `run_inference()` for simple use cases** — for multi-turn apps, use `Session` + `SessionHub`.

### Example: Minimal SDK Usage

```cpp
#include "xllama/session.h"
#include "xllama/session_hub.h"

int main() {
    xllama::SessionHub hub;
    xllama::SessionParams sp;
    sp.model_path = "models/lfm25-350m";

    std::string err;
    auto* session = hub.ensure_locked("lfm25-350m", sp, &err);
    if (!session) return 1;

    xllama::GenerateParams gp;
    gp.prompt = "Hello, world!";
    gp.n_predict = 32;

    auto result = session->generate(gp);
    if (result.success) {
        std::printf("%s\n", result.output_text.c_str());
    }
}
```

## Testing

Tests are in `tests/` and use the doctest framework.

```bash
cmake --preset linux-test
cmake --build build/linux-test -j$(nproc)
ctest --test-dir build/linux-test --output-on-failure
```

Key test files:
- `tests/test_routing_policy.cpp` — routing policy, token estimation, KV reuse gates
- `tests/test_sampling.cpp` — sampling config agreement between surfaces
- `tests/test_personalize.cpp` — personalization helpers

## Versioning

xllama follows semantic versioning: `Major.Minor.Build.Revision`.

- **Major**: Breaking API changes
- **Minor**: New features (backward compatible)
- **Build**: Bug fixes and minor improvements
- **Revision**: CI build number (stamped automatically in CI, `.0` locally)

## License

SPDX-License-Identifier: MIT

## Contributing

See [AGENTS.md](../AGENTS.md) for coding conventions and build instructions.