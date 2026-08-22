# xllama SDK

xllama is an SDK for running large language models and diffusion models on Xbox Series S|X, Linux, and Windows. It provides a unified C++ interface for inference, training, and preference capture across backends.

## Architecture

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

int main() {
    xllama::SessionHub hub;

    xllama::SessionParams sp;
    sp.model_path = "models/lfm25-350m";
    sp.n_ctx = 2048;
    sp.n_threads = 0;

    std::string err;
    auto* session = hub.ensure_locked("lfm25-350m", sp, &err);
    if (!session) return 1;

    xllama::GenerateParams gp;
    gp.prompt = "What is the capital of France?";
    gp.n_predict = 64;
    gp.reuse_kv = true;

    auto result = session->generate(gp);
    if (result.success) {
        std::printf("%s\n", result.output_text.c_str());
    }
}
```

## Build Configuration

### CMake Options

| Option | Default | Description |
|--------|---------|-------------|
| `XLLAMA_BUILD_PROBES` | `ON` | Build benchmarking probes (membw, diskbw, gpubw, gpugemv, ramceil) |
| `XLLAMA_DEVICE_TRAIN` | `OFF` | Enable on-device training engine |

### Build Presets

```bash
cmake --preset linux-release
cmake --build build/linux-release -j$(nproc)

cmake --preset linux-test
cmake --build build/linux-test -j$(nproc)
ctest --test-dir build/linux-test --output-on-failure
```

### Probe Builds (Optional)

When `XLLAMA_BUILD_PROBES=OFF`, the CLI still builds but probe commands are unavailable.

```bash
cmake --preset linux-release -DXLLAMA_BUILD_PROBES=OFF
cmake --build build/linux-release -j$(nproc)
```

## Testing

```bash
cmake --preset linux-test
cmake --build build/linux-test -j$(nproc)
ctest --test-dir build/linux-test --output-on-failure
```

Key test files:
- `tests/test_routing_policy.cpp` — routing policy, token estimation, KV reuse gates
- `tests/test_sampling.cpp` — sampling config agreement between surfaces
- `tests/test_personalize.cpp` — personalization helpers
- `tests/test_speculative.cpp` — prompt-lookup speculative decoding

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

## License

SPDX-License-Identifier: MIT

## Contributing

See [AGENTS.md](../AGENTS.md) for coding conventions and build instructions.