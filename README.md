# xllama

**xllama — local LLM inference for Xbox Series S consoles in Developer Mode**

Run a language model on your own Xbox Series S, with chat, an API on your local network, image generation and experimental on-device fine-tuning.

Tested and measured on Series S. Not yet tested on Series X.

### ⚠️ Read before installing

- **Developer Mode only.** xllama runs only on an Xbox console in Developer Mode, Microsoft's program for developing and testing apps. To activate it, you need your own Partner Center developer account, and you must accept Microsoft's [Xbox Developer Mode Activation terms](https://learn.microsoft.com/en-us/legal/windows/agreements/xbox-one-developer-mode-activation) yourself. xllama is a developer and research project, not a consumer product.
- **Retail games and apps don't run in Developer Mode.** You need to switch back to Retail Mode to play. Microsoft notes that required updates may cause "diminished or lost functionality" on a Developer Mode console, and a console reset deletes local saves that aren't synced to the cloud.
- **Unofficial.** Independent project, not affiliated with or endorsed by Microsoft. xllama is not reviewed, certified or supported by Microsoft and is not distributed through the Microsoft Store. Install it only on a console you own, at your own risk. This project makes no statement about warranty coverage or about how Microsoft may treat Developer Mode use; read Microsoft's terms and warranty for your country.
- **Install only from this repository's Releases page** and check the SHA-256. The package is signed with a test certificate (`xllama-test.cer`); trust it only if you downloaded it from here.
- **The LAN API is off by default and has no authentication.** When it is on, any device on your network, and any web page open in a browser on it, can send it requests. Enable it only on a network you trust.
- **Models are third-party.** Models are downloaded separately (from Hugging Face or this repository's `models-v1` release) and each one is under its own licence, not under xllama's MIT licence. Some licences are non-commercial or carry attribution requirements. Check each model's licence on its upstream model card (LFM models also download their LICENSE file alongside the weights) before using it.
- xllama is provided "AS IS", without warranty, under the MIT licence.

[![build-uwp](https://github.com/gianlucamazza/xllama/actions/workflows/build-uwp.yml/badge.svg)](https://github.com/gianlucamazza/xllama/actions/workflows/build-uwp.yml)
[![build-linux](https://github.com/gianlucamazza/xllama/actions/workflows/build-linux.yml/badge.svg)](https://github.com/gianlucamazza/xllama/actions/workflows/build-linux.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

<!-- XLLAMA_DOI_START -->

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22118437.svg)](https://doi.org/10.5281/zenodo.22118437)
<!-- XLLAMA_DOI_END -->

[CHANGELOG](CHANGELOG.md) · [ROADMAP](ROADMAP.md) · [Positioning](docs/positioning.md)

https://github.com/user-attachments/assets/9494a6a2-f14e-4229-afa4-ff5faf51ba67

A 4.7-second cut (three segments) of a 46-second capture made on a Series S in Developer Mode from Device Portal screenshots. The video doesn't support any speed claim. The full capture is [`xllama-demo-v1.5.6.mp4`](docs/screenshots/xllama-demo-v1.5.6.mp4), the cut points are in [`demo-manifest.json`](docs/screenshots/demo-manifest.json), and the demo is driven by [`demo/demo-script.json`](demo/demo-script.json) via `scripts/capture-demo-video.sh`. A longer [product showcase](docs/screenshots/xllama-demo-v1.5.6-showcase.mp4) is driven by [`demo/demo-showcase-script.json`](demo/demo-showcase-script.json).

---

## Quick install

<a id="build"></a>

Install in the browser (Device Portal, no credentials file): [Install from a release](docs/install-release.md#install-from-a-release-in-the-browser). (steps not yet walked through on a console)

Command-line path:

```bash
# Download release assets or a specific CI artifact first:
# docs/install-release.md (installing a newer version over the existing app keeps your data in LocalState)

# Or build (Windows host)
git clone --recursive https://github.com/gianlucamazza/xllama.git
.\scripts\build-uwp.ps1 -Configuration Release -Platform x64

# Deploy
source ~/.config/xllama/xbox-env
./scripts/deploy.sh path/to/xllama_*.msix
```

First launch downloads the default chat model LFM2.5-350M (~219 MB catalogue
size from the [model list](docs/model-selection.md), not a runtime measurement).
No model bundled in MSIX.

**Linux dev:** `cmake --preset linux-release && cmake --build build/linux-release -j`

---

## What you can do

- **LAN API** — OpenAI-compat completions, GGUF embeddings, trusted catalogue
  pull, preferences and training status. Current integration probe for games
  and assistants; **off by default and has no authentication**.
- **Bench** — headless tok/s, membw, diskbw, gpubw, gpugemv, gpustep, ramceil probes, d3d12 backend selftest
- **Diffuse** — SD-Turbo on DirectML, in-process with XAML compositor.
  Image generation uses SD-Turbo. **Powered by Stability AI.**
- **Train** — experimental on-device fine-tune of the last block (Lane B; one console marker run on SmolLM2-360M, 2026-07-20, [receipt](bench/results/phase10-console-devtrain-result.json)), host PEFT (Lane A), serve merged GGUF (Lane C). The in-app _Train on my feedback_ flow is not yet validated end to end on a console.
- **Chat (demo shell)** — multi-turn pad UI with KV-reuse, thinking models,
  coding tier. Showcase, not the product thesis.

---

## Supported models

| Model              | Params | Decode                                                  | Role                                                           |
| ------------------ | ------ | ------------------------------------------------------- | -------------------------------------------------------------- |
| LFM2.5-230M        | 230M   | **132.4** tok/s (llama.cpp CPU, 6 threads, median of 3) | Floor (236 MB peak RAM, llama.cpp CPU, 6 threads, median of 3) |
| LFM2.5-350M        | 350M   | **101.6** tok/s (llama.cpp CPU, 6 threads, median of 3) | Default chat                                                   |
| Qwen2.5-Coder-1.5B | 1.5B   | **26.1** tok/s (llama.cpp CPU, 6 threads, median of 2)  | Coding                                                         |
| LFM2-2.6B          | 2.6B   | **17.9** tok/s (llama.cpp CPU, 6 threads, median of 3)  | Quality (H9 7/8)                                               |

Tested and measured on Series S. Not yet tested on Series X. Receipts:
[132.4 / 101.6 tok/s and 236 MB](bench/results/2026-09-30-catalogue-gates.csv),
[17.9 tok/s](bench/results/2026-10-11-remeasure-1163/lfm2-2.6b-q4km-cpu-t6-rerun.csv),
[26.1 tok/s](bench/results/phase14-console.csv),
[H9 7/8](bench/results/2026-09-30-h9-rescore.jsonl).
Summary: [docs/benchmarks.md](docs/benchmarks.md).

Full catalogue + Phase 14 coding models: [model-matrix.md](docs/model-matrix.md) · [benchmarks.md](docs/benchmarks.md)

Hugging Face collection: [xllama: models measured on Xbox Series S](https://huggingface.co/collections/gianlucamazza/xllama-models-measured-on-xbox-series-s-6aca0b32275cbe99dbf02d89) — models measured in [docs/benchmarks.md](docs/benchmarks.md).

---

## LAN Endpoint — OpenAI-compatible

xllama exposes an HTTP endpoint on the local network that exposes its full
inference core (`SessionHub`) with OpenAI and Ollama-compatible APIs.
Status: v1, **off by default and has no authentication**. Dev Mode / trusted
LAN only — the current integration probe, not a public inbound service.

```bash
# Chat completions (add `"stream": true` for SSE)
curl -s http://<xbox-ip>:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"lfm25-350m","messages":[{"role":"user","content":"hi"}]}'

# Model discovery (OpenAI shape)
curl -s http://<xbox-ip>:11434/v1/models
```

| Route                         | Shape      | Note                                                             |
| ----------------------------- | ---------- | ---------------------------------------------------------------- |
| `POST /v1/chat/completions`   | OpenAI     | JSON, or SSE when `stream: true` (tool calls stay non-streaming) |
| `POST /api/pull`              | Ollama     | Trusted catalogue download; NDJSON progress                      |
| `POST /api/embed`             | Ollama     | GGUF embeddings; scalar or batch input                           |
| `POST /api/embeddings`        | Ollama     | Deprecated single-prompt embeddings                              |
| `POST /v1/embeddings`         | OpenAI     | Float32 or base64 encoded vectors                                |
| `GET /v1/models`              | OpenAI     | `"active": true` on the loaded model                             |
| `GET /api/tags`               | Ollama     | Same list, Ollama shape                                          |
| `POST /v1/preferences`        | Custom     | Append JSONL → `training/samples.jsonl`                          |
| `GET /v1/training/status`     | Custom     | Snapshot of on-device train progress                             |
| `POST /v1/images/generations` | OpenAI-ish | SD-Turbo, `b64_json` + `path`                                    |
| `GET /health`                 | Custom     | `{"status":"ok","service":"xllama"}`                             |

Full protocol — requirements, port config, enable/persistence, concurrency,
streaming status: [api-endpoint.md](docs/api-endpoint.md).

---

## Architecture at a glance

### Two pillars, one core

| Pillar        | Role                                   | Hot path                    |
| ------------- | -------------------------------------- | --------------------------- |
| **Inference** | LAN API, benches, diffusion, demo chat | `Session` / `run_inference` |
| **Training**  | PEFT adapters, merged GGUF             | `TrainingJob` → artefacts   |

Core: `src/bridge/` C++17, WinRT-free headers in `include/xllama/`, host-testable.
Two front-ends: `xllama-cli` (Linux) + UWP app.

### Backend dispatch

| Path                     | What                                                                  | Why                                                                              |
| ------------------------ | --------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **ORT GenAI + DirectML** | ONNX models, CPU int4 decode + DML fp16 prefill                       | GPU wins batch compute, long-prompt TTFT                                         |
| **llama.cpp + GGUF**     | CPU, KV-reuse, repacked GEMM; d3d12 matmul layers per catalogue entry | Zen2 wins decode at small sizes; own D3D12 kernels win on the 1.2B / 3B measured |

Unified build dispatches **per model at runtime** via `Backend::Auto`.
Llama.cpp is both benchmarking lane and shipping backend.

### Key invariants

- **One resident session** (`SessionHub`) — never 2× model in RAM
- **One budget home** (`prompt_budget.h`: `fit_prompt`, `fit_tool_transcript`) — tokens, not chars
- **One sampler chain per backend** — CLI/bench and GUI/API can't diverge
- **Single-home rule** — a decision both surfaces make lives in one header

### Platform constraints

- UWP/AppContainer: no mmap, no dlopen, no registry, no arbitrary paths
- Xbox Series S: unified memory, a per-process GPU budget and a per-file cap —
  the measured limits are in [uwp-constraints.md](docs/uwp-constraints.md)
- Patched ORT/GenAI DLLs while upstream lacks AppContainer fixes
- Dev Mode only — no retail path yet

---

## Repository map

```
include/xllama/   # WinRT-free, host-testable headers
src/bridge/       # shared implementation (Linux + UWP)
uwp/              # C++/WinRT app, LAN API, headless flags
training/         # jobs, host PEFT, datasets
shaders/          # HLSL → AOT DXIL compute shaders
tests/            # doctest suite; counts live in docs/architecture.md
scripts/          # deploy, bench, validate, crossbuild
docs/             # SSOT map → docs/README.md
```

**Doc ownership:** [docs/README.md](docs/README.md)

## Citation

The versioned research report is archived on Zenodo. The concept DOI
10.5281/zenodo.22118437 always resolves to the latest version; to cite 1.0.6 specifically:

```bibtex
@software{xllama_research_1_0,
  author  = {Mazza, Gianluca},
  title   = {Consumer Game Consoles as Local AI Compute},
  version = {1.0.6},
  doi     = {10.5281/zenodo.22119126},
  url     = {https://github.com/gianlucamazza/xllama}
}
```

The DOI badge and version DOI are synchronized from `release.toml`.

---

## Key design decisions

### Why Xbox Series S?

Zen2 CPU wins decode on the DirectML path and at small model sizes; RDNA2
wins batch prefill, and our own D3D12 kernels win GGUF decode on the 1.2B and
3B models measured (Q4_0 / Q4_K_M).
Unified memory means the GPU budget ([uwp-constraints.md](docs/uwp-constraints.md) §7) is the hard constraint.
An underexplored platform with strict memory and packaging constraints.

### Why dual backend?

Per-workload verdict on DirectML: CPU decode > GPU decode, GPU prefill > CPU
prefill. One backend can't win both. Runtime dispatch per model is the answer.
GGUF decode also runs on our own D3D12 kernels: a ggml backend `d3d12`, with
q8 numerics that match the CPU. It is the default only for LFM2.5-1.2B. The
coding option, Qwen2.5-Coder-1.5B (Apache-2.0), runs on the CPU like the
other GGUF models. The Qwen2.5-Coder-3B GPU numbers are an earlier
measurement; that model is under a non-commercial research licence and is
no longer in the model list. `gguf_gpu_layers.txt` overrides the default for an operator, and
`0` forces the CPU. Gates and per-model verdicts are in
[docs/gguf-gpu-decode.md](docs/gguf-gpu-decode.md). How that compares with
desktop RDNA2, the Steam Deck and other GPUs on published llama.cpp numbers is
in [docs/benchmarks.md](docs/benchmarks.md#gpu-decode-in-context--known-hardware).

### Why single Session owner?

A CPU model and a DML model don't fit the budget together
([uwp-constraints.md](docs/uwp-constraints.md) §7). Two models = OOM.
`SessionHub` makes this a process-wide invariant, not a per-surface convention.

### Why token-budget, not chars-per-token?

A chars-per-token estimate is not a safe stand-in for a token count: being
wrong truncates the reply or drops history that would have fit. The estimate
survives only for routing, where being wrong costs a decision, not an answer.

### Why consolidate duplicated loops?

Every copy in this codebase has eventually disagreed — silently.
`decode_loop.h` (llama), `decode_loop_ort.h` (ORT), `sampler_chain.h` (llama),
`ort_sampling.h` (ORT) — each decision lives in one header.

---

## More docs

| Topic                                              | Docs                                                                |
| -------------------------------------------------- | ------------------------------------------------------------------- |
| Full architecture (modules, backends, KV, routing) | [architecture.md](docs/architecture.md)                             |
| Training pillar (lanes A/B/C, Phase 11 UI arc)     | [training-architecture.md](docs/training-architecture.md)           |
| AppContainer constraints (§1–§13)                  | [uwp-constraints.md](docs/uwp-constraints.md)                       |
| Model catalogue + selection                        | [model-selection.md](docs/model-selection.md)                       |
| Performance numbers                                | [benchmarks.md](docs/benchmarks.md)                                 |
| Product positioning (games + assistants; limits)   | [positioning.md](docs/positioning.md)                               |
| App usage guide (demo shell)                       | [using-the-app.md](docs/using-the-app.md)                           |
| LAN API protocol (detailed)                        | [api-endpoint.md](docs/api-endpoint.md)                             |
| Console validation gates                           | [console-validation-runbook.md](docs/console-validation-runbook.md) |
| Crossbuild Linux → Xbox                            | [crossbuild-console.md](docs/crossbuild-console.md)                 |
| Release readiness                                  | [release-readiness.md](docs/release-readiness.md)                   |
| Privacy / data handling                            | [privacy.md](docs/privacy.md)                                       |
| Runtime NuGet pins                                 | [recommended-config.md](docs/recommended-config.md)                 |
| Technical report (frozen v1.0)                     | [technical-report.md](docs/technical-report.md)                     |
| Current research package and XAB                   | [paper/](paper/) · [bench/README.md](bench/README.md)               |

---

## Contributing

Areas of interest:

- UWP packaging, Xbox Dev Mode quirks
- Compact ONNX models fitting disk/GPU budgets
- Benchmark methodology and reproducibility
- Non-Windows developer documentation

See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Acknowledgements

- [`llama.cpp`](https://github.com/ggml-org/llama.cpp) — Georgi Gerganov
- [ONNX Runtime GenAI](https://github.com/microsoft/onnxruntime-genai) — Microsoft
- Xbox homebrew community
- Andrei David's `llama2.c` port to Xbox 360

## License

MIT. `llama.cpp` submodule under MIT.

---

## Support

[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-db61a2?logo=github)](https://github.com/sponsors/gianlucamazza)

xllama is free and MIT-licensed. If it's useful to you, you can support its
development on [GitHub Sponsors](https://github.com/sponsors/gianlucamazza) —
funds go to development time, Xbox Dev Mode costs, and model/storage budgets
for testing.

---

## Disclaimer

xllama is an independent open-source project by Gianluca Mazza. It is not affiliated with, endorsed, sponsored or approved by Microsoft. Xbox, Xbox Series S and Xbox Series X are trademarks of the Microsoft group of companies. Llama is a trademark of Meta; xllama is not affiliated with Meta or with the Ollama project. Other product names are trademarks of their respective owners and are used only to describe compatibility.

Independent project, not affiliated with or endorsed by Microsoft. Xbox and Xbox Series S|X are trademarks of the Microsoft group of companies.
