# xllama

> Local inference on Xbox Series S|X — GGUF / llama.cpp, dual-backend dispatch
> (ORT GenAI + DirectML), OpenAI-compat LAN endpoint, diffusion, and on-device
> training. Built for games (NPC / dialogue / agents) and AI assistants.
> Chat UI is a demo shell. Architecture under real constraints (UWP, 10 GB RAM).

[![build-uwp](https://github.com/gianlucamazza/xllama/actions/workflows/build-uwp.yml/badge.svg)](https://github.com/gianlucamazza/xllama/actions/workflows/build-uwp.yml)
[![build-linux](https://github.com/gianlucamazza/xllama/actions/workflows/build-linux.yml/badge.svg)](https://github.com/gianlucamazza/xllama/actions/workflows/build-linux.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Sponsor](https://img.shields.io/badge/Sponsor-%E2%9D%A4-db61a2?logo=github)](https://github.com/sponsors/gianlucamazza)

<!-- XLLAMA_DOI_START -->

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22118437.svg)](https://doi.org/10.5281/zenodo.22118437)
<!-- XLLAMA_DOI_END -->

[CHANGELOG](CHANGELOG.md) · [ROADMAP](ROADMAP.md) · [Positioning](docs/positioning.md)

https://github.com/user-attachments/assets/9494a6a2-f14e-4229-afa4-ff5faf51ba67

The hosted clip above is the compact technical demo recorded on a Series S in
Dev Mode. The new [52.97-second product showcase](docs/screenshots/xllama-demo-v1.5.6-showcase.mp4)
keeps the meaningful states visible through the chat **demo shell**: local
inference, feedback, completed personalization, coding, and generated image. Its [12-minute raw capture](docs/screenshots/xllama-demo-v1.5.6-showcase-raw.mp4)
and [marker log](docs/screenshots/xllama-demo-v1.5.6-showcase-markers.jsonl)
remain available for audit. The original [technical raw capture](docs/screenshots/xllama-demo-v1.5.6.mp4)
is also retained; neither video supports a throughput claim.
Reproduce it from [`demo/demo-showcase-script.json`](demo/demo-showcase-script.json)
with `scripts/capture-demo-video.sh`.

---

## Quick install

```bash
# Download release assets or a specific CI artifact first:
# docs/install-release.md (same-identity updates preserve LocalState)

# Or build (Windows host)
git clone --recursive https://github.com/gianlucamazza/xllama.git
.\scripts\build-uwp.ps1 -Configuration Release -Platform x64

# Deploy
source ~/.config/xllama/xbox-env
./scripts/deploy.sh path/to/xllama_*.msix
```

First launch: downloads default model (~219 MB). No model bundled in MSIX.

**Linux dev:** `cmake --preset linux-release && cmake --build build/linux-release -j`

---

## What you can do

- **LAN API** — OpenAI-compat completions, GGUF embeddings, trusted catalogue
  pull, preferences and training status. Current integration probe for games
  and assistants; opt-in, default OFF.
- **Bench** — headless tok/s, membw, diskbw, gpubw, gpugemv, gpustep, ramceil probes, d3d12 backend selftest
- **Diffuse** — SD-Turbo on DirectML, in-process with XAML compositor
- **Train** — on-device partial FT (Lane B), host PEFT (Lane A), serve merged GGUF (Lane C)
- **Chat (demo shell)** — multi-turn pad UI with KV-reuse, thinking models,
  coding tier. Showcase, not the product thesis.

---

## Supported models

| Model       | Params | Decode          | Role                    |
| ----------- | ------ | --------------- | ----------------------- |
| LFM2.5-230M | 230M   | **132.4** tok/s | Floor (fastest, 236 MB) |
| LFM2.5-350M | 350M   | **101.6** tok/s | Default chat            |
| LFM2-2.6B   | 2.6B   | **18.4** tok/s  | Quality (H9 7/8)        |

Full catalogue + Phase 14 coding models: [model-matrix.md](docs/model-matrix.md) · [benchmarks.md](docs/benchmarks.md)

---

## LAN Endpoint — OpenAI-compatible

xllama exposes an HTTP endpoint on the local network that exposes its full
inference core (`SessionHub`) with OpenAI and Ollama-compatible APIs.
Status: v1, opt-in, default OFF. Dev Mode / LAN research only — the current
integration probe, not a public inbound service.

```bash
# Chat completions (non-streaming)
curl -s http://<xbox-ip>:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"lfm25-350m","messages":[{"role":"user","content":"hi"}]}'

# Model discovery (OpenAI shape)
curl -s http://<xbox-ip>:11434/v1/models
```

| Route                         | Shape      | Note                                        |
| ----------------------------- | ---------- | ------------------------------------------- |
| `POST /v1/chat/completions`   | OpenAI     | Non-streaming, single-slot mutex            |
| `POST /api/pull`              | Ollama     | Trusted catalogue download; NDJSON progress |
| `POST /api/embed`             | Ollama     | GGUF embeddings; scalar or batch input      |
| `POST /api/embeddings`        | Ollama     | Deprecated single-prompt embeddings         |
| `POST /v1/embeddings`         | OpenAI     | Float32 or base64 encoded vectors           |
| `GET /v1/models`              | OpenAI     | `"active": true` on the loaded model        |
| `GET /api/tags`               | Ollama     | Same list, Ollama shape                     |
| `POST /v1/preferences`        | Custom     | Append JSONL → `training/samples.jsonl`     |
| `GET /v1/training/status`     | Custom     | Snapshot of on-device train progress        |
| `POST /v1/images/generations` | OpenAI-ish | SD-Turbo, `b64_json` + `path`               |
| `GET /health`                 | Custom     | `{"status":"ok","service":"xllama"}`        |

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

| Path                     | What                                                                | Why                                                                              |
| ------------------------ | ------------------------------------------------------------------- | -------------------------------------------------------------------------------- |
| **ORT GenAI + DirectML** | ONNX models, CPU int4 decode + DML fp16 prefill                     | GPU wins batch compute, long-prompt TTFT                                         |
| **llama.cpp + GGUF**     | CPU by default, KV-reuse, repacked GEMM; opt-in d3d12 matmul layers | Zen2 wins decode at small sizes; own D3D12 kernels win on the 1.2B / 3B measured |

Unified build dispatches **per model at runtime** via `Backend::Auto`.
Llama.cpp is both benchmarking lane and shipping backend.

### Key invariants

- **One resident session** (`SessionHub`) — never 2× model in RAM
- **One budget enforcement point** (`fit_prompt`) — tokens, not chars
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

The versioned research report is archived through Zenodo after the release
gate. Until the first archive exists, cite the tagged GitHub repository:

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
GGUF decode also has an opt-in path on our own D3D12 kernels: a ggml backend
`d3d12`, enabled with `gguf_gpu_layers.txt`. It is faster than the CPU on the
1–3B models measured, with q8 numerics that match the CPU. The default stays
CPU until the per-model decision (D3); gates and per-model verdict are in
[docs/gguf-gpu-decode.md](docs/gguf-gpu-decode.md).

### Why single Session owner?

A CPU model and a DML model don't fit the budget together
([uwp-constraints.md](docs/uwp-constraints.md) §7). Two models = OOM.
`SessionHub` makes this a process-wide invariant, not a per-surface convention.

### Why token-budget, not chars-per-token?

Measured: prose 4.6 chars/token, dense C++ 2.5. A constant trades truncated
answers for history that would have fit. The estimate survives only for routing,
where being wrong costs a decision, not an answer.

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
| Store readiness                                    | [store-readiness.md](docs/store-readiness.md)                       |
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

## Support

xllama is free and MIT-licensed. If it's useful to you, you can support its
development on [GitHub Sponsors](https://github.com/sponsors/gianlucamazza) —
funds go to development time, Xbox Dev Mode costs, and model/storage budgets
for testing.

---

## Acknowledgements

- [`llama.cpp`](https://github.com/ggml-org/llama.cpp) — Georgi Gerganov
- [ONNX Runtime GenAI](https://github.com/microsoft/onnxruntime-genai) — Microsoft
- Xbox homebrew community
- Andrei David's `llama2.c` port to Xbox 360

## License

MIT. `llama.cpp` submodule under MIT.
