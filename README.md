# xllama

> Local LLM chat + Stable-Diffusion image generation on Xbox Series S|X (UWP Dev Mode) — llama.cpp (GGUF) + ONNX Runtime GenAI, CPU and DirectML.

[![build-uwp](https://github.com/gianlucamazza/xllama/actions/workflows/build-uwp.yml/badge.svg)](https://github.com/gianlucamazza/xllama/actions/workflows/build-uwp.yml)
[![build-linux](https://github.com/gianlucamazza/xllama/actions/workflows/build-linux.yml/badge.svg)](https://github.com/gianlucamazza/xllama/actions/workflows/build-linux.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Status:** shipping (unified + PatchedGenAI + PatchedOrt) · research-grade — current version in [CHANGELOG](CHANGELOG.md) / [ROADMAP](ROADMAP.md)
**Demo:** [xllama on Xbox Series S (v1.2.0)](https://github.com/gianlucamazza/xllama/releases/download/v1.2.0.0/xllama-demo-v1.2.0.mp4) (~74 s, local chat + SD image)
**Maintainer:** [Gianluca Mazza](https://github.com/gianlucamazza)

---

## Table of Contents

- [What is this](#what-is-this)
- [Quick Start](#quick-start)
- [About the name](#about-the-name)
- [Why Xbox Series S](#why-xbox-series-s)
- [Architecture](#architecture)
- [Repository layout](#repository-layout)
- [Build](#build)
- [Models](#models)
- [Training and personalization](#training-and-personalization)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Technical report](#technical-report)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## What is this

`xllama` is a UWP application for Xbox Series S|X in Dev Mode that runs LLM chat and Stable-Diffusion image generation locally, with no cloud dependency and a gamepad-friendly UI. It also includes an opt-in, OpenAI-compatible [LAN endpoint](./docs/api-endpoint.md) and a dual-lane [training/personalization pillar](./docs/training-architecture.md): host PEFT LoRA plus an in-process partial fine-tune that runs fully on the console (ggml-opt Lane B, host + console gates PASS).

The project started as a port of [`llama.cpp`](https://github.com/ggml-org/llama.cpp)
(GGUF, CPU-only), then migrated to **ONNX Runtime GenAI + DirectML**. The measured
verdict is **per-workload**: the Zen 2 **CPU wins autoregressive decode** at this
scale; the RDNA 2 **GPU wins batch compute** (long-prompt first-turn prefill TTFT
on the parity-validated DML text asset, and diffusion). GPU **text** routing is
allowlisted for `smollm2-360m-dml-fp16-v2` only ([#91](https://github.com/gianlucamazza/xllama/issues/91)
root cause and fix: [dml-rmsnorm-fix-runbook.md](./docs/dml-rmsnorm-fix-runbook.md)).
The llama.cpp path serves Linux, CI, and modern GGUF catalogue models on
`unified` builds. Narrative snapshot: [docs/technical-report.md](./docs/technical-report.md);
routing analysis: [docs/uwp-constraints.md §5d](./docs/uwp-constraints.md).

Default chat on **unified** shipping: **LFM2.5-350M** (catalogue download on first
launch; MSIX ships no model). ORT-only builds still default to SmolLM2-360M INT4.
Throughput and RAM figures: [docs/benchmarks.md](./docs/benchmarks.md).

The LAN endpoint is experimental, unauthenticated, foreground-only and disabled
by default; it shares the one loaded model with the chat UI (single
process-wide session owner — a request during a chat turn gets "busy"). Preference samples stay in the AppContainer until the operator pulls
them for host PEFT **or** the user runs **Settings → Train on my feedback**
(Phase 11): an in-process Lane B `partial_ft` that publishes a `personalized`
GGUF to the model picker (code complete; needs a base f16 GGUF on device).

Goals:

1. Demonstrate that modern consumer console hardware is a viable, underexplored substrate for local LLM inference.
2. Publish a clean, reproducible baseline that future work (gaming AI, on-device assistants) can build on.

This is a research-grade hobby project. "Xbox" is a Microsoft trademark; this project is not affiliated with Microsoft.

---

## Quick Start

**Requirements**: Xbox Series S or X in Dev Mode (one-time ~$19 activation), Linux or Windows host.

> ⚠️ **Upgrading from ≤1.4.x?** 1.5.0.0 changed the package identity
> (`VenereLabs.xllama` → `GianlucaMazza.xllama`): it installs as a new app and
> models/history do not carry over — see the migration steps in
> [docs/install-release.md](./docs/install-release.md).

```bash
# 1. Get the pre-built MSIX from the latest CI release
./scripts/install-latest-build.sh          # fetches from GitHub Actions, deploys via Device Portal

# 2. Or build from source (Windows host / CI)
git clone --recursive https://github.com/gianlucamazza/xllama.git
.\scripts\build-uwp.ps1 -Configuration Release -Platform x64

# 3. Deploy manually
source ~/.config/xllama/xbox-env           # sets XBOX_IP, XBOX_USER, XBOX_PASS
./scripts/deploy.sh path/to/xllama_*.msix
```

On first launch the app **downloads** the default chat model (**LFM2.5-350M** on
unified builds, ~229 MB catalogue download) from the
[`models-v1` GitHub Release](https://github.com/gianlucamazza/xllama/releases/tag/models-v1)
with a progress bar, then starts the chat UI. No model is bundled in the MSIX.
Other catalogue models (SmolLM2 CPU/GPU, Qwen3.5, Gemma, SD-Turbo) download on
demand from `models-v1` or Hugging Face. Download sizes:
`uwp/models/manifest.json` (`approx_bytes`).

> `install-latest-build.sh --bench` uploads `bench.flag` for a headless benchmark.
> The default install launches the normal UI.

See [docs/install-release.md](./docs/install-release.md) to install a tagged
release, [docs/using-the-app.md](./docs/using-the-app.md) for the app guide, and
[docs/device-portal.md](./docs/device-portal.md) plus
[bench/README.md](./bench/README.md) for the developer workflow.

---

## About the name

`xllama` predates the pivot to ONNX Runtime GenAI. The name is kept for continuity, not as a claim about the engine:

- **`x`** — Xbox (UWP target) and cross-platform (Linux CLI build).
- **`llama`** — local-LLM ecosystem at large. The Linux build still uses `llama.cpp` (GGUF). The UWP "unified" build also supports GGUF models via the catalogue (`kind: "gguf"`) using the shared `xllama::Session` API and layout-aware `Backend::Auto`. Neither path ships LLaMA model weights — the shipping default chat model is LFM2.5-350M (GGUF).

---

## Why Xbox Series S

- **Capable CPU**: 8 Zen 2 cores @ 3.6 GHz, AVX2, comparable to a Ryzen 7 3700X.
- **Modern GPU**: RDNA 2, ~4 TFLOPS FP32, with INT8/INT4 hardware support.
- **Unified memory**: 10 GB GDDR6 shared between CPU and GPU.
- **Accessible Dev Mode**: one-time ~$19 activation via Partner Center unlocks unsigned UWP deployment.
- **Underexplored**: I could not find a prior LLM runtime for Series S\|X when this
  started. The closest prior art is Andrei David's `llama2.c` port to the **Xbox 360**
  (January 2025) — a proof of concept on very different hardware, credited below.

**Measured performance (Xbox Series S):** LFM2.5-350M is the fastest catalogue
chat option; larger LFM tiers trade throughput for quality. KV-cache reuse cuts
turn-2 prefill sharply on CPU paths. GPU text routing accelerates **first-turn
TTFT** on long prompts (parity-validated `-v2` DML asset); from turn 2 the CPU
wins at every reachable length (DirectML has no KV reuse). DirectML also runs
diffusion (SD-Turbo). **Numbers and rows live only in**
[docs/benchmarks.md](./docs/benchmarks.md) (generated) and the analysis in
[docs/uwp-constraints.md §5](./docs/uwp-constraints.md) — do not restate them
elsewhere.

---

## Architecture

```
┌──────────────────────────────────────────┐
│  Host: Linux (development)               │
│  ├─ xllama-cli (llama.cpp, GGUF)         │
│  ├─ Unit tests (doctest)                 │
│  └─ Deploy scripts (Device Portal REST)  │
└──────────────────┬───────────────────────┘
                   │
                   ▼  CI build (GitHub Actions, windows-2022)
┌──────────────────────────────────────────┐
│  Build: MSVC + Windows SDK + NuGet       │
│  ├─ ORT GenAI + ORT + DirectML (pins)   │
│  │  see uwp/packages.config             │
│  └─ Output: xllama_*.msix (~19 MB,      │
│     no model bundled)                    │
└──────────────────┬───────────────────────┘
                   │
                   ▼  sideload via Device Portal
┌──────────────────────────────────────────┐
│  Target: Xbox Series S|X (Dev Mode)      │
│  ├─ first launch: model download from    │
│  │  GitHub Release catalogue             │
│  ├─ per-model backend dispatch:          │
│  │  • ORT chat → CPU EP (Zen 2)         │
│  │  • long-prompt turn-1 TTFT → DML fp16 │
│  │    (auto, -v2; §5d)                    │
│  │  • GGUF chat → llama.cpp CPU +       │
│  │    KV-reuse (unified build)           │
│  └─ image gen → SD-Turbo on DirectML    │
│     (in-process; plain ORT DML — §7)     │
└──────────────────────────────────────────┘
```

The component/dataflow map — module boundaries, runtime backend dispatch, chat
templates, KV-reuse, routing, model provisioning + quant auto-upgrade, the LAN
front-end, training/personalization, the membw micro-bench, and the diffusion pipeline — lives in
[docs/architecture.md](./docs/architecture.md).

---

## Repository layout

```
xllama/
├── include/xllama/   # public C++ contracts (host-testable, no WinRT)
├── src/bridge/       # shared implementation (Linux + UWP)
├── src/main.cpp      # Linux CLI entry
├── uwp/              # C++/WinRT app (UI, LAN API, headless flags)
├── training/         # jobs, host PEFT, datasets
├── bench/            # raw results + summary policy
├── diffusion/        # SD-Turbo host toolchain
├── scripts/          # deploy, bench, validate, package
├── tests/            # doctest (xllama-tests)
├── docs/             # SSOT map: docs/README.md
├── patches/          # AppContainer / runtime patches
├── llama.cpp/        # submodule (GGUF path)
└── .github/workflows/
```

File-level map for agents/contributors: **[AGENTS.md](./AGENTS.md)**.
Doc ownership (one fact, one home): **[docs/README.md](./docs/README.md)**.

---

## Build

### Linux (development + tests)

```bash
git clone --recursive https://github.com/gianlucamazza/xllama.git
cd xllama

# Release build
cmake --preset linux-release
cmake --build build/linux-release -j

# Run with a model (llama.cpp / GGUF, Linux only)
./build/linux-release/bin/xllama-cli -m models/smollm2-360m.gguf -p "Hello"

# Full sampling control, matching the GUI/API (one shared sampler since #125):
#   --temp --top-p --top-k --repetition-penalty --seed --system --chat --greedy
./build/linux-release/bin/xllama-cli -m models/smollm2-360m.gguf --chat \
  -p "Hello" --top-p 0.9 --top-k 40 --repetition-penalty 1.1 --system "Be terse"

# Unit tests
cmake --preset linux-test
cmake --build build/linux-test -j
ctest --test-dir build/linux-test --output-on-failure
```

### Build for Xbox (Windows / CI)

The UWP package requires MSVC and the Windows SDK. The shipping `xllama-appx`
artifact from `build-uwp.yml` is **unified**: ORT GenAI and llama.cpp coexist and
the catalogue selects the backend per model. CI also builds
`xllama-appx-llamacpp`, a bench-only llama.cpp lane.

For local builds from a Windows VM, see [docs/windows-dev-vm.md](./docs/windows-dev-vm.md):

```powershell
.\scripts\build-uwp.ps1 -Configuration Release -Platform x64 -Backend unified  # shipping shape
.\scripts\build-uwp.ps1 -Configuration Release -Platform x64                   # ORT-only local build
.\scripts\build-uwp.ps1 -Configuration Release -Platform x64 -Backend llamacpp # bench-only
```

### Deploy to console

```bash
source ~/.config/xllama/xbox-env   # sets XBOX_IP, XBOX_USER, XBOX_PASS
./scripts/deploy.sh path/to/xllama_*.msix
```

No model upload is required: the app downloads the default model on first launch
from the GitHub Release catalogue. See
[docs/install-release.md](./docs/install-release.md) for release installation and
[docs/console-validation-runbook.md](./docs/console-validation-runbook.md) for
the current developer gates.

---

## Models

Text backends: **ORT GenAI** (ONNX catalogue dirs) and **llama.cpp GGUF** on
`unified` builds; diffusion is plain ORT DirectML. Catalogue data lives in
`uwp/models/manifest.json` ([models-v1 Release](https://github.com/gianlucamazza/xllama/releases/tag/models-v1));
first launch downloads the default chat model. Roles and how to add a model:
[docs/model-selection.md](./docs/model-selection.md). Full tested/shipping
inventory: [docs/model-matrix.md](./docs/model-matrix.md). Operator defaults:
[docs/recommended-config.md](./docs/recommended-config.md). **Performance
numbers:** [docs/benchmarks.md](./docs/benchmarks.md) only.

| Role                                 | Catalogue id (headline)                             |
| ------------------------------------ | --------------------------------------------------- |
| Default chat (unified)               | `lfm25-350m`                                        |
| Balanced / quality chat              | `lfm25-1.2b-instruct`, `lfm2-2.6b`                  |
| Coding chat (GGUF · n_ctx 4096)      | `qwen25-coder-0.5b` / `1.5b` / `3b`                 |
| Chat upgrade (Qwen3)                 | `qwen3-1.7b`                                        |
| Reasoning (think stripped for UI)    | `lfm25-1.2b-thinking`                               |
| ORT CPU / DML routing base           | `smollm2-360m-cpu-int4`, `smollm2-360m-dml-fp16-v2` |
| Image gen                            | `sd-turbo-fp16`                                     |
| Personalized (after on-device train) | `personalized` (LocalState override)                |

Hardware and AppContainer limits: [docs/uwp-constraints.md](./docs/uwp-constraints.md).

---

## Training and personalization

- **Lane A** — host PEFT LoRA → merged GGUF
- **Lane B** — on-device last-block `partial_ft` (available; host + console gates PASS)
- **Lane C** — serve merged GGUF / runtime LoRA
- **Preferences** — Like/Dislike/Correct → `training/samples.jsonl`
- **Phase 11 UI** — Settings → **Train on my feedback** → catalogue id `personalized`

Contracts and RE matrix: [docs/training-architecture.md](./docs/training-architecture.md)
(§11 for the UI arc). Ops: [training/README.md](./training/README.md).
Pad steps: [docs/using-the-app.md](./docs/using-the-app.md).

---

## Limitations

Headlines only — full constraints SSOT is
[docs/uwp-constraints.md](./docs/uwp-constraints.md):

- **2 GB ONNX per-file / protobuf ceiling** (not the GPU budget alone) caps large
  self-contained graphs; see §8–§9.
- **Patched runtime DLLs** while NuGet lacks GenAI #2280 and ORT AppContainer
  fixes — lifecycle in [docs/vendor-lifecycle-plan.md](./docs/vendor-lifecycle-plan.md).
- **AppContainer** filesystem (LocalState / USB / Device Portal); no arbitrary
  paths, no POSIX mmap / `dlopen`.
- **Dev Mode only** — no retail-console path.

---

## Roadmap

Current work and open items: **[ROADMAP.md](./ROADMAP.md)**. Release history:
[CHANGELOG.md](./CHANGELOG.md). Phases 1–11 product code is complete; the
2026-07-25 performance campaign shipped GGUF prefill +62% (repack), the
threads-6 CPU asset, DML warm-up + session pre-load and a single process-wide
session owner. Remaining items are the #130 root-cause profile and upstream
pin drops.

---

## Technical report

The measured narrative (CPU/GPU verdicts, falsified hypotheses, diffusion on console)
lives in [docs/technical-report.md](./docs/technical-report.md) — a **frozen v1.0
snapshot** kept for its reasoning, not current metrics. Publication entrypoint:
[Discussion #76](https://github.com/gianlucamazza/xllama/discussions/76). Live numbers:
[docs/benchmarks.md](./docs/benchmarks.md).

---

## Contributing

Issues and PRs welcome. Areas where contributions are particularly useful:

- UWP packaging and Xbox Dev Mode quirks
- Compact ONNX models that fit the disk and GPU budgets — see [docs/model-selection.md](./docs/model-selection.md)
- Benchmark methodology and reproducibility
- Documentation for non-Windows developers

---

## Acknowledgements

- [`llama.cpp`](https://github.com/ggml-org/llama.cpp) by Georgi Gerganov and contributors.
- [ONNX Runtime GenAI](https://github.com/microsoft/onnxruntime-genai) by Microsoft.
- The Xbox homebrew community for Dev Mode and Device Portal documentation.
- Andrei David's `llama2.c` port to Xbox 360, which showed this class of project is worth doing.

---

## License

MIT. See [LICENSE](./LICENSE).

`llama.cpp` is included as a submodule under its own MIT license.
