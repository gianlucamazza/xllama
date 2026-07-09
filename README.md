# xllama

> Local LLM chat + Stable-Diffusion image generation on Xbox Series S|X (UWP Dev Mode) — ONNX Runtime GenAI + DirectML.

[![build-uwp](https://github.com/gianlucamazza/xllama/actions/workflows/build-uwp.yml/badge.svg)](https://github.com/gianlucamazza/xllama/actions/workflows/build-uwp.yml)
[![build-linux](https://github.com/gianlucamazza/xllama/actions/workflows/build-linux.yml/badge.svg)](https://github.com/gianlucamazza/xllama/actions/workflows/build-linux.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Status:** v1.0 released · research-grade  
**Maintainer:** [Venere Labs](https://github.com/gianlucamazza)

---

## Table of Contents

- [What is this](#what-is-this)
- [About the name](#about-the-name)
- [Why Xbox Series S](#why-xbox-series-s)
- [Architecture](#architecture)
- [Repository layout](#repository-layout)
- [Build](#build)
- [Models](#models)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## What is this

`xllama` is a UWP application for Xbox Series S|X in Dev Mode that runs LLM chat and Stable-Diffusion image generation locally, with no cloud dependency and a gamepad-friendly UI.

The project started as a port of [`llama.cpp`](https://github.com/ggml-org/llama.cpp) (GGUF files, CPU-only), then migrated to **ONNX Runtime GenAI + DirectML**. The measured verdict is **per-workload** (see [docs/technical-report.md](./docs/technical-report.md)): the Zen 2 **CPU wins autoregressive decode** at this model scale (~66 tok/s, SmolLM2-360M int4), the RDNA 2 **GPU wins batch compute** — prefill at ~1k prompt tokens (1.8× faster) and image generation (**11.1×** on the diffusion workload: SD-Turbo 512×512 in ~7 s). The app routes per conversation (CPU / GPU / auto). The llama.cpp path is preserved for Linux development, CI, and a bench-only UWP variant (measured: decode parity with ORT, so ORT stays the text backend).

The default chat model is **SmolLM2-360M-Instruct INT4 CPU** (~403 MB), downloaded on first launch from the GitHub Release model catalogue — the MSIX itself is ~19 MB and ships no model.

Goals:

1. Demonstrate that modern consumer console hardware is a viable, underexplored substrate for local LLM inference.
2. Publish a clean, reproducible baseline that future work (gaming AI, on-device assistants) can build on.

This is a research-grade hobby project. "Xbox" is a Microsoft trademark; this project is not affiliated with Microsoft.

---

## Quick Start

**Requirements**: Xbox Series S or X in Dev Mode (one-time ~$19 activation), Linux or Windows host.

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

On first launch the app **downloads** the default model (SmolLM2-360M INT4, ~417 MB) from the [`models-v1` GitHub Release](https://github.com/gianlucamazza/xllama/releases/tag/models-v1) with a progress bar, then starts the chat UI. No model is bundled in the MSIX.

See [docs/install-release.md](./docs/install-release.md) to install a tagged release (cert + VCLibs + MSIX), [docs/using-the-app.md](./docs/using-the-app.md) for the app guide (chat, settings, image generation), and [docs/phase1-runbook.md](./docs/phase1-runbook.md) for the developer workflow.

---

## About the name

`xllama` predates the pivot to ONNX Runtime GenAI. The name is kept for continuity, not as a claim about the engine:

- **`x`** — Xbox (UWP target) and cross-platform (Linux CLI build).
- **`llama`** — local-LLM ecosystem at large. The Linux build still uses `llama.cpp`; the UWP build does not. Neither path ships LLaMA model weights — the default model is SmolLM2-360M-Instruct.

---

## Why Xbox Series S

- **Capable CPU**: 8 Zen 2 cores @ 3.6 GHz, AVX2, comparable to a Ryzen 7 3700X.
- **Modern GPU**: RDNA 2, ~4 TFLOPS FP32, with INT8/INT4 hardware support.
- **Unified memory**: 10 GB GDDR6 shared between CPU and GPU.
- **Accessible Dev Mode**: one-time ~$19 activation via Partner Center unlocks unsigned UWP deployment.
- **Underexplored**: no prior LLM port to the platform at time of writing.

**Measured performance (Xbox Series S, 2026-07-08):** chat decode **66.3 tok/s** (CPU int4, SmolLM2-360M, ORT GenAI 0.14.1); prefill at ~1k prompt tokens **354 tok/s on GPU fp16** vs 198 CPU (the crossover that motivates routing); KV-cache reuse makes turn-2 prefill **4.87×** faster; SD-Turbo generates a 512×512 image in **~6.9 s** on DirectML. Full matrices: `bench/results/` and [docs/technical-report.md](./docs/technical-report.md).

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
│  ├─ ORT GenAI 0.14.1 + ORT 1.24.4       │
│  ├─ DirectML 1.15.4 (app-local DLLs)    │
│  └─ Output: xllama_*.msix (~19 MB,      │
│     no model bundled)                    │
└──────────────────┬───────────────────────┘
                   │
                   ▼  sideload via Device Portal
┌──────────────────────────────────────────┐
│  Target: Xbox Series S|X (Dev Mode)      │
│  ├─ first launch: model download from    │
│  │  GitHub Release catalogue             │
│  ├─ chat decode → CPU EP (Zen 2)        │
│  ├─ long-prompt prefill → DML fp16      │
│  │  (per-conversation routing)           │
│  └─ image gen → SD-Turbo on DirectML    │
│     (headless flag mode, 887A0036 — §7)  │
└──────────────────────────────────────────┘
```

---

## Repository layout

```
xllama/
├── llama.cpp/              # upstream submodule (Linux path only)
├── include/xllama/         # shared public headers
│   ├── inference_params.h  # InferenceParams / InferenceResult
│   ├── inference.h         # run_inference, write_bench_csv
│   ├── session.h           # xllama::Session API (persistent model across turns)
│   ├── ort_raii.h          # RAII wrappers for OGA* types (UWP)
│   ├── llama_raii.h        # RAII wrappers for llama_* types (Linux)
│   ├── cli.h               # parse_cli_args (Linux)
│   ├── platform.h          # log_output, detect_threads, peak_working_set_mb
│   ├── path_utils.h        # resolve_model_path, resolve_local_path
│   └── utf8_utils.h        # utf8 <-> wstring (Windows)
├── src/
│   ├── main.cpp            # Linux entry point
│   └── bridge/             # shared implementation (Linux + UWP)
│       ├── inference.cpp   # #ifdef XLLAMA_USE_ORT → ORT GenAI; #else → llama_decode
│       ├── session.cpp     # xllama::Session (OrtSession + LlamaSession)
│       ├── bench.cpp
│       ├── platform.cpp
│       ├── path_utils.cpp
│       ├── cli.cpp
│       └── utf8_utils.cpp
├── uwp/                    # C++/WinRT UWP app
│   ├── App.cpp / App.h     # Application lifecycle, OnLaunched
│   ├── MainPage.cpp / .h   # MainPageController — programmatic UI (XAML-free)
│   ├── inference-bridge.cpp / .h  # UWP entry glue + bench mode main_loop()
│   ├── diffuse.cpp         # SD-Turbo diffusion pipeline (plain ORT DirectML)
│   ├── chat-history.cpp / .h      # ChatHistory: Save/Load/Delete/Clear
│   ├── model-downloader.cpp / .h  # EnsureModelAsync + LoadModelManifest (catalogue download)
│   ├── models/manifest.json # model catalogue (LocalState override supported)
│   ├── packages.config     # NuGet pins (ORT GenAI 0.14.1, ORT 1.24.4, DirectML 1.15.4)
│   ├── ggml-uwp.vcxproj    # static ggml+llama lib (bench-only llamacpp backend)
│   └── xllama.sln / .vcxproj
├── scripts/
│   ├── deploy.sh                      # Device Portal: deploy, logs, bench trigger
│   ├── build-uwp.ps1                  # Windows UWP packaging
│   ├── merge_onnx_external_data.py    # merge model.onnx.data → self-contained model.onnx
│   ├── bench-xbox.sh                  # automated benchmark runner
│   ├── install-latest-build.sh        # fetch + deploy latest CI artifact
│   ├── test-dml-config.sh             # upload DML provider_options without MSIX rebuild
│   ├── check-uwp-host.sh              # Linux host preflight
│   └── setup-windows-uwp-dev.ps1      # Windows VM setup
├── tests/                  # unit tests (doctest, incl. diffusion golden vectors)
├── bench/                  # benchmark configs + results
├── diffusion/              # SD-Turbo export/convert/validate toolchain (host)
├── patches/                # llama.cpp AppContainer guards (bench-only variant)
├── docs/                   # technical notes
├── cmake/                  # toolchain files
└── .github/workflows/      # CI: build-linux + build-uwp (default + llamacpp)
```

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

# Unit tests
cmake --preset linux-test
cmake --build build/linux-test -j
ctest --test-dir build/linux-test --output-on-failure
```

### Build for Xbox (Windows / CI)

The UWP package requires MSVC and the Windows SDK. Recommended path: push to `main` and download the `xllama-appx` artifact from the `build-uwp` GitHub Actions workflow. CI also builds `xllama-appx-llamacpp` — a **bench-only** variant with the static ggml/llama.cpp CPU text backend (`-Backend llamacpp`, `patches/`), kept for A/B benchmarking; measured decode parity with ORT, so ORT GenAI remains the shipping backend.

For local builds from a Windows VM, see [docs/windows-dev-vm.md](./docs/windows-dev-vm.md):

```powershell
.\scripts\build-uwp.ps1 -Configuration Release -Platform x64            # default (ORT GenAI)
.\scripts\build-uwp.ps1 -Configuration Release -Platform x64 -Backend llamacpp  # bench-only
```

### Deploy to console

```bash
source ~/.config/xllama/xbox-env   # sets XBOX_IP, XBOX_USER, XBOX_PASS
./scripts/deploy.sh path/to/xllama_*.msix
```

No model upload is required: the app downloads the default model on first launch from the GitHub Release catalogue. See [docs/install-release.md](./docs/install-release.md) (release install) and [docs/phase1-runbook.md](./docs/phase1-runbook.md) (developer workflow).

---

## Models

`xllama` on Xbox uses ONNX Runtime GenAI for text (model directories with `genai_config.json`, `model.onnx`, `tokenizer.json`) and plain ORT DirectML for diffusion (`text_encoder`/`unet`/`vae_decoder` components).

Models come from the catalogue `uwp/models/manifest.json` (assets hosted on the [`models-v1` GitHub Release](https://github.com/gianlucamazza/xllama/releases/tag/models-v1)): the app downloads any catalogue entry on demand, with the default chat model fetched on first launch. A `LocalState\manifest.json` uploaded via Device Portal overrides the catalogue without a reinstall — see [docs/model-selection.md](./docs/model-selection.md) for adding your own model.

| Model                          | Format        | Size    | Xbox UWP | Notes                                                            |
| ------------------------------ | ------------- | ------- | -------- | ---------------------------------------------------------------- |
| SmolLM2-360M-Instruct INT4 CPU | ONNX GenAI    | 417 MB  | ✅       | Default; decode 66.3 tok/s                                       |
| SmolLM2-360M-Instruct fp16 DML | ONNX GenAI    | ~700 MB | ✅       | Routing target (prefill 354 tok/s @1k)                           |
| SD-Turbo fp16 (image)          | ONNX DirectML | 2.4 GB  | ✅       | 512×512 in ~6.9 s ([diffusion/README.md](./diffusion/README.md)) |
| SmolLM2-1.7B-Instruct INT4 CPU | ONNX GenAI    | 1.4 GB  | ✅       | Measured 20.6 tok/s; tight on Dev Mode disk                      |
| Phi-3.5-mini CPU INT4          | ONNX GenAI    | ~2.7 GB | ❌       | Above the Dev Mode disk budget                                   |

Numbers are measured on Xbox Series S Dev Mode. See [docs/uwp-constraints.md](./docs/uwp-constraints.md) for the measured GPU budget (3801 MB), the disk budget, and the `weakly_canonical` AppContainer workaround.

---

## Limitations

- **Disk is the binding budget, not GPU memory**: the measured per-process GPU budget is **3801 MB** (package designated Game), but Dev Mode leaves only ~2.2–2.5 GB free on disk — that is what caps model size.
- **DirectML vs XAML (`887A0036`)**: ORT GenAI's DML device conflicts with the XAML compositor's D3D12 device in the same process, so GPU workloads run in headless flag modes (`bench.flag`, `diffuse.flag`); fix contributed upstream ([onnxruntime-genai#2280](https://github.com/microsoft/onnxruntime-genai/pull/2280)). Whether plain ORT DML (diffusion) shares the constraint is under test (`diffuse-inproc.flag`).
- **Sandboxed filesystem**: models are downloaded to `LocalState` or transferred via Device Portal / USB. No arbitrary path access.
- **AppContainer path traversal**: ORT 1.24.4 calls `std::filesystem::weakly_canonical()` for external ONNX data files, which traverses path segments the AppContainer cannot read. Workaround: distribute models with `model.onnx.data` merged into a self-contained `model.onnx` (`scripts/merge_onnx_external_data.py`).
- **No POSIX mmap / no `dlopen`**: NuGet-packaged ORT GenAI DLLs must be app-local (`DeploymentContent=true`); no system-wide DLL loading.
- **Dev Mode only**: no path to retail-mode consoles.

For full details see [docs/uwp-constraints.md](./docs/uwp-constraints.md).

---

## Roadmap

See [ROADMAP.md](./ROADMAP.md). Headlines:

1. **Phase 1 — CPU baseline** ✅ Working UWP, ORT GenAI, CI green.
2. **Phase 2 — GPU acceleration** ✅ GPU proven; verdict per-workload (CPU decode, GPU prefill/images).
3. **Phase 3 / 3.5 — Benchmarks + hardware ceiling** ✅ Measured matrices, routing, KV reuse, llama.cpp A/B (parity), diffusion on console.
4. **Phase 4 — In-app model download + publication** ✅ Catalogue download, v1.0.0 release, technical report (demo video pending).
5. **Phase 5 — Post-1.0 improvements** 🚧 In-proc diffusion experiment, interactive validations, upstream #2280.

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
