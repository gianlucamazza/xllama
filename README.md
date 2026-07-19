# xllama

> Local LLM chat + Stable-Diffusion image generation on Xbox Series S|X (UWP Dev Mode) — ONNX Runtime GenAI + DirectML.

[![build-uwp](https://github.com/gianlucamazza/xllama/actions/workflows/build-uwp.yml/badge.svg)](https://github.com/gianlucamazza/xllama/actions/workflows/build-uwp.yml)
[![build-linux](https://github.com/gianlucamazza/xllama/actions/workflows/build-linux.yml/badge.svg)](https://github.com/gianlucamazza/xllama/actions/workflows/build-linux.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Status:** shipping (unified + PatchedGenAI + PatchedOrt) · research-grade — current version in [CHANGELOG](CHANGELOG.md) / [ROADMAP](ROADMAP.md)
**Demo:** [xllama on Xbox Series S (v1.2.0)](https://github.com/gianlucamazza/xllama/releases/download/v1.2.0.0/xllama-demo-v1.2.0.mp4) (~74 s, local chat + SD image)
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

`xllama` is a UWP application for Xbox Series S|X in Dev Mode that runs LLM chat and Stable-Diffusion image generation locally, with no cloud dependency and a gamepad-friendly UI. It also includes an opt-in, OpenAI-compatible [LAN endpoint](./docs/api-endpoint.md) and a dual-lane [training/personalization pillar](./docs/training-architecture.md): host PEFT LoRA plus an experimental in-process partial fine-tune that runs fully on the console (ggml-opt Lane B).

The project started as a port of [`llama.cpp`](https://github.com/ggml-org/llama.cpp) (GGUF files, CPU-only), then migrated to **ONNX Runtime GenAI + DirectML**. The measured verdict is **per-workload** (see [docs/technical-report.md](./docs/technical-report.md)): the Zen 2 **CPU wins autoregressive decode** at this model scale (~66 tok/s, SmolLM2-360M int4), the RDNA 2 **GPU wins batch compute** — prefill at ~1k prompt tokens (1.8× faster) and image generation (**11.1×** on the diffusion workload: SD-Turbo 512×512 in ~7 s). GPU **text** routing is enabled for the parity-validated `smollm2-360m-dml-fp16-v2` asset only ([#91](https://github.com/gianlucamazza/xllama/issues/91): the DML `(Skip)SimplifiedLayerNormalization` kernel computes wrong results on the Series S GPU — fixed by decomposing those nodes into primitives, a data-only graph fix; see [docs/dml-rmsnorm-fix-runbook.md](./docs/dml-rmsnorm-fix-runbook.md)); diffusion stays on the GPU. The llama.cpp path serves Linux development, CI, and — in the `unified` UWP build — modern GGUF-only models (Qwen3.5, LFM2.5); for the same model, ORT stays the text backend (measured decode parity, better prefill).

The default chat model on the shipping **unified** build is **LFM2.5-350M Q4_K_M** (~219 MB, ~94 tok/s), downloaded on first launch from the GitHub Release model catalogue — the MSIX itself is ~19 MB and ships no model. ORT-only builds still default to SmolLM2-360M INT4.

The LAN endpoint is experimental, unauthenticated, foreground-only and disabled
by default. Personalization is operator-driven: preference samples remain in
the AppContainer until explicitly pulled to a host for retraining — or consumed
on-device by a Lane B `partial_ft` job (experimental, gates pending).

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

On first launch the app **downloads** the default chat model (**LFM2.5-350M** on unified builds, ~219 MB) from the [`models-v1` GitHub Release](https://github.com/gianlucamazza/xllama/releases/tag/models-v1) with a progress bar, then starts the chat UI. No model is bundled in the MSIX. Other catalogue models (SmolLM2 CPU/GPU, Qwen3.5, Gemma, SD-Turbo) download on demand from `models-v1` or Hugging Face.

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
- **Underexplored**: no prior LLM port to the platform at time of writing.

**Measured performance (Xbox Series S):** LFM2.5-350M is the fastest chat option
at **94.2 tok/s**; the larger LFM catalogue tiers trade throughput for quality.
KV-cache reuse improves measured turn-2 prefill by up to **20.0×**. GPU text
routing is live for the parity-validated `-v2` DML asset (auto above 600
tokens, **234 tok/s** prefill — #91 postmortem); DirectML also
serves diffusion, with SD-Turbo at about **6.9 s** for 512×512. Raw evidence,
atomic comparison rows and the generated dashboard are described in the
**[benchmark SSOT](./docs/benchmarks.md)** and
[comparative charts](./docs/benchmarks-charts.html).

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
│  ├─ per-model backend dispatch:          │
│  │  • ORT chat → CPU EP (Zen 2)         │
│  │  • long-prompt prefill → DML fp16    │
│  │    (auto routing > 600 tok, -v2)      │
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
├── llama.cpp/              # upstream submodule (Linux path only)
├── include/xllama/         # shared public headers
│   ├── inference_params.h  # InferenceParams / InferenceResult
│   ├── inference.h         # run_inference, write_bench_csv
│   ├── session.h           # xllama::Session API (persistent model across turns)
│   ├── training_params.h   # TrainingJob / TrainingCapability contracts
│   ├── training.h          # job validation, capability matrix, stage names
│   ├── device_train.h      # Lane B in-process partial-FT engine (ggml-opt)
│   ├── ort_raii.h          # RAII wrappers for OGA* types (UWP)
│   ├── llama_raii.h        # RAII wrappers for llama_* types (Linux)
│   ├── cli.h               # parse_cli_args (Linux)
│   ├── platform.h          # log_output, detect_threads(_llama), peak_working_set_mb
│   ├── path_utils.h        # resolve_model_path, first_gguf_in_dir, backend sniffing
│   └── utf8_utils.h        # utf8 <-> wstring (Windows)
├── src/
│   ├── main.cpp            # Linux entry point
│   └── bridge/             # shared implementation (Linux + UWP)
│       ├── inference.cpp   # #ifdef XLLAMA_USE_ORT → ORT GenAI; #else → llama_decode
│       ├── session.cpp     # xllama::Session (OrtSession + LlamaSession)
│       ├── training.cpp    # job parsing/validation + capability table
│       ├── device_train.cpp # Lane B engine: prepare → train → export → evaluate
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
│   ├── api-server.cpp / .h        # opt-in OpenAI-compatible LAN front-end
│   ├── model-downloader.cpp / .h  # ModelDownloader::DownloadAsync + LoadModelManifest (catalogue download)
│   ├── models/manifest.json # model catalogue (LocalState override supported)
│   ├── packages.config     # NuGet runtime pins (see docs/recommended-config.md)
│   ├── ggml-uwp.vcxproj    # static ggml+llama lib (shipping GGUF backend in `unified`; sole backend in `llamacpp` variant)
│   └── xllama.sln / .vcxproj
├── scripts/
│   ├── deploy.sh                      # Device Portal: deploy, logs, bench trigger
│   ├── build-uwp.ps1                  # Windows UWP packaging
│   ├── merge_onnx_external_data.py    # merge model.onnx.data → self-contained model.onnx
│   ├── bench-xbox-ort.sh              # automated benchmark orchestrator
│   ├── generate-benchmark-summary.py  # raw results → docs table + dashboard
│   ├── install-latest-build.sh        # fetch + deploy latest CI artifact
│   ├── test-dml-config.sh             # upload DML provider_options without MSIX rebuild
│   ├── check-uwp-host.sh              # Linux host preflight
│   └── setup-windows-uwp-dev.ps1      # Windows VM setup
├── tests/                  # unit tests (doctest, incl. diffusion golden vectors)
├── bench/                  # benchmark configs, raw results + comparison policy
├── diffusion/              # SD-Turbo export/convert/validate toolchain (host)
├── training/               # train jobs (host PEFT + device partial FT), datasets, ops
├── patches/                # AppContainer/runtime patches used by UWP builds
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

`xllama` on Xbox uses ONNX Runtime GenAI for text (model directories with `genai_config.json`, `model.onnx`, `tokenizer.json`) and plain ORT DirectML for diffusion (`text_encoder`/`unet`/`vae_decoder` components).

Models come from the catalogue `uwp/models/manifest.json` (assets hosted on the [`models-v1` GitHub Release](https://github.com/gianlucamazza/xllama/releases/tag/models-v1)): the app downloads any catalogue entry on demand, with the default chat model fetched on first launch. A `LocalState\manifest.json` uploaded via Device Portal is merged per-entry into the catalogue without a reinstall — see [docs/model-selection.md](./docs/model-selection.md) for adding your own model.

Catalogue overview (roles + sizes; **decode/prefill/RAM numbers live in
[docs/benchmarks.md](./docs/benchmarks.md)**, the perf SSOT):

| Model                      | Format        | Size    | Xbox UWP | Role                                                                            |
| -------------------------- | ------------- | ------- | -------- | ------------------------------------------------------------------------------- |
| LFM2.5-350M Q4_K_M         | GGUF          | 219 MB  | ✅       | **Default chat** on unified shipping; fastest+lightest                          |
| LFM2.5-1.2B Q4_K_M         | GGUF          | 697 MB  | ✅       | Balanced chat: 37.9 tok/s, 811 MB peak, H9 6/8                                  |
| LFM2-2.6B Q4_K_M           | GGUF          | 1.46 GB | ✅       | Quality chat: 18.4 tok/s, 1623 MB peak, H9 7/8                                  |
| SmolLM2-360M-Instruct INT4 | ONNX GenAI    | 417 MB  | ✅       | ORT default / routing base (CPU)                                                |
| SmolLM2-360M fp16 DML v2   | ONNX GenAI    | ~725 MB | ✅       | Routing target (auto >600 tok) — RMSNorm-decomposed graph, #91 parity-validated |
| SmolLM2-1.7B-Instruct INT4 | ONNX GenAI    | 1.4 GB  | ✅       | Larger CPU chat (~20.6 tok/s; in-app `models-v1` download)                      |
| Qwen3.5-0.8B Q4_K_M        | GGUF          | 508 MB  | ✅       | `unified` builds (llama.cpp)                                                    |
| Gemma-3-270M Q4_K_M        | GGUF          | 253 MB  | ✅       | `unified` builds; fast, tiny                                                    |
| Gemma-4-E2B Q3_K_S         | GGUF          | 2.45 GB | ✅       | `unified` builds; heavy/advanced                                                |
| Llama-3.2-3B Q3_K_S        | GGUF          | 1.54 GB | ✅       | `unified` builds; Phase 7 dense-3B comparator (CPU-only, advanced)              |
| SD-Turbo fp16 (image)      | ONNX DirectML | 2.4 GB  | ✅       | Image gen ([diffusion/README.md](./diffusion/README.md))                        |
| Phi-3.5-mini CPU INT4      | ONNX GenAI    | ~2.7 GB | ❌       | Not attempted (>2 GB single-file ONNX, §8)                                      |
| Phi-3.5-mini Q3_K_S        | GGUF          | 1.68 GB | ⚠️       | H4 A/B measured 11.3 tok/s / 2453 MB; loses to `llama32-3b` — not catalogue     |

See [docs/uwp-constraints.md](./docs/uwp-constraints.md) for the measured GPU budget (3801 MB), the disk budget, and AppContainer workarounds.

---

## Limitations

- **File-size ceiling, not GPU memory**: the measured per-process GPU budget is **3801 MB** (package designated Game), and the Dev Mode disk allocation is raised to **90 GB** via Dev Home — what caps model choice now is the **2 GB ONNX protobuf / per-file limit** (self-contained `model.onnx` must stay under it; see [docs/uwp-constraints.md](./docs/uwp-constraints.md) §8–§9).
- **DirectML vs XAML (`887A0036`)**: vanilla NuGet ORT GenAI DML conflicts with the XAML compositor's D3D12 device. Fix [#2280](https://github.com/microsoft/onnxruntime-genai/pull/2280) is **merged on Microsoft `main`** but **not** in NuGet **0.14.1** — shipping MSIX overlays a pinned patched `onnxruntime-genai.dll` from [`vendor-dlls-v1`](https://github.com/gianlucamazza/xllama/releases/tag/vendor-dlls-v1). Image generation uses plain ORT DML and runs **in-process**. Headless `bench.flag` remains for automation.
- **Sandboxed filesystem**: models are downloaded to `LocalState` or transferred via Device Portal / USB. No arbitrary path access.
- **AppContainer path traversal / large external data**: ORT 1.24.4 `weakly_canonical()` + large `ReadFile` chunks break models with external `.onnx.data` in the AppContainer. Shipping MSIX includes a **patched `onnxruntime.dll`** (same `vendor-dlls-v1` pin: path guard + 16 MB ReadFile chunks). Prefer merging external data under the 2 GB protobuf ceiling (`scripts/merge_onnx_external_data.py`); Fix B covers un-mergeable >2 GB files. See [docs/fp16-extdata-runbook.md](./docs/fp16-extdata-runbook.md) and [docs/uwp-constraints.md](./docs/uwp-constraints.md) §8.
- **No POSIX mmap / no `dlopen`**: NuGet-packaged ORT GenAI DLLs must be app-local (`DeploymentContent=true`); no system-wide DLL loading.
- **Dev Mode only**: no path to retail-mode consoles.

For full details see [docs/uwp-constraints.md](./docs/uwp-constraints.md).

---

## Roadmap

See [ROADMAP.md](./ROADMAP.md). Headlines:

1. **Phase 1 — CPU baseline** ✅ Working UWP, ORT GenAI, CI green.
2. **Phase 2 — GPU acceleration** ✅ GPU proven; verdict per-workload (CPU decode, GPU prefill/images).
3. **Phase 3 / 3.5 — Benchmarks + hardware ceiling** ✅ Measured matrices, routing, KV reuse, llama.cpp A/B (parity), diffusion on console.
4. **Phase 4 — In-app model download + publication** ✅ Catalogue download, release and technical report.
5. **Phase 5 — Post-1.0 improvements** ✅ Unified+PatchedGenAI shipping; console gates ALL PASS.
6. **Phase 6 — Publication + polish** ✅ Patched runtimes, LFM default,
   v1.2.0 release and [published demo](https://github.com/gianlucamazza/xllama/releases/download/v1.2.0.0/xllama-demo-v1.2.0.mp4).
7. **Phase 7 — Model research** 🔮 peer-class model campaigns continue.
8. **Phase 8 — Training pillar** ✅ Host PEFT, runtime LoRA, preference capture
   and console validation are frozen complete.
9. **Phase 9 — Personalization ops** ✅ Operator pull/retrain/publish loop
   and per-response preference UI delivered. See [ROADMAP.md](./ROADMAP.md).
10. **Phase 10 — Device partial FT** 🧪 Experimental ggml-opt Lane B;
    host and console marker gates are pending.

---

## Technical report

The measured narrative (CPU/GPU verdicts, falsified hypotheses, diffusion on console)
lives in [docs/technical-report.md](./docs/technical-report.md). Publication entrypoint:
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
