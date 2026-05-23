# xllama

> Local LLM inference on Xbox Series S|X (UWP Dev Mode) — ONNX Runtime GenAI on Zen 2 CPU.

**Status:** experimental · early development
**License:** MIT
**Maintainer:** [Venere Labs](https://github.com/gianlucamazza)

---

## What is this

`xllama` is a UWP application for Xbox Series S|X in Dev Mode that runs LLM inference locally, with no cloud dependency and a gamepad-friendly UI.

The project started as a port of [`llama.cpp`](https://github.com/ggml-org/llama.cpp) (GGUF files, CPU-only), then migrated to **ONNX Runtime GenAI + DirectML** to target the Xbox GPU. After confirming that the UWP GPU memory pool on Series S is ~768 MB — too small for any usable LLM — the active backend is now **CPU EP** (Zen 2, 8 cores). The llama.cpp path is preserved for Linux development and CI.

The bundled model is **SmolLM2-360M-Instruct INT4 CPU** (~403 MB), chosen to fit within the Xbox's available storage and RAM envelope.

Goals:

1. Demonstrate that modern consumer console hardware is a viable, underexplored substrate for local LLM inference.
2. Publish a clean, reproducible baseline that future work (gaming AI, on-device assistants) can build on.

This is a research-grade hobby project. "Xbox" is a Microsoft trademark; this project is not affiliated with Microsoft.

---

## About the name

`xllama` predates the pivot to ONNX Runtime GenAI. The name is kept for continuity, not as a claim about the engine:

- **`x`** — Xbox (UWP target) and cross-platform (Linux CLI build).
- **`llama`** — local-LLM ecosystem at large. The Linux build still uses `llama.cpp`; the UWP build does not. Neither path ships LLaMA model weights — the bundled model is SmolLM2-360M-Instruct.

---

## Why Xbox Series S

- **Capable CPU**: 8 Zen 2 cores @ 3.6 GHz, AVX2, comparable to a Ryzen 7 3700X.
- **Modern GPU**: RDNA 2, ~4 TFLOPS FP32, with INT8/INT4 hardware support.
- **Unified memory**: 10 GB GDDR6 shared between CPU and GPU.
- **Accessible Dev Mode**: one-time ~$19 activation via Partner Center unlocks unsigned UWP deployment.
- **Underexplored**: no prior LLM port to the platform at time of writing.

**Current performance (CPU EP, SmolLM2-360M INT4):** ~71 tok/s decode at 4 threads (bench v0.3.1; peak 771 MB RAM). Optimal: `intra_op_num_threads=4` — explicit 8-thread setting causes severe regression (~24 tok/s) due to memory bandwidth saturation on Zen 2. See `bench/results/phase1-cpu.csv` for full results.

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
│  ├─ ORT GenAI 0.13.2 + ORT 1.24.4       │
│  ├─ DirectML 1.15.4 (app-local DLLs)    │
│  ├─ SmolLM2-360M INT4 merged into MSIX   │
│  └─ Output: xllama_*.msix               │
└──────────────────┬───────────────────────┘
                   │
                   ▼  sideload via Device Portal
┌──────────────────────────────────────────┐
│  Target: Xbox Series S|X (Dev Mode)      │
│  ├─ ORT GenAI → CPU EP (Zen 2)          │
│  └─ DirectML EP: blocked by GPU OOM     │
│     (UWP pool ~768 MB, LLM > 300 MB)    │
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
│   ├── chat-history.cpp / .h      # ChatHistory: Save/Load/Delete/Clear
│   ├── model-downloader.cpp / .h  # EnsureModelAsync — HF chunked download (Exp 2)
│   ├── packages.config     # NuGet pins (ORT GenAI 0.13.2, ORT 1.24.4, DirectML 1.15.4)
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
├── tests/                  # unit tests (doctest)
├── bench/                  # benchmark configs + results
├── docs/                   # technical notes
├── cmake/                  # toolchain files
└── .github/workflows/      # CI: build-linux + build-uwp
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

The UWP package requires MSVC and the Windows SDK. Recommended path: push to `main` and download the `xllama-appx` artifact from the `build-uwp` GitHub Actions workflow.

For local builds from a Windows VM, see [docs/windows-dev-vm.md](./docs/windows-dev-vm.md):

```powershell
.\scripts\build-uwp.ps1 -Configuration Release -Platform x64
```

### Deploy to console

```bash
source ~/.config/xllama/xbox-env   # sets XBOX_IP, XBOX_USER, XBOX_PASS
./scripts/deploy.sh path/to/xllama_*.msix
```

The model is bundled inside the MSIX — no separate upload required. See [docs/phase1-runbook.md](./docs/phase1-runbook.md) for the full workflow.

---

## Models

`xllama` on Xbox uses ONNX Runtime GenAI. Models are directories containing `genai_config.json`, `model.onnx`, `tokenizer.json`, and related files — not single `.gguf` files.

The MSIX bundles **SmolLM2-360M-Instruct INT4 CPU** (403 MB on-disk, merged into a self-contained `model.onnx` for AppContainer compatibility). The model is placed under `Package.InstalledPath\models\smollm2-360m-cpu-int4\` and is copied to `LocalState\models\` on first launch.

| Model | Format | Size | Xbox UWP | Notes |
|-------|--------|------|----------|-------|
| SmolLM2-360M-Instruct INT4 CPU | ONNX GenAI | 403 MB | ✅ | Active; bundled in MSIX |
| SmolLM2-1.7B-Instruct INT4 CPU | ONNX GenAI | 1.4 GB | ⚠ | Above disk budget |
| Phi-3.5-mini CPU INT4 | ONNX GenAI | ~2.7 GB | ❌ | Disk full on Xbox Series S |
| Phi-3.5-mini GPU INT4 | ONNX GenAI DirectML | ~2.2 GB | ❌ | GPU OOM (pool ~768 MB) |

Numbers are measured on Xbox Series S Dev Mode. See [docs/uwp-constraints.md](./docs/uwp-constraints.md) for the GPU pool limit and the `weakly_canonical` AppContainer workaround.

---

## Limitations

- **GPU pool ~768 MB**: UWP apps on Xbox Series S have ~768 MB of GPU-accessible memory. Any LLM larger than ~300 MB on-device triggers an OOM in `OgaCreateModel` (SEH `0xC0000005`). DirectML EP is not viable today.
- **Sandboxed filesystem**: models must be pre-loaded into the MSIX or transferred via Device Portal. No arbitrary path access.
- **AppContainer path traversal**: ORT 1.24.4 calls `std::filesystem::weakly_canonical()` for external ONNX data files, which traverses path segments the AppContainer cannot read. Workaround: merge `model.onnx.data` into `model.onnx` at MSIX build time (`scripts/merge_onnx_external_data.py`).
- **No POSIX mmap / no `dlopen`**: NuGet-packaged ORT GenAI DLLs must be app-local (`DeploymentContent=true`); no system-wide DLL loading.
- **Dev Mode only**: no path to retail-mode consoles.

For full details see [docs/uwp-constraints.md](./docs/uwp-constraints.md).

---

## Roadmap

See [ROADMAP.md](./ROADMAP.md). Headlines:

1. **Phase 1 — CPU baseline** ✅ Working UWP, ORT GenAI, SmolLM2-360M bundled, CI green.
2. **Phase 2 — GPU acceleration** 🚫 Blocked: UWP GPU pool too small for LLM inference.
3. **Phase 3 — Benchmarks + model exploration** Populate results, tune n_threads, try sub-400 MB models.
4. **Phase 4 — In-app model download + publication** ModelSpec multi-file ONNX, demo, technical report.

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
