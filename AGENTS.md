# AGENTS.md — xllama

Quick reference for AI agents and new contributors.

## Conventions

- **Language**: English for code, comments, filenames, commit messages.
- **C++ standard**: C++17 (UWP / SDK 22621 + MSVC).
- **Style**: concise, no over-engineering. Prefer RAII and `unique_ptr`.
- **Formatting**: auto-format on save if possible; otherwise follow existing style.

## Directory structure

```
xllama/
├── include/xllama/          # Shared public headers
│   ├── inference_params.h   # InferenceParams / InferenceResult
│   ├── inference.h          # run_inference, write_bench_csv
│   ├── session.h            # xllama::Session API (persistent model across turns)
│   ├── ort_raii.h           # RAII unique_ptr for OGA* types (UWP/ORT GenAI path)
│   ├── llama_raii.h         # RAII unique_ptr for llama_* types (Linux path)
│   ├── cli.h                # parse_cli_args (Linux)
│   ├── platform.h           # log_output, detect_threads, peak_working_set_mb
│   ├── path_utils.h         # resolve_model_path, resolve_local_path
│   └── utf8_utils.h         # utf8 <-> wstring (Windows)
├── src/bridge/              # Shared implementation (Linux + UWP)
│   ├── inference.cpp        # #ifdef XLLAMA_USE_ORT → ORT GenAI; #else → llama_decode
│   ├── session.cpp          # xllama::Session (OrtSession UWP + LlamaSession Linux)
│   ├── bench.cpp            # bench CSV writer
│   ├── platform.cpp         # log_output (writes xllama.log in UWP)
│   ├── path_utils.cpp       # resolve_model_path: LocalState\models\ + InstalledPath fallback
│   ├── utf8_utils.cpp
│   └── cli.cpp
├── src/main.cpp             # Linux entry point (getopt_long)
├── uwp/                     # C++/WinRT UWP app
│   ├── App.cpp / App.h      # Application::OnLaunched
│   ├── MainPage.cpp / .h    # MainPageController (plain C++, not runtimeclass)
│   ├── inference-bridge.cpp / .h   # UWP main_loop() + bench mode
│   ├── chat-history.cpp / .h       # ChatHistory: Save/Load/Delete/Clear
│   ├── model-downloader.cpp / .h   # EnsureModelAsync — catalogue download (GitHub Release models-v1, models/manifest.json)
│   ├── packages.config      # NuGet pins (ORT GenAI 0.14.1, ORT 1.24.4, DirectML 1.15.4)
│   └── xllama.sln / .vcxproj
├── scripts/
│   ├── deploy.sh                      # Device Portal: deploy, logs, bench trigger
│   ├── build-uwp.ps1                  # Windows UWP packaging script
│   ├── merge_onnx_external_data.py    # merge model.onnx.data → self-contained model.onnx
│   ├── bench-xbox-ort.sh              # benchmark runner (ORT GenAI; model already on device)
│   ├── install-latest-build.sh        # fetch + deploy latest CI artifact
│   ├── test-dml-config.sh             # upload DML provider_options without MSIX rebuild
│   ├── check-uwp-host.sh              # Linux host preflight (qemu, libvirt)
│   └── setup-windows-uwp-dev.ps1      # Windows VM: install VS2022 + UWP workload
├── tests/                   # Unit tests (doctest, target: xllama-tests)
├── bench/                   # Benchmark configs, prompts, results
├── docs/                    # Technical notes (see docs/README.md)
├── cmake/                   # Toolchain files
└── .github/workflows/       # build-linux.yml + build-uwp.yml
```

## Build

### Linux (development + tests)

```bash
# Release
cmake --preset linux-release
cmake --build build/linux-release -j$(nproc)

# Debug with tests
cmake --preset linux-test
cmake --build build/linux-test -j$(nproc)
ctest --test-dir build/linux-test --output-on-failure

# Smoke test
./build/linux-release/bin/xllama-cli --help
```

### UWP (Windows / CI)

Recommended: push to `main` and download the `xllama-appx` artifact from the `build-uwp` GitHub Actions workflow.

For local builds (requires a Windows VM — see `docs/windows-dev-vm.md`):

```powershell
.\scripts\build-uwp.ps1 -Configuration Release -Platform x64
```

Use `-ForceNewCert` only to regenerate the test signing certificate.

## Tests

- Framework: **doctest** (header-only, fetched via CMake).
- Target: `xllama-tests`.
- Command: `ctest --test-dir build/linux-test --output-on-failure`.
- Add tests in `tests/test_*.cpp`.

## Critical notes

- **Never commit `.env`** (contains credentials). Use `.env.example` as template.
- **Never commit `.pfx` / `.cer` certificates**. They are in `.gitignore`.

- **ORT GenAI path**: UWP inference is entirely under `#ifdef XLLAMA_USE_ORT` in `src/bridge/inference.cpp`. ORT types are wrapped in `include/xllama/ort_raii.h`. Linux path (`#else`) uses llama.cpp unchanged.

- **No model in the MSIX**: the package is ~19 MB and ships no model. On first launch the app downloads the default model (`smollm2-360m-cpu-int4`) from the GitHub Release `models-v1` catalogue (`uwp/models/manifest.json`) into `LocalState\models\`. No manual upload is needed for the standard dev flow.

- **ONNX external data merge**: ORT 1.24.4 calls `std::filesystem::weakly_canonical()` for models with a separate `.onnx.data` file, which traverses path segments inaccessible inside the Xbox AppContainer (`Q:\Users\UserMgr0\...`). Fix: merge external data into a single `model.onnx` using `scripts/merge_onnx_external_data.py` before MSIX packaging. CI does this automatically.

- **app-local DLLs**: `DirectML.dll`, `onnxruntime.dll`, `onnxruntime-genai.dll` must have `<DeploymentContent>true</DeploymentContent>` in the vcxproj. Without this the MSIX silently omits them and the app crashes on load.

- **UWP constraints**: no POSIX mmap, no dlopen, no registry, no thread-affinity desktop API. See `docs/uwp-constraints.md` for the full list.
