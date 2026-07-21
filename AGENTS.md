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
│   ├── training_params.h    # TrainingJob / TrainingCapability (training pillar)
│   ├── training.h           # validate/load job, capability matrix, stage names
│   ├── routing_policy.h     # routing decision + prompt budget (threshold must stay under it)
│   ├── sampling.h           # sampling defaults shared by CLI/bench and GUI/API
│   ├── session.h            # xllama::Session API (persistent model across turns)
│   ├── ort_raii.h           # RAII unique_ptr for OGA* types (UWP/ORT GenAI path)
│   ├── llama_raii.h         # RAII unique_ptr for llama_* types (Linux path)
│   ├── cli.h                # parse_cli_args (Linux)
│   ├── platform.h           # log_output, detect_threads(_llama), peak_working_set_mb
│   ├── path_utils.h         # resolve_model_path, first_gguf_in_dir, model_uses_llama_backend
│   └── utf8_utils.h         # utf8 <-> wstring (Windows)
├── src/bridge/              # Shared implementation (Linux + UWP)
│   ├── inference.cpp        # ORT GenAI and/or llama_decode (unified: runtime dispatch)
│   ├── session.cpp          # xllama::Session (OrtSession UWP + LlamaSession Linux)
│   ├── bench.cpp            # bench CSV writer
│   ├── platform.cpp         # log_output (writes xllama.log in UWP)
│   ├── path_utils.cpp       # resolve_model_path: LocalState\models\ + InstalledPath fallback
│   ├── utf8_utils.cpp
│   ├── cli.cpp
│   └── training.cpp         # TrainingJob validate/parse (host + UWP linkable)
├── src/main.cpp             # Linux entry point (getopt_long; --train-job)
├── training/                # Training pillar ops: jobs, datasets, host PEFT
├── docs/training-architecture.md  # Training SSOT (RE + capability matrix)
├── uwp/                     # C++/WinRT UWP app
│   ├── App.cpp / App.h      # Application::OnLaunched
│   ├── MainPage.cpp / .h    # MainPageController (plain C++, not runtimeclass); incl. autopilot driver
│   ├── inference-bridge.cpp / .h   # UWP main_loop() + bench mode
│   ├── chat-history.cpp / .h       # ChatHistory: Save/Load/Delete/Clear
│   ├── model-downloader.cpp / .h   # ModelDownloader::DownloadAsync — catalogue download (GitHub Release models-v1, models/manifest.json)
│   ├── packages.config      # NuGet pins (ORT GenAI 0.14.1, ORT 1.24.4, DirectML 1.15.4)
│   └── xllama.sln / .vcxproj
├── scripts/
│   ├── deploy.sh                      # Device Portal: deploy, logs, bench trigger
│   ├── build-uwp.ps1                  # Windows UWP packaging script
│   ├── merge_onnx_external_data.py    # merge model.onnx.data → self-contained model.onnx
│   ├── bench-xbox-ort.sh              # benchmark runner (ORT GenAI; model already on device)
│   ├── validate-console.sh           # autopilot orchestrator: §2 routing / settings ops / GGUF / §7c TAESD verdicts
│   ├── package-catalogue-ort-model.sh # stage flat models-v1 assets from merged ORT GenAI dir
│   ├── install-latest-build.sh        # fetch + deploy latest CI artifact (--bench is opt-in)
│   ├── generate-benchmark-summary.py  # raw results → generated docs/dashboard
│   ├── test-dml-config.sh             # upload DML provider_options without MSIX rebuild
│   ├── vendor-genai-dml-patch.ps1     # overlay #2280 patched onnxruntime-genai.dll
│   ├── export-taesd-asset.sh          # export TAESD VAE for models-v1 release
│   ├── check-uwp-host.sh              # Linux host preflight (qemu, libvirt)
│   └── setup-windows-uwp-dev.ps1      # Windows VM: install VS2022 + UWP workload
├── tests/                   # Unit tests (doctest, target: xllama-tests)
├── bench/                   # Benchmark configs, prompts, raw results + summary policy
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

**Versioning**: `Major.Minor.Build` in `uwp/AppxManifest.xml` is the semantic
version — bump it manually per release (with `CHANGELOG.md`). The `Revision` (4th
component) is stamped automatically in CI to the workflow run number
(`build-uwp.ps1 -BuildRevision`, wired from `github.run_number` in the
workflows), so every CI package is uniquely and monotonically versioned and the
console always takes an **in-place update** — no manual per-build bump, and never
the "same identity, different contents" install block. Local builds leave `.0`.

## Tests

- Framework: **doctest** (header-only, fetched via CMake).
- Target: `xllama-tests`.
- Command: `ctest --test-dir build/linux-test --output-on-failure`.
- Add tests in `tests/test_*.cpp`.

## Critical notes

- **Never commit `.env`** (contains credentials). Use `.env.example` as template.
- **Never commit `.pfx` / `.cer` certificates**. They are in `.gitignore`.

- **ORT GenAI path**: UWP inference is entirely under `#ifdef XLLAMA_USE_ORT` in `src/bridge/inference.cpp`. ORT types are wrapped in `include/xllama/ort_raii.h`. Linux path (`#else`) uses llama.cpp unchanged.

- **No model in the MSIX**: the package is ~19 MB and ships no model. On first launch the app downloads the default chat model (`lfm25-350m` on unified builds; `smollm2-360m-cpu-int4` on ORT-only) from the GitHub Release `models-v1` catalogue (`uwp/models/manifest.json`) into `LocalState\models\`. Routing GPU (`smollm2-360m-dml-fp16-v2`, the #91 parity-validated graph) auto-downloads from `models-v1` when GPU routing is enabled. Current console gate: `./scripts/validate-console.sh all`.

- **Shipping CI**: `build-uwp.yml` publishes `xllama-appx` as **unified + PatchedGenAI #2280 + PatchedOrt** (cached `onnxruntime.dll` + `onnxruntime-genai.dll` from `vendor-dlls-v1`; hashes in `vendor/onnxruntime-patched/SHA256SUMS` and `vendor/onnxruntime-genai-patched/SHA256SUMS`). `llamacpp` lane is bench-only. Rebuild from source only for pin refresh: `build-uwp-ort-patched.yml` (ORT, 1–3 h) / `build-uwp-patched.yml` (GenAI). Poll whether pins can drop: `scripts/check-vendor-nuget-status.sh`.

- **ONNX external data merge**: ORT 1.24.4 calls `std::filesystem::weakly_canonical()` for models with a separate `.onnx.data` file, which traverses path segments inaccessible inside the Xbox AppContainer (`Q:\Users\UserMgr0\...`). Fix: merge external data into a single `model.onnx` using `scripts/merge_onnx_external_data.py` before MSIX packaging. CI does this automatically.

- **app-local DLLs**: `DirectML.dll`, `onnxruntime.dll`, `onnxruntime-genai.dll` must have `<DeploymentContent>true</DeploymentContent>` in the vcxproj. Without this the MSIX silently omits them and the app crashes on load.

- **UWP constraints**: no POSIX mmap, no dlopen, no registry, no thread-affinity desktop API. See `docs/uwp-constraints.md` for the full list.
