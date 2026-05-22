# Changelog

All notable changes to xllama are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased] — HEAD `7bdb971`

### Added
- ChatML prompt template applied for SmolLM2-360M-Instruct (system/user/assistant turns).
- `scripts/merge_onnx_external_data.py`: merges `model.onnx.data` into a self-contained `model.onnx` to bypass `weakly_canonical` AppContainer path-traversal crash (ORT 1.24.4).
- `scripts/check-uwp-host.sh`: preflight script for Arch Linux host (qemu, libvirt, KVM groups).
- `scripts/setup-windows-uwp-dev.ps1`: Windows VM setup — installs VS2022 BuildTools + UWP workload via `winget`.
- `docs/windows-dev-vm.md`: end-to-end guide for local Windows VM UWP builds.
- `build-uwp` CI step: downloads model from HF (`homen3/SmolLM2-360M-Instruct-ort-genai-int4-cpu`), merges ONNX external data, then `nuget restore` + `build-uwp.ps1`. Cache key: `smollm2-360m-ort-genai-int4-cpu-embedded-v1`.

### Changed
- ORT GenAI bumped `0.8.3 → 0.13.2`, ORT `1.22.0 → 1.24.4` (`uwp/packages.config`).
- `bench.cpp`: backend field = `directml` when `XLLAMA_USE_ORT` (define-time; CPU EP is active runtime on Series S).

### Fixed
- `weakly_canonical: Access is denied` crash (`OgaCreateModel`, status `0xC0000005`): ORT runtime walks path segments of the model directory to validate external data; `Q:\Users\UserMgr0\...` is inaccessible from UWP AppContainer. Fix: merge external data into monolithic `model.onnx` so `ValidateExternalDataPath` is never invoked. Confirmed via Win32 probes (GFA, `CreateFile2 GENERIC_READ`, `CreateFile2 FILE_READ_ATTRIBUTES|SYNCHRONIZE`).

---

## [Pivot: SmolLM2-360M + CPU EP] — commit `14e6a14`

### Added
- SmolLM2-360M-Instruct INT4 CPU as the bundled model (403 MB on-disk, ONNX opset 21, IR version 10).
- Model included as `DeploymentContent` in `uwp/xllama.vcxproj`; deployed to `Package.InstalledPath\models\smollm2-360m-cpu-int4\`.
- `resolve_model_path` (`src/bridge/path_utils.cpp`): checks `LocalState\models\<name>` first (runtime override), falls back to `Package.InstalledPath\models\<name>` with copy-on-first-launch to LocalState.

### Changed
- Default model in `uwp/inference-bridge.cpp` and `uwp/MainPage.cpp` changed from Phi-3.5 to `smollm2-360m-cpu-int4`.
- `genai_config.json` uses `"provider_options": []` → CPU EP active (no DirectML).

### Notes
- **GPU EP ruled out**: Xbox Series S UWP GPU pool is ~768 MB. `OgaCreateModel` with DirectML EP on any tested model (Phi-3.5-mini GPU INT4 ~2.2 GB, SmolLM2-1.7B ~1.4 GB) crashes with null-deref in DML allocator on OOM. See `docs/uwp-constraints.md §7`.

---

## [Pivot: ONNX Runtime GenAI + DirectML] — commit `385cb07`

### Added
- `XLLAMA_USE_ORT=1` preprocessor flag in `uwp/xllama.vcxproj`; enables ORT GenAI path in `src/bridge/inference.cpp`.
- `include/xllama/ort_raii.h`: RAII `unique_ptr` wrappers for `OgaModel`, `OgaTokenizer`, `OgaTokenizerStream`, `OgaGeneratorParams`, `OgaGenerator`, `OgaSequences`.
- `uwp/inference-bridge.cpp` / `inference-bridge.h`: replaces `llama-bridge.cpp`; thin UWP glue around `xllama::bridge::run_inference`.
- NuGet packages (`uwp/packages.config`): `Microsoft.AI.DirectML 1.15.4`, `Microsoft.ML.OnnxRuntime.DirectML`, `Microsoft.ML.OnnxRuntimeGenAI.DirectML`.
- `src/bridge/platform.cpp`: `log_output` now writes to `LocalState/xllama.log` in UWP (previously `OutputDebugStringA` only).

### Changed
- `src/bridge/inference.cpp`: `#ifdef XLLAMA_USE_ORT` path uses `OgaGenerator` loop; `#else` path retains `llama_decode` for Linux.
- Linux CI (`build-linux.yml`): `submodules: false` for UWP; llama.cpp submodule only for Linux.
- `deploy.sh`: `upload-file` auto-creates subdirectory; new subcommands `mkdir-localstate`, `upload-dir`.

### Notes
- llama.cpp submodule retained for Linux path (`CMakeLists.txt`). Three UWP patches (`uwp/patches/llama.cpp/`) kept but not applied for this build.

---

## [Pivot: XAML-free UI] — commits `77a651a`, `3f7a950`, `385cb07`

### Removed
- `App.xaml`, `MainPage.xaml`, `XamlTypeInfo_impl.cpp`: eliminated to avoid WMC9999 (`XamlC.exe` crash during `MarkupCompilePass2` in SDK 22621/26100 for C++/WinRT projects).
- `runtimeclass MainPage` from IDL: `MainPageController` is now a plain C++ class.

### Added
- `MainPageController` (`uwp/MainPage.cpp`): programmatic UI built via `Windows.UI.Xaml.Controls.*` API. Uses `enable_shared_from_this`; `shared_from_this()` must not be called from the constructor — use `Init()` post-construction.
- `runtimeclass App` retained (required by `Application::Start`).

### Notes
- Root cause of WMC9999: without MarkupCompilePass2, `XamlTypeInfoProvider::CreateXamlType` cannot provide correct metadata for `xllama.MainPage`; parser fast-fails when `LoadComponent` tries to validate the binding. No workaround existed; XAML-free was the correct fix.

---

## [Baseline: llama.cpp + Linux CI] — initial commits

### Added
- Linux build via CMake presets (`linux-release`, `linux-test`).
- Modular bridge: `src/bridge/inference.cpp`, `bench.cpp`, `platform.cpp`, `path_utils.cpp`, `utf8_utils.cpp`, `cli.cpp`.
- Shared headers under `include/xllama/` (inference, CLI, RAII, platform, path utils).
- Unit tests with doctest (target: `xllama-tests`, preset: `linux-test`).
- `xllama-cli` binary: `src/main.cpp` with `getopt_long`.
- `scripts/deploy.sh`: Device Portal REST API wrapper (deploy, install-cert, get-log, list-localstate, list-dumps, start-app, stop-app, diagnose-startup).
- `scripts/bench-xbox.sh`: automated benchmark runner (upload, trigger, poll, fetch CSV, compute median).
- `docs/`: `device-portal.md`, `uwp-constraints.md`, `phase1-runbook.md`.
- `bench/`: methodology README, config JSONs, `prompts/standard-512.txt`, `prompts/short-32.txt`.
- `.github/workflows/build-linux.yml`: clang-format, shellcheck, cmake, ctest, UWP gate.
