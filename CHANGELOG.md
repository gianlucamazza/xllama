# Changelog

All notable changes to xllama are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased] — HEAD `450e4ff`

### Added

**Multi-turn Session API** (`include/xllama/session.h`, `src/bridge/session.cpp`):
- `xllama::Session::create(SessionParams)` — loads model + tokenizer once; subsequent `generate(GenerateParams)` calls reuse them, eliminating the ~1-2s per-call reload overhead of `run_inference()`.
- `GenerateParams` exposes `top_p`, `top_k`, `repetition_penalty`, `seed`, `stop_sequences` (substring-checked against accumulated output; matching sequence stripped on hit).
- Implemented for both UWP (ORT GenAI: `OrtSession` keeps `OgaModel` + `OgaTokenizer` alive) and Linux (llama.cpp: `LlamaSession` keeps `llama_model` alive; `llama_context` rebuilt per call to respect context parameters).
- `run_inference()` is unchanged — fully back-compatible.
- Consumer: `xbox_faraday` game (FARADAY) for per-turn dialogue generation.

**Workaround experiments for Xbox UWP constraints** (see `docs/uwp-constraints.md §7, §9`):
- `uwp/model-downloader.cpp/h`: in-app Hugging Face download via `HttpClient` chunked streaming (Exp 2). `EnsureModelAsync()` in `MainPage` implements a three-step bootstrap: LocalState `.complete` marker → InstalledPath bundle → HF download. Reduces peak disk usage from ~1.4 GB to ~480 MB; frees ~900 MB on Dev Mode partition Q:\ enabling models up to ~1 GB. `internetClient` capability already present in manifest.
- `src/bridge/path_utils.cpp`: third fallback `E:\xllama\models\<name>` for NTFS USB stick (Exp 3). No UWP capability required; probe via `GetFileAttributesW`. Enables models up to 2 GB single-file (Xbox Dev Mode USB limit). Zero cost if USB absent.
- `scripts/test-dml-config.sh`: uploads DML provider_options config to Xbox via Device Portal without MSIX rebuild (Exp 1). Backs up original `genai_config.json`; `--restore` reverts.
- `bench/configs/genai_config-dml-test.json`: DML EP test config — `enable_cpu_mem_arena=0`, `enable_mem_pattern=0`, `past_present_share_buffer=false` (reduces up-front KV-cache GPU allocation; may allow SmolLM2-360M to fit ~768 MB pool).
- `docs/model-selection.md`: consolidated model evaluation checklist — hard limits, 9-step selection sequence, tested/candidate models, conservative/borderline/over-budget tables.
- `docs/uwp-constraints.md §9`: Disk Budget — ~2.2–2.5 GB Dev Mode free space, 2× peak-install rule, working budget table (empirical, Series S).
- `scripts/merge_onnx_external_data.py`: NOTE stderr (>400 MB) and WARNING stderr (>600 MB) budget thresholds post-merge.
- `docs/uwp-constraints.md §5`: split GPU OOM and disk-budget failure modes into separate tables (previously mixed under a single "Result" column).

**UX improvements** (commits `3a12bda`–`42741e1`):
- Multi-turn chat: `uwp/chat-history.cpp/h`, conversation persistence in `LocalState/chats/` (JSON, indexed by timestamp), history browser overlay, new-chat button.
- System prompt editable via settings overlay (persisted to `LocalState/settings.json`).
- Live metrics: real-time tok/s updated every flush cycle; `StatusKind` enum (`Info`, `Working`, `Success`, `Error`) for colour-coded status bar.
- `RichTextBlock` streaming: `Paragraph::Inlines` append (O(1) per token); flush timer batches token appends every 80 ms to avoid layout thrash.
- Xbox UX: TV safe-area margins (48/27 px), dark theme on Xbox hardware, B-button cancels inference, gamepad Y jumps to prompt, Reveal focus visual, `ElementSoundPlayer::On`.
- `AppxManifest.xml`: `xbox:DefaultTile`, `xbox:SplashScreen`, dark splash background `#0E1116`.
- ChatML prompt template applied for SmolLM2-360M-Instruct (system / user / assistant turns).

**Documentation:**
- `README.md`: "About the name" section — disambiguates xllama from llama.cpp engine.
- Full docs realignment: `README.md`, `ROADMAP.md`, `AGENTS.md`, `docs/phase1-runbook.md`, `docs/uwp-constraints.md`, `docs/device-portal.md`, `patches/README.md` all updated to reflect ORT GenAI CPU EP as the active path.
- `docs/windows-dev-vm.md` (new): end-to-end Windows VM build guide.
- `scripts/setup-windows-uwp-dev.ps1` (new): Windows VM setup via `winget` (VS2022 BuildTools + UWP workload).
- `scripts/check-uwp-host.sh` (new): Arch Linux host preflight (KVM, qemu, libvirt groups).
- `docs/uwp-constraints.md §7`: removed unverified architectural claims ("Game process category", "128 MB dedicated + 640 MB shared"); replaced with observed-behaviour framing and source note.
- `ROADMAP.md`: Phase 4 milestones updated — `ModelDownloader` (Exp 2) and USB fallback (Exp 3) marked done; next: validate Exp 2 on console, remove MSIX model bundle.

**CI:**
- `build-uwp` CI step: downloads model from HF (`homen3/SmolLM2-360M-Instruct-ort-genai-int4-cpu`), merges ONNX external data, then `nuget restore` + `build-uwp.ps1`. Cache key: `smollm2-360m-ort-genai-int4-cpu-embedded-v1`.

### Changed
- ORT GenAI bumped `0.8.3 → 0.13.2`, ORT `1.22.0 → 1.24.4` (`uwp/packages.config`).
- `bench.cpp`: backend field = `directml` when `XLLAMA_USE_ORT` (define-time; CPU EP is active runtime on Series S).
- `uwp/pch.h`: added `Windows.Web.Http`, `Windows.Web.Http.Filters` for model downloader.
- `ROADMAP.md Phase 2`: corrected GPU pool description to observed-behaviour framing.

### Removed
- `uwp/llama-bridge.cpp`, `uwp/llama-bridge.h`: legacy files not compiled since ORT GenAI pivot.

### Fixed
- ASCII-safe status strings: removed em-dash and ellipsis Unicode literals that caused MSVC `C4566` warnings.
- `XYFocusKeyboardNavigationMode` removed from `MainPage.cpp` (unresolvable symbol in MSVC UWP context).
- `weakly_canonical: Access is denied` crash (`OgaCreateModel`, status `0xC0000005`): ORT runtime walks path segments of the model directory to validate external data; `Q:\Users\UserMgr0\...` is inaccessible from UWP AppContainer. Fix: merge external data into monolithic `model.onnx` so `ValidateExternalDataPath` is never invoked. Confirmed via Win32 probes (`GetFileAttributesW`, `CreateFile2 GENERIC_READ`, `CreateFile2 FILE_READ_ATTRIBUTES|SYNCHRONIZE`).

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
