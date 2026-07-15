# Changelog

All notable changes to xllama are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added

- **`smollm2-1.7b-cpu-int4` published to the `models-v1` catalogue** — the four
  assets (incl. the 1.47 GB `model.onnx`) are now on the `models-v1` release, so
  the 1.7B chat model (~20.6 tok/s decode, ~2.4 GB RAM) is downloadable in-app from
  the model picker. Verified on-console (download → load → generate). No app/code
  change — the manifest entry already referenced them.
- **Architecture overview** — new `docs/architecture.md` (SSOT for system
  structure) maps module boundaries, runtime backend dispatch, chat templates,
  KV-reuse, routing, model provisioning + quant auto-upgrade, membw, and the
  diffusion pipeline. Cross-linked from `docs/README.md`, README, and ROADMAP.

- **External-data ONNX loading unblocked >2 GB on AppContainer** (patched ORT DLL,
  both fixes console-validated 2026-07-15). Goal: load ONNX models with external
  `.onnx.data` >2 GB on
  DirectML (the merge workaround caps at the 2 GB protobuf ceiling; DirectML is in
  maintenance mode, so this is the last local GPU lever). A zero-code USB spike
  (`scripts/build-fp16-dml-model.sh` + `scripts/spike-fp16-extdata-usb.sh`) was
  **refuted on-console** — `EnsureModelNamedAsync` copies USB models into LocalState
  and loads from there, so the crash is universal to external-data models, not
  USB-specific. Fix: patch ORT core (`patches/onnxruntime-extdata-appcontainer.patch`
  - `scripts/vendor-ort-extdata-patch.ps1`), built via the dispatch-only
    `build-uwp-ort-patched` CI lane. **Two fixes**: (1) `ValidateExternalDataPath`
    `weakly_canonical` guard — **validated**: the 1.86 GB-extdata model that crashed
    now loads and generates (`bench/results/phase6-fp16-extdata.csv`); (2) `env.cc`
    `ReadFileIntoBuffer` chunk 1 GB→16 MB — fixes the intermittent `errcode 1450`
    (a ~1 GB single `ReadFile` of the un-quantized embedding exhausts AppContainer
    page-lock resources). **Status**: in the dispatch-only `build-uwp-ort-patched`
    lane — **not yet promoted into the default shipping build** (`build-uwp.yml` still
    ships vanilla ORT), and no new catalogue model added. **Scope**: the unblock is a
    **CPU/int4 enabler** — a native-DML 1B fp16 (2.49 GB) loads (2878/3801 MB) but
    OOMs GPU inference (§7 budget wall), so bigger fp16 on the GPU stays out of reach.
    Runbook: `docs/fp16-extdata-runbook.md`.

### Changed

- Documentation currency/drift pass after v1.1.7.0: `smollm2-1.7b` marked
  in-app-downloadable (README/model-selection/recommended-config), the
  model-provisioning quant auto-upgrade recorded as **shipped** (was described as an
  open gap in `benchmarks.md`), the ROADMAP shipping banner + README roadmap
  refreshed with the 1.1.7.x deliverables, and `uwp-constraints.md` clarified that
  the `unified` build dispatches backends at runtime (not compile time).

## [1.1.7.0] - 2026-07-15

### Added

- **GGUF KV-cache reuse** — `LlamaSession` keeps a persistent `llama_context`, so
  a continuation turn appends only the new turn's delta instead of re-prefilling
  the whole conversation. Console-measured **4.07×** turn-2 prefill on gemma3-270m
  (`bench/results/phase6-gemma-kv.csv`); previously GGUF was stateless. Enabled in
  the Settings KV toggle (routing stays ORT-only — llama.cpp UWP is CPU-only).
- **CI: `src/models/*.cpp` MSBuild wildcard** in `ggml-uwp.vcxproj` + a
  `scripts/check-uwp-sources.sh` drift-check — new llama.cpp architectures no
  longer break the UWP link on a submodule bump (as `657e011` did).
- **Manifest per-entry merge factored + unit-tested** — extracted the
  LocalState-override merge from `LoadModelManifest` into a pure, header-only
  `xllama::merge_manifest_entries` (`include/xllama/manifest_merge.h`) with 7
  host doctest cases (`tests/test_manifest_merge.cpp`) covering replace / append /
  preserve, incl. the 2026-07-10 whole-catalogue-shadow regression.
- **CPU memory-bandwidth micro-bench** — `xllama::measure_membw` (STREAM-style
  read/copy/triad, `include/xllama/membw.h`) with `xllama-cli --membw` (host) and a
  `membw.flag` headless mode (console → `membw-result.csv`). Pins the DRAM-bandwidth
  ceiling behind the bandwidth-bound decode number. 4 host doctest cases.
  Console-measured 2026-07-15: Xbox Zen 2 read 12.35 GB/s @1t / 30.29 @8t — the
  single-thread read matches the deduced ~13 GB/s GEMV denominator.
- **In-app HuggingFace download verified on-console** (2026-07-15) — the app
  self-downloaded the 2.29 GB single `.gguf` for `gemma4-e2b` from HF and
  loaded+generated it, confirming the >2 GB self-download path (not just Device
  Portal provisioning). See `docs/benchmarks.md`.

### Changed

- **`gemma4-e2b` catalogue default: UD-IQ2_M → Q3_K_S** (2.45 GB). Console-measured
  15.3 tok/s and generates full responses on long declarative prompts, where the
  2-bit IQ2_M collapsed to an immediate EOG. See `docs/benchmarks.md`.
- **llama.cpp submodule bumped to `657e011`** (was `9a532ae4b`); `ggml-uwp.vcxproj`
  gained the new per-arch sources.
- **Stop-sequence handling unified** into one suffix-match helper
  (`apply_stop_sequences`, `chat_prompt`) shared by the llama and ORT decode loops.
- **`scripts/install-latest-build.sh`**: `bench.flag` is now `--bench` opt-in — a
  plain install launches into the UI.
- **Prefill micro-batch knobs exposed** — `n_batch`/`n_ubatch` plumbed through
  `InferenceParams`/`SessionParams` into `llama_context_params`, with
  `xllama-cli --batch/--ubatch` and a `scripts/bench-ubatch-sweep.sh` sweep helper.
  Host sweep found no reproducible ubatch win (noise-dominated); default (llama.cpp 512) unchanged. See `docs/benchmarks.md`.
- Fixed two more stale in-code comments: `ManifestEntry` (gguf KV-reuse now
  enabled) and `GenerateParams` reuse_kv/reset_kv (honored by both backends now,
  not ORT-only).
- Consolidated documentation onto single-source-of-truth docs (perf →
  `benchmarks.md`, constraints → `uwp-constraints.md`, catalogue →
  `model-selection.md` + `manifest.json`); refreshed stale version/quant/KV claims.

### Fixed

- **Model provisioning now auto-upgrades a stale quant** — `IsModelProvisioned`
  gained an expected-aware overload (compares the dir against the manifest's
  current `files[].filename` via the new pure `dir_satisfies_expected_files`,
  `include/xllama/model_provision.h`), and `EnsureModelNamedAsync` loads the
  manifest before the provisioned-check and reconciles the dir (deletes any
  non-expected `*.gguf` + drops the stale `.complete`) before re-downloading. A
  directory holding an older `.gguf` than the manifest names (e.g. a stale IQ2_M
  under `gemma4-e2b`) is no longer treated as provisioned, and the coexisting-file
  hazard in `first_gguf_in_dir` is closed. `EnsureGpuModelIfNeeded` is
  expected-aware too. 13 host doctest cases (`tests/test_model_provision.cpp`).
- Fixed a stale comment in `MainPage.cpp::StartInference`: GGUF KV-cache reuse is
  supported (persistent `llama_context`), gated by `kv_reuse_supported_for_model`;
  only EP routing stays gated off for GGUF.
- `scripts/bench-xbox-ort.sh`: verify `bench-result.csv.done` is actually deleted
  before a run, so `wait_for_done` can't return off a stale marker (silent wrong row).

### Investigated (no change shipped)

- **AppContainer file mmap** (`CreateFileMappingFromApp`/`MapViewOfFileFromApp`,
  with a loader fallback): built and deployed, but **no measured benefit** —
  GGUF load on CPU is dominated by the AVX2 tensor repack, not the file read.
  Reverted; finding recorded in `docs/benchmarks.md`.

---

## [1.1.6.0] - 2026-07-14

### Added

- **Per-architecture chat template** (`ChatFormat`, `src/bridge/chat_prompt.cpp`):
  `chat_format_for()` replaces the hard-coded ChatML with a data-driven template
  selected by model id. ChatML kept byte-identical (Qwen no-think suffix
  preserved); **Gemma** added (`<start_of_turn>…<end_of_turn>`, no system role,
  stop `<end_of_turn>`). UWP UI + bench call one abstraction.
- **Gemma family — console-validated** (`phase6-gemma.csv`). Catalogue entries
  `gemma3-270m` (253 MB) and `gemma4-e2b` (2.29 GB IQ2_M), GGUF from HF per Gemma
  Terms. On Xbox Series S: gemma3-270m **76.8 tok/s** decode (368 MB); gemma4-e2b
  loads at 2534 MB RAM, **9.9 tok/s**. The "Gemma-4 too big" verdict is
  **overturned** — the ~2 GB Dev Mode per-file limit does not apply to GGUF.
- **`--chat` flag** for `xllama-cli` + stop-sequence support in the llama path
  (`InferenceParams::stop_sequences`), so the CLI/GGUF bench stop on the chat
  format's stop token.
- **Benchmark report** — `docs/benchmarks.md` + self-contained
  `docs/benchmarks-charts.html` consolidate every tested model across backends,
  with root-cause notes on the negative performers.
- **Automated MSIX versioning** — CI stamps the Revision from the workflow
  `run_number` (`build-uwp.ps1 -BuildRevision`); no manual per-build bump.
- **Repo automation** — CI concurrency + `push`-on-`main` only (one build per
  PR, superseded runs cancel), Dependabot (actions + llama.cpp submodule), CodeQL
  (c-cpp), PR/issue templates, `CONTRIBUTING.md`.

### Changed

- Package version **1.1.6.0** (Revision auto-stamped in CI).
- `docs/model-selection.md` Gemma verdict revised (disk no longer the binding
  constraint after the 90 GB Dev Mode bump; E2B console-validated).

---

## [1.1.5.0] - 2026-07-14

### Added

- **`include/xllama/chat_prompt.h`** — Qwen no-think generation suffix and empty
  `</think>` stripping; unit tests in `test_chat_prompt.cpp`.
- **`kv_reuse_supported_for_model()`** in `routing_policy.h` — DML EP gate for
  continuous decoding.

### Fixed

- **GPU/DML multi-turn**: KV reuse disabled on `*-dml-*` models — ORT GenAI rejects
  `AppendTokenSequences` on a persistent DirectML generator (_"Continuous decoding is
  not supported on the selected device type (DirectML)"_); avoids per-turn fallback
  and spurious failures.
- **Qwen3.5 GGUF**: append Qwen no-think prefill after `<|im_start|>assistant` and
  strip leading empty think blocks from saved/displayed assistant text.
- **`validate-console.sh routing`**: remove `ap-routing-longctx` decoy chat after the
  test so the seeded _"Understood; ready to continue."_ assistant turn does not linger
  in History.

### Changed

- Package version **1.1.5.0**.

---

## [1.1.4.0] - 2026-07-14

### Added

- **`include/xllama/routing_policy.h`** — extracted per-workload routing decision
  (600-tok threshold, GGUF/routing capability gates) with unit tests.
- **`scripts/package-catalogue-ort-model.sh`** — stage flat `models-v1` assets for
  `smollm2-360m-dml-fp16` and `smollm2-1.7b-cpu-int4` catalogue entries.
- **Catalogue download URLs** for `smollm2-360m-dml-fp16` and `smollm2-1.7b-cpu-int4`
  (`hf_base_url` + prefixed `remote` names on `models-v1`).

### Changed

- **Shipping CI default** (`build-uwp.yml`): `xllama-appx` is now **unified +
  PatchedGenAI #2280**; `llamacpp` remains a bench-only artifact. `build-uwp.ps1
-PatchedGenAI` fails closed if the vendor step fails.
- **Recommended modern settings** (`bench/configs/settings-modern.json`): default chat
  model **LFM2.5-350M** (GGUF) on unified builds; routing GPU unchanged.
- Package version **1.1.4.0**.

### Fixed

- **Model provisioning layer** (`IsModelProvisioned`, `EnsureModelNamedAsync`,
  background `gpu_model` download when `routing≠0`, routing guards, Settings
  model-change re-provision). See commit `f3d733e`.
- **`validate-console.sh` preflight/upload**: `model_provisioned` WDP check uses
  basename `genai_config.json`; `upload_file` mkdirs remote dirs before POST
  (chats + nested model paths).

### Measured (2026-07-14, unified patched on console)

- **`validate-console.sh all` → ALL PASS** (autopilot, no pad):
  routing A/B (959 tok GPU / short CPU, no `887A0036`), GGUF `lfm25-350m` via
  llama.cpp session, TAESD VAE **593–625 ms** (sub-second gate). Verified on
  **1.1.3.0** (full suite) and **1.1.4.0** (routing re-check after CI deploy).
- **`smollm2-360m-dml-fp16` on `models-v1`** — ORT GenAI builder `-p fp16 -e dml`,
  691 MB merged self-contained; catalogue in-app download operational.

### Docs

- **Drift pass after the Fase-2b/patched-lane work**: README (GGUF rows with
  measured decode, 90 GB disk supersession, in-process image gen in the
  architecture diagram, helper listings), `docs/uwp-constraints.md` intro
  aligned to the §9 supersession, `docs/recommended-config.md` (GGUF measured
  - llama thread cap + sharper "do not use"), runbook §2 (CI-lane artifact
    alternative; settings must upload as `settings.json`), vendor README +
    script message (rel-0.14.1 branch pin, ORT_HOME build), AGENTS.md map.

### Fixed

- **Stale `LocalState\manifest.json` override no longer hides the catalogue**:
  the override used to replace the bundled catalogue _entirely_, so an old
  override (e.g. the Exp-2-era single-entry file found on console 2026-07-10)
  made `sd-turbo-fp16` and the GGUF entries invisible — the Image dialog
  failed with "catalogue entry missing hf_base_url". `LoadModelManifest` now
  **merges per entry**: same-name entries replace bundled ones, new names are
  appended, unmentioned bundled entries stay.

- **Headless bench could not load catalogue GGUF models**: `run_inference_llama`
  passed the resolved model _directory_ straight to
  `llama_model_load_from_file`, which needs the `.gguf` _file_ — every bench
  run of `qwen35-0.8b` / `lfm25-350m` failed on console (found 2026-07-10 on
  the first unified-build bench). The chat path (`create_llama`) already
  descended into the directory; that logic is now the shared
  `first_gguf_in_dir()` (`path_utils`) used by both, with host tests.
- **llama.cpp default thread count livelocked on console**: with no explicit
  `n_threads`, both llama paths used `detect_threads()` (= 8 on Series S),
  hitting the known ggml spin-wait livelock at t7/t8 (phase35-llamacpp-scaling;
  re-hit 2026-07-10 — bench runs loaded the model then hung past the 300 s
  watchdog). New `detect_threads_llama()` caps the llama default at 6 on UWP
  (t6 is the measured optimum); explicit `n_threads` still wins, ORT paths
  unchanged.
- **Bench CSV mislabelled GGUF runs in unified builds** as
  `int4`/`ort-genai-cpu`: the label block keyed on compile-time
  `XLLAMA_USE_ORT` only. It now keys on the backend that actually ran
  (`model_uses_llama_backend`), keeping the llamacpp-lane labels
  (`Q4_K_M`/`cpu`), and the CSV `n_threads` column reports the llama-capped
  default.

### Measured (2026-07-10, unified 1.1.1.0 on console)

- **Fase 2b GGUF benches** (`bench/results/phase5-gguf.csv`, t6,
  standard-512): **Qwen3.5-0.8B Q4_K_M decode 35.1 tok/s** (prefill 98.1,
  peak 718 MB, load 4.1 s) — **LFM2.5-350M Q4_K_M decode 94.2 tok/s**
  (prefill 241.4, peak 321 MB, load 1.4 s). LFM2.5-350M beats the ORT int4
  360M baseline (66.3 @ t8) by ~42% at similar size; Qwen3.5-0.8B is in line
  with size scaling (1.7B CPU int4: 20.6). Unified promotion is now
  bench-unblocked.

### Changed

- **`diffuse.cpp` is timestep-shape-aware**: the UNet `timestep` tensor is fed
  with the rank the model declares — `[1]` (optimum ≤ 1.23, the deployed
  artifacts, unchanged path) or scalar `[]` (optimum-onnx 0.1.0+ exports) —
  from the same 1-element buffer, mirroring `validate_pipeline.py`. Removes
  the last code blocker for promoting new-generation diffusion artifacts;
  the remaining gate is runbook §7 on console. The unet log line now reports
  `ts rank 0|1`.
- **Diffusion export toolchain bumped** (host-only, never shipped): torch
  2.4.1→2.9.1, optimum 1.23.3→optimum-onnx 0.1.0, transformers 4.46.3→4.57.6,
  diffusers 0.31.0→0.39.0. Clears the actionable dependabot alerts (torch
  CVE-2025-32434, transformers ReDoS batch, diffusers CVE-2026-44513/45804);
  residual non-applicable alerts documented in `diffusion/requirements.txt`.
  Full recipe re-validated (export → fp16 convert → validate_pipeline → golden
  vectors byte-identical → ctest → TAESD). `validate_pipeline.py` is now
  timestep-shape-aware (new exports declare a scalar UNet `timestep`);
  `gen_golden_vectors.py` stamps installed versions. Deployed console
  artifacts are unchanged.
- **In-app image Generate runs in-process (no restart).** The Image dialog runs
  SD-Turbo on a background MTA thread (the path console-validated in §7b): live
  stage in the status bar via `diffuse-progress.txt` (200 ms poll), Cancel
  writes `diffuse-cancel.flag` (honored between UNet steps). The headless
  `diffuse.flag` path is kept for bench/WDP parity.
- **Routing `auto` counts real tokens.** New `Session::count_tokens()` (ORT
  tokenizer / llama_tokenize) replaces the `size/4` estimate; threshold updated
  to 600 tokens (crossover between ~285 and ~1050 in the v0.3.6 matrix). The
  routing decision is logged (`routing: auto → gpu/cpu (N tok, ...)`).
- UWP inference threads pinned to `t=4` (Series S optimum; `t=7/8` livelock).

### Added

- **Console autopilot + `validate-console.sh`.** Dev Mode gives the console no
  working text-input path, so §2 routing / §7c TAESD / GGUF-chat validations
  needed a person at the pad. A flag-driven in-app driver (`autopilot.flag` →
  `App::OnLaunched`) now replays a JSON action list
  (`load_chat|send|new_chat|set_model|generate_image|quit`) against the same
  controller methods the buttons call — real XAML process, no UI code
  duplicated — writing `autopilot-done.txt` (`ok`/`error:`). The host
  orchestrator `scripts/validate-console.sh <routing|gguf|taesd|all>` uploads
  the script, restarts, polls, and emits deterministic PASS/FAIL from the log
  (no LLM judgment in the verdict). The §2 automated PASS is the official gate
  for promoting `-PatchedGenAI` to default. Package bumped to **1.1.3.0** (also
  carries the #44 manifest-merge fix on-device). `create_llama` now logs a
  distinct GGUF-session load marker.
- **GGUF assets in the catalogue (Fase 2b).** `Qwen3.5-0.8B-Q4_K_M.gguf`
  (508 MB, unsloth quant, Apache-2.0) and `LFM2.5-350M-Q4_K_M.gguf` (219 MB,
  LiquidAI official) published on `models-v1`; catalogue entries `qwen35-0.8b`
  and `lfm25-350m` now download in-app (unified builds). The LFM Open License
  v1.0 permits redistribution (§4) provided the license accompanies the work —
  `LICENSE.txt` is published on the release and downloaded alongside the model.
  Host smoke test: both load and generate via `xllama-cli`. On-console
  decode/prefill benches remain (bench-gated promotion to default).
- **Patched GenAI DLL pipeline** ([#2280](https://github.com/microsoft/onnxruntime-genai/pull/2280)):
  `patches/onnxruntime-genai-2280-dml-fallback.patch`,
  `scripts/vendor-genai-dml-patch.ps1`, and `build-uwp.ps1 -PatchedGenAI` to
  overlay the NuGet `onnxruntime-genai.dll` for XAML + DML chat routing.
- **CI lane for -PatchedGenAI packages** — default `build-uwp.yml` now ships
  `xllama-appx` as unified+#2280 (2026-07-14). `build-uwp-patched.yml` remains
  a manual dispatch fallback.
- **Recommended configurations** — [`docs/recommended-config.md`](docs/recommended-config.md)
  and [`bench/configs/settings-modern.json`](bench/configs/settings-modern.json)
  for console validation.
- **TAESD fast VAE** for image generation: toggle in the Image dialog downloads
  `sd-turbo-fp16_taesd_vae_decoder_model.onnx` (~5 MB) from `models-v1` over the
  full VAE in-place; setting `diffuse_taesd_vae` persists in `settings.json`.
  Host export: `scripts/export-taesd-asset.sh` (asset published on `models-v1`).
  Console-validated 2026-07-14 (runbook §7c, VAE **593–625 ms**).
- **GGUF catalogue plumbing (Fase 2).** Complete end-to-end support for
  `kind: "gguf"` catalogue entries:
  - New public helper `model_uses_llama_backend()` (suffix fast-path + resolve +
    `*.gguf` directory scan) makes `Backend::Auto` layout-aware. Bare catalogue
    names (e.g. "qwen35-0.8b") now correctly select the llama.cpp backend in
    unified builds.
  - `resolve_model_path` treats directories containing `*.gguf` as valid models
    (LocalState primary + USB fallback).
  - `run_kv_bench` (ORT-only) now guards GGUF models and emits a clear skip.
  - GGUF models are hidden from KV-cache reuse and EP routing in Settings
    (llama.cpp path is stateless + CPU-only on Xbox).
  - Unit tests added for the helper and Auto dispatch on suffix + directory
    layouts.
  - Example placeholder entry (`qwen35-0.8b`) added to `uwp/models/manifest.json`
    (manual provisioning until Fase 2b).

  Catalogue asset upload + on-console benches for Qwen3.5/LFM2 remain Fase 2b
  (bench-gated). See PR #30.

## [1.1.0] - 2026-07-09

**Image self-serve + faster in-app diffusion + dual-backend-ready build.** The
image model now downloads itself from the catalogue; diffusion is proven to run
in-process (no restart needed); and the binary can now compile both the ORT
GenAI and llama.cpp backends into one MSIX, chosen per model at runtime — the
groundwork for modern GGUF-only models (Qwen3.5, LFM2) the ORT builder can't
produce. Full measured story: `docs/technical-report.md`.

### Measured — plain ORT DML coexists with the XAML compositor (in-app diffusion)

The `887A0036` device conflict that forced headless image generation was
specific to ORT **GenAI**'s Agility device factory. `diffuse-inproc.flag` ran
the full SD-Turbo pipeline **on a background thread inside the live XAML
process** on console: coherent 512×512 PNG, compositor alive, no conflict, no
OOM — **total 5.57 s, faster than the 6.9 s headless run** (warm GPU). Image
generation no longer needs the restart flow (`docs/uwp-constraints.md` §7,
runbook §7b). Wiring the in-app Generate is the follow-up; the restart flow
stays as a fallback.

### Added — runtime backend dispatch (ORT GenAI ⊕ llama.cpp in one binary)

`src/bridge/session.cpp` + `inference.cpp` no longer select the backend with a
mutually-exclusive `#ifdef XLLAMA_USE_ORT / #else`. Two independent capability
macros (`XLLAMA_USE_ORT`, `XLLAMA_USE_LLAMA`) let both backends compile together;
`Session::create` / `run_inference` become runtime dispatchers keyed on
`SessionParams::backend` (`Auto` infers `.gguf` vs ORT-dir layout). A new
`XllamaBackend=unified` UWP build links the static ggml lib alongside ORT and
ships both — CI-validated (`build (unified)` green). Single-backend builds stay
behaviorally identical. This unblocks modern GGUF-only models via llama.cpp
while ORT remains the default; UI wiring (`kind:gguf`) is the next phase.

### Evaluated — modern small models (host-validated; console benches pending)

Surveyed the post-Qwen3/Gemma3 landscape (the ORT GenAI builder is frozen at
those архitectures). Confirmed loading through our stack:

- **Qwen3.5-0.8B** (Feb 2026, Apache-2.0) — current-gen, Q4_K_M 507 MB, runs via
  our llama.cpp submodule (`qwen35` arch).
- **LFM2.5-350M** (LiquidAI) — hybrid edge, Q4_K_M 218 MB, runs via llama.cpp.
- **Qwen3-0.6B int4** — builds with the ORT GenAI builder (969 MB merged; the
  151k-vocab embedding dominates — heavier than SmolLM2-360M's 417 MB).
- **Gemma-3-270M** — gated on HF, build blocked pending a token.
- **TAESD decoder** (madebyollin/taesd, MIT) — a 4.9 MB drop-in VAE that replaces
  SD-Turbo's 94 MB decoder; `validate_pipeline.py --vae` confirms a coherent
  image. On console the VAE stage is 2.6 s of 6.9 s → TAESD targets ~4.5 s.
  Export recipe: `diffusion/export_taesd.py` (falsified the [0,1] output
  assumption; the diffusers decoder already emits SD `[-1,1]`).

### Removed — purpose-served legacy (image spike, GGUF-era bench, dead switches)

- **Image spike** (`uwp/image-spike.cpp`, the `image.flag` headless mode,
  `scripts/gen_imgspike_model.py`): the hypothesis it existed to test was
  confirmed on console 2026-07-08 (DML 11.1× CPU) and the real diffusion
  pipeline shipped in 1.0.0. The result CSV stays; the tooling lives in git
  history at `v1.0.0`.
- **`CheckBenchMode`** (MainPage): superseded by the `wWinMain` headless
  dispatch — an in-XAML bench would run with the compositor alive and produce
  numbers not comparable with every recorded CSV.
- **`bench-xbox.sh` + `bench/config/phase1-*.json`**: the pre-pivot GGUF-era
  orchestrator (single-`.gguf` upload, qwen3/llama-3.2 configs never used by
  the ONNX app); docs already invoked it with arguments it rejected.
  `bench-xbox-ort.sh` is the orchestrator (docs updated).
- **`build-uwp.ps1 -NoBundledModel`**: the `XllamaNoBundledModel` MSBuild
  property no longer exists (the model ItemGroup was removed in 1.0.0) — the
  switch was a no-op.

### Added — SD-Turbo in the download catalogue (image model self-serve)

The image model no longer requires Device Portal provisioning: the catalogue
gains a `kind: "diffusion"` entry for `sd-turbo-fp16` (validated artifact —
`validate_pipeline.py` end-to-end pass; the `-ort-fp16` candidate is the
CUDA-only NhwcConv trap and stays excluded), and the **Image dialog downloads
it on the first Generate** (~2.4 GB, progress in the status bar).

- Manifest schema: optional `kind` (`ort-genai` default / `diffusion`) and
  per-file `remote` (flat release asset name; `filename` may now carry a
  subpath like `unet/model.onnx` — the downloader creates subfolders).
- Diffusion entries are hidden from the chat model picker.
- `run_diffuse` reads the CLIP tokenizer from the model's own `tokenizer/`
  dir first (what the download provides), falling back to the legacy
  `LocalState\clip\` upload.
- Requires the 5 `sd-turbo-fp16_*` assets on the `models-v1` GitHub Release.

### Docs — full drift audit + user guides (2026-07-09)

Three-way audit (docs↔code, docs↔measured state, gaps) and fixes:

- **Retired-narrative drift**: README, `phase1-runbook`, `device-portal`,
  `windows-dev-vm`, `model-selection` still described the bundled-model MSIX,
  the "~768 MB GPU pool", the build-time model merge, and pre-1.0 versions —
  all rewritten to the current reality (19 MB no-model MSIX, first-launch
  catalogue download from the `models-v1` Release, measured 3801 MB GPU
  budget, per-workload verdict, image generation).
- **Status drift**: ROADMAP Phase 3/3.5 headers and stale items (load_ms
  baseline, 1.7B CPU bench, int4 variant location) flipped to their measured
  state; three historical "On-console validation pending" CHANGELOG lines
  flipped per the runbook convention; runbook section headers now carry
  ✅ MEASURED / ⏳ PENDING markers and the v0.4.0.0 anchor is gone.
- **New user docs**: `docs/using-the-app.md` (chat, settings, routing,
  KV reuse, image generation) and `docs/install-release.md` (cert + VCLibs +
  MSIX from a tagged release); `docs/README.md` index completed;
  "Add your own model (manifest override)" how-to in `model-selection.md`;
  `deploy.sh fetch-file` and the llamacpp CI variant documented.

### Added — in-process diffusion experiment (`diffuse-inproc.flag`)

The 887A0036 device conflict (§7) was measured with ORT **GenAI**'s
Agility-factory device; the diffusion pipeline uses **plain ORT DML**, which may
coexist with the XAML compositor device. `diffuse-inproc.flag` runs
`run_diffuse()` on a background MTA thread inside the live XAML process to
falsify the inherited headless requirement. If it passes on console, image
generation becomes in-app — no restart flow. Runbook §7b; on-console validation
pending.

### Added — diffusion progress + cancellation

`diffuse-progress.txt` reports the live stage (`start` / `text_encoder` /
`unet s/N` / `vae` / `done` / `cancelled` / `error`); `diffuse-cancel.flag`
(consumed) aborts between UNet steps. Works in both the headless and the
in-process paths; stale `.done`/cancel artifacts are cleared at run start.

## [1.0.0] - 2026-07-08

**First stable release.** Local LLM chat (ORT GenAI, CPU int4 + per-conversation
GPU routing, KV-cache reuse) and Stable-Diffusion image generation (SD-Turbo
fp16 on DirectML, 512×512 in ~7 s) on a retail Xbox Series S in Dev Mode.
Models are downloaded on first launch from the GitHub Release catalogue (no
bundled model, 19 MB MSIX). Full measured story: `docs/technical-report.md`.
Sections below record everything that shipped and was measured on the way.

### Measured — in-app model download validated end-to-end on console (Exp 2 ✅)

The nobundle app on the Xbox downloaded the full SmolLM2-360M model (4 files,
417 MB merged `model.onnx`) from the **GitHub Release catalogue** inside the
AppContainer — `[manifest] using LocalState\manifest.json override` → all files
byte-exact on device → `.complete` written. Distribution is now self-hosted:

- **`models-v1` GitHub Release** carries the merged, AppContainer-safe model
  assets (the upstream HF repo ships a non-merged `model.onnx` stub + external
  data that §8 cannot load — that path stays broken upstream and is no longer
  referenced).
- `uwp/models/manifest.json` and the built-in fallback now point at the release
  URL; the LocalState manifest override was exactly the mechanism used to
  validate before flipping the default.
- **The MSIX no longer bundles a model** (ROADMAP Phase 4 milestone): first
  launch downloads from the catalogue; USB/Device-Portal provisioning unchanged.
  CI matrix simplified to `default` (nobundle) + `llamacpp`; the `xllama-appx`
  artifact is now the 19 MB no-model package.

### Measured — diffusion steps/seed plumbing on console

`diffuse-steps.txt=2` / `diffuse-seed.txt=777` with a new prompt produced a
coherent new 512×512 image in **7.7 s** (UNet 2.08 s/step ×2 — per-step cost
scales as expected; te/vae unchanged).

### Measured — llama.cpp CPU A/B on console: parity, not 2× (hypothesis falsified)

llama.cpp **runs on the Xbox in AppContainer** (first time): static ggml+llama
lane (`uwp/ggml-uwp.vcxproj`, `patches/0001-uwp-appcontainer-guards.patch`, CI
`build (llamacpp)`), SmolLM2-360M **Q4_K_M**, standard-512 ChatML prompt
(`bench/results/phase35-llamacpp-scaling.csv`):

| threads | prompt tok/s | decode tok/s |
| ------- | ------------ | ------------ |
| 1       | 141.6        | 19.9         |
| 4       | 141.3        | 51.5         |
| 6       | 141.5        | **62.9**     |
| 7–8     | —            | **livelock** |

- Versus ORT GenAI int4 @8t (66.3 decode / 220 prefill): **decode parity
  (−5 %), prefill worse** — Q4_K_M does not extract more bandwidth than ORT's
  AVX2 `MatMulNBits` on this machine; both saturate ~13 GB/s effective. The
  ROADMAP's "~2× → ~130 tok/s" hypothesis is falsified. **ORT GenAI stays the
  text backend**; the llamacpp lane remains in CI as a bench-only variant.
- t7/t8 livelock: ggml's spin-wait threadpool oversubscribes the ~6 cores Dev
  Mode leaves the app (no thread affinity in AppContainer). t6 is the ceiling.
- Real bugs fixed on the way: `#ifdef XLLAMA_USE_ORT` selected ORT even with
  `=0`; llama_tokenize size-query sign treated as error; `no_perf` (default)
  hid all timings (own chrono now, like the ORT path); ggml.c/ggml.cpp and
  ggml-cpu.c/.cpp same-dir obj collisions silently dropped every C symbol; the
  128 `src/models/*.cpp` per-arch files were missing from the static lib.

### Measured — image generation on console (v0.4.2.0, 2026-07-08) 🎨

**The flagship GPU workload is live**: SD-Turbo fp16 generates a coherent
512×512 image on the Xbox Series S GPU (DirectML) in **6.9 s** — text_encoder
1.0 s, UNet **3.3 s/step** (1 step), VAE decode 2.6 s; session load ~7.5 s
excluded (`bench/results/phase5-diffuse.csv`; image at
`docs/screenshots/diffuse-sd-turbo-xbox.png`, matches the local CPU validation
image for the same prompt/seed). Fixes required on the way, all measured:

- **VAE OOM with all sessions resident** (first run, 8007000E at
  `/decoder/up_blocks.3` InstanceNormalization): the 3801 MB budget doesn't fit
  ~2.4 GB of weights + the VAE's 512×512 activations. `run_diffuse` now creates
  and destroys each session per stage — text_encoder → UNet loop → VAE.
- **Deployable fp16 artifacts**: produced with `onnxruntime.transformers`'
  `convert_float_to_float16` (onnxconverter_common emits ORT-rejected mixed-type
  graphs for all 3 SD components — falsified options documented in
  `diffusion/README.md`); sessions cap graph optimization at **EXTENDED**
  (ORT_ENABLE_ALL crashes on these graphs). Each component self-contained
  < 2 GB: unet 1.65 GB, text_encoder 0.65 GB, vae 0.09 GB.
- **`deploy.sh mkdir_localstate` dropped the last path component** (while-read
  on an unterminated printf stream) — the true root cause of the earlier silent
  model-upload losses; fixed, uploads verified byte-exact.
- `run_diffuse` gained input-dtype adaptivity (fp16/fp32 floats, i32/i64 ids,
  i64/f32/f16 timestep), `diffuse-steps.txt`/`diffuse-seed.txt`, and per-stage
  telemetry (`diffuse-result.csv`).

### Added — model catalogue (`models/manifest.json`), model management de-hardcoded

The model list, download source, and downloadability gate were hardcoded in four
places; they now come from one catalogue file.

- `uwp/models/manifest.json`: name/display/`hf_base_url`/file-list per model,
  bundled in the MSIX (both variants) at `InstalledPath\models\manifest.json`;
  a `LocalState\manifest.json` uploaded via Device Portal **overrides it without
  a reinstall**. Parsed with WinRT `Windows::Data::Json` (`LoadModelManifest`,
  `uwp/model-downloader.cpp`) with a built-in fallback so the app never starts
  with an empty catalogue.
- Settings ComboBox is populated from the catalogue (plus the active model as a
  "(custom)" entry if it isn't listed — e.g. a dir uploaded under a new name);
  replaces the static 3-entry `kModels[]`.
- `EnsureModelAsync` downloads **any** catalogue entry with an `hf_base_url`
  (replaces the single-model `kDownloadableModel` gate + hardcoded repo URL +
  `SmolLM2_360M_Files()`); entries without a URL keep the USB/Device-Portal
  guidance error.

### Measured — Phase 3.5 console validation (Xbox Series S, v0.4.0.0, 2026-07-08)

The pending on-console checks for the merged 0.3.7–0.4.0 features, run in one session
per `docs/console-validation-runbook.md`. CSVs under `bench/results/phase35-*.csv`.

- **Image spike (flagship hypothesis) — CONFIRMED.** On a compute-bound fp16 batch
  (309 GFLOP, one UNet-step proxy), DirectML is **11.1× faster than CPU** (128.7 ms /
  2403 GFLOP/s vs 1428 ms / 216 GFLOP/s). The exact inverse of text decode — image
  generation plays to the GPU's strength, greenlighting the diffusion pipeline.
- **Decode matrix (ORT GenAI 0.14.1), SmolLM2-360M, 285-tok prompt:** CPU int4
  **66.3** tok/s, DML fp16 **46.8** (real GPU: engines ~46–57 %, `gpu_mem` 793 MB),
  DML int4 **8.82** (real GPU: engines ~87–90 %, `gpu_mem` 307 MB). Versus the v0.3.6
  baselines (68.0 / 46.8 / 8.8) the 0.14.1 bump is **flat on decode** — a valid,
  recorded result (its win, if any, is CPU-overhead at higher token rates, not here).
- **int4 DML floor — §12 desk-check CONFIRMED on hardware.** DML int4 decode is
  8.82 tok/s with the **GPU compute engines saturated (87–90 %)** — not a CPU
  fallback. The non-fused `MatMulNBits` (dequantize→fp16 + full GEMM) is
  bandwidth-bound; CPU int4 stays the decode default. Kernel-design limit, confirmed.
- **KV-cache reuse (Stage 2) — CONFIRMED.** Turn-2 prefill with reuse is **4.87×**
  faster than cold (103.7 ms / 22-tok delta vs 505.2 ms / 114-tok full re-prefill) —
  continuous decoding processes only the new turn's tokens as designed.
- **1.7B scale (§6) — CONFIRMED.** SmolLM2-1.7B cpu-int4 runs on the console:
  prefill 54.9 tok/s, **decode 20.6 tok/s**, peak WS 2423 MB, load 6.2 s. Decode
  scales ~3.2× down from 360M (66.3 → 20.6) — memory-bandwidth-bound, as expected;
  CPU int4 stays usable at 1.7B. (fp16-DML 1.7B remains undeployable: 3.4 GB weights
  exceed the 2 GB protobuf limit — a serialization constraint, not the GPU.)
- ~~Still pending~~ **Closed 2026-07-14:** CPU/GPU **routing** (§2, autopilot PASS);
  **diffusion** (§7, measured 2026-07-08); **TAESD** (§7c, autopilot PASS).

### Fixed — deploy/bench tooling gaps surfaced during console validation

- `scripts/deploy.sh upload-dir`: now verifies the target dir exists (WDP folder
  creation can fail silently) and checks each file's `Success` flag, failing loudly
  instead of reporting "Uploaded N" while landing nothing (the 1.7B model.onnx
  silently vanished this way until the missing subdir was created). Mirrors the
  `upload-file` check.
- `src/bridge/bench.cpp`: the CSV `backend`/`quant` columns were hardcoded
  `ort-genai-cpu`/`int4`, mislabelling DML and fp16 runs; now derived from the model
  dir name (+ `gpu_mem_mb` corroboration). The model label also no longer truncates
  at the first dot (`smollm2-1.7b-cpu-int4` was logged as `smollm2-1`).

### Added — C++ diffusion pipeline (`diffuse.flag`, host-validated)

The console image-generation pipeline. `uwp/diffuse.cpp` runs three ORT DirectML
sessions (text_encoder → 1× UNet denoise → VAE decode) behind a `diffuse.flag`
headless mode, mirroring the image spike's D3D12-clean host (887A0036-safe).

- Correctness-critical logic is **header-only, dependency-free, and unit-tested on
  the host** against golden vectors from the diffusers/transformers reference:
  `include/xllama/diffusion/clip_tokenizer.h` (CLIP byte-level BPE),
  `euler_scheduler.h` (EulerDiscreteScheduler, SD-Turbo 1-step), `half.h` (fp16),
  `png_writer.h` (self-contained PNG). `tests/test_diffusion.cpp` asserts all four
  against `diffusion/golden_vectors.json` — **638 assertions, all green** — so the
  logic ships in the un-runtime-testable console C++ only after matching Python.
- `diffusion/gen_golden_vectors.py` captures the reference (token ids for several
  prompts incl. empty + non-ASCII; scheduler sigmas/timesteps + one deterministic
  step). The CLIP tokenizer assets (`diffusion/clip_tokenizer/{vocab.json,merges.txt}`)
  are vendored so the test is hermetic and the console upload uses the same files.
- The tokenizer parses `vocab.json` with a minimal scanner (no JSON lib) so the
  header stays dependency-free for the UWP build (which skips the llama.cpp
  submodule); the host test uses vendored nlohmann only to load the golden file.
- Model contract: an **fp16** SD-Turbo-class model (each component < 2 GB,
  self-contained). The ORT DirectML orchestration is CI-compile-validated; runtime
  validation is on console per `docs/console-validation-runbook.md §7`.

### Corrected — int4 DML decode diagnosis (desk-check 2026-07-08)

The [0.3.6] entry below (and the docs at the time) attributed the int4 GPU decode
collapse (8.8 tok/s) to a "missing fused int4 DML kernel" that, if added, would
imply ~180 tok/s. A source-level desk-check **refines that**: `MatMulNBits` is
present in the graph (225 nodes) and **runs on the DML GPU** — the profiled run
shows one `DmlFusedNode` on `DmlExecutionProvider` (96%), no CPU fallback. The
real limit is that DirectML's `MatMulNBits` is **non-fused**
(`DML_DEQUANTIZE`→fp16 + full `DML_GEMM`, materialising fp16 weights), so int4
moves more bandwidth than fp16 and cannot beat it on DML; the builder also gives
DML `accuracy_level=0` vs CPU's fused-int8 `=4`. Full analysis + config tests in
`docs/uwp-constraints.md §12`; `ROADMAP.md` Phase 3.5 updated. Net: GPU int4
decode is blocked by a DirectML kernel-design limit (out of our control), not a
kernel we could contribute — CPU int4 stays the decode default.

### Added — diffusion model toolchain (image generation, `diffusion/`)

Model-side toolchain for running SD-Turbo (1-step distilled diffusion) on the
console GPU — the workload that plays to DirectML's strength (compute-bound fp16
batch), unlike LLM decode.

- **Validated on the owned layer (2026-07-08)**: SD-Turbo exported to ONNX and
  run through **ONNX Runtime (CPU EP)** generates a coherent 512×512 image in
  ~13 s / 1 step — the same artifact + runtime that runs on Xbox via the
  DirectML EP.
- `diffusion/export_sd_turbo.sh` + `requirements.txt`: reproducible, **pinned**
  toolchain (Python 3.10, torch 2.4.1 legacy ONNX tracer, optimum 1.23.3,
  transformers 4.46.3, diffusers 0.31.0). The pins are load-bearing: newer torch
  uses the dynamo/onnxscript exporter that optimum 1.23 can't consume (external-
  data-naming + LayerNormalization opset-downgrade bugs).
- `diffusion/generate_onnx.py`: validates the exported ONNX through ORT.
- `diffusion/convert_fp16.py`: confirms the fp16 components fit the 3801 MB
  budget — each < 2 GB (UNet ~1.65 GB, ~2.4 GB total), all self-contained,
  AppContainer-safe (unlike a 1.7B LLM fp16 blob). **Caveat**: the CPU fp16 pass
  leaves a mixed-type node in the UNet timestep embedding (`/time_proj/Mul`) that
  ORT rejects at load — a _runnable_ fp16 model needs a GPU export
  (`optimum-cli --fp16 --device cuda`) or Olive. fp32 is fully validated.
- Next: the C++ pipeline (3 ORT DirectML sessions + scheduler + CLIP tokenizer)
  behind a `diffuse.flag` headless mode, on the plain-ORT DirectML foundation
  already proven by the image spike (PR #3).

## [0.4.0.0] - 2026-07-08

### Added — image-generation spike (plain ORT DirectML, new model axis)

First step toward image/vision models on the console. The desk-check (§12) showed
the GPU loses at text decode because that workload is M=1, dispatch-bound, and
int4-DML is non-fused — the GPU's _worst_ case. Image generation (diffusion) is
the opposite: compute-bound fp16 batch, the case where DML already wins (prefill).
This spike tests that hypothesis cheaply before building a full diffusion pipeline.

- `uwp/image-spike.cpp`: runs a compute-bound conv model through the **plain
  ONNX Runtime DirectML** EP (not ORT GenAI — `onnxruntime.dll` was already a
  dependency) plus a CPU EP control, measuring forward-pass latency + GFLOP/s.
  Model `imgspike.onnx` (`scripts/gen_imgspike_model.py`, deterministic): 17
  Conv(3×3,64ch)+Relu on 1×3×512×512 fp16, ~309 GFLOP/forward — a faithful proxy
  for one diffusion UNet step.
- `uwp/App.cpp`: generalised the headless flag dispatch — `image.flag` →
  `run_image_spike()`, `bench.flag` → `main_loop()` (shared `HeadlessView`, same
  D3D12-clean host that avoids the 887A0036 compositor conflict).
- `uwp/xllama.vcxproj`: added the ORT DirectML NuGet include dir + the new source.
  `onnxruntime.lib` was already linked.
- Writes `imgspike-result.csv` (+ `.done`): DML vs CPU ms, GFLOP/s, GPU speedup.
- On-console validation: ✅ measured 2026-07-08 — DML **11.1× faster** than CPU on
  the compute-bound fp16 batch (2403 vs 216 GFLOP/s), the inverse of text
  decode; hypothesis confirmed and the SD-Turbo diffusion pipeline shipped
  in [1.0.0].

## [0.3.9.0] - 2026-07-08

### Added — per-conversation CPU/GPU routing (Stage 3, default off)

The v0.3.6 matrix showed the EP choice is per-workload: DML fp16 wins prompt
prefill at scale (1.8× at ~1k tokens), CPU int4 wins decode. The app can now
route between them.

- Settings `routing` (0 = CPU only / default = unchanged behaviour, 1 = GPU
  only, 2 = auto: route first prompts over ~500 est. tokens to GPU) + `gpu_model`
  (DML fp16 model dir, default `smollm2-360m-dml-fp16`); a routing ComboBox in
  the Settings dialog.
- The decision is made once at a conversation's first turn and **sticky** for its
  lifetime (the KV cache is per-EP): `m_active_model` holds the routed dir,
  cleared on new/loaded chat so each conversation re-decides. Reuses the existing
  single-slot `EnsureSession` (one model resident at a time) — switching
  conversations may reload, which is memory-safe on the 3801 MB budget.
- Default routing = CPU, so behaviour is identical until a user opts in. GPU
  routing requires the DML fp16 model present on device (LocalState/USB).
- On-console validation pending: confirm auto-routing picks GPU for long prompts
  and that TTFT improves for prompt-heavy conversations.

## [0.3.8.0] - 2026-07-08

### Added — KV-cache reuse across chat turns (continuous decoding)

The interactive app re-prefilled the **entire** ChatML history on every turn
(`BuildChatMLPrompt` + a fresh `OgaGenerator` per `generate()`), so turn-N TTFT
paid to re-process ~all prior tokens (~1.8k at budget → seconds). `OrtSession`
now keeps its generator alive across turns and appends only the new turn's
tokens (`OgaGenerator_AppendTokenSequences` on the persistent generator), so the
per-turn prefill covers just the delta.

- `GenerateParams` gains `reuse_kv` / `reset_kv`; `InferenceResult` gains
  `ended_with_stop` and now populates prefill telemetry (`n_p_eval`/`t_p_eval_ms`)
  on the interactive path too (previously bench-only). Decode timing excludes
  prefill, matching the bench convention.
- `OrtSession` holds `m_chat_gen`/`m_chat_params`/`m_chat_stream` + the bound
  sampling signature; the stateless path is preserved unchanged and still used
  when `reuse_kv` is false.
- `MainPageController::BuildDeltaPrompt` builds the incremental turn; the KV is
  reused only when valid and no context turn was evicted (RewindTo truncates the
  tail, not the head → eviction forces a full re-prefill). Reuse is invalidated
  on new/loaded chat, settings change, abort, and any generator failure.
- **Correctness guard**: a continuation that fails before emitting a token
  auto-falls back to a full re-prefill (no UI double-streaming) — worst case is
  the previous behaviour, never a wrong result.
- Settings toggle `kv_reuse` (ToggleSwitch + `settings.json`, default on) so the
  win can be A/B'd on console.
- On-console validation: ✅ measured 2026-07-08 — turn-2 prefill **4.87× faster**
  with reuse (103.7 ms vs 505.2 ms cold re-prefill); coherence confirmed.

### Added — multi-turn TTFT bench (Stage 2b)

- Headless bench gains a KV-reuse measurement: when `bench_turns.txt` is present
  (turn-2 user prompt; `prompt.txt` supplies turn 1), `main_loop` runs both turns
  on one persistent `Session` and measures turn-2 prefill **with reuse** (append
  only the delta) vs the **cold** baseline (full re-prefill of the 2-turn
  context), writing `bench-kv-result.csv` (+ `.done`) with a `speedup` column and
  logging the numbers. This measures the Stage 2 win on console instead of
  assuming it.

## [0.3.7.0] - 2026-07-08

### Changed — ONNX Runtime GenAI 0.13.2 → 0.14.1

- Bumped `Microsoft.ML.OnnxRuntimeGenAI.DirectML` NuGet from 0.13.2 to **0.14.1**
  (`uwp/packages.config`, `uwp/xllama.vcxproj` — 8 package-path references).
  0.14.x reduces CPU-side per-token overhead in `GenerateNextToken`/`SampleTopP`
  (directly relevant to the decode bottleneck measured in the v0.3.6 matrix) and
  is the prerequisite for continuous decoding / KV-cache reuse (`RewindTo`,
  generator reuse — Stage 2). No breaking C-API changes vs 0.13; existing model
  directories (built with model builder 0.14.1) load unchanged.
- On-console validation: ✅ measured 2026-07-08 — decode flat vs v0.3.6 (CPU int4
  66.3 vs 68.0; DML fp16 46.8 = 46.8): the bump is a prereq/overhead win, not a
  decode-rate win at this scale.

### Docs

- `ROADMAP.md`: new Phase 3.5 — Hardware Ceiling with the ordered unlock
  levers (Game-mode designation, 1B+ models via no-bundle, int4-AWQ proxy,
  upstream `MatMulNBits` desk check, llama.cpp CPU A/B, per-workload routing,
  optional upstream kernel contribution); Phase 2 heading refined to the
  per-workload verdict; "GPU vs CPU same model" milestone closed by the
  v0.3.6 matrix.
- `docs/uwp-constraints.md` §5: stale "future work" paragraph replaced with
  the measured per-workload approach; documented the App-vs-Game designation
  lever — **settled 2026-07-08**: the package was found already designated
  Game, so all measured figures are Game-mode numbers (the interim "all
  numbers are App-mode" assumption was wrong and has been rectified in
  §5/§7/§11); the GPU decode gap is a DML/kernel issue, not platform
  scheduling.
- Environment change (2026-07-08): Dev Mode storage allocation raised to
  90 GB — the Q:\ ~2.2–2.5 GB disk budget (§9) is superseded as the binding
  constraint for model sizing; ROADMAP Phase 3.5 updated accordingly.
- `docs/device-portal.md`: new "Lifecycle gotchas" section — LocalState purge
  semantics (only forward upgrades preserve it), LocalState absent before
  first launch, WDP file APIs returning HTTP 200 with `"Success": false`,
  stale PFN staging window, portal unreachable while a game is running.

## [0.3.6] - 2026-07-07

### Measured — Hardware utilization matrix (prefill vs decode, CPU vs GPU)

SmolLM2-360M, 3 variants × 2 prompts (285 / ~1050 tok), v0.3.6 timing:

| Variant  | prefill 285 | prefill ~1050 | decode short | decode long |
| -------- | ----------- | ------------- | ------------ | ----------- |
| CPU int4 | **220**     | 198           | **68.0**     | **50.9**    |
| GPU int4 | 152         | 334           | 8.8          | 8.3         |
| GPU fp16 | 169         | **354**       | 46.8         | 36.5        |

- **Prefill crossover**: GPU scales with batch, CPU doesn't — at ~1k prompt
  tokens the GPU is 1.8× faster (TTFT 3.0 s vs 5.3 s). DML fp16 wins
  prompt-heavy workloads today.
- **int4 GPU decode collapse = missing fused int4 kernel**, not dispatch: fp16
  decode is 5.3× int4 on DML. Effective bandwidth: GPU fp16 ~34 GB/s vs CPU
  ~13 GB/s (bus ~224 GB/s) — the GPU exploits memory 2.6× better; a
  `MatMulNBits`-class DML kernel would imply ~180 tok/s (2.5× CPU).
- Verdict in `docs/uwp-constraints.md §5`; matrix rows appended to
  `bench/results/phase2-dml.csv`. New model on-device:
  `smollm2-360m-dml-fp16` (builder `-p fp16 -e dml`, 691 MB merged).

### Fixed

- Prefill timing (0.3.5) started the clock after `AppendTokenSequences`, but
  on-console measurement showed the prompt prefill runs **inside**
  `AppendTokenSequences` in ORT GenAI 0.13.2 (the first `GenerateNextToken`
  returns in ~40 µs) — `prompt_tok_s` came out in the millions. The clock now
  starts before `AppendTokenSequences`, covering the prefill under either
  implementation. Decode numbers were unaffected (measured after prefill in
  both layouts).

## [0.3.5] - 2026-07-07

### Added

**Prefill measurement in the ORT path** (`prompt_tok_s` was always 0.00):

- `src/bridge/inference.cpp`: ORT GenAI runs the prompt prefill inside the
  _first_ `GenerateNextToken` call; it was never timed (`n_p_eval`/`t_p_eval_ms`
  were only populated by the legacy llama.cpp path). Now the first loop
  iteration is timestamped: `prompt_tok_s` lands in the bench CSV and the log
  gains `prefill=N tok/s (M tok, T ms)`.
- `decode_tok_s` is now prefill-free (rate over `n_generated-1` tokens from the
  end of the first iteration) — previous decode numbers were slightly
  underestimated because they included prefill time.
- `bench/prompts/long-1k.txt`: ~1k-token prompt to exercise prefill in the
  CPU-vs-GPU utilization matrix.

### Upstream fix validated on hardware (2026-07-07, evening)

The `887A0036` DML init failure was fixed at the source and validated on
console before contributing upstream:

- Patch to `onnxruntime-genai` `CreateDmlObjects`: fall back to the system
  D3D12 runtime when the Agility SDK device factory cannot create a device
  (upstream PR [microsoft/onnxruntime-genai#2280](https://github.com/microsoft/onnxruntime-genai/pull/2280)).
- Validated with a throwaway 0.3.9 test MSIX (branch
  `test/dml-fallback-validation`: patched DLL injected over the NuGet one,
  headless detection disabled to force the XAML repro path):
  - XAML + DML + vanilla DLL: `887A0036` at `OgaCreateModel` (known repro)
  - XAML + DML + **patched DLL: loads in 886 ms, 315 MB on GPU, full decode
    8.8 tok/s** — identical to the headless/Agility path
  - XAML + CPU + patched DLL: 67.2 tok/s — no regression
- Console restored to the official 0.3.4 MSIX afterwards.

## [0.3.4] - 2026-07-07

### Added

**Headless bench mode — unblocks DML EP init** (root cause of the 0.3.3
`887A0036` finding, now identified at the exact source line):

- Root cause: ORT GenAI 0.13.2 creates its D3D12 device through the **Agility
  SDK device factory** (`dml_helpers.cpp` `CreateDmlObjects`:
  `CreateDeviceFactory(614, module_path)` succeeds on Xbox OS 26100 via the
  in-box runtime ≥ 614, no app-local `D3D12Core.dll` needed — verified absent
  from our MSIX). The factory's `CreateDevice` then collides with the
  process-wide D3D12 device the **XAML compositor** (D3D11on12) created at
  `Window.Activate()` — two different D3D12 runtimes cannot share a process →
  `DXGI_ERROR_ALREADY_EXISTS`. This also reconciles Exp 1 (May): the then-OS
  had in-box < 614, so ORT fell back to plain `D3D12CreateDevice` (line 144)
  which coexists with the compositor device — the OS update flipped the branch.
  No upstream fix on `main`; v0.14.0 has identical code.
- `uwp/App.cpp`: when `LocalFolder\bench.flag` exists at startup, `wWinMain`
  skips `Application::Start` entirely and runs a minimal `IFrameworkView`
  (`BenchView`) via `CoreApplication::Run` — CoreWindow activated (PLM watchdog
  satisfied) but **no XAML compositor, no in-process D3D12 device** — then runs
  `xllama::bridge::main_loop()` on an MTA thread and exits via
  `CoreApplication::Exit()`. Interactive path unchanged (flag absent →
  `uninit_apartment` + `Application::Start` as before); `CheckBenchMode` in
  MainPage stays as fallback if the early detection throws.

### Fixed

- `bench/configs/genai_config-dml-{test,profile,profile.tpl}.json`:
  `past_present_share_buffer` `false` → `true` — the DML EP uses graph capture,
  which rejects `false` at `OgaCreateGenerator` ("Graph capture is not
  supported with past_present_share_buffer set to false"). The `false` was an
  Exp 1 leftover that only ever ran on the silent-CPU-fallback path.

### Measured — GPU-truth verdict on console (2026-07-07, headless bench mode)

**`VERDICT: GPU`** — first proven DML GPU execution in the Xbox UWP sandbox:

- ORT profiler: `DmlExecutionProvider` 10.7 ms kernel time (fused node 9.7 ms),
  `CPUExecutionProvider` 0.4 ms → 96% DML.
- In-app probe: `gpu-mem post-load: current=411MB` ≈ model size (403 MB) —
  weights resident on the RDNA 2 GPU. CPU control run: `current=0MB`,
  69.7 tok/s.
- The bundled CPU-int4 `MatMulNBits` model fails inside the fused DML node
  (`80070057`) — a DML model variant is required for full decode (below).

### Measured — Phase 2 end-to-end DML bench (2026-07-07, evening)

SmolLM2-360M INT4 **DML** variant, built with the ORT GenAI model builder
(0.14.1, `-p int4 -e dml`, 285 MB merged), uploaded to
`LocalState\models\smollm2-360m-dml-int4`:

| Backend            | decode tok/s | load ms | gpu_mem MB | peak WS MB |
| ------------------ | ------------ | ------- | ---------- | ---------- |
| DML EP (RDNA 2)    | **8.83**     | 927     | 307        | 999        |
| CPU EP (Zen 2 t=8) | **70.9**     | 1518    | 8          | 771        |

- Full decode completes on GPU (739 tokens, `VERDICT: GPU`, GPU compute engine
  ~88% saturated via `systemperf`) — but the **CPU is ~8× faster**: per-token
  DML dispatch overhead dominates a 360M model while `MatMulNBits`/AVX2 is
  highly optimised. **Phase 2 closed: CPU EP stays the production backend.**
- GPU pool estimate corrected: measured `Budget` is **3801 MB** (the "~768 MB
  pool" was OOM-bracketing inference). Disk (`Q:\` ~2.2–2.5 GB) is the real
  model-size constraint.
- Known cosmetic issue: bench CSV `backend` column says `ort-genai-cpu` even on
  DML runs (label hardcoded in `bench.cpp`, does not detect the configured EP).

## [0.3.3] - 2026-07-07

### Changed

- `src/bridge/inference.cpp`: dropped the `gpu-mem pre-load` probe that ran
  _before_ `OgaCreateModel`. `gpu_mem_info()` opens and caches an
  `IDXGIAdapter3` on adapter 0; not holding that handle open across model load
  is cleaner. GPU memory is still sampled `post-load`/`post-decode`, after the
  session exists — where the GPU-truth signal (`current` ≈ model size) matters.
  **Note:** this was first hypothesised to be the cause of the DML init failure
  below, but the rebuild (0.3.3) disproved it — see _Investigated_. The change
  is kept as harmless cleanup, not a fix.

### Investigated — GPU-truth DML experiment on console (2026-07-07)

First profiled DML run on Xbox Series S, using the 0.3.2 GPU-truth toolkit.
**Result: the DirectML EP does not initialise at all** on this stack — it is
neither GPU execution nor a CPU fallback.

- `OgaCreateModel` throws `887A0036` "The desired element already exists" at
  `onnxruntime-genai .../dml/dml_helpers.cpp(140)`, **before any kernel runs**,
  so no `ort_profile_*.json` is produced and the profiler emits no `VERDICT:`.
- Reproduced 3×: profiling config (0.3.2), plain `dml-test` config (0.3.2), and
  after the pre-load-probe removal (0.3.3) — identical signature each time.
- **Not OOM** (`avail_phys` 5.0 GB, `budget` 3801 MB), **not the profiling
  config** (plain config fails identically), **not our telemetry** (0.3.3 fix
  changes nothing). Cause is inside the ORT GenAI DML EP device creation.
- Likely a D3D12 single-device-per-adapter conflict: a UWP XAML app already
  holds a D3D12 device (compositor) on adapter 0 before `OgaCreateModel`.
- Corroborated by telemetry: GPU engines flat (only the display engine ~99%),
  `gpu_dedicated` ~100 MB (Dev Home noise), no compute spike.
- Reconciles Exp 1 (2026-05-23): its "loads without OOM, 71.7 tok/s ≈ CPU
  baseline, GPU execution unconfirmed" was almost certainly a **silent CPU
  fallback**, never real DML execution. Same ORT GenAI 0.13.2, same config.
- **Conclusion:** DML EP is not viable on ORT GenAI 0.13.2 in the Xbox UWP
  sandbox; the CPU EP is the only working backend (70.9 tok/s control run, same
  session). Supersedes the earlier "GPU EP ruled out via OOM" with a precise
  init-failure signature.

## [0.3.2] - 2026-07-07

MSIX carrying the GPU-truth toolkit + in-app `gpu_mem` telemetry (CWD pin to
LocalState) for the first profiled DML run on console.

### Added

**Bench infrastructure — ORT GenAI thread tuning** (`feat(bench): ORT GenAI bench infrastructure`):

- `scripts/bench-xbox-ort.sh`: new bench orchestrator for ORT GenAI models already on device (no model upload). Supports `--threads N`, `--runs N`, `--prompt file`. Appends median row to `bench/results/phase1-cpu.csv`. Drops warmup run automatically.
- `bench/configs/genai_config-threads-{4,6,8}.json`: `genai_config.json` variants with `intra_op_num_threads` set for Zen 2 thread-count tuning.
- `uwp/inference-bridge.cpp`: reads optional `bench_threads.txt` (uploaded per bench variant) to set `params.n_threads` for CSV tracking and suffix host_label (`xbox-series-s-tN`).

**GPU-truth debug toolkit — DML EP attribution without PIX** (see `docs/uwp-constraints.md §11`):

- `bench/configs/genai_config-dml-profile.json` (+ `.tpl.json` absolute-prefix variant): DML EP config with ORT `enable_profiling` + `log_severity_level: 0`.
- `scripts/profile-dml-run.sh`: one profiled DML run — config swap, bench run, fetch `ort_profile_*.json` + log tail into `bench/results/profiles/<ts>/`, restore config, analyze.
- `scripts/analyze_ort_profile.py`: per-provider kernel-time summary from the ORT profiling JSON; greppable `VERDICT: GPU | MIXED | CPU-FALLBACK` line; tolerates truncated traces.
- `scripts/xbox-gpu-sample.sh`: WDP `systemperf` sampler (per-engine GPU utilization + VRAM used → CSV + max/mean summary); `--gpu-sample` integration in `profile-dml-run.sh` and `bench-xbox-ort.sh`.
- `gpu_mem_info()` (`src/bridge/platform.cpp`): per-process `IDXGIAdapter3::QueryVideoMemoryInfo` (LOCAL); logged pre-load/post-load/post-decode; `dxgi.lib` linked in `uwp/xllama.vcxproj`.
- `set_cwd_to_local_folder()`: bench mode pins CWD to LocalState so the relative ORT profiling prefix lands in a writable, WDP-fetchable location.
- Bench CSV schema: new `gpu_mem_mb,gpu_budget_mb` columns before `host,date` (header updated in `bench.cpp`, both bench scripts, `bench/README.md`; existing `phase1-cpu.csv` rows backfilled with `0,0`); `--out FILE` flag in `bench-xbox-ort.sh` (DML runs → `bench/results/phase2-dml.csv`).
- CI (`build-linux`): smoke step for the analyzer (fixtures in `tests/fixtures/`) and the sampler parser.

**No-bundle MSIX build variant — unblocks Exp 2 validation**:

- `uwp/xllama.vcxproj`: model `ItemGroup` now also conditioned on `'$(XllamaNoBundledModel)' != 'true'`.
- `scripts/build-uwp.ps1 -NoBundledModel`: builds an MSIX without the bundled SmolLM2 model, so `EnsureModelAsync` reaches the USB/HF-download fallbacks on console.
- `.github/workflows/build-uwp.yml`: matrix `variant: [bundled, nobundle]`; the `nobundle` job skips model download/merge and uploads artifact `xllama-appx-nobundle`.

### Fixed

- `src/bridge/bench.cpp`: backend label was `"directml"` even on CPU EP → corrected to `"ort-genai-cpu"`; quant `"int4-awq"` → `"int4"`.
- `src/bridge/inference.cpp` (ORT path): `load_ms` was always 0 in the bench CSV — `run_inference` never measured model load. Now times `OgaCreateModel` wall-clock and logs `ORT model loaded in N ms`.
- Docs drift: `bench/README.md` results table said "pending" for the populated `phase1-cpu.csv`; `bench/README.md` + `docs/phase1-runbook.md` still documented the old `directml` backend label.

### Measured — Phase 1 bench results (Xbox Series S Zen 2, 2026-05-23)

SmolLM2-360M-Instruct INT4 CPU, ORT GenAI 0.13.2, n=990:

| n_threads          | decode tok/s | peak RAM MB | notes                                           |
| ------------------ | ------------ | ----------- | ----------------------------------------------- |
| auto (ORT default) | 66.9         | 704         | baseline, no `intra_op_num_threads`             |
| 4 (explicit)       | **71.4**     | 771         | **best**                                        |
| 6 (explicit)       | 68.0         | 772         |                                                 |
| 8 (explicit)       | 28.2         | 771         | severe regression — memory bandwidth saturation |

**Recommendation**: use `intra_op_num_threads: 4` in `genai_config.json` for SmolLM2-360M on Zen 2.

### Investigated — Exp 1 DirectML (2026-05-23)

- DML `genai_config.json` (provider: dml, `enable_cpu_mem_arena=0`, `enable_mem_pattern=0`) loads without SEH 0xC0000005 on SmolLM2-360M INT4.
- Performance: 71.7 tok/s — indistinguishable from CPU baseline.
- Conclusion: SmolLM2-360M INT4 (~200 MB ONNX) likely fits within the 768 MB UWP GPU pool. Cannot confirm GPU execution vs CPU fallback without D3D profiling tools. Phase 2 "blocked" status revised: **360M model fits; larger models still blocked**.
- Exp 2 (HF in-app download): unreachable with current bundled MSIX. `EnsureModelAsync` checks InstalledPath before HF download; model is always found there. Requires a separate build without bundled model to validate — now available via `build-uwp.ps1 -NoBundledModel` / CI artifact `xllama-appx-nobundle`.

### Investigated — model candidates via HF Hub file sizes (2026-07-02)

- Qwen2.5-0.5B INT4 CPU: ~822 MB `model.onnx.data` (rtn-block-32) — the ~200 MB estimate was wrong (151k-vocab embedding not INT4-quantized). Ruled out for disk and GPU pool. DML int4-awq variant ~507 MB: borderline GPU-pool fit, possible DML retry via USB.
- Llama-3.2-1B INT4 CPU: ~1.77 GB — USB-only, same class as SmolLM2-1.7B. Details in `docs/model-selection.md`.

---

## [0.3.0] — 2026-05-23

### Added

**Settings dialog — sampling parameters** (`feat(uwp): Settings dialog — sampling params`):

- `ShowSettings` now exposes `temperature` (Slider 0–2, default 0.8), `top_p` (Slider 0–1, default 0.9), `top_k` (NumberBox 1–200, default 40), `repetition_penalty` (Slider 1–2, default 1.1), and `n_predict` (NumberBox 16–2048, default 512).
- New `MainPageController` members `m_temperature`, `m_top_p`, `m_top_k`, `m_repetition_penalty`, `m_n_predict` wired into `StartInference` → `GenerateParams`.
- `settings.json` schema extended with a nested `"sampling"` object; back-compat preserved for existing 0.2.x files.

**Settings dialog — model selection ComboBox** (`feat(uwp): Settings dialog — model selection ComboBox`):

- ComboBox with three entries: SmolLM2-360M (bundled MSIX), SmolLM2-1.7B (USB `E:\xllama\models\`), SmolLM2-360M (HF download in `LocalState`).
- Selected model persisted to `settings.json`; `EnsureSession` detects model change at next `StartInference` and rebuilds transparently.
- `LoadModelName` now reads `m_model_filename` from `settings.json`; falls back to `LocalState/model.txt` for 0.2.x installations.

**History dialog enhancements** (`feat(uwp): History dialog — delete, clear all, timestamps`):

- Per-item ✕ Delete button: click closes the dialog and opens a confirmation ContentDialog; on confirm calls `ChatHistory::Delete(id)`. If deleted entry was the active conversation, `NewChat()` is called.
- Clear all (Secondary button): confirmation ContentDialog → `ChatHistory::Clear()` → `NewChat()`.
- Current conversation indicator: ● prefix on the active history entry.
- Relative timestamps via `FormatRelativeTs` helper: "today HH:MM", "yesterday HH:MM", "DD Mon HH:MM".
- Index refreshed from disk on every `ShowHistory` call.
- Empty-state ContentDialog with placeholder TextBlock instead of silent no-op.

**ChatHistory::Delete / Clear** (`feat(uwp/chat-history): add Delete and Clear methods`):

- `Delete(id)` removes `<id>.json` from `LocalState/chats/` and updates the in-memory index + `index.json`.
- `Clear()` removes all conversation files and writes an empty `index.json`.

**Tests** (`test: add ChatHistory helpers and TitleFrom smoke tests`):

- `tests/test_chat_history.cpp`: Linux CI tests exercise `TitleFrom` logic (truncation, newline stop, empty fallback). UWP `#ifdef` branch tests `Save`/`Load` roundtrip, `Delete`, and `Clear` with a temp directory.

### Fixed

- **Newline rendering**: `AppendOutput` now splits text on `\n` and inserts `LineBreak` inlines; previously `\n` in a `Run` was rendered as a space by WinUI `RichTextBlock`.
- **Prompt not cleared after Run**: `OnRunClick` clears `m_promptInput.Text(L"")` before handing off to `StartInference`.
- **NewChat did not clear prompt**: `NewChat()` now resets `m_promptInput.Text(L"")`.
- **Double FocusEngagement on Xbox**: removed `IsFocusEngagementEnabled(true)` from `m_outputScroll`; only the TextBox retains it, eliminating the extra A-press required to engage text input.
- **Focus not returned after generation**: `SetRunning(false)` now calls `m_promptInput.Focus(FocusState::Programmatic)` so the cursor returns to the input after inference completes.
- **Smart autoscroll**: `AppendOutput` auto-scrolls only when the user is already at the bottom (within 24 px of `ScrollableHeight`). A `ViewChanged` handler tracks `m_at_bottom`; `SetRunning(false)` resets it to `true`.
- **Status shows "Loading model…" at startup**: `BuildUI` initialises `m_statusText` to `L"Loading model..."` and disables the Run button. `EnsureModelAsync` sets `L"Ready"` and re-enables Run when the model is confirmed loaded.
- **Context trim overflow**: `kMaxEstimatedTokens` lowered from 3500 to 1800 to stay within `n_ctx = 2048` (leaves ~250 token generation headroom). Trim events now surface a `"Context trimmed: N turns dropped"` status message.
- **Partial save on cancel**: `StartInference` completion path sets `ChatMessage::partial = true` when the user pressed Cancel (`m_abort.load() == true`) before saving the conversation.

---

## [0.2.1] — 2026-05-23

### Added

- ChatML stop sequence `<|im_end|>` in UI inference path (`uwp/MainPage.cpp`). SmolLM2-360M does not always emit EOS naturally; without this the model would continue generating filler or hallucinate the next user turn up to `n_predict=512`. Bench path unchanged.
- `tests/test_session.cpp`: smoke tests for `Session::create` error paths (non-existent path, empty path) — covers the Linux/llama.cpp path in CI.

### Fixed

- `CHANGELOG.md` 0.2.0 section: collapsed duplicate `### Added` blocks; removed stale empty `[Unreleased]` header.

---

## [0.2.0] — 2026-05-22

### Added

**Persistent inference session** (`feat(uwp): integrate xllama::Session into MainPage`):

- `MainPageController` now keeps an `xllama::Session` alive across chat turns; subsequent turns skip model reload entirely (~1–2 s overhead eliminated after first turn).
- `EnsureSession()` private helper: lazy-build on first turn, transparent rebuild on model change (Settings), free-then-alloc to avoid 2× RAM during transitions.
- Bench mode (`inference-bridge.cpp`) unchanged — continues to call `run_inference()` for cold-load measurement.

**Bench diagnostics** (`bench(inference): log prompt token count + bump bench n_predict`):

- `inference.cpp`: logs `[xllama] prompt=N tok, max_length=M (new≤K)` after tokenisation — makes `n` in bench CSV self-explanatory.
- `inference-bridge.cpp`: bench `n_predict` raised `128 → 512` (effective `max_length` 640 → 1024 total tokens); gives SmolLM2-360M room to show natural generation length while 1.7B still exits at EOS.

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

- `fix(bridge): OrtModelPtr → OgaModelPtr` typo in `OrtSession` UWP build (MSVC `C2065`; GCC/clang skip the `XLLAMA_USE_ORT` block on Linux).
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
