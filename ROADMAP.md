# xllama Roadmap

## Phase 1 — CPU Baseline ✅ DONE

**Goal**: working UWP build, ORT GenAI CPU EP inference on Xbox Series S, MSIX self-contained.

Milestones:

- [x] Linux build green (CMake presets + CI `build-linux`)
- [x] Modular C++ bridge with RAII wrappers, CLI parser, unit tests (doctest)
- [x] Pivot from llama.cpp to ONNX Runtime GenAI + DirectML (`XLLAMA_USE_ORT`)
- [x] XAML-free programmatic UI (`MainPageController`, no WMC9999 workarounds)
- [x] GPU EP evaluated and ruled out: UWP pool ~768 MB → OOM on any usable LLM
- [x] CPU EP adopted: SmolLM2-360M-Instruct INT4 (403 MB), Zen 2 compatible
- [x] ONNX external data merged into self-contained `model.onnx` (AppContainer `weakly_canonical` fix)
- [x] SmolLM2-360M bundled inside MSIX as `DeploymentContent`; CI `build-uwp` green
- [x] ChatML prompt template applied for SmolLM2 Instruct

## Phase 2 — GPU Acceleration ❌ NOT VIABLE (DML EP does not initialise)

**Goal**: DirectML EP inference using Xbox RDNA 2 hardware.

**Status (2026-07-07, xllama v0.3.3)** — GPU-truth experiment run on console:

_DML EP does not initialise._ `OgaCreateModel` throws `887A0036` "The desired
element already exists" at `onnxruntime-genai .../dml/dml_helpers.cpp(140)`,
**before any kernel runs** — so the ORT profiler produces no `VERDICT:`. Neither
GPU execution nor CPU fallback: the EP fails at device creation. Reproduced 3×
(profiling config, plain `dml-test`, and after the 0.3.3 pre-load-probe removal).
Not OOM (`avail_phys` 5.0 GB, `budget` 3801 MB), not our telemetry, not the
profiling config. Likely a D3D12 single-device-per-adapter conflict with the UWP
XAML compositor. See `CHANGELOG.md` [0.3.3] and `docs/uwp-constraints.md §7`.

_Exp 1 reconciled_: its "loads without OOM, **71.7 tok/s ≈ CPU baseline**, GPU
execution unconfirmed" (2026-05-23) was almost certainly a **silent CPU
fallback**, never real DML execution — same ORT GenAI 0.13.2, same config.

**Conclusion**: on ORT GenAI 0.13.2 in the Xbox UWP sandbox, the CPU EP is the
only working backend (70.9 tok/s control run). GPU acceleration would require a
different ORT GenAI version, a GDK (non-UWP) path, or resolving the DML device
conflict. Superseded the earlier "GPU EP ruled out via OOM" with a precise
init-failure signature.

Milestones:

- [x] Exp 1: test DML provider_options on SmolLM2-360M — loads without OOM; later found to be silent CPU fallback (see Status)
- [x] GPU-truth toolkit in place of PIX (unavailable in Dev Mode): ORT profiler with per-kernel EP attribution (`profile-dml-run.sh` + `analyze_ort_profile.py`), WDP `systemperf` GPU counters (`xbox-gpu-sample.sh`), in-app `QueryVideoMemoryInfo` (`gpu_mem_mb` CSV columns) — see `docs/uwp-constraints.md §11`
- [x] Run the profiled DML experiment on console — ❌ DML EP fails to init (`887A0036` at `OgaCreateModel`); no `VERDICT:` obtainable; CPU control run 70.9 tok/s
- [x] Evaluate Qwen2.5-0.5B INT4 ONNX as GPU EP candidate — ❌ CPU-int4 is ~822 MB (not ~200 MB); only the DML int4-awq variant (~507 MB) borderline fits the pool (see `docs/model-selection.md`)
- [~] Validate DirectML EP inference end-to-end — ❌ blocked: DML EP does not initialise on this stack
- [~] Measure GPU vs CPU tok/s for the same model — ❌ blocked: no GPU execution possible

## Phase 3 — Benchmarks + Model Exploration 🔄 IN PROGRESS

**Goal**: reproducible results, n_threads tuning, evaluate alternative compact models.

Milestones:

- [x] Populate `bench/results/phase1-cpu.csv` — SmolLM2-360M INT4 at t=4,6,8; optimal: **t=4 → 71.4 tok/s**
- [x] Tune `n_threads` on Zen 2 Xbox: t=4 optimal; t=8 causes severe regression (28 tok/s — memory bandwidth saturation)
- [x] `bench-xbox-ort.sh` + `bench_threads.txt` mechanism for automated multi-thread bench runs
- [x] Evaluate Qwen2.5-0.5B INT4 ONNX — ❌ ~822 MB real (vocab embedding dominates); exceeds disk borderline and GPU pool
- [x] Evaluate Llama-3.2-1B INT4 ONNX CPU — ❌ ~1.77 GB real; USB-only, same class as SmolLM2-1.7B
- [x] `load_ms` in bench CSV: column existed but ORT path never measured it (always 0) — `run_inference` now times `OgaCreateModel`; baseline pending next bench run on console

## Phase 4 — In-App Download + Publication 🔮 FUTURE

**Goal**: remove bundled-model constraint; demo video; technical write-up.

Milestones:

- [x] `ModelDownloader` with chunked `HttpClient` download from HF endpoint (Exp 2, `uwp/model-downloader.cpp`)
- [x] `EnsureModelAsync()` bootstrap: LocalState → InstalledPath → HF download fallback chain
- [x] USB external drive fallback at `E:\xllama\models\<name>` in `resolve_model_path` (Exp 3)
- [x] No-bundle build variant to unblock Exp 2: `build-uwp.ps1 -NoBundledModel` (MSBuild `XllamaNoBundledModel=true` excludes the model ItemGroup); CI publishes `xllama-appx-nobundle` alongside `xllama-appx`
- [ ] Validate Exp 2 on console: deploy the `xllama-appx-nobundle` MSIX, confirm HF HTTPS download works from Xbox AppContainer
- [ ] Remove model bundle from MSIX once Exp 2 is validated (delete ItemGroup in `xllama.vcxproj`)
- [ ] `model-manifest.json`: configurable model list (HF repo + file list) for model switching
- [ ] Demo video: model loaded and running on Xbox hardware
- [ ] Technical report (arXiv or GitHub Discussions)
- [ ] Tagged v1.0.0 release with pre-built MSIX and model manifest
