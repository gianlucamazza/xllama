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

## Phase 2 — GPU Acceleration ⚠️ PARTIALLY UNBLOCKED

**Goal**: DirectML EP inference using Xbox RDNA 2 hardware.

**Status (2026-05-23, xllama v0.3.1)**:

_Exp 1 completed_: SmolLM2-360M INT4 loads with DML provider_options (`enable_cpu_mem_arena=0`, `enable_mem_pattern=0`, `past_present_share_buffer=false`) WITHOUT `0xC0000005` OOM. Performance: **71.7 tok/s** ≈ CPU baseline (t=4, 71.4 tok/s). Phi-3.5-mini (~2.2 GB) still causes OOM as expected.

**Open question**: whether ORT actually routes SmolLM2-360M INT4 to the RDNA 2 GPU or silently falls back to CPU cannot be determined without D3D performance profiling (PIX or hardware GPU counters). No profiling infrastructure is available on-device yet.

**Revised blocker**: models ≥ ~1 GB remain blocked by the ~768 MB GPU pool. Sub-400 MB INT4 models physically fit the pool. Actual GPU utilisation is unconfirmed.

**Conditions to confirm Phase 2 complete**:

- D3D performance profiling confirms GPU kernel execution (not CPU fallback)
- tok/s on GPU visibly exceeds CPU baseline for the same model/quant

Milestones:

- [x] Exp 1: test DML provider_options on SmolLM2-360M — loads without OOM; GPU execution unconfirmed
- [x] GPU-truth toolkit in place of PIX (unavailable in Dev Mode): ORT profiler with per-kernel EP attribution (`profile-dml-run.sh` + `analyze_ort_profile.py`), WDP `systemperf` GPU counters (`xbox-gpu-sample.sh`), in-app `QueryVideoMemoryInfo` (`gpu_mem_mb` CSV columns) — see `docs/uwp-constraints.md §11`
- [ ] Run the profiled DML experiment on console: `VERDICT:` from `profile-dml-run.sh` + control run on CPU config
- [x] Evaluate Qwen2.5-0.5B INT4 ONNX as GPU EP candidate — ❌ CPU-int4 is ~822 MB (not ~200 MB); only the DML int4-awq variant (~507 MB) borderline fits the pool (see `docs/model-selection.md`)
- [ ] Validate DirectML EP inference end-to-end: confirm GPU tok/s > CPU tok/s
- [ ] Measure GPU vs CPU tok/s for the same model at same quant level

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
