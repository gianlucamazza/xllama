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

## Phase 2 — GPU Acceleration ✅ COMPLETE — GPU proven, CPU wins 8× (DML not competitive)

**Goal**: DirectML EP inference using Xbox RDNA 2 hardware.

**Status (2026-07-07, xllama v0.3.4)** — end-to-end DML bench completed:

**`VERDICT: GPU` + full decode** with a purpose-built SmolLM2-360M INT4 **DML**
variant (285 MB, ORT GenAI model builder `-p int4 -e dml`): weights resident on
GPU (`gpu-mem post-load: 307 MB`), GPU compute engine ~88% saturated during
decode, 739 tokens generated. **Result: 8.83 tok/s on GPU vs 70.9 tok/s on CPU
— the Zen 2 CPU is ~8× faster** (median of 3 runs, `phase2-dml.csv`).
Autoregressive decode of a 360M model is dominated by per-token DML dispatch
overhead; `MatMulNBits` on AVX2 wins decisively at this scale. The bundled
CPU-int4 variant is not DML-compatible (`80070057` in the fused node).

_Root cause history_ (see `CHANGELOG.md` [0.3.3]/[0.3.4],
`docs/uwp-constraints.md §7`): `887A0036` at `dml_helpers.cpp(140)` was the
**Agility SDK device factory** (`CreateDeviceFactory(614)`, in-box runtime)
colliding with the process-wide D3D12 device the XAML compositor creates at
`Window.Activate()`. Fixed architecturally in 0.3.4: **headless bench mode**
(`bench.flag` → no XAML, D3D12-clean process). Exp 1's "71.7 tok/s ≈ CPU"
(May) was a silent CPU fallback on a pre-614 OS. Config note: DML graph
capture requires `past_present_share_buffer: true` (all `genai_config-dml-*`
updated).

**Conclusion — Phase 2 closed, refined by the utilization matrix (v0.3.6)**:
DML EP works on GPU in the Xbox UWP sandbox (headless path). **CPU int4 stays
the default for decode-heavy chat** (68 tok/s vs GPU fp16 46.8), but the
picture is workload-dependent:

- **Prefill: GPU wins at scale** — 354 vs 198 tok/s at ~1k prompt tokens
  (1.8×, TTFT 3.0 s vs 5.3 s). DML fp16 is the better choice for prompt-heavy
  workloads (long context / RAG) already today.
- **The int4 GPU decode collapse (8.8 tok/s) is a missing fused int4 DML
  kernel, not a hardware limit**: fp16 decode is 5.3× faster; effective
  bandwidth GPU ~34 GB/s vs CPU ~13 GB/s. A `MatMulNBits`-class DML kernel
  would imply ~180 tok/s (2.5× CPU) — upstream kernel-coverage issue.

GPU pool estimate corrected: measured budget **3801 MB** (was "~768 MB");
disk is the real constraint. Upstream fix for the `887A0036` init failure
contributed and validated on console:
[microsoft/onnxruntime-genai#2280](https://github.com/microsoft/onnxruntime-genai/pull/2280).
Future options: DML int4-AWQ variants (e.g. block-128) as a fused-kernel
proxy; larger models (1B+) where GPU compute amortises further — blocked
today by disk budget, not GPU memory.

Milestones:

- [x] Exp 1: test DML provider_options on SmolLM2-360M — loads without OOM; later found to be silent CPU fallback (see Status)
- [x] GPU-truth toolkit in place of PIX (unavailable in Dev Mode): ORT profiler with per-kernel EP attribution (`profile-dml-run.sh` + `analyze_ort_profile.py`), WDP `systemperf` GPU counters (`xbox-gpu-sample.sh`), in-app `QueryVideoMemoryInfo` (`gpu_mem_mb` CSV columns) — see `docs/uwp-constraints.md §11`
- [x] Run the profiled DML experiment on console — ✅ **`VERDICT: GPU`** via headless bench mode (0.3.4); weights on GPU (411 MB); CPU control run 70.9 tok/s
- [x] Root-cause and fix DML EP init failure — `887A0036` = Agility-factory vs XAML-compositor device conflict → headless bench mode (`uwp/App.cpp`, 0.3.4)
- [x] Evaluate Qwen2.5-0.5B INT4 ONNX as GPU EP candidate — ❌ CPU-int4 is ~822 MB (not ~200 MB); only the DML int4-awq variant (~507 MB) borderline fits the pool (see `docs/model-selection.md`)
- [x] Procure a DML-compatible model variant and run the end-to-end DML bench — ✅ SmolLM2-360M INT4 DML build (285 MB): decode completes, **8.83 tok/s GPU vs 70.9 CPU** (`phase2-dml.csv`)
- [x] Stage B (only if DML tok/s beats CPU) — ❌ dropped: DML is 8× slower than CPU at this scale; interactive app stays on CPU EP
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
