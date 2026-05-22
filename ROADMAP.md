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

## Phase 2 — GPU Acceleration 🚫 BLOCKED

**Goal**: DirectML EP inference using Xbox RDNA 2 hardware.

**Blocker**: UWP GPU-accessible memory pool on Xbox Series S is ~768 MB (observed total at `OgaCreateModel` OOM). `OgaCreateModel` with DirectML EP crashes with SEH `0xC0000005` (null-deref on OOM in the DirectML allocator) for any LLM that exceeds this budget including activation, KV-cache, and staging buffers.

**Active workaround experiments** (see `docs/uwp-constraints.md §7`):
- *Exp 1*: DML provider_options `enable_cpu_mem_arena=0` + `enable_mem_pattern=0` + `past_present_share_buffer=false` may reduce allocator footprint enough for SmolLM2-360M. Test script: `scripts/test-dml-config.sh` (no rebuild needed).
- *Exp 3*: If Qwen2.5-0.5B INT4 ONNX (~200 MB merged) can be identified, it would fit the GPU budget with room to spare.

**Conditions to reopen**:
- DML provider_options reduce footprint enough for an existing model (Exp 1)
- A sub-300 MB on-device ONNX model is validated (e.g., Qwen2.5-0.5B INT4)
- Microsoft expands the UWP GPU pool allocation for Dev Mode apps

Milestones (blocked):
- [ ] Run Exp 1: test DML provider_options on SmolLM2-360M via `scripts/test-dml-config.sh`
- [ ] Evaluate Qwen2.5-0.5B INT4 ONNX for GPU EP fitness (est. ~200 MB merged)
- [ ] Validate DirectML EP inference end-to-end: tok/s vs CPU EP baseline
- [ ] Measure GPU vs CPU tok/s for the same model at same quant level

## Phase 3 — Benchmarks + Model Exploration 📋 NEXT

**Goal**: reproducible results, n_threads tuning, evaluate alternative compact models.

Milestones:
- [ ] Populate `bench/results/phase1-cpu.csv` with 3+ models × 3 runs (median tok/s)
- [ ] Tune `n_threads` on Zen 2 Xbox (6–7 available cores); find optimal value for SmolLM2-360M
- [ ] Evaluate Qwen2.5-0.5B INT4 ONNX (estimate ~200 MB): fits Phase 2 GPU budget?
- [ ] Evaluate Llama-3.2-1B INT4 ONNX CPU: fits disk budget with MSIX overhead?
- [ ] Add `load_ms` to bench CSV if not already present; baseline model load time

## Phase 4 — In-App Download + Publication 🔮 FUTURE

**Goal**: remove bundled-model constraint; demo video; technical write-up.

Milestones:
- [x] `ModelDownloader` with chunked `HttpClient` download from HF endpoint (Exp 2, `uwp/model-downloader.cpp`)
- [x] `EnsureModelAsync()` bootstrap: LocalState → InstalledPath → HF download fallback chain
- [x] USB external drive fallback at `E:\xllama\models\<name>` in `resolve_model_path` (Exp 3)
- [ ] Validate Exp 2 on console: confirm HF HTTPS works from Xbox AppContainer
- [ ] Remove model bundle from MSIX once Exp 2 is validated (delete ItemGroup in `xllama.vcxproj`)
- [ ] `model-manifest.json`: configurable model list (HF repo + file list) for model switching
- [ ] Demo video: model loaded and running on Xbox hardware
- [ ] Technical report (arXiv or GitHub Discussions)
- [ ] Tagged v1.0.0 release with pre-built MSIX and model manifest
