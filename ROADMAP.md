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

**Blocker**: UWP GPU memory pool on Xbox Series S is ~768 MB (128 MB dedicated + 640 MB shared). `OgaCreateModel` with DirectML EP crashes with SEH `0xC0000005` (null-deref on OOM in the DirectML allocator) for any LLM with on-device weights > ~300 MB. SmolLM2-360M INT4 on-device weighs ~400 MB after merging external data, which still exceeds the threshold.

**Conditions to reopen**:
- A sub-300 MB on-device ONNX model (e.g., < 250 M parameter INT4) is identified and validated
- Microsoft expands the UWP GPU pool allocation for Dev Mode apps
- An alternative GPU path that bypasses the ORT DirectML allocator is found

Milestones (blocked):
- [ ] Identify a model whose DirectML memory footprint fits in ~600 MB (conservative budget)
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
- [ ] `ModelSpec` with `vector<string> files` for multi-file ONNX downloads in-app
- [ ] Device Portal–based model swap UI (or HTTP download from known HF endpoint)
- [ ] Demo video: model loaded and running on Xbox hardware
- [ ] Technical report (arXiv or GitHub Discussions)
- [ ] Tagged v1.0.0 release with pre-built MSIX and model manifest
