# xllama Roadmap

## Phase 1 — CPU Baseline

**Goal**: reproducible, measurable CPU-only inference on Xbox Series S via UWP Dev Mode.

Milestones:
- [ ] `cmake -B build -DXLLAMA_TARGET=linux && cmake --build build` green on Ubuntu 22.04
- [ ] `llama-bridge.cpp` compiles clean with MSVC under UWP constraints (no mmap, no dlopen)
- [ ] `xllama.appx` deploys and launches on Xbox Series S in Dev Mode
- [ ] Inference runs end-to-end for Qwen3-1.7B Q4_K_M; tok/s recorded to `bench/results/`
- [ ] `CreateFileMappingFromApp`-based model loading replaces POSIX mmap
- [ ] `bench/results/phase1-cpu.csv` published with median tok/s across 3 models

Target models: Qwen3 1.7B, Qwen3 8B, Llama 3.2 3B.

## Phase 2 — Vulkan Backend

**Goal**: GPU acceleration on Xbox RDNA 2 via Mesa/libgallium Vulkan driver.

Milestones:
- [ ] Mesa / libgallium builds for UWP ARM64/x64 (or source Mesa Vulkan driver that works in UWP sandbox)
- [ ] `GGML_VULKAN=1` backend compiles and links against Mesa Vulkan ICD
- [ ] Vulkan device enumeration succeeds inside the UWP container
- [ ] Inference runs on GPU; tok/s measured and compared to Phase 1 CPU baseline
- [ ] Target: ≥2× speedup for Qwen3-8B Q4_K_M vs CPU-only

Dependencies: Xbox homebrew Mesa/libgallium work (external project).

## Phase 3 — Optimization

**Goal**: squeeze the hardware for best tok/s within the 8 GB envelope.

Milestones:
- [ ] Profile `ggml` kernel hot-paths on Zen 2 with `perf` (Linux) and PIX (Xbox)
- [ ] Evaluate IQ-quant variants (IQ3_XS, IQ4_XS) for quality/speed tradeoff
- [ ] KV-cache memory layout tuning for GDDR6 bandwidth
- [ ] Sliding-window / grouped-query attention paths validated on target hardware
- [ ] `bench/results/phase3-optimized.csv` showing improvement deltas

## Phase 4 — Publication

**Goal**: reproducible results, public release, technical write-up.

Milestones:
- [ ] All benchmark CSVs reviewed and reproducibility instructions validated by at least one external contributor
- [ ] Technical report published (target: arXiv or GitHub Discussions)
- [ ] Demo video: model loaded and running on Xbox hardware
- [ ] Tagged v1.0.0 release with pre-built .appx and GGUF download instructions
- [ ] Submission to llama.cpp upstream tracker / Xbox homebrew community forums
