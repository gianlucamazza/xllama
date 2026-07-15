# xllama — LLM and Diffusion Inference on Xbox Series S: What the Hardware Actually Gives You

**Venere Labs · 2026-07 · v1.0 draft**

xllama is a UWP application that runs local LLM chat and Stable-Diffusion-class
image generation on a retail Xbox Series S in Dev Mode. This report is the
measured story: which workloads the console's hardware serves well, which it
does not, and why — with every claim backed by a benchmark row or a falsified
hypothesis recorded along the way.

> The numbers below are a narrative snapshot (v1.0 draft, 2026-07). The current,
> consolidated performance tables are maintained in
> [benchmarks.md](benchmarks.md) (the perf SSOT) — including the GGUF/Gemma models
> and KV-reuse added after this draft. For the current system structure (how the
> two backends dispatch, provisioning, membw, etc.) see
> [architecture.md](architecture.md). Subsystems that post-date this draft — the
> `unified` build shipping llama.cpp as a real GGUF text backend (not just an A/B
> lane), the per-architecture chat template, and model-provisioning quant
> auto-upgrade — are documented there, not below.

## 1. The machine, as seen from an app

| Resource | Available to a Dev Mode UWP app                                                                                                                                      |
| -------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CPU      | AMD Zen 2, 8 cores — **~6 usable** (system reserves the rest; no thread affinity in AppContainer)                                                                    |
| GPU      | RDNA 2, 4 TFLOPS fp32 / ~8 fp16 — **3801 MB** local budget (Game designation)                                                                                        |
| Memory   | 10 GB GDDR6 unified, 224 GB/s bus — **~13–14 GB/s effective** from CPU int4 GEMV (measured)                                                                          |
| Storage  | Dev Mode allocation (raised to 90 GB here); `LocalState` survives only forward upgrades                                                                              |
| Sandbox  | AppContainer: no mmap for large models, no desktop Win32 (registry, affinity, `SetErrorMode`), external-data ONNX crashes path canonicalisation (§8 constraints doc) |

Toolchain: ONNX Runtime GenAI 0.14.1 (DirectML EP) for text; plain ONNX Runtime
DirectML 1.24.4 for vision; llama.cpp (static ggml, CPU) as an A/B lane.

## 2. Text generation: the CPU wins decode, and nothing beats it

SmolLM2-360M, identical prompts (285/~1050 tok), Game-mode, measured on the 0.4.x→1.0 line:

| Backend                  | prefill 285 | prefill ~1k | decode   |
| ------------------------ | ----------- | ----------- | -------- |
| ORT **CPU int4** @8t     | **220**     | 198         | **66.3** |
| ORT DML fp16             | 169         | **354**     | 46.8     |
| ORT DML int4             | 152         | 334         | 8.8      |
| llama.cpp **Q4_K_M** @6t | 141         | —           | 62.9     |

Three hypotheses died against these numbers:

1. **"The GPU should win decode."** LLM decode is M=1 GEMV: dispatch-bound and
   bandwidth-bound, the GPU's worst case. DML fp16 loses to CPU int4 (46.8 vs
   66.3); the crossover exists only in prefill, where batch amortises dispatch
   (1.8× at ~1k tokens). The app therefore routes per-workload (long-prompt
   conversations → DML fp16, decode → CPU int4), sticky per conversation.
2. **"DML int4 decode collapses because a kernel is missing."** Falsified by
   desk-check and confirmed on hardware: `MatMulNBits` _is_ on the GPU (compute
   engines 87–90 % busy at 8.8 tok/s) but DirectML implements it **non-fused**
   (dequantise to fp16, then full GEMM) — int4 moves _more_ bandwidth than
   fp16. A kernel-design limit in DirectML itself, not fixable app-side.
3. **"llama.cpp extracts ~2× the bandwidth of ORT's AVX2 kernels."** Falsified
   on hardware: Q4_K_M reaches 62.9 tok/s at 6 threads — decode parity (−5 %)
   with worse prefill. Both stacks saturate the same ~13 GB/s effective CPU
   bandwidth. (llama.cpp does run in AppContainer — a first, requiring five
   Win32 partition guards — but its spin-wait threadpool livelocks at 7–8
   threads on the ~6 usable cores; 6 is the ceiling.)

Multi-turn chat is made interactive by **KV-cache reuse** (persistent generator,
append-only per turn): turn-2 prefill is **4.87×** faster than a cold re-prefill
(103.7 ms for the 22-token delta vs 505.2 ms for the full context).

Scale: SmolLM2-1.7B cpu-int4 decodes at **20.6 tok/s** (bandwidth-bound scaling,
~3.2× down from 360M) — usable. A 1.7B fp16 single-file ONNX exceeds the 2 GB
protobuf limit and cannot be deployed self-contained (a serialization
constraint, not a GPU one).

## 3. Image generation: the GPU's actual job

If decode is the GPU's worst case, compute-bound fp16 batch is its best. A
309-GFLOP conv proxy (one UNet-step stand-in) ran **11.1× faster on DirectML
than on the CPU** (2403 vs 216 GFLOP/s) — the exact inverse of text decode.

Built on that: a from-scratch C++ diffusion pipeline (CLIP BPE tokenizer, Euler
scheduler, fp16 conversion, PNG writer — all header-only and unit-tested against
the diffusers/transformers reference before shipping; 638 assertions) driving
three ORT DirectML sessions. **SD-Turbo fp16 generates a coherent 512×512 image
on the console in 6.9 s** (text encoder 1.0 s, UNet 3.3 s/step × 1, VAE decode
2.6 s; ~7.5 s session load excluded).

Hardware lessons encoded in the pipeline:

- **Sequential session lifetime**: the 3801 MB budget does not fit all three
  components' weights (~2.4 GB) _plus_ the VAE's 512² activations — the first
  run OOM'd inside the VAE. Create → run → destroy, per stage.
- **fp16 artifacts are a minefield**: `onnxconverter_common` emits ORT-rejected
  mixed-type graphs for all three SD components; the ORT-team HF pre-exports
  use `NhwcConv` (CUDA-only, no CPU kernel — unvalidatable); the working recipe
  is `onnxruntime.transformers`' converter with graph optimisation capped at
  `EXTENDED` (ALL crashes the session optimiser on these graphs).
- The XAML compositor's D3D12 device conflicts with **ORT GenAI** DML init
  (887A0036), but plain ORT DirectML (diffusion) coexists with the compositor
  in-process (validated 2026-07-09, ~5.6 s). The **Image** dialog runs the
  pipeline on a background thread without restarting the app.

## 4. Method: validate on the layer you own

Every correctness-critical piece was validated off-console before touching
hardware, because console runtime errors are expensive to attribute:

- **Golden vectors**: the C++ tokenizer/scheduler are asserted bit-for-value
  against diffusers/transformers outputs in host unit tests, in CI.
- **Model-side gates**: every ONNX artifact must load _and_ generate through
  ORT CPU locally (`validate_pipeline.py`) before upload. This caught the
  broken fp16 converters and the CUDA-only pre-exports.
- **Same-code local runs**: the llama.cpp bridge bugs (tokenize sign, hidden
  perf counters) were reproduced and fixed on the Linux CLI — the same source
  path — before the console retry.
- **Fail-loud tooling**: the Device Portal upload path silently lost files
  twice (HTTP 200 with `Success:false`; a while-read that dropped the last
  path component). Both were found because a later step _verified_ instead of
  trusting the reporter.

## 5. What ships in v1.0.0

- Chat UI (ChatML, history, settings) with per-conversation CPU/GPU routing and
  KV-cache reuse; models described by a `manifest.json` catalogue (bundled +
  Device-Portal-overridable) with in-app download from the `models-v1` GitHub
  Release (the upstream HF path shipped a non-merged model and was retired).
- Image generation UI over the in-process SD-Turbo fp16 pipeline.
- Headless bench/validation modes (`bench.flag`, `diffuse.flag`)
  with CSV artifacts; a console-validation runbook; CI producing the default
  (no-model, ~19 MB) and llamacpp MSIX variants.

**Bottom line.** On a 2020 console in a sandbox: a 360M-class chat model at
66 tok/s, a 1.7B at 21 tok/s, ~5 s multi-turn latency turned interactive by KV
reuse, and a 512² diffusion image in ~7 s on the GPU. The Series S is a
perfectly good local-AI box — as long as you give each processor the workload
it is actually shaped for.
