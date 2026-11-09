# Xbox GPU acceleration baseline

Discovery date: 2026-11-09. Audited main:
`20e79727c3074507f51717596abb699ed184c655`; llama.cpp:
`d81235049384534c167caea52b85a694f6103d14`.

**Program status: in progress.** This is a source/evidence audit, not a fresh
console performance baseline. No Xbox credentials/device or Windows SDK are
configured in this execution. The previous execution records the Series S back
in retail mode. Do not deploy, switch modes or reuse old timing rows as current
measurements. Series X acceptance remains [#374](https://github.com/gianlucamazza/xllama/issues/374).

## Discovery inventory

The checkout includes all default-branch source, tests, CI, raw evidence and the
pinned submodule. Remote branches at discovery: main, the C4 prefill experiment,
FloppyLM training and the decision-model spike. The latter two are unrelated to
this GPU program and were not integrated. Current open work is #313, #374,
#395, upstream overlays #84/#85/#86 and architecture watches #273/#274/#275.
The GitHub API returned no native milestones. This execution can create issues
and PRs but has no milestone-creation connector or authenticated CLI; the browser
fallback also failed before opening the milestone page. The stages below are
tracked backlog stages, not a claim that GitHub milestones were created.

| Requested PR | Discovery disposition | Reuse / consequence                                                                           |
| ------------ | --------------------- | --------------------------------------------------------------------------------------------- |
| #381         | Merged                | TAESD fresh-result/integrity gate; diffusion acceptance is separate from text GPU correctness |
| #369         | Merged                | Candidate package acceptance and scoped limitations; not a stable release                     |
| #414         | Open draft            | Opt-in C4 shaders, compiler/blob provenance, tails and UI diagnostics; do not promote         |
| #417         | Merged                | Preserve failed original-kernel Coder control, raw dumps and stopped campaign                 |
| #419         | Merged                | Standard selftest propagates failure and rejects non-finite native results                    |

All current main Linux/UWP/CodeQL runs observed at discovery completed
successfully. Shipping workflow requires Linux, both UWP variants and C++
CodeQL on an up-to-date PR head; optional jobs are not substitutes. Stable
release remains v1.6.0.0; the reviewed 1.7 candidate and its receipts are owned
by [release notes](../releases/v1.7.0.md). No release is created by this program.
Existing fork analysis is in [community-forks](../community-forks.md) and the
[previous execution ledger](../audit/excellence/execution-2026-11-07.md).

## Actual execution paths

Source authorities: `src/bridge/session.cpp`, `inference.cpp`, `llama_gpu.h`,
`decode_loop.h`, `decode_loop_ort.h`, `sampler_chain.h`, `ort_sampling.h`,
`ggml_d3d12.cpp`, `include/xllama/routing_policy.h` and the shipping catalogue.
The complete model-by-model status remains in [model-matrix](../model-matrix.md).

| Family / path                                                     | Load and placement                                                                                          | Prefill / decode                                                                       | KV, attention and sampling                                                                                                 |
| ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| GGUF CPU (including small LFM and shipping coding model)          | llama.cpp buffered weights, CPU/repack buffers; Linux supports this path                                    | llama.cpp graph, CPU kernels                                                           | Persistent llama context; CPU attention/KV and shared llama sampler                                                        |
| GGUF custom D3D12 (shipping LFM2.5-1.2B; explicit layer override) | Register static ggml backend; select supported weight buffers; unavailable device falls back before loading | GPU Q4_0/Q4_K/Q6_K MUL_MAT; Q6_K tied embedding GET_ROWS; unsupported types/ops on CPU | CPU q8 activation quantization, CPU norms/RoPE/attention/KV, CPU logits/sampling; CPU Flash Attention explicitly preserved |
| ONNX ORT CPU                                                      | ORT GenAI model/config and CPU provider                                                                     | ORT graph via shared decode loop                                                       | ORT generator; continuous KV reuse where supported; ORT sampling path                                                      |
| ONNX ORT DirectML                                                 | Verified eligible DML asset/provider; independent of GGUF layer offload                                     | Actual EP placement requires an ORT profile; fused nodes can conceal internals         | DML text route does not support continuous KV reuse; no claim that every node or sampling stage is GPU-resident            |

The custom backend is deliberately hybrid, not a complete GPU transformer.
Q5_K/Q8_0 weight kernels, GPU RoPE/attention/KV and a general GPU GEMM are not
implemented by main's custom backend. The requested Q8_0 _weight_ kernel must
not be confused with its existing q8 _activation_ path. Prefill reuses the
single-column kernel across columns; C4 is only on the draft branch.
Weights are uploaded through staging; CPU-cached coherent-UMA buffers share
activations/results, so not every CPU/GPU boundary is a discrete device copy.
Submissions still wait for completion; shared-input q/k/v and gate/up reuse
quantization, while dependent chains flush before use.

GGUF catalogue layers and explicit `gguf_gpu_layers.txt` override are separate
from ORT CPU/DML first-turn routing. The latter is allowlisted, readiness-gated
and tested; it is not a general model-independent router. Do not migrate an
active KV state across backends or change shipping defaults from projections.

## Platform evidence boundary

Use [UWP constraints](../uwp-constraints.md), especially §7, as the resource and
compatibility authority. `d3d12_compute.cpp` uses the system D3D12 device path;
`ggml_d3d12.cpp` requires cache-coherent UMA and creates the embedded pipeline
states. HLSL/DXIL sources and generated blobs are versioned separately. Existing
PSO creation/console receipts show the tested package's capabilities, not every
RDNA 2 feature exposed in UWP or a Series X guarantee.

Microsoft documents UWP Xbox resource allocation separately from desktop D3D12:
[resource allocation](https://learn.microsoft.com/en-us/windows/uwp/xbox-apps/system-resource-allocation).
Wave intrinsics require feature checks according to the
[Shader Model 6 documentation](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/hlsl-shader-model-6-0-features-for-direct3d-12).
Check shader model, WaveOps/lane bounds and relevant packed-integer/16-bit
capabilities on the actual adapter before adding feature-dependent paths.
Do not infer wave width, accessible compute units or memory budget from the
console's retail hardware name. A desktop or GDK capability is not a UWP receipt.

## Profiling coverage and bottlenecks

| Requested metric / bottleneck             | Existing source                                            | Current limitation / next gate                                                                    |
| ----------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------- |
| Prefill/decode throughput, load, TTFT     | `bench.cpp`, shared decode loops, bench CSV/sidecars       | CSV TTFT starts at prefill, excludes load; retained timing is package-specific                    |
| API TTFB / end-to-end request latency     | API validation and client observations                     | No common benchmark field; define request start, header byte and first content byte separately    |
| GPU time, graph wall time, dispatch count | D3D12 query pair per submission, backend-lifetime counters | Aggregated across phases; missing timestamp support must not be represented as measured zero      |
| CPU compute time                          | CPU inference durations, mixed-graph wall time             | Wall minus GPU is not CPU compute; separate quantization, command recording and waits             |
| Per-operator/layer                        | ORT provider/op traces; ggml tensor names available        | Fused ORT nodes hide layers; D3D12 has no per-node timestamp/capture export                       |
| Synchronization / dispatch overhead       | `gpustep` round-trip and sync/nosync probes                | Historical synthetic probe, not current per-token attribution                                     |
| Transfer bytes / effective bandwidth      | `gpugemv`, `gpubw`, weight buffers and UMA architecture    | Packed weight BW is not total inference BW; current graph transfer counters absent                |
| Allocation / peak memory                  | Process working-set helpers and GPU budget observations    | Distinguish lifetime OS peak from sampled maxima and reset scope; no full allocation ledger       |
| KV utilization                            | Session/context budget and KV persistence                  | No unified occupancy/bytes/fragmentation export across backends                                   |
| Shader compile / warmup                   | Offline shader compilation and PSO init                    | Shader compiler time is host build time; PSO creation/warmup require separate on-device intervals |

Measured historical bottlenecks and their values belong only in
[GPU decode design](../gguf-gpu-decode.md), [Phase 15](../phase15-re-opt.md) and
[generated benchmarks](../benchmarks.md), with raw CSV provenance. They motivate
submission-overhead and multi-column work; they do not quantify current main.
The cost-model bandwidth and projected step times in those documents are
estimates, not measurements of a full decode or GPU occupancy.

Reusable harnesses: XAB, benchmark sidecars, `bench-xbox-ort.sh`,
`bench-d3d12-selftest.sh`, `bench-gpustep.sh`, `profile-dml-run.sh`,
`analyze_ort_profile.py`, `compare-logits.py`, H9 and the console/API runbooks.
New host-side integrity work is [#423](https://github.com/gianlucamazza/xllama/pull/423)
and [#424](https://github.com/gianlucamazza/xllama/pull/424). Their merge/CI state
must be checked rather than inferred from this dated audit.

## C4 control diagnosis

The negative source/package binding and untouched raw float32 dumps are in
[the stopped preflight](../gguf-gpu-decode.md#2026-11-08-preflight--control-logit-failure).
The original Coder GPU arm fails the frozen maximum-error and NMSE gates even
though its top token matches. The new kernel's Coder logits were not collected;
no C4 performance campaign or causal attribution is available. Synthetic kernel
passes do not supersede an end-to-end parity failure.

A host replay can reproduce the **comparison failure**, not rerun GPU inference:

```bash
E=docs/audit/excellence/execution-2026-11-08
python3 scripts/compare-logits.py "$E/prefill-logits-coder-CPU.bin" "$E/prefill-logits-coder-A.bin"
# Expected exit 1. Keep it as a negative control, never override it to succeed.
python3 scripts/compare-logits.py "$E/prefill-logits-lfm-CPU.bin" "$E/prefill-logits-lfm-A.bin"
# Expected exit 0.
```

The [host replay receipt](evidence/2026-11-09-logit-gate-replay.json) binds the
retained binary hashes and candidate comparator. It also reproduces the
shape-mismatch false success before the correction and failure after it. These
are correctness verdicts, not before/after throughput benchmarks. The host
Debug build and ctest passed (413 registered doctest cases); the parity tranche
Python discovery passed 201 tests with 20 optional dependency skips. No model
weights were provisioned here, so model-dependent opt-in coverage is absent.

Next hardware diagnostic protocol, before resuming #313:

1. Bind the exact MSVC package/source/model/tokenizer/prompt hashes, context,
   ubatch, threads, greedy parameters and observed device identity. Reproduce
   CPU and original GPU control without C4, retries or changed prompts.
2. Add a disabled-by-default bounded ggml scheduler evaluation capture for the
   identical graph/token positions. Copy only selected tensors after completion;
   do not change the llama.cpp submodule. Retain capture-on/off timing separately.
3. Compare matching named/shaped tensors in graph dependency order: embedding,
   normalization, q/k/v projections, attention outputs, residuals, feed-forward
   and output logits. Mixed-backend nodes still need a CPU reference. A first
   observed mismatch at a capture point is not yet proof of its causal operator.
4. Refine the first mismatching interval down to quantized activation bytes,
   weight blocks, scales, strides, reduction order and rounding. Distinguish
   expected quantized arithmetic drift from indexing/lifetime/sync defects.
5. Keep the frozen comparator and all raw attempts. If arithmetic requirements
   need revision, propose an explicit methodological ADR before a new campaign;
   do not retroactively change the failed verdict.
6. Only after control parity passes, qualify C4 correctness/H9/resource gates
   and then the existing preregistered paired performance experiment. Abandon
   C4 if no benefit is established; preserve the negative outcome.

## Tracked stages and order

The 6–8 week horizon is conditional on hardware availability, not a completion
promise. No arbitrary speedup target is introduced. The existing #313 margins
remain frozen for that experiment only.

| Stage         | Work / owner lane                            | Acceptance                                                                          | Status / dependency                                            |
| ------------- | -------------------------------------------- | ----------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| M0, week 1    | Source/evidence audit, principal/runtime     | This baseline, route/coverage map and triage reviewed                               | Audit produced; no fresh console baseline                      |
| M1, weeks 1–2 | Integrity and profiling, QA/performance      | Fail-closed CLI/trace tests; complete attributable current-device capture           | #420/#421; host PRs #423/#424; hardware capture blocked        |
| M2, weeks 2–3 | Coder control RCA, numerical/GPU             | First causal divergence isolated and frozen parity passes                           | #420/#313; blocked on device and tensor-capture implementation |
| M3, weeks 3–5 | Measured kernel/overhead work, HLSL          | Independent reversible PRs, numerical and resource gates, paired gains              | #313/#414; blocked by M2, no Q5/Q8 or GPU KV promotion         |
| M4, weeks 4–6 | Deterministic routing, runtime/platform      | Measured envelope, logged reasons, fallback/override and repeatable tests           | #422; existing defaults preserved pending data                 |
| M5, weeks 5–7 | Matched S/X suite, benchmark/reproducibility | Raw attempts, device receipts, uncertainty, quality and memory gates                | #422/#374; Series X absent                                     |
| M6, weeks 6–8 | Hardening/report, QA/release/reviewer        | Required CI plus device lost, concurrency, upgrade/download and final-package gates | #395/#422; publication blocked until evidence sufficient       |

Highest-impact/lowest-risk next action is host integrity, followed by control
RCA and attributable measurement. New weight types, kernel fusion, attention
and GPU KV are higher-risk experiments with no current evidence of a benefit.
Do not replace validated CPU paths merely to increase the fraction on GPU.

Technical report working draft:
[Optimizing Quantized LLM Inference on Xbox RDNA 2 GPUs](gpu-acceleration-report.md).
It is not a publication-ready result or an added research-registry claim.
