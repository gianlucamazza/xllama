# Xbox GPU acceleration baseline

Discovery date: 2026-11-09. Audited main:
`20e79727c3074507f51717596abb699ed184c655`; llama.cpp:
`d81235049384534c167caea52b85a694f6103d14`.

**Program status: in progress.** The November 9 discovery ran without a
console and cited an earlier retail-mode report. The November 27 assessment
below reconciles later package receipts. The independent Series X campaign
completed [#374](https://github.com/gianlucamazza/xllama/issues/374);
the matched project-device and routing gates remain open under
[#422](https://github.com/gianlucamazza/xllama/issues/422).

## Discovery inventory (2026-11-09)

The checkout includes all default-branch source, tests, CI, raw evidence and the
pinned submodule. Remote branches at discovery: main, the C4 prefill experiment,
FloppyLM training and the decision-model spike. The latter two are unrelated to
this GPU program and were not integrated. At discovery, open work included #313, #374,
#395, upstream overlays #84/#85/#86 and architecture watch #273 (MoE).
[#274](https://github.com/gianlucamazza/xllama/issues/274) BitNet and
[#275](https://github.com/gianlucamazza/xllama/issues/275) SSM/RWKV + mm
closed completed: gates documented in [model-matrix.md](../model-matrix.md)
§I2 / §I3; latest scan found no candidate.
The GitHub API returned no native milestones. This execution can create issues
and PRs but has no milestone-creation connector or authenticated CLI; the browser
fallback also failed before opening the milestone page. The stages below are
tracked backlog stages, not a claim that GitHub milestones were created.

| Requested PR | Discovery disposition | Reuse / consequence                                                                              |
| ------------ | --------------------- | ------------------------------------------------------------------------------------------------ |
| #381         | Merged                | TAESD fresh-result/integrity gate; diffusion acceptance is separate from text GPU correctness    |
| #369         | Merged                | Candidate package acceptance and scoped limitations; not a stable release                        |
| #414         | Closed, not merged    | C4 shaders correct but slower on LFM2.5-1.2B (2026-11-16); branch kept, see the end of this page |
| #417         | Merged                | Preserve failed original-kernel Coder control, raw dumps and stopped campaign                    |
| #419         | Merged                | Standard selftest propagates failure and rejects non-finite native results                       |

All current main Linux/UWP/CodeQL runs observed at discovery completed
successfully. Shipping workflow requires Linux, both UWP variants and C++
CodeQL on an up-to-date PR head; optional jobs are not substitutes. At discovery the stable
release was v1.6.0.0; the reviewed 1.7 candidate and its receipts are owned
by [release notes](../releases/v1.7.0.md). No release is created by this program.
Existing fork analysis is in [community-forks](../community-forks.md) and the
[previous execution ledger](../audit/excellence/execution-2026-11-07.md).

## Actual execution paths

Source authorities: `src/bridge/session.cpp`, `inference.cpp`, `llama_gpu.h`,
`decode_loop.h`, `decode_loop_ort.h`, `sampler_chain.h`, `ort_sampling.h`,
`ggml_d3d12.cpp`, `include/xllama/routing_policy.h` and the shipping catalogue.
The complete model-by-model status remains in [model-matrix](../model-matrix.md).

| Family / path                                                     | Load and placement                                                                                          | Prefill / decode                                                                                           | KV, attention and sampling                                                                                                 |
| ----------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| GGUF CPU (including small LFM and shipping coding model)          | llama.cpp buffered weights, CPU/repack buffers; Linux supports this path                                    | llama.cpp graph, CPU kernels                                                                               | Persistent llama context; CPU attention/KV and shared llama sampler                                                        |
| GGUF custom D3D12 (shipping LFM2.5-1.2B; explicit layer override) | Register static ggml backend; select supported weight buffers; unavailable device falls back before loading | GPU Q4_0/Q4_K/Q6_K/Q5_K/IQ4_XS/IQ3_XXS MUL_MAT; Q6_K tied embedding GET_ROWS; unsupported types/ops on CPU | CPU q8 activation quantization, CPU norms/RoPE/attention/KV, CPU logits/sampling; CPU Flash Attention explicitly preserved |
| ONNX ORT CPU                                                      | ORT GenAI model/config and CPU provider                                                                     | ORT graph via shared decode loop                                                                           | ORT generator; continuous KV reuse where supported; ORT sampling path                                                      |
| ONNX ORT DirectML                                                 | Verified eligible DML asset/provider; independent of GGUF layer offload                                     | Actual EP placement requires an ORT profile; fused nodes can conceal internals                             | DML text route does not support continuous KV reuse; no claim that every node or sampling stage is GPU-resident            |

The custom backend is deliberately hybrid, not a complete GPU transformer.
Plain F32 RoPE and a narrow F32 bias ADD can dispatch only with their separate
LocalState opt-ins, successful PSO creation and eligible operands. See the
[operator contracts and matched gates](../gguf-gpu-decode.md#bias-add-on-the-gpu-opt-in-experimental).
Q8_0 weight kernels, GPU attention/KV and a general GPU GEMM remain outside
the custom backend. The requested Q8_0 _weight_ kernel must
not be confused with its existing q8 _activation_ path. Prefill reuses the
single-column kernel across columns. C4 was tested on the closed #414 branch.
Weights are uploaded through staging; CPU-cached coherent-UMA buffers share
activations/results, so not every CPU/GPU boundary is a discrete device copy.
Submissions still wait for completion; shared-input q/k/v and gate/up reuse
quantization, while dependent chains flush before use.

GGUF catalogue layers and explicit `gguf_gpu_layers.txt` override are separate
from ORT CPU/DML first-turn routing. The latter is allowlisted, readiness-gated
and tested; it is not a general model-independent router. Do not migrate an
active KV state across backends or change shipping defaults from projections.

## Platform evidence boundary

Assessment on 2026-11-27 at `62a577e00859ec381463707f3baf1430edda5ddc`.
The [Series S](https://www.xbox.com/en-US/consoles/xbox-series-s) and
[Series X](https://www.xbox.com/en-US/consoles/xbox-series-x) product specifications
describe the silicon. [Microsoft's UWP allocation table](https://learn.microsoft.com/en-us/previous-versions/windows/uwp/xbox-apps/system-resource-allocation)
distinguishes foreground App and Game designations and debugging. Record actual
process resources at measurement time. [UWP constraints](../uwp-constraints.md)
owns measured budgets and API restrictions.

| Capability                                       | Xbox Series S silicon                             | Xbox Series X silicon                               | API / UWP path                                                                                                                                                                                                                                                                                                                                                                                                                         | Evidence and Windows x64 control                                                                                                                                                                                                                                                                                                                                                                                        |
| ------------------------------------------------ | ------------------------------------------------- | --------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CPU / SIMD / caches                              | Eight Zen 2 cores                                 | Eight Zen 2 cores                                   | UWP build uses `/arch:AVX2`; the available CPU set and designation constrain usable workers.                                                                                                                                                                                                                                                                                                                                           | [Series S CPU/thread evidence](../uwp-constraints.md#6-limited-thread-count) and [external X receipt](../../bench/results/2026-11-16-series-x-422/README.md). On PC record CPU, CPUID/OS-enabled vector state, cache topology and effective threads. No per-console cache-residency or VNNI result in those receipts.                                                                                                   |
| Compute and shaders                              | RDNA 2, 20 CUs, 4 TFLOPS nominal                  | RDNA 2, 52 CUs, 12 TFLOPS nominal                   | Series UWP lists D3D12 feature level 11_0. The system device, individual shader-model/features and actual PSO creation determine which shaders execute.                                                                                                                                                                                                                                                                                | [Series S package selftest](../../bench/results/2026-11-27-v1110-series-s-release/receipts/d3d12-selftest-ee6a718-off.csv) and [external X grid](../../bench/results/2026-11-16-series-x-422/README.md) cover their own arms. On PC log adapter LUID, driver, OS, features and physical hardware vs software adapter.                                                                                                   |
| Waves / instruction selection                    | RDNA 2 SIMD                                       | RDNA 2 SIMD                                         | `gpugemv.cpp` queries [OPTIONS1](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_feature_data_d3d12_options1) for WaveOps and min/max lanes. An intrinsic-dependent variant needs an eligible lane range and a scalar or groupshared control.                                                                                                                                                                 | The [Series S wave32 and dot4 probe](../../bench/results/phase15-gpugemv-h63.csv) explicitly records `wave_ops=0`. The wave32 kernel uses groupshared reduction. Capture the capability report and a dispatch on each device before selecting a native wave path.                                                                                                                                                       |
| FP32, FP16, INT8, packed 4-bit                   | GPU arithmetic and GDDR6                          | GPU arithmetic and GDDR6                            | FP32 output, unpacking FP16 scales and packed INT8 dot arithmetic have distinct requirements. `gpugemv` checks SM 6.4 for [packed dot4](https://learn.microsoft.com/en-us/windows/win32/direct3dhlsl/hlsl-shader-model-6-4-features-for-direct3d-12). For native-half candidates query [OPTIONS4.Native16BitShaderOpsSupported](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_feature_data_d3d12_options4). | The S dot4 rows have GPU execution, timestamps and a passing numerical check. Quantized weights in the D3D12 selftest and FP16 DirectML graph execution cover the named paths. Native opcode throughput, INT4 arithmetic and arbitrary DML operators need their own evidence on S/X/PC.                                                                                                                                 |
| Memory segments, GPU caches and shared resources | 10 GB GDDR6, 8 GB at 224 GB/s and 2 GB at 56 GB/s | 16 GB GDDR6, 10 GB at 560 GB/s and 6 GB at 336 GB/s | `MemoryManager` reports process limits and usage; `QueryVideoMemoryInfo(LOCAL)` reports adapter budget and current usage. The backend requires [CacheCoherentUMA](https://learn.microsoft.com/en-us/windows/win32/api/d3d12/ns-d3d12-d3d12_feature_data_architecture1). Shared mapped activations still require ordering. Weights use bounded upload/readback staging.                                                                 | [S UMA and heap probe](../../bench/results/phase15-gpustep-d1.csv), [process and GPU memory pair](evidence/2026-11-18-d3d12-layer-overhead/README.md). X has an external process-limit receipt. PC can be discrete or UMA; a discrete adapter fails this custom backend's eligibility. Segment placement, cache sizes/hit rates and sustained model bandwidth cannot be recovered from retail bandwidth or working set. |
| SSD / app-visible I/O                            | Custom NVMe, up to 2.4 GB/s raw                   | Custom NVMe, up to 2.4 GB/s raw                     | AppContainer GGUF uses buffered Win32 loading. `diskbw` can read aligned blocks unbuffered. The [DirectStorage Windows SDK](https://github.com/microsoft/DirectStorage) is a separate desktop experiment pending supported package APIs and on-device completion.                                                                                                                                                                      | [S sequential and random reads](../../bench/results/diskbw.csv) specify block size, workers and buffering. [CPU expert-streaming receipts](evidence/2026-11-16-moe-24b-console/README.md) count logical bytes. Collect comparable X/PC storage traces with volumes, queue depth and cache state. Physical bytes and random 4 KiB IOPS are unavailable from those runs.                                                  |
| DirectML, timing and profiling                   | Integrated GPU                                    | Integrated GPU                                      | ORT profiles identify assigned EP/fused node; GPU timestamps on the D3D12 queue need valid pairs and its frequency. PIX/ETW require an accessible capture target and appropriate tooling.                                                                                                                                                                                                                                              | [Series S DirectML attribution](../uwp-constraints.md#11-gpu-truth--ep-attribution-without-pix) and [dispatch calibration](evidence/2026-11-16-d3d12-dispatch-timing/README.md). On capable Windows hardware capture queue/CPU timelines with PIX/ETW and compare uninstrumented runs. Desktop captures and Windows builds leave console validation open.                                                               |

### Reproduce the hardware and bottleneck assessment

1. Record the package hash, source, DLLs, model and prompt digests, shader build, OS, hardware receipt, device/driver, App/Game and foreground state. Collect App runs from the UI after the CPU cap settles (at least 20 seconds after launch per [#422](https://github.com/gianlucamazza/xllama/issues/422)). Use the [device runbook](../console-validation-runbook.md) for Series X.
2. Run `bench-d3d12-selftest.sh` at defaults, then gate an individual optional operator separately. For each eligible model run `bench-xbox-ort.sh` with fixed prompt/decode/context and CPU vs requested offload. Bind the observed GPU dispatch or ORT provider to the same run. Check output with [ADR 0010](../adr/0010-logit-parity-reference-spread.md) before comparing speed.
3. Separate model load, tokenization, prefill, decode, sampling and stream delivery. Use the layer sidecar for dispatches and `measure-api-ttfb.py` for first visible content over LAN. Treat an uninstrumented interval as unknown. Perform wall-time trials with diagnostic knobs off and a paired on/off calibration for each timing capture.
4. For a representative matmul shape, compute expected operations (two per multiply-add) and declared bytes of weights, activations and output. Use their ratio in FLOP/byte with sustained bandwidth and arithmetic throughput measured on the same path. At one column, weights amortize over one token; extra prefill columns can increase reuse. Absolute DRAM traffic, cache hits, utilization, clock rate and energy require corresponding counters. Use model-level throughput/latency and a CPU or paired knob control to test any roofline hypothesis.

A November 27 host replay of the [six historical C4 cells](evidence/2026-11-16-c4-performance/README.md) reproduced every archived per-gate median, 95% bootstrap interval and verdict from the checked-in A/B rows and ledgers. All three LFM cells failed and all three Coder cells passed, with 20 pairs per cell. To repeat, use each `id` and its ordered `gates` in [the frozen manifest](evidence/2026-11-16-c4-performance/manifest.json).

```sh
# Example failing cell. Exit 1 is the recorded performance verdict.
python3 scripts/paired-bench-summary.py \
  bench/results/2026-11-16-c4-paired/cells/lfm25-1.2b-instruct/long-1k \
  --gates prefill,decode,ttft,peak
# Default-profile integrity of the previously measured v1.11.0 Series S package.
python3 scripts/verify-d3d12-selftest.py \
  bench/results/2026-11-27-v1110-series-s-release/receipts/d3d12-selftest-ee6a718-off.csv
```

The latter replay verified 24 distinct GPU selftests. Both checks reuse historical console records. Fresh runs still require the matching package and device.

### GPU opportunities and decisions

| Candidate                                                                               | Implemented baseline / cost to isolate                                                                                                                                                                             | Decision and acceptance                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Command lists, allocators, PSOs, descriptors                                            | A process-wide queue, allocator, list, root descriptors and per-type/width PSOs are already reused. Submission waits protect reuse; opt-in per-dispatch timing adds UAV barriers.                                  | Profile the number of dependency-driven splits and fence cost per token in #421. Require completion proof for every in-flight allocator/buffer before proposing asynchronous reuse. Persistent disk PSO caching needs measured cold PSO cost.                                                                                                                                                                                                                         |
| GPU-resident intermediates, fusion, tiled prefill, wave and thread-group specialization | Supported operand shapes gate placement, shared q8 inputs avoid repeated quantization, and thread widths already vary. Bias ADD and RoPE have independent opt-ins. Q4_0 C4 was correct but lost end-to-end on LFM. | Retain defaults; benchmark hypotheses per model/quant/phase. The [community review](../audit/excellence/standards-2026-11-27/forks-and-gpu.md) credits Chorss and Christopher Bearman. [#589](https://github.com/gianlucamazza/xllama/issues/589) owns FA/small-op/quantizer assessment, [#590](https://github.com/gianlucamazza/xllama/issues/590) owns remaining RoPE/timing deltas. Compare upstream support and array alias/barrier requirements before any port. |
| Attention, sampling, KV and expert placement                                            | GGUF shipping attention/KV/sampler remain on CPU; DirectML owns its ONNX graph. Moving one op may add synchronization or a full logits copy.                                                                       | Quantify round trips, bytes, memory and end-to-end token latency. Use matched EOS/seed and teacher-forced logits where applicable. Keep streamed experts on #479 → #480 → #481; GPU expert kernels need their own dispatch, parity and resource receipts.                                                                                                                                                                                                             |

## Profiling coverage and bottlenecks

Instrumentation checkpoint (2026-11-10): #428 adds opt-in prefill tensor
capture and control probes; #430 adds per-prefill/decode counters for CPU q8
quantization, command recording, execute-through-fence, submissions, dispatches
and valid/missing GPU timestamps; #431 adds client-side API stream timing.
Tensor capture is diagnostic, not temporal operator/layer profiling. Fresh
console captures, overhead calibration and CSV/sidecar integration remain pending.
Implemented tools do not establish a new performance result.

Console checkpoint (2026-11-15): the Coder-3B control was reproduced on a
Series S with package 1.7.0.1533 and diagnosed with tensor capture. The dumps
are byte-identical to 2026-11-08, the frozen verdict is still FAIL, and two
host CPU builds show the same spread between themselves. The same run gives
the first console `d3d12-phase` lines. Details and limits:
[gguf-gpu-decode.md](../gguf-gpu-decode.md#2026-11-15-control-reproduction-and-tensor-diagnosis-420);
[receipt](evidence/2026-11-15-coder-control-diagnosis/receipt.json).

Console checkpoint (2026-11-18): package 1.7.0.1667 on a Series S. The
shared timing opt-in and a CPU arm of the working set are in
[evidence/2026-11-18-d3d12-layer-overhead](evidence/2026-11-18-d3d12-layer-overhead/README.md).
Published tok/s stay in [benchmarks.md](../benchmarks.md).

| Requested metric / bottleneck             | Existing source                                            | Current limitation / next gate                                                                                                                                     |
| ----------------------------------------- | ---------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Prefill/decode throughput, load, TTFT     | `bench.cpp`, shared decode loops, bench CSV/sidecars       | CSV TTFT starts at prefill, excludes load; retained timing is package-specific                                                                                     |
| API TTFB / end-to-end request latency     | Client stream timing probe (#431)                          | Probe records connect, complete header, first body byte and first content; console runs and common CSV integration pending                                         |
| GPU time, graph wall time, dispatch count | D3D12 query pair per submission, backend-lifetime counters | Per-phase counters integrated in #430; Series S layer-line calibration is in [2026-11-18-d3d12-layer-overhead](evidence/2026-11-18-d3d12-layer-overhead/README.md) |
| CPU compute time                          | CPU inference durations, mixed-graph wall time             | Wall minus GPU is not CPU compute; separate quantization, command recording and waits                                                                              |
| Per-operator/layer                        | ORT provider/op traces; ggml tensor names available        | Fused ORT nodes hide layers. GGUF d3d12 layer lines are in [2026-11-18-d3d12-layer](evidence/2026-11-18-d3d12-layer/README.md)                                     |
| Synchronization / dispatch overhead       | `gpustep` round-trip and sync/nosync probes                | Historical synthetic probe, not current per-token attribution                                                                                                      |
| Transfer bytes / effective bandwidth      | `gpugemv`, `gpubw`, weight buffers and UMA architecture    | Packed weight BW is not total inference BW; current graph transfer counters absent                                                                                 |
| Allocation / peak memory                  | Process working-set helpers and GPU budget observations    | Distinguish lifetime OS peak from sampled maxima and reset scope; no full allocation ledger                                                                        |
| KV utilization                            | Session/context budget and KV persistence                  | No unified occupancy/bytes/fragmentation export across backends                                                                                                    |
| Shader compile / warmup                   | Offline shader compilation and PSO init                    | Shader compiler time is host build time; PSO creation/warmup require separate on-device intervals                                                                  |

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

| Stage         | Work / owner lane                            | Acceptance                                                                          | Status / dependency                                                                             |
| ------------- | -------------------------------------------- | ----------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| M0, week 1    | Source/evidence audit, principal/runtime     | This baseline, route/coverage map and triage reviewed                               | Audit produced; no fresh console baseline                                                       |
| M1, weeks 1–2 | Integrity and profiling, QA/performance      | Fail-closed CLI/trace tests; complete attributable current-device capture           | #420/#421; tools integrated; Series S calibration and layer sidecar delivered                   |
| M2, weeks 2–3 | Coder control RCA, numerical/GPU             | First causal divergence isolated and frozen parity passes                           | #420 closed; control traced to layer 0, ADR 0010 amended, both gate models PASS (2026-11-15/16) |
| M3, weeks 3–5 | Measured kernel/overhead work, HLSL          | Independent reversible PRs, numerical and resource gates, paired gains              | #313 closed, #414 closed unmerged: C4 correct, paired campaign FAIL on LFM2.5-1.2B (2026-11-16) |
| M4, weeks 4–6 | Deterministic routing, runtime/platform      | Measured envelope, logged reasons, fallback/override and repeatable tests           | #422; existing defaults preserved pending data                                                  |
| M5, weeks 5–7 | Matched S/X suite, benchmark/reproducibility | Raw attempts, device receipts, uncertainty, quality and memory gates                | #422; Series X supported from an external tester's runs (#455), no project measurements         |
| M6, weeks 6–8 | Hardening/report, QA/release/reviewer        | Required CI plus device lost, concurrency, upgrade/download and final-package gates | #395/#422; publication blocked until evidence sufficient                                        |

Host integrity and diagnostic tools are integrated, and the control has been
reproduced and traced on the console, down to single int8 rounding flips in the
activation quantizer. The comparison of
[ADR 0010](../adr/0010-logit-parity-reference-spread.md) is implemented
(`scripts/control-parity.py`), and the first control under it
[failed on 2026-11-15](evidence/2026-11-15-adr0010-control/README.md): no
first-layer kernel error, but the GPU arm's largest maximum error is 1.37
times the CPU reference spread against a declared 1.22, and one prompt's top
token differs. A [reference check](evidence/2026-11-15-gpu-kernel-reference-check/README.md)
then found every GPU weight product correct in all 36 layers, so the gap is
amplified rounding between two correct executions, not a kernel defect. The
ADR was amended on that basis, and a fresh control under the amendment
[passed](evidence/2026-11-15-adr0010-control-2/README.md) on both parts, as
did LFM2.5-1.2B and the four-column kernel. The preregistered #313
performance campaign then
[failed](evidence/2026-11-16-c4-performance/README.md): 1.5 times faster
prefill on Coder-3B and 15 % slower on LFM2.5-1.2B, so the kernel is not
promoted; #313 is closed as completed and #414 closed without merge. The #493 parser emits a layer CSV sidecar. #421 still tracks
the externally reported Series X submission cost and the conditional Q4_0
shader follow-up. A KV sidecar is a separate proposal. Layer lines separate GPU timestamps from `wait_ms`
(`exec_wait_ms − submission GPU time, divided by n`). That remainder can
come out slightly negative because the CPU and GPU clocks are different
domains. A K-quant-only variant would need a new issue
and preregistration, and serves no catalogue model today. New weight types, kernel fusion, attention
and GPU KV are higher-risk experiments with no current evidence of a benefit.
Do not replace validated CPU paths merely to increase the fraction on GPU.

Technical report working draft:
[Optimizing Quantized LLM Inference on Xbox RDNA 2 GPUs](gpu-acceleration-report.md).
It is not a publication-ready result or an added research-registry claim.
