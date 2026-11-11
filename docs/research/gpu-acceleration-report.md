# Optimizing Quantized LLM Inference on Xbox RDNA 2 GPUs: Kernel Design, CPU/GPU Trade-offs and Reproducible Measurements

Working draft, 2026-11-09. **Not publication-ready.** The
[baseline](gpu-acceleration-baseline.md) binds the audited source and distinguishes
current implementation, historical observations and unmeasured hypotheses.
No new console performance measurements are presented here.

## Research question

Which model/quantization/prompt/context/device envelopes benefit from custom
D3D12 offload, once CPU quantization, graph boundaries, synchronization, memory,
KV behavior and user-visible latency are included? Faster isolated GEMV is
insufficient to answer this question.

## Implemented system

xLlama combines llama.cpp CPU execution with an xLlama-owned ggml D3D12 backend
for selected quantized weight operations, and a separate ORT GenAI/DirectML path
for compatible ONNX assets. The custom path retains CPU attention, KV, norms,
RoPE and sampling. Its GPU weight kernels and q8 activation arithmetic are
specified in [the backend design](../gguf-gpu-decode.md); the
[model matrix](../model-matrix.md) identifies current shipping eligibility.
The system uses public UWP/system D3D12 interfaces without firmware changes.
This does not imply access to every feature of the retail RDNA 2 hardware.

## Demonstrated evidence and negative findings

Historical synthetic bandwidth and full inference are separate experiments;
consult [generated benchmarks](../benchmarks.md) for selectors and raw sources.
Prior matched Series S gates established scoped model defaults. They do not
measure the exact audited main under a fresh campaign and do not establish
Series X behavior.

The latest C4 prefill attempt stopped on its original-kernel Coder control's
numerical failure. Raw dumps remain accessible in the dated execution ledger.
LFM control/experimental logits passed, but this cannot override a failed Coder
cell. The Coder C4 arm and paired performance trials were not executed. A C4
regression, benefit or cause cannot be concluded from these observations.

## Reproducible method

Use the baseline's execution map and the existing #313 preregistration. Freeze
source/submodule/shader/compiler/model/tokenizer/prompt/package/device bindings
before measuring. Run CPU, applicable ORT and custom D3D12 configurations while
preserving quantization/asset differences in the interpretation. Use all failed
attempts, warmup and cold-start policy, matched AB/BA ordering and the already
specified paired uncertainty method. Never pool away a failed model/prompt cell.

Report prefill/decode throughput, TTFT with its actual start boundary, API TTFB
with a distinct byte boundary, end-to-end latency, process-peak and sampled
memory scopes, numerical gates and missing instrumentation. Use hardware GPU
timestamps only when valid; ORT provider trace durations are host event
intervals. Fused node names cannot prove internal layer placement. Every figure
must resolve to immutable raw inputs and an executable aggregation command.

## Instrumentation checkpoint — 2026-11-10

Control probes and opt-in prefill tensor capture are integrated through #428.
Phase counters through #430 distinguish CPU q8 quantization, command recording,
execute-through-fence and valid GPU timestamp time; missing timestamps remain
unavailable. Wall minus GPU time is not CPU compute. Client-side API timing
through #431 distinguishes response-header, first body byte and first content
boundaries from internal benchmark TTFT.

These tools do not complete #420 or #421: fresh console control reproduction,
causal diagnosis, temporal operator/layer attribution, overhead calibration and
CSV/sidecar integration remain pending. Tensor capture does not measure
operator execution time. C4 (#313/#414) remains stopped at control parity;
routing qualification (#422) still depends on correctness and matched evidence.

## Hypotheses and required experiments

| Hypothesis                                        | Required evidence                                                    | Current disposition                                   |
| ------------------------------------------------- | -------------------------------------------------------------------- | ----------------------------------------------------- |
| Multi-column packed weight reuse improves prefill | Control parity, C4 parity/resource/H9 then frozen matched experiment | Stopped at control failure                            |
| Submission overhead limits further decode gains   | Current per-phase submission/quantization/wait/GPU attribution       | Tools integrated in #430; fresh console trace pending |
| Larger quantized models benefit more from offload | Matched model/context matrix within memory constraints               | No universal crossover established                    |
| Static routing can improve user latency           | First/multi-turn, cold/warm and KV-aware matched measurements        | Existing scoped rules; general router unqualified     |
| Series X supports the same inference envelope     | Independent exact-package device receipt and execution               | No inference measurements in this execution           |

## Limitations and publication gate

This audit cannot isolate the first divergent tensor without new captures.
Host regression tests do not validate HLSL execution, device removal behavior,
Xbox memory pressure, power, thermals or Series X compatibility. Model-dependent
host tests and optional Python dependencies must retain explicit skip status.
The existing H9 screen is a regression screen, not broad output-quality proof.

Publication requires a closed control RCA, sufficient matched console evidence,
complete provenance, reproducible figures, supported conclusions and peer
review. Until then, keep this draft out of accepted claims, Zenodo updates and
release performance announcements. Preserve rejected kernels and null results
as part of the research record.
