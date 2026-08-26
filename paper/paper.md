# Consumer Game Consoles as Local AI Compute

## LLM and diffusion inference on Xbox Series S

**Gianluca Mazza — xLlama Research Report 1.0**

## Abstract

This engineering report evaluates an Xbox Series S as a local-AI platform under
the constraints of UWP Dev Mode. It measures quantized LLM inference through
llama.cpp, ONNX Runtime GenAI and DirectML, together with a Stable-Diffusion
class pipeline. The central result is that the console is workload-shaped:
CPU execution is effective for autoregressive decode, GPU execution is useful
for sufficiently large batched workloads, and persistent KV state is more
important to interactive latency than a single peak throughput number.

The report is a reproducibility artifact rather than a claim about retail
console certification. Exact figures are generated from the benchmark SSOT and
the claim registry in this directory.

## 1. Research question

How usable is a fixed consumer game console as a general-purpose local-AI
platform when the application must run inside a sandboxed UWP environment?

The question includes both performance and engineering usability: model
admissibility, memory limits, backend correctness, repeatability, and
multi-turn latency.

## 2. Platform and constraints

The target is an Xbox Series S running xLlama in Dev Mode. The relevant
constraints are shared unified memory, a limited application GPU budget,
approximately six practical CPU threads, AppContainer filesystem/API limits,
and the absence of desktop profiling facilities. The complete constraint and
runtime description is maintained in [`../docs/architecture.md`](../docs/architecture.md)
and [`../docs/uwp-constraints.md`](../docs/uwp-constraints.md).

## 3. Methodology

Prompts, model identities, quantization, context, thread count and backend are
recorded with each benchmark row. Repeated runs discard the warm-up and retain
individual measurements; published tables use the selector policy in
`../bench/benchmark-summary.json`. Model hashes, build identity, ambient
conditions, power readings and thermal rules belong in benchmark sidecars.

The report distinguishes host checks, console measurements, generated
summaries, and end-user or retail validation. A passing CI job is evidence that
checks passed for that revision, not proof of runtime behaviour on a console.

## 4. LLM inference

Prefill processes a prompt as a batch and can benefit from GPU parallelism.
Autoregressive decode processes one token at a time and is dominated by weight
streaming, dispatch overhead and memory latency. The measured model and
backend matrix is available in [`../docs/benchmarks.md`](../docs/benchmarks.md).
The claim-level values used by this report are listed in the generated
[`generated/benchmarks.md`](generated/benchmarks.md) table.

The catalogue spans a fast 230M/350M tier through models around 3B. Larger
models trade throughput and memory for capability. KV-cache reuse changes the
second-turn cost materially, so TTFT and turn-2 prefill are reported alongside
decode throughput.

DirectML text routing is allowlisted by model because a successful GPU session
is not sufficient evidence of correct logits. The parity harness and RMSNorm
decomposition are documented in [`../docs/dml-rmsnorm-fix-runbook.md`](../docs/dml-rmsnorm-fix-runbook.md).

## 5. Diffusion

Diffusion is a better GPU-shaped workload than single-token LLM decode. The
pipeline uses sequential ORT DirectML sessions to stay within the console
budget and validates tokenizer, scheduler and image-path components against
host reference vectors. Diffusion measurements remain separate from text
throughput because they have different units, warm-up behaviour and memory
profiles.

## 6. Engineering findings

The useful runtime is not a single backend. It is a policy over model,
workload, conversation state and memory: llama.cpp is effective for CPU decode
and KV reuse; ORT/DirectML can serve selected batched work; and diffusion uses
the GPU-oriented path. A resident session owner avoids loading the same model
twice for the UI and LAN API.

## 7. Limitations and reproducibility boundary

This edition does not claim retail certification, sustained thermally
equilibrated production throughput, power efficiency, or parity with a PC,
mini-PC or Raspberry Pi. Those claims require separately sidecar-backed
measurements. Store submission, external customer/UAT evidence and hardware
comparison are outside this edition.

The exact evidence boundary and selectors are recorded in [`claims.json`](claims.json)
and [`research-manifest.json`](research-manifest.json).
