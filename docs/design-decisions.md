# Design decisions

Why xllama is built the way it is. Accepted, dated decisions live in
[adr/](adr/). Structure lives in [architecture.md](architecture.md). This page
is the narrative that used to sit on the README.

## Why Xbox Series S?

Zen2 CPU wins decode on the DirectML path and at small model sizes; RDNA2
wins batch prefill, and our own D3D12 kernels win GGUF decode on LFM2.5-1.2B
(Q4_0).
Unified memory means the GPU budget ([uwp-constraints.md](uwp-constraints.md) §7) is the hard constraint.
An underexplored platform with strict memory and packaging constraints.

## Why dual backend?

Per-workload verdict on DirectML: CPU decode > GPU decode, GPU prefill > CPU
prefill. One backend can't win both. Runtime dispatch per model is the answer.
GGUF decode also runs on our own D3D12 kernels: a ggml backend `d3d12`, with
q8 activation quantization like the CPU path. It is the default only for
LFM2.5-1.2B; every other GGUF model runs on the CPU. Every GPU weight
product is checked against an exact reference
([result](research/evidence/2026-11-16-adr0010-control-lfm25-1.2b/README.md)).
A four-column prefill kernel was tried and not adopted: it made a 3B
research model 1.5 times faster to prefill and the shipping LFM2.5-1.2B 15 %
slower ([result](research/evidence/2026-11-16-c4-performance/README.md)).
`gguf_gpu_layers.txt` overrides the default for an operator, and
`0` forces the CPU. Gates and per-model verdicts are in
[gguf-gpu-decode.md](gguf-gpu-decode.md). How that compares with
desktop RDNA2, the Steam Deck and other GPUs on published llama.cpp numbers is
in [benchmarks.md](benchmarks.md#gpu-decode-in-context--known-hardware).
Logit-parity policy: [ADR 0010](adr/0010-logit-parity-reference-spread.md).

## Why single session owner?

A CPU model and a DML model don't fit the budget together
([uwp-constraints.md](uwp-constraints.md) §7). Two models = OOM.
One process-wide resident session makes this an invariant, not a per-surface
convention. GUI and API resolve the same load through
[ADR 0007](adr/0007-session-spec.md) and share a compute permit
([ADR 0008](adr/0008-process-compute-admission.md)).

## Why token-budget, not chars-per-token?

A chars-per-token estimate is not a safe stand-in for a token count: being
wrong truncates the reply or drops history that would have fit. The estimate
survives only for routing, where being wrong costs a decision, not an answer.

## Why consolidate duplicated loops?

Every copy in this codebase has eventually disagreed — silently.
`decode_loop.h` (llama), `decode_loop_ort.h` (ORT), `sampler_chain.h` (llama),
`ort_sampling.h` (ORT) — each decision lives in one header.
