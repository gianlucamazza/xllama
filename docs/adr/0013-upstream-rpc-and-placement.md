# ADR 0013: Evaluate pinned GGML RPC with explicit tensor ownership

Status: Proposed for maintainer review. [RPC inventory](../research/distributed/upstream.md) · [GGML compatibility work item](https://github.com/gianlucamazza/xllama/issues/690).

## Context

GGML already supplies backend registration, remote buffers, tensor transport and graph computation. The pinned code has desktop and failure assumptions. The xLlama GPU path is a separately registered local D3D12 backend. Upstream version agreement, a socket connection and a successful model load each test different properties.

## Decision proposed

Start with CPU and static registration in a disposable host process at the exact pinned revisions. Reuse the upstream transport for a compatibility experiment. Compare a newer upstream build in a separate matched-version pair. Keep proposed changes to general error handling, placement evidence or platform abstraction small enough to offer upstream under the existing patch lifecycle. [ADR 0011, Proposed](0011-standard-first-and-upstream-first.md).

For a two-stage dense model, the coordinator holds the artifact identity, tokenizer, sampler and placement plan. It needs its own loading and staging budget. Assign contiguous whole layers to workers. Each worker owns allocated weights, scratch and any KV allocated with its attention layers. Trace the actual placement at load, prefill and decode, including embeddings, final projection, logits and any CPU fallback. Provisioning the GGUF once per HTTP replica and loading GGUF tensors into remote RPC buffers are separate operations.

## Alternatives and decision boundary

Pinning a third-party engine, implementing a new distribution protocol, or trusting aggregate memory before tracing the largest worker are poor initial experiments. If the pinned backend cannot recover or cannot express the required boundary, keep the HTTP lane usable and record the unmet contract. Evaluate an upstream fix or a separately reviewed host-only alternative. DC-4 requires stable placement evidence, per-node fit at peak, a contained failure and a resident numerical reference. Record a maintainer decision before adding an Xbox build path or changing a pin.
