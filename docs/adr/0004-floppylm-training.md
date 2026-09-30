# ADR 0004: FloppyLM scalar training on the shared CPU engine

Status: Accepted (2026-09-30), implementation plan approved by the owner.

## Context

Lane B trains a subset of a GGUF model. FloppyLM E0 v2 requires training every parameter from
random initialization, tied embeddings, quantization-aware training, isolated WSD cooldowns,
and an independently counted FLP2 artifact. The llama inference graph's cache writes prevent
that backward path. The pinned ggml also lacks the exact GELU backward.

## Decision

Add a separate `floppylm` training method shared by Linux and UWP. Build a complete ggml CPU
graph without KV cache. Quantized forward tensors are graph parameters; their straight-through
gradients update fp32 master weights with AdamW. Carry the minimal exact-GELU backward patch
in patches and apply it to host and UWP builds. Preserve the existing inference and partial-FT
paths. Support the scalar E0 configuration space, not VQ or GPU training.

Pin the independent Python reference to FloppyLM commit 2c2e737. Host preparation exports
initial master weights and batch offsets; neither Python nor a pretrained model is required
on console. Native checkpoints contain optimizer moments, trunk and cooldown state, phase,
batch cursor and cumulative accounting. Checkpoints are distinct from inference-only FLP2.

Require reference fixture parity before device validation, an exact MSVC CI package on Xbox,
and restoration of console state. Benchmark/training budget is eight hours cumulatively;
functional correctness does not imply an acceleration claim. The branch ends in a draft PR,
not a merge or public release.

## Consequences

The backend owns its model graph, codec, optimizer and checkpoint contract. Dependency changes
are small carried patches, never unrecorded submodule edits. Numeric comparisons use identical
initial weights and data: fp32 tolerance 1e-5 absolute / 1e-4 relative for forward, and 1e-5 /
1e-3 for gradients and updates. Codec fixture bytes and serialization round trips are exact.

## Alternatives

The existing GGUF partial-FT graph cannot train embeddings and earlier layers. Replacing GELU
with an approximation changes the experiment. Deploying Python/PyTorch inside UWP introduces
an unsupported runtime. These approaches are rejected.
