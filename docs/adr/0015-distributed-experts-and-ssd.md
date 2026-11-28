# ADR 0015: Preserve router semantics before combining remote experts and SSD

Status: Proposed for maintainer review. [Single-node plan of record](../superpowers/plans/2026-11-24-moe-engineering-program.md) · [OOC-6 tracker](https://github.com/gianlucamazza/xllama/issues/687).

## Context

The single-node work already owns expert loading, bounded cache leases, native CPU parity and the separate hardware acceptance stages. Distributing selected experts adds remote activation traffic, ordering, queue skew and a failure boundary. An expert cached on another worker is not a hit in a local SSD cache.

## Decision proposed

Qualify the constituent mechanisms separately. DC-7 may place immutable expert replicas near demand only after checking a resident router reference. Preserve original selected IDs, weights, tie behavior, combine order, shared experts and arithmetic. Send activations only to selected owners. A cache miss fetches the selected expert under its bounded lease; it never substitutes a different expert. Account for in-flight reads, weights, working copies and scratch per node. Avoid partial aggregate output after a worker fails.

DC-8 can compare resident expert parallelism, local SSD-backed inference, contiguous-layer placement and their composition after DC-7 and the relevant [#479](https://github.com/gianlucamazza/xllama/issues/479) → [#480](https://github.com/gianlucamazza/xllama/issues/480) → [#481](https://github.com/gianlucamazza/xllama/issues/481) evidence. GPU has its own dispatch and parity gate. Reuse the existing cache/lease contract when it qualifies.

## Alternatives and decision boundary

Replicate a hot expert only when a trace shows that memory overhead buys fewer remote exchanges or lower tail latency. Cache-aware rerouting changes model semantics and belongs to the independently proposed OOC-5. Stop the hybrid lane if it fails frozen numerical, resource or economic criteria against the best admissible single-mechanism baseline. A failure leaves the local MoE program with its existing owners.
