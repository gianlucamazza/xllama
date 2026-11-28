# ADR 0012: Start distributed research with an external request coordinator

Status: Proposed for maintainer review. [Program and gates](../research/distributed/README.md).

## Context

A second console could serve another request even if neither console could serve a larger model. A single token stream split across workers has different memory ownership, failure and latency costs. The existing foreground app shares one resolved session and one compute permit between callers. [ADR 0007](0007-session-spec.md) · [ADR 0008, Proposed](0008-process-compute-admission.md) · [LAN contract](../api-endpoint.md).

## Decision proposed

1. Carry OOC-6 as a separate research program. Retain the standalone application as the baseline in every experiment. Scope the initial service to operator-selected, independent replicas of an identical model.
2. Use an external Linux or Windows coordinator first. Reuse an existing streaming HTTP client or proxy with bounded admission. It owns the queue, health observations, session affinity, cancellation and timing. Each console owns its model, local KV, compute permit and response stream.
3. Separate this control plane from the tensor data plane. A replica's application data stays in its HTTP request and response. A future sharded experiment would let the upstream graph driver own tensor traffic. Discovery or a compatible HTTP response gives no evidence about remote tensor support.

## Alternatives and decision boundary

An Xbox coordinator would remove a PC from the deployment. It would also place queues, sampling, staging and local inference under one foreground application budget and failure boundary. Compare that configuration only after measuring external coordination. A heterogeneous PC worker requires its own per-stage fit and kernel reference. Power the external coordinator and switch through the wall meter when comparing cost.

Review this ADR against DC-1 and a two-replica DC-2 receipt. Reconsider the placement if a packaged Xbox can bound coordination overhead, release all resources on suspension, and restore single-node service within the declared recovery deadline. The experiment is opt-in. Its acceptance does not change the catalogue or release policy.
