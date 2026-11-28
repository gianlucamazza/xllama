# ADR 0014: Constrain distributed transports and contain worker failures

Status: Proposed for maintainer review. [Threat and failure model](../research/distributed/security.md).

## Context

The existing opt-in HTTP service assumes a trusted LAN. Pinned GGML RPC has process-fatal peer error paths and no authenticated encrypted peer transport. Model buffers, prompts, KV, routing and intermediate tensors carry sensitive data. AppContainer capabilities determine network access, not peer identity. [Repository security policy](../../SECURITY.md) · [Upstream inventory](../research/distributed/upstream.md).

## Decision proposed

Use an explicitly selected, isolated network with operator-owned endpoints, an allowlist, synthetic public fixtures and a finite run window. Keep raw RPC bound to loopback for host inventory wherever possible. For a proposed isolated multi-host lab, document which endpoints, interfaces and data are exposed, enforce isolation at the network boundary, and close access on teardown. No automatic discovery or public binding. Before any broader use, verify mutual identity, role authorization, integrity and confidentiality end to end with a standard transport on the exact SKU. A host-side tunnel that ends before a console leaves a separate leg to qualify.

Enforce per-run byte, memory, time and concurrency budgets. A lost shard invalidates the in-flight graph, remote leases and session KV. Terminate that attempt once and allow new local work only after clean recovery. The external coordinator must contain a crashing RPC client in a disposable process. A desktop process model does not establish an Xbox isolation mechanism.

## Alternatives and decision boundary

A private IP, a model/cache hash, or an HTTP health response cannot prove peer identity. Adding custom cryptography would introduce a new security protocol and maintenance obligation. A production decision needs the DC-9 evidence for end-to-end protection, teardown, operator consent and independent review. Until then only the narrow research lane with its stated trust assumptions may be proposed.
