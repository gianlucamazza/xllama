# Structural proposals from the audit

All proposals are **Proposed**, not accepted or implemented. Number them in the
project ADR sequence only when promoted for owner review. Existing accepted ADRs
remain unchanged. The host exporter already has [ADR 0003](../../adr/0003-diffusion-export-first-party.md).

## Proposal A: one resolved session specification for GUI and API

### Context

[F03](README.md#f03--p1-api-created-sessions-omit-catalogue-lora-policy) shows that
GUI and API assemble different load parameters for the same catalogue ID. The hub
compares only that ID. A correct sampler does not repair a wrong resident model or
missing adapter.

### Decision proposed

Resolve catalogue data once into a WinRT-free session specification containing the
model ID and all load-affecting parameters, including backend, context and adapter
path/scale. GUI, chat API, embedding API and pull use the same resolver. Embedding
context overrides are validated before constructing the specification.

The hub reuses a session only when its resolved specification matches. Successful
model/adapter promotion, rollback or import invalidates the affected resident
specification, even when the catalogue ID and file paths have not changed. Acquire
the hub at that boundary according to the existing ownership contract; never hold
two resident models during replacement. Generation-only sampling remains outside
the load identity.

### Consequences

Adapter behavior becomes independent of entry surface and load order. Some
previously silent reuses become explicit reloads. Unit tests can compare resolved
policies without WinRT; MSVC/device tests still prove the actual adapter load and
memory behavior. The exact public type name is a follow-up implementation detail,
not an API introduced by this audit.

### Alternatives

- Copy the GUI's LoRA assignments into the API: smaller patch, but retains multiple
  policy implementations and ID-only compatibility.
- Always reload: avoids stale reuse but discards the resident-session benefit and
  does not by itself supply missing adapter parameters.

## Proposal B: portable HTTP framing policy, bounded UWP transport

### Context

[F02](README.md#f02--p2-http-framing-is-accepted-without-exact-validation) is in a
WinRT-only reader, so the Linux unit suite cannot exercise its framing rules.
Current request caps limit stored bytes but do not establish a valid message or a
time bound.

### Decision proposed

Extract request-line/header/framing decisions into a WinRT-free parser. Keep
socket reads and writes in UWP. Require complete header names, checked decimal
length parsing, a single unambiguous framing decision and exact body completion.
Reject oversized bodies instead of clamping them. Preserve the explicit rejection
of chunked requests until a separate supported contract is adopted.

Add application-level header/body deadlines and bounded connection admission to
the UWP transport. Set the actual limits in the follow-up ADR review using the
largest supported embedding request and measured slow-client behavior; this audit
does not invent production timeout numbers without those measurements.

### Consequences

Parser tests run on Linux, including malformed and boundary inputs; bounded wire
tests still verify UWP cancellation and resource release. Some previously accepted
malformed clients receive explicit 4xx errors. New limits become part of the LAN
contract and must be documented and measured before acceptance.

### Alternatives

- Add only length checks in `read_request`: repairs immediate framing defects but
  leaves the policy outside host tests and does not bound blocked reads.
- Replace the server with a general HTTP dependency: larger AppContainer and
  packaging change without evidence that it is necessary for this narrow probe.

## Proposal C: process-wide compute-job admission

### Context

F08 reproduces API chat execution during GUI diffusion. GUI running flags protect
only the GUI, while the API assumes the session mutex expresses process-wide
admission. Image generation also uses shared LocalState input/output files.

### Decision proposed

Introduce one process-wide, RAII-owned job permit for text generation, embeddings,
diffusion and on-device training across GUI and API. Acquire before changing job
inputs or starting work, and retain it through completion/cancellation cleanup.
Use a permit whose ownership can cross worker/coroutine boundaries, not a mutex
unlocked by a different thread. Keep `SessionHub::mtx` responsible for resident
session integrity, with a documented order: job admission before session access.

Keep model downloading under its existing writer permit; it is not itself a
compute job. The post-pull load must follow the same admitted model-swap policy.
Read-only health/discovery/status requests remain available. API compute requests
receive the existing busy error format; GUI operations use the same admission
decision before modifying shared inputs.

### Consequences

Work accepted through different surfaces obeys one resource policy. The change
needs tests for opposite entry orders, cancellation and exceptional exits; a
single successful sequential diffusion test cannot validate it. This is a
proposed structural contract, not a permit added by the audit.

### Alternatives

- Check the GUI's booleans in API handlers: retains races and ties transport to UI
  lifetime without an owned job boundary.
- Lock the session mutex throughout every job: conflates session ownership with
  asynchronous resource admission and risks cross-thread unlock or nested locking.
