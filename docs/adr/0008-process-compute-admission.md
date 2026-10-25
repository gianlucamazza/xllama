# ADR 0008: Process-wide compute-job admission

Status: Proposed

Proposed for owner acceptance. Resolves audit finding F08 and records Proposal C
([adr-proposals.md](../audit/excellence/adr-proposals.md)).

## Context

The SessionHub mutex protects the resident model and serializes its use, but it
does not cover expensive work that does not use a Session. GUI diffusion could
run while an API chat request acquired the SessionHub and started inference.
Diffusion also reads and writes shared LocalState files. Device training frees
the resident Session before running and therefore could similarly overlap with
API work.

## Decision

Add a process-wide, single-slot RAII permit in
`include/xllama/compute_admission.h`. Chat, embeddings, diffusion, device
training, startup pre-load, and the model load after an API pull acquire the
permit before touching the SessionHub or shared diffusion inputs. API compute
requests return the existing 503 `busy` response when another admitted job is
active. API requests and the first GUI send may wait up to 15 seconds while the
startup pre-load is active, preserving the first-request behavior. A streamed
`/v1/chat/completions` reply keeps the permit until the last SSE event is
written. The LAN listener must dispatch each connection off
`ConnectionReceived` (a bounded thread-pool work item in
`include/xllama/api_connection.h`) so a second request can run the busy check
while the first still holds the permit. A client disconnect mid-SSE aborts
generation; the stack-owned permit then releases. A non-streaming pull that
has finished downloading but cannot acquire the permit returns 503; a
streaming pull emits its existing NDJSON error record.

The permit is a shared RAII handle so ownership may safely span worker threads
and UI dispatcher callbacks. Releasing the final handle ends the lease, even
when its destructor runs on a different thread. `SessionHub::mtx` remains
responsible for resident-session identity and access. The lock order is always
compute permit, then SessionHub mutex; never acquire a permit while holding the
SessionHub mutex.

Model downloads and preference writes are not compute jobs and keep their
existing writer and file-append policies. API pull acquires compute admission
only for the post-download Session invalidation and model load.

## Consequences

GUI and API jobs now share one resource policy, including image generation and
training. A busy API pull can finish its download but receive 503 before the
resident Session is swapped; the caller can retry the load request. GUI actions
that lose the admission race report a busy status without changing shared job
inputs. The host test covers exclusivity, opposite entry order, cancellation,
release on error, streamed-handle lifetime, the preload wait, and two
overlapping API compute admits (one 200 and one 503). Console validation
must still exercise both GUI/API overlap directions, LAN/LAN overlap while
an SSE reply is in flight, training overlap, cancellation, and exceptional
exits before release.

## Alternatives

- Checking GUI running flags from API handlers retains races and couples the
  transport layer to UI lifetime.
- Holding `SessionHub::mtx` through every job conflates resident-session
  ownership with compute admission and risks cross-thread mutex release.
