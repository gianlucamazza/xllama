# ADR 0001: Serialize model writes at the downloader boundary

Status: Proposed

## Context

The GUI and LAN API both invoke `ModelDownloader::DownloadAsync`. The API's
`ApiPullGate` serializes API pulls only; it does not cover GUI downloads.
Both paths use the same model directory, `.part` files, rollback files, and
completion marker. Concurrent calls can therefore overwrite another writer's
staging file even though each caller is independently serialized.

## Decision

Enforce one active download inside `ModelDownloader`, shared by GUI and API.
Use an atomic RAII permit rather than a mutex held across coroutine suspension:
callbacks can resume on different threads. A second download fails explicitly
before writing any model file. Keep the API pull gate for its existing HTTP 409
contract and hold it through model loading. A competing GUI download produces
an explicit download-busy error; it must not appear as a successful pull.

Protect the entire downloader operation, including verification, promotion,
rollback backup, and completion-marker publication. Do not alter the SessionHub
inference lock or permit a second resident model.

## Consequences

All callers share the same writer policy. UI and API retries remain explicit;
there is no hidden queue, automatic overwrite, or cross-thread mutex unlock.
Unit tests must cover exclusive acquisition and release after failure/move.
Xbox testing must cover concurrent API pulls and GUI/API overlap.

## Alternatives

- API-only locking leaves the GUI writer outside the invariant.
- Unique staging filenames avoid one collision but leave promotion and rollback
  publication racing.
- A queued download service adds scheduling and cancellation contracts that are
  unnecessary for this fix.
