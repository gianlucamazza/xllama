# v1.9.0 release package: Series S validation (2026-11-26)

Package-bound validation of the v1.9.0 release package on a project Xbox Series S in Developer Mode.
Times are UTC.

## Binding

- Source commit: `8c538c62efd2122f665fc1cd96dd1025c231d126` (main, merge of #604).
- Package: CI MSVC unified `xllama-appx` from the `build-uwp` push workflow on that commit (run `45846298666`),
  `xllama_1.9.0.1963_x64.msix`, SHA-256 `eed3ea3890b7a7c62841b3836a0da099c251a4b81c10cfbfae5aff7a2756e74e`.
  Sideloaded through Device Portal as an in-place update over `1.8.1.1956`. Models, chats and settings were kept.
- Package SHA-256: `receipts/package-SHA256SUMS`. Device and OS: `receipts/bindings.json`. Summary: `release-validation.json`.

## Outcomes

| Gate | Command | Result | Receipt |
| --- | --- | --- | --- |
| In-place update | `deploy.sh <msix>` | PASS: the LocalState listing (top level, 12 model folders, 18 chats, KV snapshots) was identical before and after the update | `localstate-before.txt`, `localstate-after.txt`, `restoration.json` |
| Console suite (10 gates) | `validate-console.sh all` | PASS, 10/10 | `console-gates.json`, `validate-console.log` |
| LAN API suite (12 gates) | `validate-api.sh all`, `ORT_MODEL=smollm2-360m-cpu-int4` | PASS, 12/12 (see below) | `api-gates.json`, `validate-api.log`, `validate-api-embed.log` |
| Client disconnect during prefill (#581) | `validate-api.sh disconnect` | PASS: the session served a probe 8 s after the client left (limit 30 s) | `validate-api-disconnect.log` |
| D3D12 selftest, knobs absent | `bench-d3d12-selftest.sh` | PASS, 24/24 cases | `d3d12-selftest-8c538c6-off.csv`, `d3d12-selftest.log` |

The `embed` gate first failed before sending a request: its model, `embed-nomic-v2-moe`, was not on the
console, and `/api/embed` answered 404 naming it. The model was pulled through the app's own
`POST /api/pull`, and the unmodified gate then passed with 17 live requests. Every other gate passed in
the first run. The `/health` reply names the package (`"version":"1.9.0.1963"`) and the new capabilities
(`reasoning_content`, `reasoning_tokens`, `cached_tokens`).

The validation host reached the console directly on its local network for the API and through the
same network for Device Portal.

## Restoration

The six chats the gates created were removed afterwards, leaving the 18 chats there were before. Gate
outputs left at the top level of LocalState are listed in `restoration.json`.

## Not in scope

- Throughput, speed, power and thermal figures. This record makes no performance claim; the release's
  measured prefill figures are in `../2026-11-26-prefill-knobs/`.
- The opt-in D3D12 bias ADD and ROPE paths: unchanged since v1.8.1, whose record covers them
  (`../2026-11-24-v181-series-s-release/`).
