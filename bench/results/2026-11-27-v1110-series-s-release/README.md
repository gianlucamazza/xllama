# v1.11.0 release package: Series S validation (2026-11-27)

Package-bound validation of the v1.11.0 release package on a project Xbox Series S in Developer Mode.
Times are UTC.

## Binding

- Source commit: `ee6a718d151465e357dc6f3e2b5091bb2a35416e` (main, merge of #682).
- Package: CI MSVC unified `xllama-appx` from the `build-uwp` push workflow on that commit (run
  `46070519230`), `xllama_1.11.0.2203_x64.msix`, SHA-256
  `ed6ac5a66a0d5925c7b6103dc27b73a9f320715b5eb2237c12464f12be2f5651`.
- Provenance: the first release package with a GitHub artifact attestation (#637).
  `gh attestation verify` against `build-uwp.yml`, `refs/heads/main` and the source commit exits 0.
- Install: sideloaded through Device Portal as an in-place update over `1.10.0.2117`, a higher
  revision (#603). Models, chats and settings were kept.
- Package SHA-256: `receipts/package-SHA256SUMS`. Device and OS: `receipts/bindings.json`. Summary:
  `receipts/release-validation.json`.

## Outcomes

| Gate                              | Command                                         | Result                                                                                                                 | Receipt                                                |
| --------------------------------- | ----------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------ |
| In-place update                   | `deploy.sh <msix>`                              | PASS: the LocalState top-level listing was identical before and after the update, except `xllama.log`                  | `localstate-before.txt`, `localstate-after.txt`        |
| Console suite (10 gates)          | `validate-console.sh all`                       | PASS, 10/10; `settings.json` restored at the end (#654)                                                                | `validate-console.log`                                 |
| LAN API suite (12 gates)          | `validate-api.sh all`                           | PASS, 12/12                                                                                                            | `validate-api.log`                                     |
| Client disconnect (#581, #594)    | `validate-api.sh disconnect`                    | PASS: a probe was served 9 s after a plain stream's client left and 9 s after a tool-proposal stream's (limit 30 s)    | `validate-api-disconnect.log`                          |
| Native Ollama (#663, #666)        | `validate-api.sh compat`                        | PASS: NDJSON/JSON framing, refused fields and `think`, load and busy, tool replay, `keep_alive`, `/api/ps`, `num_ctx`  | `validate-api-compat.log`                              |
| D3D12 selftest, knobs absent      | `bench-d3d12-selftest.sh --out …`               | PASS, 24/24 cases, under the complete-receipt gate (#673)                                                              | `d3d12-selftest-ee6a718-off.csv`, `d3d12-selftest.log` |
| Reasoning control (#648)          | requests in `feature-checks.json`               | PASS: qwen3-4b `none` → no reasoning, `high` → 223 reasoning tokens; lfm25-8b-a1b `none` → 400; Ollama `think:false`   | `feature-checks.json`                                  |
| Tool reasoning (#655)             | qwen3-4b tool proposal                          | PASS: `get_weather {"city":"Rome"}`, `finish_reason: tool_calls`, 0 reasoning tokens                                   | `feature-checks.json`                                  |
| Catalogue and replay (#658, #664) | `/api/catalog`, `/api/pull`, replayed tool turn | PASS: `lfm2-2.6b` listed as pullable, pulled in 53 s, and answered from the replayed tool result without calling again | `pull-lfm2-2.6b.json`, `replay-lfm2-2.6b.json`         |

The first `compat` run failed with HTTP 503. The gate sent its next request as soon as it closed a
stream, before the server had released the session (1.2 s on this console). #683 makes the gate wait
for the release within the `disconnect` limit; the receipt is the run with that script. The first
selftest run stopped before starting because the driver refuses to overwrite its recorded CSV under
`bench/results/`; the receipt is the run with `--out`.

`GET /health` (`receipts/health.json`) names the package (`"version":"1.11.0.2203"`, `"api":1`) and
this release's new capabilities: `ollama_chat`, `ollama_keep_alive`, `chat_context`,
`reasoning_control` and `catalog`. The llama.cpp runtime lists patches 0006 and 0007 (#625).

The validation host reached the console directly on its local network, for the API and for Device
Portal. The console's address and the Device Portal user are replaced with `console` and `<user>` in
the receipts.

## Restoration

`validate-console.sh` saved `settings.json` before the first gate, restored it at the end and removed
the chats the gates created (#654). `lfm2-2.6b` stays installed from the replay check.

## Not in scope

- The GUI half of the Ollama residency check in `docs/console-validation-runbook.md` (a pad turn
  while an API lease runs; `keep_alive: 0` on the app's model returning 409). It needs the pad and
  was not run here. The API half is covered by `compat`.
- Throughput, speed, power and thermal figures. This record makes no performance claim. The
  release's measured figures are in the receipts of the pull requests it ships, for example #660
  (`lfm25-8b-a1b` at n_ctx 8192) and #643 (catalogue sampling).
