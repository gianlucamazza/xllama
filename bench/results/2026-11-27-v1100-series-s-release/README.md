# v1.10.0 release package: Series S validation (2026-11-27)

Package-bound validation of the v1.10.0 release package on a project Xbox Series S in Developer Mode.
Times are UTC.

## Binding

- Source commit: `7c36f95ec20d3cd2f28e53436f4fecdb12795f6d` (main, merge of #619).
- Package: CI MSVC unified `xllama-appx` from the `build-uwp` push workflow on that commit (run
  `45935512254`), `xllama_1.10.0.2000_x64.msix`, SHA-256
  `839c2c5b9cb70f170ca80241ee24a276b1c23825e73c0750571c109acbb8460f`. Sideloaded through Device Portal
  as an in-place update over `1.9.0.1982`, a higher revision (#603). Models, chats and settings were
  kept.
- Package SHA-256: `receipts/package-SHA256SUMS`. Device and OS: `receipts/bindings.json`. Summary:
  `receipts/release-validation.json`. The scripts that ran are those of the source commit.

## Outcomes

| Gate                           | Command                                                  | Result                                                                                                                              | Receipt                                                             |
| ------------------------------ | -------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------- |
| In-place update                | `deploy.sh <msix>`                                       | PASS: the LocalState listing (top level, 13 model folders, 17 chats plus `index.json`, KV snapshots) was identical before and after | `localstate-before.txt`, `localstate-after.txt`, `restoration.json` |
| Console suite (10 gates)       | `validate-console.sh all`                                | PASS, 10/10                                                                                                                         | `console-gates.json`, `validate-console.log`                        |
| LAN API suite (12 gates)       | `validate-api.sh all`, `ORT_MODEL=smollm2-360m-cpu-int4` | PASS, 12/12                                                                                                                         | `api-gates.json`, `validate-api.log`                                |
| Client disconnect (#581, #594) | `validate-api.sh disconnect`                             | PASS: a probe was served 9 s after a plain stream's client left and 10 s after a tool-proposal stream's (limit 30 s)                | `validate-api-disconnect.log`                                       |
| D3D12 selftest, knobs absent   | `bench-d3d12-selftest.sh`                                | PASS, 24/24 cases                                                                                                                   | `d3d12-selftest-7c36f95-off.csv`, `d3d12-selftest.log`              |

Every gate passed in the first run. The `embed` gate ran 17 live requests against
`embed-nomic-v2-moe`, which was already on the console. The disconnect gate now has a tool-proposal
round: that is #594's acceptance criterion, measured here on the release package.

`GET /health` (`receipts/health.json`) names the package (`"version":"1.10.0.2000"`, `"api":1`) and
this release's two new fields:

- `sampling`, which lists `top_k` and `repetition_penalty` among the accepted chat fields (#609);
- `runtime`, which names the llama.cpp commit `d812350` and its four patches (#611).

The validation host reached the console directly on its local network, for the API and for Device
Portal. The console's address and the Device Portal user are replaced with `console` and `<user>` in
the receipts.

## Restoration

The six chats the gates created were removed afterwards. Their titles are the gates' prompts, such as
"Review this code." and "What is 2+2? …". That leaves the 17 chats (plus `index.json`) there were before
(`localstate-restored.txt`). The only remaining difference from `localstate-before.txt` is the #170b
gate's KV snapshot (`ap-170b-switch-*.kv`), which the new run replaced with its own.
`localstate-end.txt` is the listing before the gate chats were removed.

## Not in scope

- Throughput, speed, power and thermal figures. This record makes no performance claim. The release's
  measured figures are in the receipts of the pull requests it ships: #613, #614 and #615.
- The opt-in D3D12 bias ADD and ROPE paths: unchanged since v1.8.1, whose record covers them
  (`../2026-11-24-v181-series-s-release/`).
