# v1.8.0.0 release package: Series S validation (2026-11-23)

Package-bound validation of the v1.8.0.0 release package on a project Xbox Series S
in Developer Mode. Times in the logs are Europe/Rome (UTC+1) unless they end in `Z`.

## Binding

- Source and tag commit: `47d1cee3648eb5e157e67382c81366caed26c344` (main, merge of #510).
- Package: CI MSVC unified `xllama-appx` from the `build-uwp` push workflow on that commit,
  `xllama_1.8.0.1721_x64.msix`. It was sideloaded through Device Portal as an in-place update
  over `1.7.0.1702`, and LocalState was preserved. The installed, registered and running package was
  `GianlucaMazza.xllama_1.8.0.1721_x64__pj67f1fcj4n14`.
- Package SHA-256 values: `receipts/package-SHA256SUMS`. Device: `receipts/bindings.json`.
- `XLLAMA_EXPECTED_PFN` was set to that package for the console and API suites.
- `release-validation.json` is the record attached to the GitHub release.

## Outcomes

| Gate | Command | Result | Receipt |
| --- | --- | --- | --- |
| Console suite (10 gates) | `validate-console.sh all` | PASS | `receipts/validate-console-all.log` |
| LAN API suite (11 gates) | `validate-api.sh all`, `ORT_MODEL=smollm2-360m-cpu-int4` | PASS on attempt 2 (see below) | `receipts/validate-api-all-attempt1.log`, `receipts/validate-api-all-attempt2.log`, `receipts/api-embed-nomic-v2-moe.json` |
| ONNX tool proposal refusal | `validate-api.sh` tools step E | PASS: 400 `tools_unsupported_for_model`, GGUF kept resident | attempt 2 log |
| D3D12 selftest, bias ADD knob absent | `bench-d3d12-selftest.sh` | PASS, 14/14 cases | `receipts/d3d12-selftest-bias-off.*` |
| D3D12 selftest, bias ADD knob `1` | `bench-d3d12-selftest.sh --require-bias-add` | PASS, 15/15 cases, first row `add_f32` ok | `receipts/d3d12-selftest-bias-on.*` |
| Bias ADD on/off logit parity | `logits.flag`, stride 16, 99 GPU layers, ubatch 512 and 1, fresh process per arm | PASS: 12/12 byte-identical comparisons (final and per-position logits) for `qwen25-coder-3b`, `qwen25-coder-1.5b`, `lfm25-1.2b-instruct` | `receipts/bias-parity/` |
| SSD MoE 8B parity on the console | n/a | N/A | see below |

API attempt 1 passed every gate except `embed`. That gate failed because `embed-nomic-v2-moe`
was not on the console, and the API returned 404 for it. The model was then pulled with the product pull path
(`PULL_MODEL=embed-nomic-v2-moe PULL_MODEL_KIND=embedding validate-api.sh pull`, PASS). After that,
attempt 2 of `validate-api.sh all` passed all 11 gates.

The bias parity cells log `d3d12: bias_add=0` or `=1` as requested. Each cell resolved the 99 GPU layers it
requested. Per-cell prefill submission counts are in `receipts/bias-parity/summary.json`.

## Not in scope

- The 20-pair bias ADD performance campaign. It was accepted on CI package 1.7.0.1702 (#504) and was not repeated here.
- SSD MoE parity on the console. The 8B declaration is `PREPARED_NOT_EXECUTABLE`, and the loader integration is not in this package
  (`docs/research/moe-upstream-readiness.md`).
- Throughput, power and thermal figures.
- Logit binaries and full device logs. The operator retains them; they are not committed because of their size.

The console settings, model, prompt and API flag were restored and compared byte for byte with the
pre-run copies. The probe files were removed, the app was stopped and the console was left on the Developer Mode home screen.
