# v1.8.1 release package: Series S validation (2026-11-24)

Package-bound validation of the v1.8.1 release package on a project Xbox Series S in Developer Mode.
Times are Europe/Rome (UTC+1) unless they end in `Z`.

## Binding

- Source commit: `b47fe340951cc1bb79a8d1ce8a9434185ff48a38` (main, merge of #529).
- Package: CI MSVC unified `xllama-appx` from the `build-uwp` push workflow on that commit,
  `xllama_1.8.1.1764_x64.msix`, SHA-256 `7126169533a9e0dedfe1c9cc00500f28fc14087727c5e5d6a200ed91904f99fd`.
  Sideloaded through Device Portal as an in-place update over `1.8.0.1754`. Models, chats and settings were kept.
- Package SHA-256 values: `package-SHA256SUMS`. Device and OS: `bindings.json`. Summary: `release-validation.json`.

## Outcomes

| Gate | Command | Result | Receipt |
| --- | --- | --- | --- |
| In-place update | `deploy.sh <msix>` | PASS: the model folders, the existing chats and the settings files were unchanged after the update | `bindings.json`, `restoration.json` |
| Console suite (10 gates) | `validate-console.sh all` | PASS, 10/10 | `console-gates.json` |
| LAN API suite (11 gates) | `validate-api.sh all`, `ORT_MODEL=smollm2-360m-cpu-int4` | PASS, 11/11 (see below) | `api-gates.json`, `validate-api-framing-lan.log` |
| D3D12 selftest, knobs absent | `bench-d3d12-selftest.sh` | PASS, 24/24 cases | `d3d12-selftest-b47fe34-off.csv` |
| D3D12 selftest, bias ADD knob `1` | `bench-d3d12-selftest.sh --require-bias-add` | PASS, 25/25 cases | `d3d12-selftest-b47fe34-bias.csv` |
| D3D12 selftest, ROPE knob `1` | `bench-d3d12-selftest.sh --require-rope` | PASS, 34/34 cases; all ten `rope_*` rows bit-identical to the CPU reference | `d3d12-selftest-b47fe34-rope.csv` |
| ROPE on/off logit parity | `logits.flag`, stride 16, 99 GPU layers, ubatch 512 and 1, fresh process per arm | PASS: 12/12 byte-identical comparisons | `rope-parity/` |

Ten LAN API gates passed in a run from the validation host, which reaches the console through a proxy.
That proxy closes a connection when the client half-closes it, which the framing gate does on purpose,
so framing could not be judged on that route. The unmodified framing gate was then run from a host on
the console's local network and passed all seven cases (`validate-api-framing-lan.log`). The
transport-limited attempt is kept in `api-gates.json`.

## ROPE logit parity

The criterion was fixed before the run: the logits must be identical bit for bit in 12 of 12 comparisons.
Both arms had the bias ADD knob set to `1`. Arm A had no `d3d12-rope.txt`, and arm B had it set to `1`.
The prompt was `bench/prompts/long-1k.txt`. Each cell dumped the final logits and every 16th prompt
position. Every arm's log shows `bias_add=1`, `rope=0` in A and `rope=1` in B, and 99 requested and
resolved GPU layers.

| Model | ubatch | File | SHA-256 A (ROPE off) | SHA-256 B (ROPE on) |
| --- | ---: | --- | --- | --- |
| qwen25-coder-3b | 512 | `logits.bin` | `493aba0b8fecd0cf50cd84ce80069078a1bf35043f4a82666f9bc131690068d8` | `493aba0b8fecd0cf50cd84ce80069078a1bf35043f4a82666f9bc131690068d8` |
| qwen25-coder-3b | 512 | `logits.bin.positions.bin` | `e138e2b26513263c986d4e5debf0d9cacc2f4386996b2ffdb0deb59dd631bdc2` | `e138e2b26513263c986d4e5debf0d9cacc2f4386996b2ffdb0deb59dd631bdc2` |
| qwen25-coder-3b | 1 | `logits.bin` | `f9f354f9efec613947af86114f01f81db21a14772afdba01e4a08c163799c57d` | `f9f354f9efec613947af86114f01f81db21a14772afdba01e4a08c163799c57d` |
| qwen25-coder-3b | 1 | `logits.bin.positions.bin` | `68cfacd8131301e22a6dcc723904be2a937b2e9f2261b622ec52292ad398ad31` | `68cfacd8131301e22a6dcc723904be2a937b2e9f2261b622ec52292ad398ad31` |
| qwen25-coder-1.5b | 512 | `logits.bin` | `975b69db7ab9eadf43dfdf2a08be7bff60ff5277d3e74cc2023f391e88c50c3b` | `975b69db7ab9eadf43dfdf2a08be7bff60ff5277d3e74cc2023f391e88c50c3b` |
| qwen25-coder-1.5b | 512 | `logits.bin.positions.bin` | `bc60fad9f3ddbba7ae347e67c7a009ef1dc7a22437d873295e880425b5192ca7` | `bc60fad9f3ddbba7ae347e67c7a009ef1dc7a22437d873295e880425b5192ca7` |
| qwen25-coder-1.5b | 1 | `logits.bin` | `5abf3d1e64ca3d55e4d803981f58926f906ea31ae992d465659878fe4587bf2f` | `5abf3d1e64ca3d55e4d803981f58926f906ea31ae992d465659878fe4587bf2f` |
| qwen25-coder-1.5b | 1 | `logits.bin.positions.bin` | `ad1b74eace7b594f2494f08121c8e35936b8adcf76f3a8c9bd5c78a959736adc` | `ad1b74eace7b594f2494f08121c8e35936b8adcf76f3a8c9bd5c78a959736adc` |
| lfm25-1.2b-instruct | 512 | `logits.bin` | `a6a2aa7241a59336066478afff6e3ba0e16671e19d80c1d4135c4cacba55d3cd` | `a6a2aa7241a59336066478afff6e3ba0e16671e19d80c1d4135c4cacba55d3cd` |
| lfm25-1.2b-instruct | 512 | `logits.bin.positions.bin` | `7d2348470470bf1881d934e8d18ce6ceb569bc70532a45064ad677d04ff5f62c` | `7d2348470470bf1881d934e8d18ce6ceb569bc70532a45064ad677d04ff5f62c` |
| lfm25-1.2b-instruct | 1 | `logits.bin` | `126cff5fd49669442675e549283c5f3b2020ad2b5cfe6b6cc2f58d6d5f0ce068` | `126cff5fd49669442675e549283c5f3b2020ad2b5cfe6b6cc2f58d6d5f0ce068` |
| lfm25-1.2b-instruct | 1 | `logits.bin.positions.bin` | `123d80876cb4cc9ac87cea479227591f8bcbf7fdec0ea07bbc0d1bd9201d2023` | `123d80876cb4cc9ac87cea479227591f8bcbf7fdec0ea07bbc0d1bd9201d2023` |

Models: Qwen2.5-Coder-3B and Qwen2.5-Coder-1.5B Q4_K_M, LFM2.5-1.2B Instruct Q4_0. Per-cell details are in
`rope-parity/summary.json`.

## Not in scope

- Throughput, speed, submission-count, power and thermal figures. This record makes no performance claim
  for the ROPE or bias ADD knobs, which stay off by default.
- Logit binaries and full device logs. The operator retains them; they are not committed because of their size.

The console's knob files, settings, model and prompt were restored and checked byte for byte against
copies taken before the run. The app was stopped and the console was left on the Developer Mode home screen.
