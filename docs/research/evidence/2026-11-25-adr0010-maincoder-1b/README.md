# Maincoder-1B Q4_K_M — ADR 0010 CPU-only preparation (25 Nov 2026)

CPU-only preparation of an [ADR 0010](https://github.com/gianlucamazza/xllama/blob/main/docs/adr/0010-logit-parity-reference-spread.md) declaration (`xllama.control-declaration/v2`) for Maincoder-1B Q4_K_M, following the 2026-11-15 control-2 and 2026-11-16 c4 (`declaration-coder.json`) templates. The prompt set, its hashes, the three CPU arms and the draft declaration were frozen before any GPU arm on these prompts. **This is not a GPU-arm result and not a control verdict.** The GPU arm was not run.

Session: 25 Nov 2026, 18:50–19:09 Europe/Rome for the three CPU arms (host first, then the console), on an Xbox Series S in Developer Mode. Approval: Gianluca, 25 Nov, after the Maincoder Series S run was stopped for a missing declaration. No repo change and no PR from this session; Coder commits the artefacts.

## Bindings

- App on the console: package **1.8.1.1819** (CI build-uwp run 1819, `workflow_dispatch` on a PR branch, head `ded444fd`). It was already installed and was **not** replaced. It is **not** the `main` release build. Treat the numbers as measured on that build.
- Model: `Maincode/Maincoder-1B-GGUF`, `Maincoder-1B-Q4_K_M.gguf`, HF revision `602967f81be467f79f7b7caf345b98471da6d724`, 672,108,032 B, SHA-256 `8dc2a67c138a038c9c284d82e039f8570dadfaa8f0273f1fa30930a4d036ee47`. Size and hash were read from the HF API before the download, then checked on the measurement-host copy and on a copy fetched back from the console after the app download (`model-hash-check.txt`). The app fetched the file through a LocalState manifest override (`manifest.override.json`, per-entry merge; not a catalogue entry). Architecture in the GGUF: `maincoder`, 32 blocks (layers 0–31), vocab 151,936.
- Settings, all three CPU arms: `gpu_layers` 0 (console request and resolved count both 0), `n_ctx` 2048, 6 threads, `n_ubatch` 512, greedy, logits stride 2. Host builds of `xllama-cli`: linux-release preset (repack on) and Release with `GGML_CPU_REPACK=OFF`, AVX2, same source head as the console package (`ded444fd`, llama.cpp `d812350`). The build tree was not retained; the dumps were hashed and are not kept in this directory.

## Prompts

[`prompts/`](prompts/) holds seventeen prompts written or cut for this declaration. None has been run on any GPU arm. `u01` to `u16` are passages of repository documents not used for earlier ADR 0010 prompt sets, with an instruction in front. Token counts on this model, stride 2:

| Prompt | Tokens | Positions | Source (passage further into the file than earlier controls, or a file not used before) |
| ------ | -----: | --------: | --- |
| `c1`   |     29 |         — | Original short prompt (last tile 1 column) |
| `u01`  |    674 |       337 | `docs/shape-gate-plan.md` |
| `u02`  |    672 |       336 | `docs/research/moe-streaming-candidates.md` |
| `u03`  |    601 |       301 | `docs/superpowers/plans/2026-11-24-moe-engineering-program.md` |
| `u04`  |    888 |       444 | `docs/gguf-gpu-decode.md` |
| `u05`  |    585 |       293 | `docs/architecture.md` |
| `u06`  |    871 |       436 | `docs/superpowers/plans/2026-11-24-moe-m0-issue-alignment.md` |
| `u07`  |    572 |       286 | `docs/superpowers/plans/2026-11-16-ssd-moe-upstream.md` |
| `u08`  |    913 |       457 | `docs/phase7-hypotheses.md` |
| `u09`  |    611 |       306 | `docs/install-release.md` |
| `u10`  |    652 |       326 | `docs/training-architecture.md` |
| `u11`  |    820 |       410 | `docs/ssd-inference-assessment.md` |
| `u12`  |    676 |       338 | `docs/adr/0010-logit-parity-reference-spread.md` |
| `u13`  |    750 |       375 | `docs/model-selection.md` |
| `u14`  |    542 |       271 | `docs/releases/v1.7.0.md` |
| `u15`  |    728 |       364 | `docs/superpowers/plans/2026-11-16-moe-streaming-all-models.md` |
| `u16`  |    739 |       370 | `docs/phase16-model-scouting.md` |

`u01`–`u16` together: **5,650** positions (declared minimum 4,000). Range 542–913 tokens. `c1` (29 tokens, 29 mod 4 = 1) and `u03` (601 = 512 + 89, 89 mod 4 = 1, second ubatch) both end in a one-column partial tile of the four-column kernel, so those two are the declared capture prompts.

SHA-256 of each prompt is in `declaration.json`.

## The floor

Three CPU executions were measured on these prompts before the declaration ([`cpu-reference-pairs.json`](cpu-reference-pairs.json)):

| Pair                          | Mean KLD | Top-1 agreement |
| ----------------------------- | -------: | --------------: |
| Console CPU vs host Release   | 3.057e-3 |           0.963 |
| Console CPU vs host no-repack | 3.088e-3 |           0.956 |
| Host Release vs host no-repack| 3.066e-3 |           0.956 |

The floor is the host build further from the console CPU arm on mean KL divergence: **host no-repack** (3.088e-3, against 3.057e-3 for Release).

## Draft declaration

[`declaration.json`](declaration.json), schema `xllama.control-declaration/v2`, model `maincoder-1b`.

| Value                    | Declared | Basis |
| ------------------------ | -------- | ----- |
| `reference.nmse_limit`   | 1e-7     | As in the single-column and c4 controls. |
| `reference.captures`     | `c1` to 31, `u03` to 5 | Last block of this GGUF is 31; `u03` to layer 5, as in the templates. |
| Floor (host arm)         | host no-repack | The host build further from the console CPU arm. |
| `kld_ratio_limit`        | 1.50     | 5 % of 0.031 over the floor, rounded down. Floor 3.088e-3 plus 1.55e-3 is a ratio of 1.502, declared as 1.50. |
| `top1_margin`            | 0.01     | As before. |
| Confidence, resamples    | 0.95, 2,000, seed 1 | As before. |
| `min_positions`          | 4000     | 5,650 positions at a stride of 2. |

The GPU arm named in the declaration is the console with `gpu_layers` 99. It has not been measured on these prompts.

The 0.031 figure is the mean KL divergence llama.cpp publishes for Q4_K_M against FP16 (for a different model; no FP16 copy of this one was measured). The same anchoring rule as Coder-3B and LFM2.5-1.2B.

## Dumps and captures

Per-position logit dumps are about 3.3 GB per arm and are not kept here. Their SHA-256 values are in [`receipt.json`](receipt.json).

Part 1 tensor captures are **not** in this receipt. A console CPU capture of `c1` was started with spec `l_out-0;mode=last;max_mb=1024;matmul_inputs=l_out-31`. The console produced 354 files, 81,012,578 bytes, and that set was still present after config restore. The fetch was interrupted at 33 files, so no complete capture hash is recorded. `u03` was not captured. The abbreviated form `matmul_inputs=l_out-31` alone matches no tensor; the working spec is the full form already used in the 2026-11-15 control-2 receipt. That is not a code change, and no NMSE is reported. Part 1 of the control is not claimed here. A later session can recapture both prompts on the CPU arm and hash them before the GPU arm.

## Console restore

After the CPU dumps, diagnostic knobs and the manifest override were removed and the pre-session LocalState files (`api.flag`, `autopilot-done.txt`, `autopilot.json`, `diffuse-model.txt`, `model.txt`, `prompt.txt`, `settings.json`) were restored and checked against the pre-backup SHA-256 (all seven identical). The provisioned GGUF was left in place so a later session does not have to fetch it again. The package was not replaced. A `tensor-capture` directory of 354 files (81,012,578 B) from the interrupted `c1` capture remains on the console; it was not fully fetched, so it has no hash in this receipt and was not deleted. The app was left stopped.

## What this does not do

- No GPU arm, no distribution or reference verdict, no Series S T3 / coding-suite / H9 numbers for Maincoder.
- No catalogue change.
- The words «validated» and «certified» are not used.
