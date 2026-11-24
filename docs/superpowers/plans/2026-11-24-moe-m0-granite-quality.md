# M0 / E1 — Granite quality campaign (host-prepared, console NOT_RUN)

**Disposition:** **NOT_RUN** for the #273 same-quality bar.  
No PASS or FAIL is recorded here. Console H9 was not run (no Device Portal
/ console access in this slice). Host-side H9 is not supported by the
existing harness.

**Date:** 2026-11-24  
**Issues:** [#525](https://github.com/gianlucamazza/xllama/issues/525),
[#273](https://github.com/gianlucamazza/xllama/issues/273)  
**Already on `main` (not re-measured):** T3 speed+memory PASS in
[model-matrix.md](../../model-matrix.md) §I1 and
`bench/results/2026-11-24-granite40-h-tiny/`.

This file is the repo-side preparation: the frozen protocol, the peer
score that already exists, and a ready-to-run console command list. It
does not add a catalogue id, change defaults, or invent numbers.

## Frozen quality bar (do not retune after seeing results)

#273 criterion 3: decode strictly above the shipping dense peer **at the
same quality bar** (Llama-3.2-3B / Phi-4-mini class). For catalogue
candidates the product screen is **H9**, the deterministic 8-task suite.

SSOT for how to run it:

- Tasks: [`bench/eval/phase7-h9.json`](../../../bench/eval/phase7-h9.json)
  (temperature 0, seed 42, `max_tokens` per task).
- Scorer + LAN client:
  [`scripts/eval-xbox-models.sh`](../../../scripts/eval-xbox-models.sh)
  (requires `XBOX_IP` from `~/.config/xllama/xbox-env`, `curl`, `jq`).
- Operator recipe: [console-validation-runbook.md](../../console-validation-runbook.md)
  “Campaign T3 — new GGUF catalogue candidate” (H9 through the LAN API;
  bench mode never binds the port).
- `--out` must **not** be the canonical published H9 file
  (`2026-09-30-h9-rescore.jsonl`). `eval-xbox-models.sh` truncates `--out`.
- H9 is a small regression screen, not broad quality proof
  ([docs/model-matrix.md](../../model-matrix.md) H9 definition).

Peer for this candidate (named in #525 / §I1): **`llama32-3b`**. Phi-4-mini
is the other class named in #273; it is not the T3 pair already measured.

### Task ids (frozen)

From `bench/eval/phase7-h9.json`: `it_exact_capital`,
`arithmetic_multistep`, `json_extraction`, `grounded_qa`,
`constrained_summary`, `translation_fidelity`, `multi_turn_memory`,
`unknown_from_context`.

Do not edit patterns to make a candidate pass.

### Existing measured peer H9 (not a Granite result)

`llama32-3b` is **5/8** in
[`bench/results/2026-09-30-h9-rescore.jsonl`](../../../bench/results/2026-09-30-h9-rescore.jsonl)
(the published H9 column; [model-matrix.md](../../model-matrix.md) §A1).
Per-task from that file:

| task_id              | pass  |
| -------------------- | ----- |
| it_exact_capital     | true  |
| arithmetic_multistep | false |
| json_extraction      | false |
| grounded_qa          | true  |
| constrained_summary  | true  |
| translation_fidelity | true  |
| multi_turn_memory    | false |
| unknown_from_context | true  |

Granite must be scored **on the same tasks, same scorer, same greedy
seed**, then compared to a **paired** `llama32-3b` run from that same
`eval-xbox-models.sh` invocation (runbook: one paired call). Do not quote
the 2026-09-30 peer row as the pair if a fresh pair is collected; the
fresh pair is the comparison. The 5/8 figure above is only the published
baseline, not a Granite score.

## Host harness

`eval-xbox-models.sh` talks to `http://${XBOX_IP}:11434/v1/chat/completions`.
There is no host CLI path in that script and no other in-tree H9 runner
that scores `phase7-h9.json` against `xllama-cli`.

A homemade host loop would not be the product bar (different process,
no LAN admission, host tok/s are never product numbers). **Not run.**

The existing Granite host artefacts remain T1 / snapshot only:

- `bench/results/2026-11-24-granite40-h-tiny/t1-host-smoke.log`
- `bench/results/2026-11-24-granite40-h-tiny/hybrid-snapshot.log`

Those logs are not H9 and are not used as a quality verdict.

## Ready-to-run console list (NOT_RUN)

Operator machine with `xbox-env` and a current unified `xllama-appx`.
Override recipe already in
`bench/results/2026-11-24-granite40-h-tiny/README.md`. New tests in this
program run on **Series S only**.

```bash
source ~/.config/xllama/xbox-env

# 1. Sideload current unified xllama-appx if needed, launch once.
PFN=$(./scripts/deploy.sh pfn)

# 2. LocalState per-entry merge; bundled catalogue stays. Not a catalogue ship.
./scripts/deploy.sh upload-file \
  bench/results/2026-11-24-granite40-h-tiny/manifest.override.json \
  "$PFN" "" manifest.json

# 3. Select granite40-h-tiny in the app so it fetches official Q3_K_S
#    (ibm-granite/granite-4.0-h-tiny-GGUF). Device Portal bulk upload of the
#    ~3 GB GGUF is fallback only.

# 4. LAN API must be listening. Bench mode (bench.flag) does not bind the port.
./scripts/validate-api.sh spike

# 5. Paired H9. Write to a *new* jsonl (never truncate the published file).
./scripts/eval-xbox-models.sh --models granite40-h-tiny,llama32-3b \
  --out bench/results/2026-11-24-granite40-h-tiny/h9-series-s.jsonl
```

After a real run: commit the jsonl, record each task PASS/FAIL, compare
totals, and write the #273 quality verdict in §I1. Do not add a catalogue
id from this file.

Optional context/template note for the operator (not a second bar):
Granite uses `ChatFormatKind::Granite`; `llama32-3b` uses Llama-3. The
LAN API applies `chat_format_for` per model id. That difference is
expected and must be recorded, not "fixed" after seeing scores.

## Verdict table

| Gate                                | Disposition                   | Evidence                                     |
| ----------------------------------- | ----------------------------- | -------------------------------------------- |
| T3 speed (decode > peer)            | PASS (prior #527, not re-run) | §I1; `t3-series-s.csv` / `t3-llama32-3b.csv` |
| T3 memory (peak ≤ 3584 MiB)         | PASS (prior #527, not re-run) | same                                         |
| T3 TTFT not several× worse          | PASS (prior #527, not re-run) | same                                         |
| H9 same-quality bar vs `llama32-3b` | **NOT_RUN**                   | no jsonl from this slice                     |
| #273 overall reopen                 | **NOT_RUN** (blocked on H9)   | —                                            |
| Catalogue id                        | not requested                 | —                                            |

**E1 disposition: NOT_RUN.**
