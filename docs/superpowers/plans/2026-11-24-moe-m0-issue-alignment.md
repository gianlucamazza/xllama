# M0 / E0 — Proposed issue alignment after #527

**Status:** proposed GitHub text only. This file does **not** edit, comment
on, or close GitHub issues. Apply (or reject) the text below in a later
maintainer pass.

**Date:** 2026-11-24  
**Program:** [2026-11-24-moe-engineering-program.md](2026-11-24-moe-engineering-program.md)
(accepted for M0 only)  
**Do not create a new umbrella.** Keep GPU work under #421/#481 and
cache/loader work under #479/#480. Coordinate with #395.

No IPs, hostnames, device serials, or credentials.

## Sources used (already on `main`)

| Item                                                       | State                                   | Evidence                                                                                        |
| ---------------------------------------------------------- | --------------------------------------- | ----------------------------------------------------------------------------------------------- |
| [#527](https://github.com/gianlucamazza/xllama/pull/527)   | Merged 2026-11-24                       | Granite 4.x renderer, T1 host smoke, #512 snapshot on `granitehybrid`, Series S T3 speed+memory |
| [#525](https://github.com/gianlucamazza/xllama/issues/525) | Open; body still lists T1/T3 unchecked  | Gate issue for Granite-4.0-H-Tiny                                                               |
| [#524](https://github.com/gianlucamazza/xllama/issues/524) | Open umbrella                           | MoE architecture extensions; 0 comments                                                         |
| [#273](https://github.com/gianlucamazza/xllama/issues/273) | Open reopen gate                        | Quality bar still the #273 text; last comment 2026-11-16                                        |
| [#395](https://github.com/gianlucamazza/xllama/issues/395) | Open excellence tracker                 | Last body checkpoint 2026-11-22; last comment 2026-11-18                                        |
| [#522](https://github.com/gianlucamazza/xllama/issues/522) | Open; body says pin bump is not planned | Owner has since approved a pin bump **after** 1.8.1, first a full-catalogue regression PR       |
| v1.8.1.0                                                   | Published                               | https://github.com/gianlucamazza/xllama/releases/tag/v1.8.1.0 package **1.8.1.1764**            |
| Model-matrix §I1                                           | Speed+memory PASS; quality NOT MEASURED | [docs/model-matrix.md](../../model-matrix.md) §I1                                               |

T3 numbers already in §I1 (not re-measured here): Granite-4.0-H-Tiny Q3_K_S
decode 21.73 tok/s, TTFT 5442.3 ms, peak 3100 MiB vs `llama32-3b` Q3_K_S
14.06 tok/s / 7950.9 ms / 1832 MiB. Every Granite run `peak_ws_mb` ≤ 3584.

---

## Proposed: comment on #524 (do not replace the umbrella)

Do not open a second MoE architecture umbrella. Post this as a checkpoint
comment, and tick only the Granite _speed+memory_ work in the candidates
list if the maintainer wants a checkbox split. Leave the Granite candidate
open until #273 quality is measured.

```markdown
## Checkpoint 2026-11-24 — #527 results, no new umbrella

[#527](https://github.com/gianlucamazza/xllama/pull/527) merged. This issue
stays the architecture umbrella. The evidence-gated program is
`docs/superpowers/plans/2026-11-24-moe-engineering-program.md` (accepted for
**M0 only**). GPU work stays under #421/#481; cache/loader stays under
#479/#480. Coordinate with #395. Do not start a parallel extraction.

### What #527 changed in this table

- `granitehybrid` (Granite-4.0-H-Tiny, #525): T1 host smoke, #512 snapshot
  on this Mamba2 checkpoint, and Series S T3 speed+memory are on `main`
  (`bench/results/2026-11-24-granite40-h-tiny/`, model-matrix §I1).
  Decode 21.73 tok/s > dense peer `llama32-3b` 14.06; every run
  `peak_ws_mb` 3100 ≤ 3584; TTFT not several× worse. **#273 same-quality
  bar is NOT MEASURED.** Overall reopen still pending quality. Not a
  catalogue id. Routed experts, Mamba2 layers and Q3_K stay on the CPU.
- Hybrid prefix snapshots (#512) have now been exercised on
  `granitehybrid`, not only LFM2-shaped caches. The #527 follow-up
  (`can_shift=1` on this Mamba2 hybrid) is M0/E2, not a new issue.
- `MUL_MAT_ID` / Q3_K D3D12 still belong under #421/#481. M1 GPU work is
  **not** authorized from this checkpoint: it waits on Granite quality,
  a new owner approval, and the #522 pin-bump regression (or an explicit
  deferral of that bump).

### OLMoE trigger (do not start F0–F5 in M0)

The body said to reopen `olmoe` only if Granite-4.0-H-Tiny shows a decode
win over the dense peer. That **speed** trigger is now met by the §I1 T3
row. That is not permission to spend device budget. Park OLMoE until
Granite quality has a #273 verdict and the owner explicitly asks for F0–F5
under this umbrella.

### #522

A llama.cpp pin bump is owner-approved to happen **after 1.8.1**, first as
a full-catalogue regression PR. That is not an automatic bump and not a
new MoE umbrella. Sequence it before any M1 GPU work because it changes
the measurement baseline. Blocked candidates on #522 stay on #522.
```

Optional body edit on #524 (same facts, if the maintainer prefers in-place
text over a comment):

- Classification row `granitehybrid`: change "partly covered" to "T1 + T3
  speed+memory on `main` via #527; quality pending #525/#273".
- "Hybrid prefix snapshots … only been exercised on LFM2-shaped caches":
  replace with "exercised on LFM2-shaped caches and on `granitehybrid`
  (#527); Mamba2 `can_shift=1` is an open E2 hypothesis, not a confirmed
  bug".
- Candidates checkbox for Granite: leave unchecked, or split into
  `[x] T3 speed+memory (#527)` / `[ ] #273 quality`.
- "Not pursued now / olmoe": add that the decode-win trigger is met and
  F0–F5 remain parked.

---

## Proposed: new checkpoint on #395 (prepend to the body)

#395 is the excellence tracker, not a second MoE umbrella. Prepend a
dated checkpoint; do not rewrite older checkpoints.

```markdown
## Current checkpoint — 2026-11-24

- **v1.8.1.0 is published** (package 1.8.1.1764):
  https://github.com/gianlucamazza/xllama/releases/tag/v1.8.1.0
  Receipts #529 / #530.
- [#527](https://github.com/gianlucamazza/xllama/pull/527) merged: Granite
  4.x renderer, Granite-4.0-H-Tiny host smoke and Series S T3 speed+memory
  PASS vs `llama32-3b`. #273 quality **NOT MEASURED**. No catalogue id.
- MoE architecture work stays under #524/#525/#526. The evidence-gated
  program (`docs/superpowers/plans/2026-11-24-moe-engineering-program.md`)
  is accepted for **M0 only** (E0–E2). It is not a new umbrella and does
  not replace #395, #421, #422, or #479–#481.
- #522 (llama.cpp pin bump) is owner-approved to happen after 1.8.1, first
  a full-catalogue regression PR. Not an automatic bump. Sequence it
  before M1 GPU work.
- #421 and #422 remain open on their existing Series X / matched-matrix
  gates. This program's new tests run on Series S only. Any Series X
  figure is attributed exactly as "Series X figures measured by Joshua
  Henne (Eidon Labs Inc.), xllama contributor"; never "validated" or
  "certified".
- #479 still awaits an agreed cache candidate. No parallel extraction.
- M1 GPU, M2 contributor SSD, and M3–M5 are **not** authorized from this
  checkpoint.

The 2026-11-22 checkpoint below stays as history.
```

---

## Proposed: checkbox / body updates on #525

Tick only what #527 actually landed. Do not tick a #273 verdict.

Funnel:

- [x] **F0 dedup** — recorded in model-matrix §I1 (2026-11-24 desk note).
- [x] **F1 size** — file sizes in the issue body; **peak measured** on
      Series S T3 (`peak_ws_mb` 3100 ≤ 3584).
- [x] **F2 architecture** — `granitehybrid` on pin `d812350` (unchanged).
- [x] **F3 chat template** — `ChatFormatKind::Granite` in #527, shared
      with #521.
- [x] **F4 licence** — Apache 2.0 (unchanged).
- [x] **F5 surface** — chat (unchanged).

Ladder:

- [x] **T1 host smoke** — `bench/results/2026-11-24-granite40-h-tiny/t1-host-smoke.log`.
- [x] Host test: #512 snapshot restore — `hybrid-snapshot.log`; CI skips
      without `XLLAMA_GRANITE_H_TINY`.
- [x] **T3 Series S** speed+memory — `t3-series-s.csv` / `t3-llama32-3b.csv`;
      §I1 row written.
- [ ] **T4 / #273 same-quality bar** — **NOT MEASURED**. Use the existing
      H9 protocol (`scripts/eval-xbox-models.sh` + `bench/eval/phase7-h9.json`)
      vs `llama32-3b`. Ready-to-run console list:
      [2026-11-24-moe-m0-granite-quality.md](2026-11-24-moe-m0-granite-quality.md).
- [ ] Record an overall #273 reopen verdict in §I1 (PASS or FAIL). Do not
      add a catalogue id from speed+memory alone.

Follow-up (not a new architecture issue): session logs `can_shift=1` on
this Mamba2 hybrid. Investigation:
[2026-11-24-moe-m0-e2-can-shift.md](2026-11-24-moe-m0-e2-can-shift.md).

---

## Proposed: comment on #273 (do not close, do not reopen)

```markdown
## Desk note, 2026-11-24: Granite-4.0-H-Tiny T3 speed+memory, quality still open

[#527](https://github.com/gianlucamazza/xllama/pull/527) / #525 measured
Granite-4.0-H-Tiny Q3_K_S on Series S against `llama32-3b` Q3_K_S
(model-matrix §I1, `bench/results/2026-11-24-granite40-h-tiny/`):

- Official GGUF on the current pin — criterion 1 holds.
- Peak working set 3100 MiB ≤ 3584 — criterion 2 holds for this T3.
- Decode 21.73 > 14.06 — the **speed** half of criterion 3 holds.
- TTFT 5442.3 ms vs 7950.9 ms — criterion 4 holds for this T3.
- Criterion 3's **same quality bar** is **NOT MEASURED**. H9 vs
  `llama32-3b` (peer recorded 5/8 in `2026-09-30-h9-rescore.jsonl`) is
  the existing product screen. No reopen and no catalogue id until that
  measurement exists.

The resident gate text in this issue is unchanged. Streamed #479–#481
work is a different path and does not reopen H2.
```

---

## Proposed: comment on #522 (do not treat the issue as the pin-bump PR)

```markdown
## 2026-11-24 — pin bump approved after 1.8.1; still not automatic

v1.8.1.0 is published (package 1.8.1.1764). The owner approved a llama.cpp
pin bump **after** that release. First step remains a full-catalogue
regression PR (UWP/AppContainer patches included). This issue still does
not _be_ that PR.

The MoE engineering program
(`docs/superpowers/plans/2026-11-24-moe-engineering-program.md`) sequences
#522 **before any M1 GPU work** because a new pin changes the measurement
baseline. No automatic bump. Blocked candidates on this issue still wait
for the pin to land, then F0–F5 / T0–T5 as written.
```

---

## What not to do

- Do not close #524, #525, #273, #395, or #522 from M0.
- Do not file a new MoE umbrella or a duplicate GPU issue.
- Do not start OLMoE, AFMoE, M1 kernels, or a pin bump from this file.
- Do not paste Device Portal addresses, credentials, or host paths into
  issue text.
