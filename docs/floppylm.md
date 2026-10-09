# FloppyLM — not pursued in this repository

> **SSOT for xllama's relationship to FloppyLM.** This page records the
> decision and the historical objective. It does **not** ship an engine, a
> quality score, or a tok/s number.
>
> Product training lanes A/B/C stay in
> [training-architecture.md](training-architecture.md). Module map:
> [architecture.md](architecture.md). Checklist:
> [`../ROADMAP.md`](../ROADMAP.md) (FloppyLM experimentation).

**Currency:** 2026-10-01. Closed here. No FloppyLM runtime on `main`, no
resume of [PR #301](https://github.com/gianlucamazza/xllama/pull/301), and no
quality or speed claim.

## What it is

FloppyLM is a **Tiny GPT** research line aimed at an extreme compression
budget of the `floppy_4mb` class: a model and codec small enough that the
weights themselves are the experiment, not a catalogue chat tier.

The native path drafted for this repository (Linux host CLI and Xbox UWP,
**shared C++**, no Python on device) was:

- a complete CPU training graph (every parameter, not a last-block subset)
- scalar codecs at **ternary / 2-bit / 4-bit**
- **quantization-aware training (QAT)** of those codecs
- **WSD** (warmup–stable–decay) with **isolated cooldowns**
- export to a counted **FLP2** artifact (inference-only weights, distinct
  from native resume checkpoints)

That design lived only on
[PR #301](https://github.com/gianlucamazza/xllama/pull/301)
(`feat/floppylm-training`), closed without merge on 2026-10-01. It is not a
`main` product surface, and this repository does not resume it.

Ownership is [floppylm ADR 0012](https://github.com/gianlucamazza/floppylm/blob/main/docs/adr/0012-repo-boundaries.md)
(accepted 2026-10-01): the floppylm repository owns semantics, FLP2 and
contracts; [xbox-gpu-training](https://github.com/gianlucamazza/xbox-gpu-training)
is the only native training backend. A second native backend requires a
superseding ADR there.

## Why the draft existed

xllama already has a training pillar:

| Lane                    | What it is today                                    | Limit relative to FloppyLM                                                          |
| ----------------------- | --------------------------------------------------- | ----------------------------------------------------------------------------------- |
| **A** Host PEFT         | Python LoRA on a workstation, then merge to GGUF    | Host-side; not a full-model QAT of a Tiny GPT                                       |
| **B** Device partial FT | In-process ggml-opt on a **filtered** tensor subset | Last-block class; not every parameter; not the FloppyLM codec/QAT/WSD/FLP2 contract |
| **C** Serve             | Load merged GGUF / runtime LoRA                     | Inference only                                                                      |

FloppyLM is **not** Lane D and is **not** a rewrite of A/B/C. The draft's
point was to demonstrate, under the limits below:

1. **Extreme compression** at the `floppy_4mb` budget (scalar low-bit
   codecs, not a second catalogue backend).
2. **Full-model QAT** of that Tiny GPT (every parameter, random init, tied
   embeddings) rather than adapter or last-block personalization.
3. A **native console training path** — the same CPU engine on Linux and
   Xbox UWP — **without Python inside the MSIX**, going beyond Lane B's
   partial fine-tune.

That work left this repository on 2026-10-01. The shipping product remains
local inference for games and assistants
([positioning.md](positioning.md)). Lanes A/B/C are unchanged.

## What the draft set out to prove

Capabilities, not leaderboard wins. These targets are **not** milestones of
this repository after the boundary. This file does not publish measurements.

- A Tiny GPT at the `floppy_4mb` compression class can be **trained
  natively** (QAT + WSD isolated cooldowns + FLP2 export).
- That path is **honestly bounded**: ≤16M parameters, CPU only, cumulative
  experimental wall budget of 8 hours, no Python in the MSIX.
- Console evidence uses an exact Windows/MSVC CI package and live Xbox
  validation — a Linux-green draft is not a Series S result.
- Optional later: a **public comparison** against peer Tiny-GPT toys
  (below). Comparison was a draft milestone, not a present claim, and it is
  not an xllama milestone.

Do not invent tok/s, H9 scores, loss tables, or “FloppyLM beats X”.
Functional checks on the closed PR are not an acceleration or
scientific-quality claim, and they are not on `main`.

## Known limits

These bounds belonged to the draft. They are not a live engine in this tree:

| Limit                             | Meaning                                                                                                           |
| --------------------------------- | ----------------------------------------------------------------------------------------------------------------- |
| **≤16M parameters**               | Tiny GPT class. Not a 350M–3B catalogue model and not Lane B on those sizes.                                      |
| **CPU only**                      | The rejected draft was a shared host + UWP CPU graph. No GPU / DirectML FloppyLM training here.                   |
| **8-hour cumulative wall budget** | Experimental compute budget across the campaign, not a per-run SLA and not a speed claim.                         |
| **No Python in the MSIX**         | Host preparation may use an independent reference; a console path must not require Python/PyTorch.                |
| **Closed here**                   | Linux checks on #301 are not a merge, not a release, and not Xbox validation in this repository.                  |
| **Not a shipped lane**            | Lanes A/B/C in [training-architecture.md](training-architecture.md) are unchanged. No FloppyLM runtime on `main`. |

Non-goals (same honesty as the training pillar): no pretraining of
catalogue SLMs on Xbox, no Python/PyTorch inside the package, no quality
or speed headline before measured evidence in the repository that owns the
experiment.

## Current status (2026-10-01)

Status is **closed in this repository**. xllama keeps inference and lanes
A/B/C. It does not keep a FloppyLM engine.

| Fact                     | State                                                                                                                                                                           |
| ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Decision                 | 2026-10-01. [floppylm ADR 0012](https://github.com/gianlucamazza/floppylm/blob/main/docs/adr/0012-repo-boundaries.md) accepted. This repository contains no FloppyLM logic.     |
| Implementation PR        | [PR #301](https://github.com/gianlucamazza/xllama/pull/301) on `feat/floppylm-training`. **Closed without merge.** The branch stays on the remote as history. Do not reopen it. |
| Linux validation         | The PR description called CTest / parity green. That evidence stays on the closed PR, not on `main`.                                                                            |
| Windows / MSVC and Xbox  | Stopped with the close. This repository publishes no FloppyLM console campaign.                                                                                                 |
| ADRs 0004 / 0005         | On the historical branch only (`docs/adr/0004-floppylm-training.md`, `docs/adr/0005-floppylm-numeric-validation.md`). They are not accepted ADRs of this tree.                  |
| Owning repositories      | floppylm: semantics, FLP2, contracts, scientific gates. xbox-gpu-training: the only native backend.                                                                             |
| This repository (`main`) | This page and the ROADMAP checklist. No FloppyLM runtime.                                                                                                                       |

For accepted-on-`main` training contracts, stay on
[training-architecture.md](training-architecture.md).

## Peer context (historical orientation only)

These were **comparison targets of the draft**, not baselines this
repository claims to beat and not dependencies of the xllama product:

- [Le Chaton Floppe](https://github.com/Pondsiders/le-chaton-floppe) —
  a language model that fits on a floppy (peer experiment).
- [Soul Player C64](https://github.com/gizmo64k/soulplayer-c64) —
  a small transformer that runs on a Commodore 64.
- **TinyStories-scale toys** — the well-known small-story training size
  class, not a catalogue quality bar and not a single repo.

A public comparison against those peers is not an xllama milestone. Naming
them here is orientation only.

## See also

- Closed draft, history only: [PR #301](https://github.com/gianlucamazza/xllama/pull/301)
- Boundary: [floppylm ADR 0012](https://github.com/gianlucamazza/floppylm/blob/main/docs/adr/0012-repo-boundaries.md)
- Training pillar (shipped lanes A/B/C): [training-architecture.md](training-architecture.md)
- Training ops: [`../training/README.md`](../training/README.md)
- Platform ceilings: [uwp-constraints.md](uwp-constraints.md) §13
- Product intent: [positioning.md](positioning.md)
- Checklist: [`../ROADMAP.md`](../ROADMAP.md)
