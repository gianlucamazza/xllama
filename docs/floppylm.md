# FloppyLM — official research / experimentation objective

> **SSOT for the FloppyLM experiment.** This page records the _goal_, the
> honest limits, and the current status. It does **not** ship an engine, a
> quality score, or a tok/s number.
>
> Product training lanes A/B/C stay in
> [training-architecture.md](training-architecture.md). Module map:
> [architecture.md](architecture.md). Checklist:
> [`../ROADMAP.md`](../ROADMAP.md) (FloppyLM experimentation). Implementation
> lives on draft [PR #301](https://github.com/gianlucamazza/xllama/pull/301)
> (`feat/floppylm-training`) and is **not** on `main`.

**Currency:** 2026-09-30. Official experiment goal. Draft PR only. No merge,
no release, no E0 campaign, and no quality or speed claim.

## What it is

FloppyLM is a **Tiny GPT** research line aimed at an extreme compression
budget of the `floppy_4mb` class: a model and codec small enough that the
weights themselves are the experiment, not a catalogue chat tier.

The intended native path (Linux host CLI and Xbox UWP, **shared C++**, no
Python on device) is:

- a complete CPU training graph (every parameter, not a last-block subset)
- scalar codecs at **ternary / 2-bit / 4-bit**
- **quantization-aware training (QAT)** of those codecs
- **WSD** (warmup–stable–decay) with **isolated cooldowns**
- export to a counted **FLP2** artifact (inference-only weights, distinct
  from native resume checkpoints)

That path is the design on [PR #301](https://github.com/gianlucamazza/xllama/pull/301).
Until that PR merges, none of it is a `main` product surface.

## Why it exists

xllama already has a training pillar:

| Lane                    | What it is today                                    | Limit relative to FloppyLM                                                          |
| ----------------------- | --------------------------------------------------- | ----------------------------------------------------------------------------------- |
| **A** Host PEFT         | Python LoRA on a workstation, then merge to GGUF    | Host-side; not a full-model QAT of a Tiny GPT                                       |
| **B** Device partial FT | In-process ggml-opt on a **filtered** tensor subset | Last-block class; not every parameter; not the FloppyLM codec/QAT/WSD/FLP2 contract |
| **C** Serve             | Load merged GGUF / runtime LoRA                     | Inference only                                                                      |

FloppyLM is **not** Lane D and is **not** a rewrite of A/B/C. It is a
separate experiment whose point is to demonstrate, under the limits below:

1. **Extreme compression** at the `floppy_4mb` budget (scalar low-bit
   codecs, not a second catalogue backend).
2. **Full-model QAT** of that Tiny GPT (every parameter, random init, tied
   embeddings) rather than adapter or last-block personalization.
3. A **native console training path** — the same CPU engine on Linux and
   Xbox UWP — **without Python inside the MSIX**, going beyond Lane B's
   partial fine-tune.

The shipping product remains local inference for games and assistants
([positioning.md](positioning.md)). FloppyLM does not change that thesis.

## What we want to prove

Capabilities, not leaderboard wins. The experiment is a success only if
each claim is backed by measured evidence in the owning homes
(`bench/results/` → [benchmarks.md](benchmarks.md) when a number is
published; this file for status). Until then, the targets are:

- A Tiny GPT at the `floppy_4mb` compression class can be **trained
  natively** (QAT + WSD isolated cooldowns + FLP2 export) on the shared
  Linux/UWP CPU engine.
- That path is **honestly bounded**: ≤16M parameters, CPU only, cumulative
  experimental wall budget of 8 hours, no Python in the MSIX.
- Console evidence uses an exact Windows/MSVC CI package and live Xbox
  validation — a Linux-green draft is not a Series S result.
- Optional later: a **public comparison** against peer Tiny-GPT toys
  (below). Comparison is a future milestone, not a present claim.

Do not invent tok/s, H9 scores, loss tables, or “FloppyLM beats X”.
Functional correctness on the draft PR is not an acceleration or
scientific-quality claim.

## Known limits

These are part of the experiment, not temporary footnotes:

| Limit                             | Meaning                                                                                                       |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **≤16M parameters**               | Tiny GPT class. Not a 350M–3B catalogue model and not Lane B on those sizes.                                  |
| **CPU only**                      | Shared host + UWP CPU graph. No GPU / DirectML FloppyLM training.                                             |
| **8-hour cumulative wall budget** | Experimental compute budget across the campaign, not a per-run SLA and not a speed claim.                     |
| **No Python in the MSIX**         | Host preparation may use an independent reference; the console path must not require Python/PyTorch.          |
| **Experimental until gates pass** | Linux CTest/parity on the draft PR is not a merge, not a release, and not Xbox validation.                    |
| **Not a shipped lane**            | Lanes A/B/C in [training-architecture.md](training-architecture.md) are unchanged. FloppyLM is not on `main`. |

Non-goals (same honesty as the training pillar): no pretraining of
catalogue SLMs on Xbox, no Python/PyTorch inside the package, no quality
or speed headline before measured evidence.

## Current status (2026-09-30)

Status is **draft implementation, not shipped**.

| Fact                            | State                                                                                                                                                                                                          |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Implementation PR               | Draft [PR #301](https://github.com/gianlucamazza/xllama/pull/301) on `feat/floppylm-training`. **Open, not merged.**                                                                                           |
| Linux validation                | Described as **green** on that PR (CTest / parity). That evidence lives on the PR, not on `main`.                                                                                                              |
| Windows / MSVC CI package       | **In progress.** Not a product or measurement package yet.                                                                                                                                                     |
| Live Xbox / Series S validation | **In progress.** No console campaign result is published.                                                                                                                                                      |
| ADRs 0004 / 0005                | Exist **on the PR branch** (`docs/adr/0004-floppylm-training.md`, `docs/adr/0005-floppylm-numeric-validation.md`). They are **not on `main`** until #301 merges. Do not treat them as accepted tree ADRs here. |
| Merge / release / E0 campaign   | **Not done.** The PR body states no merge, release, E0 campaign, or scientific quality/speed claim.                                                                                                            |
| This repository (`main`)        | Documentation of the _objective_ only (this file + ROADMAP). No FloppyLM runtime.                                                                                                                              |

For implementation detail, read the PR. For accepted-on-`main` training
contracts, stay on [training-architecture.md](training-architecture.md).

## Peer context (comparison targets only)

These are **future comparison targets**, not baselines we claim to beat
and not dependencies of the xllama product:

- [Le Chaton Floppe](https://github.com/Pondsiders/le-chaton-floppe) —
  a language model that fits on a floppy (peer experiment).
- [Soul Player C64](https://github.com/gizmo64k/soulplayer-c64) —
  a small transformer that runs on a Commodore 64.
- **TinyStories-scale toys** — the well-known small-story training size
  class, not a catalogue quality bar and not a single repo.

A public comparison against those peers is an **optional later milestone**
in [`../ROADMAP.md`](../ROADMAP.md). It requires measured evidence on both
sides. Until then, naming them here is orientation only.

## See also

- Draft implementation: [PR #301](https://github.com/gianlucamazza/xllama/pull/301)
- Training pillar (shipped lanes A/B/C): [training-architecture.md](training-architecture.md)
- Training ops: [`../training/README.md`](../training/README.md)
- Platform ceilings: [uwp-constraints.md](uwp-constraints.md) §13
- Product intent: [positioning.md](positioning.md)
- Experiment checklist: [`../ROADMAP.md`](../ROADMAP.md)
