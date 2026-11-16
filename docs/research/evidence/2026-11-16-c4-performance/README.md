# #313 paired performance campaign of the four-column kernel, 2026-11-16

The paired campaign preregistered for #313 on 2026-11-07
([protocol](../../../gguf-gpu-decode.md#preregistered-313-experiment--2026-11-07)),
for the four-column prefill kernel of #414. [`manifest.json`](manifest.json)
freezes the package, models, prompts, settings, cells and analysis. It was
committed before any campaign cell ran. This directory is committed in two
steps: the manifest and the prerequisite gates first, the measurements after.

## Prerequisite gates on the exact package (1.7.0.1595)

| Gate                        | Result                                                                                                                       |
| --------------------------- | ---------------------------------------------------------------------------------------------------------------------------- |
| Gate A, UI-process selftest | Knob absent: 14/14 cases. Knob = 4: 50/50 cases, 252 four-column dispatches. All decode cases ≥ 100 GB/s.                    |
| Logit control (ADR 0010)    | PASS on both models, output byte-identical to the single-column kernel ([#468](../2026-11-16-adr0010-control-c4/README.md)). |
| H9, LAN API                 | LFM2.5-1.2B 6/8 with and without the kernel, 8/8 replies identical; the kernel ran (712 dispatches).                         |

Coder-3B's H9 is also 7/8 with and without the kernel, with identical
replies, but it does not test the kernel: the API loads that research model
on the CPU (0/37 layers), because it has no catalogue GPU setting. Its
campaign cells run it on the GPU through the bench knob.

## Amendment of the 2026-11-07 protocol, before measurement

The owner approved, on 2026-11-16, the plan that proposed these changes.

1. **"Logit parity" means the amended ADR 0010 control.** The protocol was
   written before ADR 0010 and named the frozen gate, which fails CPU against
   CPU on Coder-3B. Both models pass the amended control.
2. **No power or thermal instrumentation, and a sampled peak.** No wattmeter
   is available. The memory gate uses the app's sampled `peak_ws_mb`: the
   upper bound of the paired difference must be at most 32 MiB. Each arm runs
   a dropped warmup; there is no thermal sidecar.
3. **Prompt sizes are the existing bench prompts.** `short-32`,
   `standard-512` and `long-1k`. The device tokens are in each row's
   `n_prompt_tok`: about 300 for `standard-512` and about 1,000 for
   `long-1k`, so the "about 512" planning target is met at about 300.
4. **Decode is a fixed 128 tokens** (`--ignore-eog`), so both arms decode the
   same length.

The margins, pair count, ordering, bootstrap and per-cell rule are the
preregistered ones, unchanged.

## Disclosure

Before this manifest, two kinds of speed information about the kernel were
seen:

- The diagnostic counters in the logs of the control runs (prefill wall time
  of single runs with many logits rows requested).
- A two-pair smoke of the runner on LFM2.5-1.2B with `spec-chat-open.txt`, a
  prompt the campaign does not use (116 tokens). Its graph wall time was
  about 1.70 s with the kernel against 1.61 s without.

Neither changes a margin: the margins were frozen on 2026-11-07.
