# #313 paired performance campaign of the four-column kernel, 2026-11-16

The paired campaign preregistered for #313 on 2026-11-07
([protocol](../../../gguf-gpu-decode.md#preregistered-313-experiment--2026-11-07)),
for the four-column prefill kernel of #414. [`manifest.json`](manifest.json)
freezes the package, models, prompts, settings, cells and analysis. It was
committed before any campaign cell ran (PR #469, merged 03:08:12 UTC); the
campaign ran from 03:08:17 to 06:15:24 UTC. **Verdict: FAIL.** The kernel
passes every cell on Coder-3B and fails every cell on LFM2.5-1.2B, and the
preregistered rule requires every cell.

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

## Result

240 arms, none failed or replaced; every arm's log showed the intended kernel
(four-column dispatches in every B arm, none in any A arm). Raw rows,
ledgers, per-arm dispatch lines and per-cell verdicts are in
[`bench/results/2026-11-16-c4-paired/cells/`](../../../../bench/results/2026-11-16-c4-paired/cells/),
the summary in [`result.json`](result.json).

Median paired ratios, experiment over baseline, with the 95 % bootstrap
interval in brackets. TTFT and memory: lower is better.

| Cell                  | Device tokens | Prefill              | Decode               | TTFT                 | Peak delta | Verdict |
| --------------------- | ------------: | -------------------- | -------------------- | -------------------- | ---------- | ------- |
| LFM2.5-1.2B, short    |            60 | not gated            | 0.999 [0.997, 1.004] | 1.176 [1.173, 1.177] | +1.5 MiB   | FAIL    |
| LFM2.5-1.2B, standard |           298 | 0.842 [0.840, 0.844] | 1.003 [0.997, 1.007] | 1.187 [1.185, 1.190] | +2.0 MiB   | FAIL    |
| LFM2.5-1.2B, long     |         1,000 | 0.849 [0.848, 0.850] | 1.003 [1.000, 1.007] | 1.178 [1.177, 1.179] | +2.0 MiB   | FAIL    |
| Coder-3B, short       |            59 | not gated            | 0.997 [0.994, 1.005] | 0.683 [0.675, 0.693] | +5.0 MiB   | PASS    |
| Coder-3B, standard    |           292 | 1.501 [1.499, 1.510] | 1.000 [0.993, 1.002] | 0.666 [0.663, 0.667] | +5.5 MiB   | PASS    |
| Coder-3B, long        |           948 | 1.468 [1.467, 1.471] | 0.998 [0.993, 1.005] | 0.681 [0.680, 0.682] | +6.0 MiB   | PASS    |

Margins: prefill at least 1.10, decode at least 0.97, TTFT at most 1.05,
peak delta at most 32 MiB, each on the interval bound.

In absolute terms (medians): on the long prompt LFM2.5-1.2B prefills at 250
tok/s without the kernel and 212 with it, and its time to first token goes
from 4.0 s to 4.7 s. Coder-3B prefills at 74 tok/s without and 109 with it,
and its time to first token goes from 12.8 s to 8.7 s. Decode does not move
on either model.

### Reading

- The kernel does what it was built for on Coder-3B: about 1.5 times the
  prefill and a third off the time to first token, with decode and memory
  unchanged.
- It makes LFM2.5-1.2B slower: about 15 % less prefill and 18 % more time to
  first token. That is the model that ships on the GPU, so the kernel has no
  product benefit as built.
- All of LFM2.5-1.2B's weight products are Q4_0; Coder-3B's are Q4_K and
  Q6_K. A plausible reading is that the four-column Q4_0 shader is slower
  than its single-column version while the K-quant shaders are faster. That
  was not tested: no per-shader timing was taken.
- A variant that enables four columns only for K-quant weights would be a new
  experiment. These data cannot accept it, because choosing the cells after
  seeing them is what the per-cell rule forbids.
- One console, one package, sampled peak memory, no power measurement.

All twelve CSVs pass `scripts/validate-benchmark.py`. The CI schema check
covers nested campaign directories since 2026-11-16.
