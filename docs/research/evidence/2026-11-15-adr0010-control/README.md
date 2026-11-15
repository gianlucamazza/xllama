# ADR 0010 control, Coder-3B, 2026-11-15

A GPU control for `qwen25-coder-3b` judged under
[ADR 0010](../../../adr/0010-logit-parity-reference-spread.md) with
`scripts/control-parity.py`. The declaration and prompts were committed first
(PR #452, `7e8f146`, 18:28:30 UTC); the GPU arm ran from 18:28:40 to 18:32:40
UTC. **Verdict: FAIL.** Part 2 fails on maximum error and on one top token;
part 1 finds no kernel error and fails one prompt on coverage only.

## Declaration

[`declaration.json`](declaration.json) was committed before the GPU arm was
measured on these prompts. Its values were chosen from CPU-only runs.

| Value                            | Choice  | Basis                                                                                                                                                                                                                            |
| -------------------------------- | ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `kernel.nmse_limit`              | 1e-9    | Console CPU against host CPU: every first-layer product is at or below 2.9e-13 on its comparable columns. 1e-9 is the largest operand difference seen on comparable columns (host Debug against Release, 27-token smoke prompt). |
| `kernel.min_comparable_fraction` | 0.75    | Console CPU against host CPU on these prompts: at least 465 of 512 columns (0.908) comparable for every product.                                                                                                                 |
| `spread.margin`                  | 1.22    | See below.                                                                                                                                                                                                                       |
| `spread.top_k`, `top_k_min`      | 10, 0.8 | Unchanged from the frozen gate, as the ADR requires.                                                                                                                                                                             |

The four prompts in [`prompts/`](prompts/) were written for this control and
were not used in the 2026-11-08 or 2026-11-15 diagnosis: 40, 404, 614 and 713
tokens, so two of them span two 512-token evaluations.

### Margin

The reference spread is the console CPU arm against a host Release build of
the same source. A host Debug build is a third CPU execution of that source.
Over the four prompts:

| Pair                        | Largest max error | Largest NMSE | Against the reference spread |
| --------------------------- | ----------------: | -----------: | ---------------------------- |
| Console CPU vs host Release |            0.7201 |     1.997e-3 | reference                    |
| Console CPU vs host Debug   |            0.7322 |     2.421e-3 | x1.017, x1.212               |
| Host Release vs host Debug  |            0.7880 |     2.938e-3 | x1.094, x1.471               |

The margin is the larger ratio of the pair that, like the GPU arm, is
measured against the console CPU arm: 1.212, rounded up to 1.22. The GPU arm
may therefore sit as far from the console CPU arm as another CPU build of the
same source does. The pair that does not involve the console arm is wider
(x1.471); it was not used, which makes the criterion stricter.

Every CPU-only pair above exceeds the frozen gate's limits (0.5 and 1e-3) on
these prompts, and the Debug build changes the top token of one prompt.

Disclosure: when this margin was chosen, two GPU results for this model were
already known from other prompts, x1.38 on NMSE for the 2026-11-15 diagnosis
prompt and x1.01 for a 27-token tool smoke. The rule above uses neither.

## Result

Series S, CI package 1.7.0.1563, `gpu_layers` 99 (37/37 layers offloaded)
against `gpu_layers` 0. Bindings and hashes are in
[`receipt.json`](receipt.json); raw logits are in [`logits/`](logits/) and the
verdict files in [`verdicts/`](verdicts/). On every prompt and both arms, the
logits of the capture run are byte-identical to those of the run without
capture.

### Part 2, logits against the reference spread: FAIL

| Prompt         | Tokens | Reference max error | Reference NMSE | GPU max error | GPU NMSE | Top token |
| -------------- | -----: | ------------------: | -------------: | ------------: | -------: | --------- |
| p1-code-short  |     40 |              0.6031 |       1.007e-3 |        0.5970 | 1.027e-3 | same      |
| p2-prose-it    |    404 |              0.7201 |       1.573e-3 |        0.7411 | 1.845e-3 | differs   |
| p3-code-review |    614 |              0.6314 |       1.980e-3 |        0.5420 | 1.834e-3 | same      |
| p4-prose-long  |    713 |              0.7046 |       1.997e-3 |        0.9865 | 1.952e-3 | same      |

- Maximum error: 0.9865 against a spread of 0.7201, x1.370, over the x1.22
  limit. This fails.
- NMSE: 1.952e-3 against 1.997e-3, x0.978. This is inside the limit.
- Top token: differs on p2. The console CPU arm ends on a space followed by
  two newlines (token 4710), the GPU arm on `" Ris"` (token 53363). This
  fails on its own. The host Debug build also changes the top token of p2
  against the console CPU arm, so that prompt ends on a near tie.
- Top-10 overlap is 1.00 on all four prompts.

### Part 1, first-layer products on equal operands: FAIL on coverage, no kernel error

Seven weight products per evaluation, 42 records over the four prompts.

- On the comparable columns every record is at or below 2.9e-12 NMSE, against
  a limit of 1e-9. The three products fed directly by the layer input (Q, K,
  V) have bit-identical operands on every column and agree to 1e-13 or better.
- 41 of 42 records meet the 0.75 coverage requirement. The exception is
  `blk.0.ffn_down.weight` on p1: 29 of 40 columns comparable (0.725). Its
  NMSE on those 29 columns is 2.9e-12. p1 has 40 tokens, so each column is
  2.5 points of coverage.
- The GPU arm has fewer comparable columns than the CPU-to-CPU pair that set
  the requirement: at worst 0.725 against 0.908.

So no first-layer kernel was found wrong, and the control still fails part 1
as declared, because one record could not be checked on enough columns.

### Under the frozen gate

`scripts/compare-logits.py` (console CPU against console GPU) fails all four
prompts, as it does for the CPU-only pairs above.

### What this does and does not show

- It does not show a kernel defect in layer 0.
- It shows the GPU arm further from the console CPU arm than the two other
  CPU builds are, on maximum error for one prompt, and a changed top token on
  a prompt where another CPU build changes it too.
- It says nothing about layers 1 to 35, which part 1 does not capture.
- Four prompts and one run per arm. The runs are deterministic (capture on
  and off give identical logits), so repeating them would not add information;
  more prompts would.
- The verdict is not rewritten: the criterion was declared first and stays.
