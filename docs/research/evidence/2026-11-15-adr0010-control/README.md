# ADR 0010 control, Coder-3B, 2026-11-15

A GPU control for `qwen25-coder-3b` judged under
[ADR 0010](../../../adr/0010-logit-parity-reference-spread.md) with
`scripts/control-parity.py`. This directory is committed in two steps: the
declaration and prompts first, the measurements after.

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
