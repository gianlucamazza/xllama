# GPU kernel reference check and spread survey, 2026-11-15

A diagnostic for #420, run after the
[first ADR 0010 control](../2026-11-15-adr0010-control/README.md) failed. It is
not a control: nothing was declared in advance and no criterion changes. Same
console, package (1.7.0.1563) and model (Coder-3B Q4_K_M) as that control;
bindings are in [`receipt.json`](receipt.json).

The control left one question open: its first-layer check found no kernel
error, but it could say nothing about layers 1 to 35, and the GPU logits were
wider than the CPU reference spread on one prompt.

## 1. Every weight product, against an exact reference

`scripts/check-matmul-reference.py` compares each captured weight `MUL_MAT`
with a float64 product of the dequantized weight and the q8_K-quantized
operand that the arm itself fed in. It needs no second arm, so it covers every
token column of every record.

| Capture                                 | Records per arm | CPU arm, largest NMSE | GPU arm, largest NMSE |
| --------------------------------------- | --------------: | --------------------: | --------------------: |
| p1-code-short, 40 tokens, all 36 layers |             252 |               2.3e-12 |               4.1e-14 |
| p4-prose-long, 713 tokens, layers 0–5   |              84 |               9.6e-13 |               3.1e-14 |

The largest per-token NMSE is 1.3e-11 on the CPU arm and 4.3e-13 on the GPU
arm. Files: [`reference-p1-all-layers.json`](reference-p1-all-layers.json),
[`reference-p4-layers-0-5.json`](reference-p4-layers-0-5.json).

Both arms compute the products they are given. The GPU arm is closer to the
exact value than the CPU arm in every weight family, which is what an integer
dot product followed by one scaling predicts against float32 accumulation.

The d3d12 backend runs `MUL_MAT` and `GET_ROWS` only; every other node of the
GPU arm runs the same CPU code as the CPU arm. With the products correct, the
GPU arm has no remaining operator that could be wrong on its own for these
captures. Layers 6 to 35 of the long prompt were not captured (about 130 MB
per layer and arm).

## 2. Why two-arm comparison stops working after a few layers

[`equal-operand-p1-all-layers.json`](equal-operand-p1-all-layers.json) is the
`control-parity.py kernel` comparison over all 36 layers of the short prompt.
The least-covered record has 29 of 40 comparable token columns in layer 0, 7
in layer 1 and 1 in layer 2; from layer 8 on no record has any. Where columns are comparable (54 records, layers 0 to 7) the two arms
agree to 1.3e-11 or better.

So the ADR's part 1, as implemented, can only test the first layers. The
reference check above does not have that limit.

## 3. Is the GPU arm further from the console CPU than another CPU build is?

Fifteen prompts: the control's four and eleven more
([`prompts/`](prompts/), documentation passages of 144 to 551 tokens). For
each, the console GPU arm and a host Release build are compared with the
console CPU arm. Two prompts end in a space that the console trims, so the
host ran a different prompt; they are excluded from the host comparison,
leaving thirteen. File: [`spread-15-prompts.json`](spread-15-prompts.json).

| Against the console CPU arm, 13 prompts | Host Release | Console GPU |
| --------------------------------------- | -----------: | ----------: |
| Maximum error, median                   |        0.631 |       0.629 |
| Maximum error, mean                     |        0.630 |       0.678 |
| Maximum error, largest                  |        0.720 |       0.987 |
| NMSE, median                            |      1.72e-3 |     1.83e-3 |
| NMSE, largest                           |      2.53e-3 |     2.63e-3 |
| Prompts with a different top token      |            2 |           4 |
| Prompts failing the frozen gate         |           13 |          13 |

The GPU arm is wider than the host on 9 of 13 prompts, by 8 % on average for
both measures. Nine of thirteen is not a significant majority (a fair coin
gives nine or more about 13 % of the time), so this sample does not establish
that the GPU arm is systematically wider. The one large value (0.987, on
p4-prose-long) is what failed the control.

## What this does and does not show

- The GPU kernels are numerically correct in all 36 layers for the short
  prompt and in the first six layers for the long one.
- The divergence between the arms is therefore the amplification of
  rounding-level differences between two correct executions, the mechanism
  already traced to int8 flips in the activation quantizer.
- Every pair, including CPU against CPU, fails the frozen gate on all thirteen
  prompts. For this model the frozen gate does not separate a correct arm
  from an incorrect one.
- It does not change the control's verdict, which stays FAIL under the
  declared criterion.
- It is one model, one console and fifteen prompts, with one deterministic
  run per arm.
