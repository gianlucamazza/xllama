# ADR 0010 control under the amendment, Coder-3B, 2026-11-15

The first control judged under the
[2026-11-15 amendment of ADR 0010](../../../adr/0010-logit-parity-reference-spread.md#amendment-2026-11-15),
with `scripts/control-parity.py reference` and `distribution`. This directory
is committed in two steps: the declaration, prompts and CPU-only figures
first, the GPU measurements after.

## Declaration

[`declaration.json`](declaration.json) was committed before the GPU arm was
measured on these prompts. Every value comes from CPU-only runs or from an
outside figure.

| Value                          | Choice | Basis                                                                                                                                            |
| ------------------------------ | ------ | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| `reference.nmse_limit`         | 1e-7   | llama.cpp's default NMSE limit in its backend tests. The console CPU arm's own products are at 8.7e-13 or better on these captures.              |
| `reference.captures`           | 2      | `k1` (40 tokens) through layer 35, so every layer is covered; `t01` (675 tokens) through layer 5, so a long prompt spanning two ubatches is too. |
| `distribution.kld_ratio_limit` | 1.25   | See below.                                                                                                                                       |
| `distribution.top1_margin`     | 0.01   | One percentage point. The three CPU pairs' top-token agreement spans 0.947 to 0.953.                                                             |
| `distribution.confidence`      | 0.95   | One-sided upper bound from 2,000 bootstrap resamples of the sixteen prompts, seed fixed.                                                         |
| `distribution.min_positions`   | 4000   | The sixteen prompts give 5,024 positions at a stride of 2.                                                                                       |

The prompts in [`prompts/`](prompts/) were written or cut for this control and
were not run on the GPU arm before: `k1` is new text, `t01` to `t16` are
passages of this repository's documentation with an instruction in front, 509
to 789 tokens each.

### The KL-divergence limit

Three CPU executions of the same source were measured on the sixteen prompts
before the declaration: the console CPU arm, a host Release build and a host
build without the weight repack (`GGML_CPU_REPACK=OFF`). Figures are in
[`cpu-reference-pairs.json`](cpu-reference-pairs.json).

| Pair                           | Mean KL divergence | Top-token agreement |
| ------------------------------ | -----------------: | ------------------: |
| Console CPU vs host Release    |            5.59e-3 |               0.947 |
| Console CPU vs host no-repack  |            5.66e-3 |               0.951 |
| Host Release vs host no-repack |            5.66e-3 |               0.953 |

The three pairs are within 1.3 % of one another. Put through the declared
test with the no-repack build in the GPU arm's place, the ratio is 1.012 with
an upper bound of 1.035. So the floor is stable, and a limit has to be chosen
on some other ground than the floor's own scatter.

The limit is anchored to what the weights' quantization already costs.
llama.cpp publishes a mean KL divergence of 0.031 for Q4_K_M against FP16 (for
a different model; no FP16 copy of this one was measured). The GPU arm may add
at most 5 % of that, 1.55e-3, on top of the CPU floor of 5.59e-3: a ratio of
1.28, declared as 1.25.

Disclosure: when this was chosen, earlier results for this model's GPU arm on
other prompts were known. On last-token logits over thirteen prompts its NMSE
was 8 % wider than a host build's on average. No KL divergence of the GPU arm
had been computed. The rule above uses neither.

### CPU arm through part 1

The console CPU arm's captures pass the reference check: `k1`, 252 records,
worst NMSE 8.7e-13; `t01`, 84 records, worst NMSE 6.5e-13.
