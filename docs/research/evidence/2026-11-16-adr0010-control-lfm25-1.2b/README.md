# ADR 0010 control, LFM2.5-1.2B, 2026-11-16

A GPU control for `lfm25-1.2b-instruct` (QAD Q4_0), the one catalogue model
that runs on the GPU by default, under the
[2026-11-15 amendment of ADR 0010](../../../adr/0010-logit-parity-reference-spread.md#amendment-2026-11-15).
It uses the shipping single-column kernels. This directory is committed in two
steps: the declaration, prompts and CPU-only figures first, the GPU
measurements after.

## Declaration

[`declaration.json`](declaration.json) was committed before the GPU arm of
this model was measured on these prompts.

| Value                          | Choice | Basis                                                                                                                      |
| ------------------------------ | ------ | -------------------------------------------------------------------------------------------------------------------------- |
| `reference.nmse_limit`         | 1e-7   | As for Coder-3B: llama.cpp's default NMSE limit in its backend tests. The console CPU arm is at 8.2e-14 on these captures. |
| `reference.captures`           | 2      | `k1` (41 tokens) through layer 15, the last; `t01` (538 tokens) through layer 5.                                           |
| `distribution.kld_ratio_limit` | 1.95   | See below.                                                                                                                 |
| `distribution.top1_margin`     | 0.01   | One percentage point, as for Coder-3B.                                                                                     |
| `distribution.confidence`      | 0.95   | One-sided upper bound from 2,000 bootstrap resamples of the sixteen prompts, seed fixed.                                   |
| `distribution.min_positions`   | 4000   | The sixteen prompts give 5,284 positions at a stride of 2.                                                                 |

The prompts are the seventeen used for the Coder-3B control
([there](../2026-11-15-adr0010-control-2/README.md)). They have not been run
on this model's GPU arm. With this model's tokenizer they are 538 to 824
tokens, and `k1` is 41.

All weight products of this model are Q4_0, whose operands go through ggml's
q8_0 quantizer; the reference check reproduces it since #464.

### Which CPU build is the floor

Three CPU executions were measured before the declaration
([`cpu-reference-pairs.json`](cpu-reference-pairs.json)):

| Pair                           | Mean KL divergence | Top-token agreement |
| ------------------------------ | -----------------: | ------------------: |
| Console CPU vs host Release    |            1.26e-3 |               0.980 |
| Console CPU vs host no-repack  |            1.61e-3 |               0.973 |
| Host Release vs host no-repack |            1.61e-3 |               0.972 |

Unlike Coder-3B, the pairs are not alike here. The console and the host
Release build both use the repacked Q4_0 kernels and sit closer together; the
build without the repack (`GGML_CPU_REPACK=OFF`, llama.cpp's plain kernels) is
27 % further on KL divergence and 0.7 points lower on top-token agreement.

Put through the declared test in the GPU arm's place, the no-repack build
fails against the host Release floor: its top-token deficit has an upper
bound of 1.09 points against the 1-point margin. A test that fails
llama.cpp's own plain CPU kernels is not a test of the GPU arm.

So the floor for this model is the wider CPU pair: the host arm of the test
is the no-repack build. The d3d12 kernels are a kernel family of their own,
as the plain kernels are, and the question the test answers is whether the
GPU arm is further from the console CPU arm than those are. With that floor,
the host Release build passes in the GPU arm's place (ratio 0.785, upper
bound 0.804).

### The KL-divergence limit

The rule is the one used for Coder-3B: the GPU arm may add at most 5 % of the
KL divergence that llama.cpp publishes for Q4_K_M quantization itself (0.031,
for a different model and a different quantization than this one's Q4_0) on
top of the floor. With a floor of 1.61e-3 that is a ratio of 1.96, declared
as 1.95.

This limit is loose in relative terms, because this model's floor is small:
it would pass an arm nearly twice as far from the console CPU arm as the
plain CPU kernels are. The top-token margin is the tighter of the two
conditions here, and the measured ratio is reported whatever the verdict.

Disclosure: no result for this model's GPU arm under either part of the
amended criterion was known when these values were chosen. Its last-token
logits had passed the frozen gate in the 2026-11-08 campaign.

### CPU arm through part 1

The console CPU arm's captures pass the reference check: `k1`, 92 records,
worst NMSE 8.2e-14; `t01`, 68 records, worst NMSE 7.9e-14.
