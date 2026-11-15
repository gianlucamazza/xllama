# ADR 0010: Judge GPU logit parity against the measured CPU reference spread

Status: Accepted (2026-11-15) by the owner, and amended the same day ([Amendment](#amendment-2026-11-15)). New controls are judged under the amendment; the first one, Coder-3B on 2026-11-15, [passed](../research/evidence/2026-11-15-adr0010-control-2/README.md). The control run before the amendment [failed](../research/evidence/2026-11-15-adr0010-control/README.md) under the original text and stays failed.

It answers step 5 of the diagnostic protocol in
[gpu-acceleration-baseline.md](../research/gpu-acceleration-baseline.md#c4-control-diagnosis)
and issue #420. Accepting it does not change the 2026-11-08 control verdict,
which stays FAIL.

Implementation: the capture option `matmul_inputs=` and
`scripts/control-parity.py` (`kernel` for part 1, `spread` for part 2),
described in
[architecture.md](../architecture.md#intermediate-tensor-capture-420). Three
points the text below leaves open are fixed there:

- The declaration is a file with no defaults: the per-record NMSE limit, the
  minimum fraction of comparable token columns, the margin, the top-k rule and
  the sha256 of every prompt. A verdict carries the file's sha256.
- Part 1 covers every `MUL_MAT` on a model weight in the first layer, on
  either backend. That is a superset of the products the GPU executes; a
  product both arms ran on the CPU compares equal.
- The reference spread "for that model" is the largest maximum error and the
  largest NMSE between the two CPU references over the whole prompt set, and
  the GPU arm is judged on its own largest values over the same set.

Equal int8 values do not make two operands equal, because each 256-value
block keeps its own float scale. On the comparable columns the result
therefore inherits the operand's remaining difference, which the `kernel`
mode reports next to the result error. The declared limit has to sit above
that and far below what a wrong kernel produces.

Known limit of part 1 as implemented, found on 2026-11-15: the two arms stop
sharing operands after a few layers (no comparable column from layer 8 on for
a 40-token prompt), so the two-arm comparison cannot test deeper layers. A
single-arm check against an exact reference
(`scripts/check-matmul-reference.py`) can, and found every GPU product correct
in all 36 layers
([evidence](../research/evidence/2026-11-15-gpu-kernel-reference-check/README.md)).
It is a diagnostic and not part of this decision.

## Context

The GPU prefill campaign (#313, #414) stopped on its control: Coder-3B on the
CPU against the original single-column GPU kernels failed the frozen logit
gate (`scripts/compare-logits.py`: maximum error at most 0.5, NMSE at most
1e-3, same top token, top-10 overlap at least 0.8).

The 2026-11-15 console run reproduced that failure byte for byte and traced it
with tensor capture
([gguf-gpu-decode.md](../gguf-gpu-decode.md#2026-11-15-control-reproduction-and-tensor-diagnosis-420),
[receipt](../research/evidence/2026-11-15-coder-control-diagnosis/receipt.json)):

- With equal inputs, the GPU Q, K and V projections of layer 0 match the CPU to
  NMSE about 1e-14.
- The first differences are in a few token positions in layer 0. Attention
  spreads them, and the per-layer NMSE settles near 1e-3 from layer 6 on.
- Two CPU builds of the same model show a similar spread. Of the three
  CPU-only pairs, two fail the 0.5 maximum-error limit (0.507 and 0.534); their
  NMSE stays between 7.3e-4 and 8.5e-4.
- The three pairs that include the GPU arm are the widest on NMSE (7.4e-4,
  1.02e-3 and 1.12e-3) and on maximum error (0.52 to 0.64).

So the gate has two problems for this model and prompt. Its maximum-error limit
rejects pairs of accepted CPU paths, and it reads a deep network's amplified
rounding as if it were a kernel defect. It still carries information: the GPU
pairs are consistently the widest.

## Decision

Split the question the single gate was asked to answer.

1. **Kernel correctness is judged where the comparison is exact.** A control
   run captures the first transformer layer with every token position on the
   CPU arm and on the GPU arm. Each `MUL_MAT` the GPU executes must match the
   CPU result for the same input to a per-record NMSE limit declared before the
   run. This extends Gate A in `docs/gguf-gpu-decode.md` from the synthetic
   selftest to the real model and prompt. A failure here is a kernel finding.
2. **End-to-end agreement is judged against a measured reference spread.**
   Before a campaign, the same prompt set is run on two accepted CPU
   references: the console CPU arm and a host build of the same source with
   the same kernel family. Their pairwise maximum error and NMSE are the
   reference spread for that model. The GPU arm passes when its maximum error
   and NMSE against the console CPU arm do not exceed the spread by more than
   a margin declared before the GPU arm is measured. Same top token and top-10
   overlap at least 0.8 stay as they are.
3. **The margin and the prompt set are frozen before measurement.** A new
   campaign uses prompts that were not used in this diagnosis. With the
   2026-11-15 data the GPU arm is 1.08 times the console-to-host spread on
   maximum error and 1.38 times on NMSE, so any margin below 1.38 would still
   fail that prompt. This ADR does not pick the margin to make it pass.
4. **Recorded verdicts are not rewritten.** The 2026-11-08 and 2026-11-15
   results stay FAIL under the criterion they were measured with.

`scripts/compare-logits.py` keeps its current defaults. The spread comparison
is a separate, explicit mode with its own tests, added in the PR that
implements this ADR.

## Consequences

- #313 and #414 can resume only after a fresh control passes both parts.
- A campaign costs two more runs per model: the second CPU reference and the
  layer-0 full capture. The partial two-layer capture of this diagnosis was
  97 MB for a 958-token prompt.
- A real kernel defect is caught by part 1 at the operator, instead of being
  inferred from a number at the end of 36 layers.
- A GPU arm that is consistently wider than the CPU spread still fails part 2.
  The first differences are now traced to single int8 rounding flips in the
  activation quantizer
  ([analysis](../research/evidence/2026-11-15-coder-control-diagnosis/quantization-flip-analysis.json)),
  a mechanism any two executions share. That explains where the gap starts,
  not how wide it may be, so part 2 still decides.
- Part 1 must compare kernels on the same quantized input, or on inputs whose
  int8 values are checked to be equal. Comparing outputs of bit-different float
  inputs would report the quantizer's flips as kernel error.
- Models whose reference spread already passes the absolute limits (the LFM
  pairs do) are unaffected in practice.

## Alternatives

- **Keep the gate and drop Coder-3B as the control.** No method change, but
  the 3B class is where the prefill work matters most, and the same effect
  would return with the next deep model.
- **Raise the absolute limits.** Simple, but a number chosen after seeing the
  data, with no link to what the CPU paths actually do.
- **Judge top-token and top-k agreement only.** Tolerant of amplified
  rounding, but blind to a kernel that is wrong in a way that happens not to
  move the top tokens on the test prompt.

## Amendment, 2026-11-15

Decided by the owner after the first control and its diagnosis. It replaces
parts 1 and 2 of the decision for controls declared from now on. Parts 3 and 4
stand: the criterion and the prompts are frozen before the GPU arm is
measured, and recorded verdicts are not rewritten.

### Why

- The first control failed part 2 on the largest single-logit error over four
  prompts (1.37 times the reference spread against a declared 1.22) and
  part 1 on coverage alone.
- A check against an exact reference then found every GPU weight product
  correct in all 36 layers, closer to exact than the CPU's
  ([evidence](../research/evidence/2026-11-15-gpu-kernel-reference-check/README.md)).
  The criterion had failed an arm with no defect.
- Part 1 as written cannot test past the first layers: the two arms stop
  sharing operands (no comparable column from layer 8 on).
- Part 2 as written rests on a maximum over a few prompts. Two CPU builds
  move that statistic by tens of percent between themselves, and over
  thirteen prompts the GPU arm was wider than a CPU build on nine, which that
  sample cannot tell from chance.
- Other projects judge the same question per operator against a
  higher-precision reference (llama.cpp's backend tests at 1e-7 NMSE by
  default, TOSA and WebNN with exact integer results), and end to end with
  distribution statistics over many tokens (llama.cpp's KL-divergence tool),
  not with a maximum over a handful of prompts. Sources are listed in #420.

### Decision

1. **Kernel correctness: every weight product against an exact reference.**
   For each declared capture, every weight `MUL_MAT` the arm ran must be
   within a declared NMSE limit of a float64 product of the dequantized
   weight and the arm's own q8_K-quantized operand. One arm is enough, so
   every token column and every captured layer is judged. The declaration
   names the prompts to capture and the last layer each capture must reach;
   at least one capture covers every layer. The GPU arm must pass. The CPU
   arm is run through the same check and reported.
2. **End-to-end agreement: a distribution over many positions, against the
   CPU noise floor.** For a declared prompt set, the logits at many prompt
   positions are dumped on the console CPU arm, the console GPU arm and a
   host CPU build. Per position: the KL divergence from the console CPU arm,
   and whether the top token agrees. The GPU arm passes when, at a declared
   confidence from a bootstrap over prompts,
   - its mean KL divergence is at most a declared multiple of the host CPU
     build's, and
   - its top-token agreement is not below the host CPU build's by more than a
     declared margin,

   over at least a declared number of positions.

3. **Maximum error, NMSE and per-prompt figures are reported, not judged.**
4. **`scripts/compare-logits.py` is unchanged.** It remains the gate for
   publishing ORT model assets, where it has not shown this problem.

### What is and is not claimed

- Passing means: the GPU arm computes the products it is given, and its
  output distribution is no further from the console CPU arm than the
  declared multiple of what another CPU build shows.
- It does not mean the two arms produce the same tokens. Two CPU builds do
  not either.
- The choice of KL divergence, the multiple and the number of positions are
  this project's. No external source prescribes them for a new backend.

### Implementation

`scripts/control-parity.py reference` and `distribution`, with declaration
schema `xllama.control-declaration/v2`; the per-position dump is
`--dump-logits-stride` (CLI) or `logits_stride.txt` (console). The `kernel`
and `spread` modes stay, so the 2026-11-15 verdict can be reproduced.
