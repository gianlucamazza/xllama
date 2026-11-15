# ADR 0010: Judge GPU logit parity against the measured CPU reference spread

Status: Proposed

Proposed for owner acceptance. It answers step 5 of the diagnostic protocol in
[gpu-acceleration-baseline.md](../research/gpu-acceleration-baseline.md#c4-control-diagnosis)
and issue #420. Nothing here takes effect until it is accepted. The 2026-11-08
control verdict stays FAIL either way.

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
