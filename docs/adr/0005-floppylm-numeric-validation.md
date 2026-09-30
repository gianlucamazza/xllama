# ADR 0005: Separate optimizer correctness from backend rounding

Status: Accepted (2026-09-30), explicitly approved by the owner.
Supersedes the end-to-end optimizer-update acceptance criterion in ADR 0004.

## Context

ADR 0004 requires an absolute tolerance of 1e-5 and relative tolerance of 1e-3 for
updates when both backends compute their own gradients. The 36 small configurations
pass. A larger fixture (d=80, ff=267, two layers, SwiGLU, 2-bit, row16, QK-norm,
seed 7, batch 2, length 8) fails that update gate despite passing forward and gradient
gates. At `blocks.0.qkv.weight[8999]`, the gradients are 2.631563855715058e-10
(PyTorch) and -1.3154931366443634e-08 (ggml). AdamW with epsilon 1e-8 amplifies
that cancellation-scale difference into a 0.0003085266798734665 update difference.

This is a failing gate, not proof of compatibility. Changing epsilon, dropping QK-norm,
ignoring the coordinate, or choosing a different fixture would change the experiment
or conceal the failure. None is allowed.

## Decision

Keep the independent frozen reference and all configurations. Validate four separate
contracts:

1. Forward and gradients keep ADR 0004 tolerances unchanged.
2. Feed identical externally supplied gradients to native AdamW and PyTorch AdamW.
   Compare clipping, moments and updates at ADR 0004 tolerances, including near-zero,
   signed, zero, large, and multiple-step gradients. This isolates optimizer correctness.
3. End-to-end update differences must be bounded by an independently calculated
   sensitivity envelope from the two measured gradients, clipping factors, optimizer
   moments and epsilon. Compare native updates against the reference optimizer applied
   to native gradients as an additional mandatory gate. Publish the maximum coordinate
   difference and the near-zero-gradient cases, rather than dropping them.
4. Keep byte-exact FLP2 checks for identical master weights. Require continuous versus
   resumed native training to produce identical state and artifacts, plus independent
   sliding evaluation. No speed or scientific-quality claim follows from numeric tests.

All adversarial fixtures remain in the test corpus. No compiler-dependent fast-math,
perturbed gradients, increased optimizer epsilon, or relaxed blanket tolerance is added.

## Consequences

This revises the end-to-end update acceptance criterion in ADR 0004. The component gates and sensitivity envelope
are mandatory before Xbox validation and an extension-complete claim. Native training
remains experimental until all platform gates pass.

## Alternatives

- Preserve the original gate: investigate whether ggml can reproduce the relevant
  PyTorch arithmetic closely enough across all supported shapes and platforms. There
  is no demonstrated fix yet; passing the smaller fixtures is insufficient.
- Emulate the entire PyTorch CPU math implementation on Xbox: creates a large dependency
  maintenance obligation and does not constitute the planned minimal ggml extension.
- Change optimizer hyperparameters or remove configurations: rejected because it changes
  the E0 experiment.
