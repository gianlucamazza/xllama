# ADR 0009: Qualify an isolated host PEFT toolchain

Status: Accepted (2026-11-08), implemented and qualified in [#410](https://github.com/gianlucamazza/xllama/pull/410), tracked in [#405](https://github.com/gianlucamazza/xllama/issues/405).

## Context

The host training environment is separate from diffusion export. Its current
`training/host/requirements.txt` excludes Transformers 5 and does not pin Torch.
`training/host/run_job.sh` creates a venv with `--system-site-packages` and may
bootstrap `torch>=2.2`. A patched dependency resolution in a clean laboratory
venv does not therefore establish what the documented operator path executes.
Monitoring (#399) and refusing model-selected Python code (#401) improve the
boundary but do not remediate the dependency graph.

An isolated candidate with PEFT 0.21.2, Transformers 5.10.1, CPU Torch 2.13.0,
Accelerate 1.15.0 and Datasets 3.6.0 resolves without overriding constraints.
The unmodified trainer can train, save, reload and merge a tiny random built-in
Llama with that graph. This is API feasibility, not qualification of the
documented SmolLM2 training/merge/GGUF path. The subsequent audit rejected this
candidate: Datasets 3.6.0 and fsspec 2025.3.0 have known advisories, with patched
floors 5.0.1 and 2026.6.0 respectively. The current `datasets<4` bound excludes
that patch. Temporary bootstrap pip 25.0.1 also has findings, and the CPU Torch
wheel is skipped by the installed-graph auditor. No current training-path
exploit is claimed; this graph cannot be promoted on API smoke alone. The dated
[program evidence](../audit/excellence/execution-2026-11-07.md) owns the receipt.

A second isolated candidate explicitly changes the proposed Datasets bound to
5.0.1 and resolves fsspec 2026.6.0. The same synthetic trial passes; its installed
graph audit reports zero known findings in 58 queried distributions, with the
CPU Torch wheel still skipped. This supports a feasible migration path, not
whole-recipe or binary security acceptance. Both the failed first candidate
and the second candidate remain in the dated evidence package.

## Decision

Use a coherent, hash-recorded host training environment with an explicit
compute profile. The first qualification target is CPU, to match the current
trainer's float32 CPU path. Do not silently inherit arbitrary system Torch or
install a CUDA graph through the CPU path. A future GPU profile must state and
qualify its own wheel index, driver/runtime compatibility and dependency graph;
it is not certified by the CPU trial.

After the acceptance below, replace the loose dependency bootstrap with an
isolated environment and supported pins/lock. Record the Python version, wheel
sources/hashes, resolved transitive packages and vulnerability-scan omissions.
Keep `trust_remote_code=False`; a local directory is not an exception. Freeze
the first-party model revision, dataset and seed. Preserve the existing job
format and adapter interchange; no runtime or UWP architecture change is proposed.

## Acceptance before implementation promotion

1. Agree the CPU installation/lock ownership and GPU out-of-scope boundary;
   verify `run_job.sh`, `run_spike.sh` and conversion callers use that policy.
2. Reproduce a full isolated, hash-checked install with no constraint overrides
   or system-site fallback; run `pip check` and whole-graph security scanning.
   A skipped wheel is not a successful scan. For the official CPU wheel, separately
   verify the published wheel hash and every installed hashed payload record,
   bind its compiled Git identity to the upstream base-release tag, and query
   base-release advisories. Retain the raw local-version skip alongside this
   explicit bounded disposition; it is not whole-binary security certification.
3. Before the production trial, freeze the supported SmolLM2 revision,
   dataset/job hashes, seed and numerical/quality acceptance in a trial manifest.
   A random one-layer Llama is not an interchangeable reference.
4. Execute the documented first-party train → adapter save/reload → merge →
   GGUF/conversion → inference path. Check its artifact shapes/configuration,
   fixed-input logits/outputs and marker-quality behaviour against a pinned
   reference. Retain negative results and all exact commands and hashes.
5. Run downstream adapter/parity/device checks where the change affects them;
   distinguish host PEFT from on-device training. Update the canonical runbook,
   requirements and CI, then close #405 only with the qualified receipts.

The [dated CPU qualification receipt](../../training/host/qualification-2026-11-08.json)
records the full first-party recipe, adapter/merge parity, GGUF marker A/B and
new-artifact Series S checks. It also binds the official CPU wheel and its
installed payload to the base-release advisory disposition. Earlier synthetic
candidate receipts remain historical; they do not establish these later results.
The CPU decision is promoted through the normal protected PR path. #405 closes
only after that integration; GPU profiles remain separately unqualified.

## Alternatives and consequences

- Keep inheriting system packages: convenient for an existing GPU workstation,
  but an external environment can satisfy the loose bounds with vulnerable or
  incompatible versions. Rejected as the future default after qualification.
- Upgrade Transformers alone: violates the current declared range and leaves
  PEFT/adapter/conversion compatibility unproven. Rejected.
- Share the diffusion venv: unrelated dependency lifecycles and validation
  contracts would become coupled. Keep the environments separate.

Isolation adds installation/storage work and explicit profile maintenance. It
buys a graph that an operator can identify and reproduce. Rollback must restore
the prior coherent toolchain/recipe together; do not roll back only one package
or describe a rollback to vulnerable pins as a security remediation.

## Qualification — 2026-11-08

The CPU/Python 3.12 hash lock installs without system-site inheritance or an
unpinned fallback. The deprecated spike wrapper delegates to the same runner;
conversion uses its selected interpreter. The full canonical first-party job,
FP32 adapter/merge parity and GGUF marker A/B pass. Its converted artifacts
were served on Series S, the runtime adapter produced the marker on a fresh
greedy API request, and device files were read back and matched to the candidate.
The installed CPU wheel matches the official artifact and upstream release Git
identity; the base-release advisory query covers its declared source release.
The scanner's raw local-version skip remains visible in the receipt. These are
bounded known-advisory and compatibility checks, not general model-quality or
whole-binary security certification.
