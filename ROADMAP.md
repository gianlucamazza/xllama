# xllama Roadmap

Current work only. Completed release history belongs in `CHANGELOG.md`; measured
performance belongs in `docs/benchmarks.md`.

## Current product state

- Latest semantic release: **v1.2.1.0** (training pillar + console ALL PASS).
- Shipping artifact: unified ORT GenAI + llama.cpp, with pinned patched runtime
  DLLs while upstream fixes have not reached NuGet.
- Default chat: `lfm25-350m`; balanced and quality tiers: LFM2.5-1.2B and
  LFM2-2.6B.
- Text DML remains gated by #91; CPU serves text and DirectML serves diffusion.
- Phase 1–6 delivery, console validation and demo publication are complete.

## Phase 7 — Peer-class model research

Detailed hypotheses and measured verdicts: `docs/phase7-hypotheses.md`.

- [x] H1 efficient LFM campaign and deterministic H9 suite.
- [x] H4 usable dense 3B campaign; Llama-3.2-3B is the preferred comparator.
- [ ] H2 MoE candidate, only when a supported GGUF fits below the measured
      memory ceiling.
- [ ] H3 speculative decoding spike, gated on a concrete target/draft pair and
      a predeclared throughput-quality threshold.
- [ ] H5 BitNet/low-bit survey before any runtime work.
- [ ] H6/H7 GPU or hybrid GGUF experiments only after a credible UWP backend
      path exists.

## Phase 8 — Training pillar (exploration) ✅ FROZEN complete

**Exit criteria met 2026-07-17.** Architecture + host + console evidence for
lanes A/C; device train remains rejected. SSOT:
[`docs/training-architecture.md`](docs/training-architecture.md). Ops:
[`training/README.md`](training/README.md). Platform §13:
[`docs/uwp-constraints.md`](docs/uwp-constraints.md).

- [x] C++ contracts, host PEFT, RE capability matrix, CLI.
- [x] Runtime LoRA (llama) + catalogue `lora` field; prefer `model.gguf` over adapter.
- [x] Preference capture (`rate` → `samples.jsonl`); console **PASS** on 1.2.0.546.
- [x] Console training serve + lora-rt **PASS**; hybrid operator loop = future
      **Phase 9** (optional), not Phase 8 architecture work.
- [x] Device train **rejected by default** (no ODT in shipping MSIX).

## Phase 9 — Personalization ops (optional)

- [x] Operator hybrid: `pull_console_samples.sh` + `from-console-samples` job +
      publish manifest snippet (`training/README.md`).
- [ ] UI like/dislike (not required for exploration)
- [x] GitHub Release **v1.2.1.0** (MSIX 1.2.1.550, console ALL PASS 2026-07-17)

## Upstream and vendor lifecycle

Operational details live in `docs/vendor-lifecycle-plan.md`.

- [ ] Drop PatchedGenAI after a NuGet release includes GenAI #2280.
- [ ] Land ORT ReadFile 16 MB chunk PR #29732 and drop PatchedOrt only after the
      required fixes reach NuGet.
- [ ] Keep the DML text gate until logit parity passes on the target device.
- [ ] Re-evaluate fused low-bit DirectML GEMM only after upstream capability
      changes; it is not a local application task.

## Definition of done

Any roadmap item that changes runtime behavior needs host tests, the relevant
Windows CI build, on-console validation when hardware-specific, raw evidence
under `bench/results/` when measured, and documentation in the owning SSOT.
