# xllama Roadmap

Current work only. Completed release history belongs in `CHANGELOG.md`; measured
performance belongs in `docs/benchmarks.md`.

## Current product state

- Latest semantic release: **v1.2.0.0**.
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

## Phase 8 — Training pillar (exploration)

SSOT: [`docs/training-architecture.md`](docs/training-architecture.md)
(RE inventory + capability matrix). Ops: [`training/README.md`](training/README.md).
Platform: [`docs/uwp-constraints.md`](docs/uwp-constraints.md) §13.

- [x] C++ training contracts, job validation and host/device gate.
- [x] Host PEFT LoRA job runner, merge pipeline and A/B evaluation (marker PASS).
- [x] CLI `--validate-train-job` / `--train-job` / `--training-capabilities`.
- [x] RE capability matrix + `scripts/re-training-stack.sh` (inference-only NuGet,
      GenAI OgaLoadAdapter + DML adapter block, llama adapter API, ODT research,
      llama-finetune rejected).
- [x] Runtime LoRA load (llama) — `SessionParams.lora_path` / CLI `--lora`
      applies `llama_set_adapters_lora` (Lane C serve without full merge).
- [ ] Preference capture on console → host retrain (JSONL schema in SSOT).
- [ ] Catalogue publication contract for fine-tuned models/adapters.
- [ ] Device train research (ORT ODT) only with measured memory plan; default
      remains `device=device` rejected.

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
