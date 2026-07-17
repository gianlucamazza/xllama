# xllama Roadmap

Current work only. Completed release history belongs in `CHANGELOG.md`; measured
performance belongs in `docs/benchmarks.md`.

## Current product state

- Latest published release: **v1.2.1.0** (training pillar + console ALL PASS).
  **v1.3.0.0** (expanded product UI + experimental Lane B) is cut in
  `CHANGELOG.md` (2026-07-17) and drafted as a GitHub Release; publication
  pending.
- Shipping artifact: unified ORT GenAI + llama.cpp, with pinned patched runtime
  DLLs while upstream fixes have not reached NuGet.
- Default chat: `lfm25-350m`; balanced and quality tiers: LFM2.5-1.2B and
  LFM2-2.6B.
- Text DML remains gated by #91; CPU serves text and DirectML serves diffusion.
- Phases 1–9 (delivery, console validation, demo, training pillar,
  personalization ops) are complete; Phase 10 (Lane B) is in progress.

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
lanes A/C; full device fine-tuning and ORT ODT remain rejected. The later,
bounded ggml-opt Lane B work is tracked separately in Phase 10. SSOT:
[`docs/training-architecture.md`](docs/training-architecture.md). Ops:
[`training/README.md`](training/README.md). Platform §13:
[`docs/uwp-constraints.md`](docs/uwp-constraints.md).

- [x] C++ contracts, host PEFT, RE capability matrix, CLI.
- [x] Runtime LoRA (llama) + catalogue `lora` field; prefer `model.gguf` over adapter.
- [x] Preference capture (`rate` → `samples.jsonl`); console **PASS** on 1.2.0.546.
- [x] Console training serve + lora-rt **PASS**; personalization operations are
      isolated in Phase 9, not in the frozen Phase 8 architecture.
- [x] Full device fine-tune and ORT ODT **rejected by default**.

## Phase 9 — Personalization ops ✅ complete

- [x] Operator hybrid: `pull_console_samples.sh` + `from-console-samples` job +
      publish manifest snippet (`training/README.md`).
- [x] Per-response Like/Dislike/Correct UI with persisted, single-write feedback.
- [x] GitHub Release **v1.2.1.0** (MSIX 1.2.1.550, console ALL PASS 2026-07-17)

## Phase 10 — Lane B: on-device training (in progress)

Training runs **fully on the console** via the in-process ggml-opt engine
(`llama_opt_init`/`llama_opt_epoch`, name-filtered partial fine-tune). Design +
pin constraints: [`docs/training-architecture.md`](docs/training-architecture.md)
§10.

- [x] `partial_ft` method + device-lane job schema (`param_filter`, `n_ctx_train`,
      `epochs`, `checkpoint_every`); `device=device` accepted on
      `XLLAMA_DEVICE_TRAIN` builds.
- [x] Engine `src/bridge/device_train.cpp`: prepare (selective f32 upcast GGUF)
      → train (llama_opt, last-block filter) → export (merged GGUF) → evaluate
      (in-process marker A/B); result.json + checkpoints.
- [ ] Host engine marker job **PASS** (same code path as console; validation in
      progress, do not treat implementation-only coverage as evidence).
- [x] UWP glue: `train.flag` headless mode, LocalState-relative job paths,
      `training/result.done`; `device-train` mode in
      `validate-console-training.sh`.
- [ ] Console `device-train` **PASS** on a llamacpp/unified MSIX (RSS < 3 GB,
      wall time recorded) → flip `DeviceGgmlPartialFt` to `available`.
- [ ] Widen the trainable filter beyond the last block when the llama.cpp pin
      gains backward support for the KV-cache `set_rows` write (or carry a patch).

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
