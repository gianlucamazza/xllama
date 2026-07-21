# xllama Roadmap

Current work only. Completed release history belongs in `CHANGELOG.md`; measured
performance belongs in `docs/benchmarks.md`.

## Current product state

- Latest published release: **v1.4.0.0** (MSIX 1.4.0.588 — GPU text routing
  re-enabled on the `-v2` DML asset, #91 root cause fixed).
- Shipping artifact: unified ORT GenAI + llama.cpp, with pinned patched runtime
  DLLs while upstream fixes have not reached NuGet.
- Default chat: `lfm25-350m`; balanced and quality tiers: LFM2.5-1.2B and
  LFM2-2.6B.
- Text DML routing re-enabled for the parity-validated
  `smollm2-360m-dml-fp16-v2` asset (#91 root cause: broken DML RMSNorm kernel,
  fixed by graph decomposition — `docs/dml-rmsnorm-fix-runbook.md`); CPU still
  serves decode and short prompts, DirectML serves long-prompt prefill and
  diffusion.
- Phases 1–9 (delivery, console validation, demo, training pillar,
  personalization ops) are complete; Phase 10 (Lane B on-device training) has
  passed both marker gates — `DeviceGgmlPartialFt` is `available` — with only
  the pin-blocked filter-widening left. Phase 11 (headless ↔ UI gap) is next.

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

## Phase 10 — Lane B: on-device training (marker gates PASS; filter-widening blocked on pin)

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
- [x] Host engine marker job **PASS** (2026-07-20): LR 2e-4 + shortened
      `XLLAMA-LORA-OK.` marker converge the greedy eval at epoch 8; recipe and
      convergence notes in `docs/training-architecture.md` §10.
- [x] UWP glue: `train.flag` headless mode, LocalState-relative job paths,
      `training/result.done`; `device-train` mode in
      `validate-console-training.sh`.
- [x] Console `device-train` **PASS** (2026-07-20, MSIX 1.4.0.595): RSS peak
      1195 MB (< 3 GB gate), wall 446 s, marker reproduced —
      `DeviceGgmlPartialFt` flipped to `available`. Evidence:
      `bench/results/phase10-console-devtrain-result.json`. (Eval fix: the
      device merged model loads by absolute path; the last-block fine-tune has
      no LR decay, so evaluate at the epoch-8 convergence point, not later.)
- [ ] Widen the trainable filter beyond the last block when the llama.cpp pin
      gains backward support for the KV-cache `set_rows` write (or carry a patch).

## Phase 11 — Close the headless ↔ UI gap (planned; Phase 10 gates now PASS, unblocked)

Gap analysis 2026-07-20: the training loop is half-invisible (UI captures
preferences, but launch/progress/serving of the fine-tuned model are
headless-only), and the test/API surfaces trail the UI.

- [ ] In-app personalization arc: trigger training from Settings, surface
      progress, serve `merged.gguf` from the model picker (#116) — now unblocked
      (the `DeviceGgmlPartialFt` flip landed).
- [x] Autopilot ops for routing / sampling / KV-reuse so the harness exercises
      the real UI code paths (#117, PR #120 merged): `set_routing`,
      `set_sampling` and `set_kv_reuse` drive the real controller state and
      persist through `SaveSettings()`; the new `validate-console.sh settings`
      gate asserts all seven persisted values — PASS on Xbox Series S
      (MSIX 1.4.0.606, 2026-07-21).
- [ ] LAN API parity (images, preferences, training status) — #118, low
      priority while the API remains a demo.

## Upstream and vendor lifecycle

Operational details live in `docs/vendor-lifecycle-plan.md`.

- [ ] Drop PatchedGenAI after a NuGet release includes GenAI #2280.
- [ ] Land ORT ReadFile 16 MB chunk PR #29732 and drop PatchedOrt only after the
      required fixes reach NuGet.
- [x] Lift the DML text gate (#110): `dml_text_model_ok` allowlist behind
      `token_threshold`, `-v2` asset published on models-v1
      (`docs/dml-rmsnorm-fix-runbook.md`).
- [ ] Upstream the RMSNorm kernel finding (#111) — fork/analyze/fix where
      needed, minimal single-op repro; re-test at every GameOS/driver update.
- [ ] Re-evaluate fused low-bit DirectML GEMM only after upstream capability
      changes; it is not a local application task.

## Definition of done

Any roadmap item that changes runtime behavior needs host tests, the relevant
Windows CI build, on-console validation when hardware-specific, raw evidence
under `bench/results/` when measured, and documentation in the owning SSOT.
