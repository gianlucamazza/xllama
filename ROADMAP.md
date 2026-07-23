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
- [ ] **Blocked** — widen the trainable filter beyond the last block. Waits on
      the llama.cpp pin gaining backward support for the KV-cache `set_rows`
      write, or on us carrying a patch. Not actionable as it stands; listed so
      the gap is visible, not as work anyone can pick up.

## Phase 11 — Close the headless ↔ UI gap (planned; Phase 10 gates now PASS, unblocked)

Gap analysis 2026-07-20: the training loop is half-invisible (UI captures
preferences, but launch/progress/serving of the fine-tuned model are
headless-only), and the test/API surfaces trail the UI.

- [ ] In-app personalization arc: trigger training from Settings, surface
      progress, serve `merged.gguf` from the model picker (#116) — now unblocked
      (the `DeviceGgmlPartialFt` flip landed).
- [x] Autopilot ops for every setting the GUI exposes, so the harness exercises
      the real UI code paths instead of writing `settings.json` behind the app's
      back. `set_routing` / `set_sampling` / `set_kv_reuse` (#117, PR #120),
      then `set_taesd` / `set_system_prompt` (#126, PR #137) — all drive the real
      controller state and persist through `SaveSettings()`. The
      `validate-console.sh settings` gate seeds the opposite of every target and
      asserts all nine persisted values, so an op that silently does nothing
      fails — PASS on Xbox Series S (MSIX 1.4.0.633, 2026-07-21).
- [ ] LAN API parity (images, preferences, training status) — #118, low
      priority while the API remains a demo.

## Phase 12 — DirectML routing calibration (measurement done; one product call open)

A measurement campaign on the shipping `-v2` asset, prompted by the question of
whether DirectML is worth routing to at all. The measurement is done and the
answer is a product judgement, not a constant: the GPU wins first-turn TTFT on a
long prompt and the CPU wins from the second turn (§5d). Evidence under
`bench/results/phase12-*.csv`; analysis in `docs/uwp-constraints.md` §5b–§5f.

- [x] Instrument the bench path so a row can be interpreted: `n_prompt_tok` and
      `n_gen_tok` (#128), then `max_length` — without the variable under study a
      row carries no information, which cost one throwaway dataset to learn
      twice. `n_ctx` / `n_predict` controllable from the console (#131).
- [x] Prompt-length sweep, 10 lengths x 2 backends (#129). Found a reproducible
      DirectML prefill collapse and retuned `token_threshold` 600 -> 1550 on it.
- [x] **Root-caused (#130):** the collapse tracks `max_length`, not prompt
      length. One byte-identical 1289-token prompt varying only `n_predict`
      swings prefill 130 -> 611 tok/s; a control at `n_ctx` 3072 with
      `max_length` held fixed reproduces the slow figure, so `n_ctx` has no
      effect of its own. `Session::generate` now saturates `max_length` at
      `n_ctx` — faster and less memory. The sweep's numbers stand; the reading
      that produced 1550 does not.
- [x] **Fixed (#133):** the 1550 retune put `token_threshold` above the context
      trimmer's real-token ceiling, so auto GPU routing was unreachable for every
      input. The two constants now live together with a test pinning the
      relation. Found by running the console routing gate, which had not been
      run since the retune.
- [x] Close the CLI/GUI sampling split (#125, PR #136). Not a roadmap item when
      the phase started, but the same class of defect the phase kept surfacing:
      two surfaces holding their own copy of a shared quantity. The CLI ran
      `temp -> dist` while the GUI ran the full chain, so a generation seen on
      one could not be reproduced on the other. Defaults and the chain builder
      now have one home each.
- [x] **Threshold re-derivation done (#139, §5d) — it is a product call, not a
      constant.** The full criterion (adding the asymmetric model load and the
      KV-reuse asymmetry the sweep omitted) shows there is no correct single
      prompt-length threshold: DirectML wins turn 1 above ~1250 tokens and the
      CPU wins from turn 2 at every reachable length. Whether Auto is worth
      leaving on is "how single-turn are long-prompt conversations", which §5d
      calls not measurable from here — so the UI now surfaces TTFT to make it
      observable in real use. The shipping 1550 is left as-is, its rationale
      corrected.
- [x] **Threading recommendation corrected 4 → 6 in the docs (§5f).** The old
      recommendation was a decode-only optimum; measuring prefill too puts 6
      ahead, and `t8` is a trap (it sinks prefill as well as decode).
- [ ] **Ship `intra_op_num_threads: 6` on the CPU asset — deferred, conditional.**
      The shipped `smollm2-360m-cpu-int4` sets none, so the recommendation is
      doc-only. Not shipped yet on purpose: the +8.5% is rigorous at one prompt
      length (1380), below Phase 12's own bar (10 lengths, closing control), and
      a republish now would move the baseline every Phase 12 measurement used
      while #130 is open. Ship condition: a proper thread sweep (≥3 lengths incl.
      short, 3 runs, closing control), ideally with the next models-v1 republish.
      See `docs/recommended-config.md`. The **"3 runs" half of that bar is now
      supported**: the bench records each repeat individually (`run_index`) and
      the summary reports a median + decode min–max spread instead of a
      pre-averaged point, so a re-run at ±8.5% can be told from run-to-run noise
      rather than asserted against a single number (see the measurement-integrity
      item below).
- [x] **Measurement integrity: per-run variance is now recoverable (W1.1).** Every
      published decode number was a single row with no spread reported anywhere —
      the ORT driver ran repeats but appended only a median, discarding the data
      needed to judge it. The bench CSV now carries a `run_index`; the drivers
      (`bench-xbox-ort.sh`, `bench-prompt-sweep.sh`, `bench-xbox-kv.sh`) append
      each recorded run; `scripts/generate-benchmark-summary.py` derives the
      median and a min–max spread and marks single-run rows as such. Host-tested
      (`tests/test_bench.cpp`); the committed CSVs regenerate their current numbers
      unchanged (all pre-existing rows now show _single run_). Re-measuring any
      row on-console with the default 3 recorded runs is what turns a marker into
      a spread — no such row exists yet.
- [x] **Duplicated-state audit (the #125 / #133 / #141 class) — nothing divergent.**
      Three Phase 12 defects were one logical quantity with two homes. A read-only
      sweep of the sampling defaults, the invariant constant pairs
      (`token_threshold`↔trimmer ceiling, `max_length`↔`n_ctx`), chat-template /
      stop-token selection, and every `SaveSettings()` value versus its CLI/bench
      counterpart found **zero divergent cases**: the three prior defects are each
      consolidated to a single home with a guarding test, and the residual
      duplications are doc-only or unconstrained per-surface defaults. The one item
      worth a future hardening is the `max_length` saturation, expressed in two
      files (`session.cpp`, `inference.cpp`) that agree by comment rather than by a
      shared helper — not a defect today.
- [ ] Confirm the `max_length` valley mechanism. §5e gives **strong evidence for
      per-process lazy kernel compilation** (DirectML warms 1.64×/1.72× on the
      second call in a process; CPU control 1.00×), which is the leading
      hypothesis over WDDM residency — but not yet a conclusive profile. The
      per-node profiler cannot localise it alone (the graph is one
      `DmlFusedNode_0_0` at 96% of kernel time, §12). Tracked in #130.

## Upstream and vendor lifecycle

Operational details live in `docs/vendor-lifecycle-plan.md`.

- [ ] Drop PatchedGenAI after a NuGet release includes GenAI #2280.
- [ ] Land ORT ReadFile 16 MB chunk PR #29732 and drop PatchedOrt only after the
      required fixes reach NuGet.
- [x] Lift the DML text gate (#110): `dml_text_model_ok` allowlist behind
      `token_threshold`, `-v2` asset published on models-v1
      (`docs/dml-rmsnorm-fix-runbook.md`).
- [~] Upstream the RMSNorm kernel finding (#111 — **closed**; the repro and
  analysis are done and the workaround ships). What remains is not a task but
  a standing watch: re-test at every GameOS/driver update, since a driver fix
  would let `dml_text_model_ok` widen beyond the single `-v2` allowlist entry.
- [ ] Re-evaluate fused low-bit DirectML GEMM only after upstream capability
      changes; it is not a local application task.

## Definition of done

Any roadmap item that changes runtime behavior needs host tests, the relevant
Windows CI build, on-console validation when hardware-specific, raw evidence
under `bench/results/` when measured, and documentation in the owning SSOT.
