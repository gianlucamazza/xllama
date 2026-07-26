# xllama Roadmap

Current work only. Completed release history belongs in `CHANGELOG.md`; measured
performance belongs in `docs/benchmarks.md`.

## Current product state

- Current manifest: **1.5.0.0** under the new **`GianlucaMazza.xllama`**
  identity (PR #163 — breaking: no in-place update from ≤1.4.x, see
  `docs/install-release.md`). Released as **v1.5.0.0** (2026-07-26): the
  2026-07-25 perf campaign + rebrand (PR #155-#165), shipping the
  console-validated MSIX 1.5.0.698 (all gates PASS).
- Shipping artifact: unified ORT GenAI + llama.cpp, with pinned patched runtime
  DLLs while upstream fixes have not reached NuGet. The UWP ggml build now
  enables `GGML_USE_CPU_REPACK` (PR #155): **GGUF prefill +62%** on Q4_K.
- Default chat: `lfm25-350m`; balanced and quality tiers: LFM2.5-1.2B and
  LFM2-2.6B. The models-v1 CPU asset ships `intra_op_num_threads: 6`
  (confirmed on-device, `bench/results/t6-shipped-confirm.csv`).
- Text DML routing re-enabled for the parity-validated
  `smollm2-360m-dml-fp16-v2` asset (#91 root cause: broken DML RMSNorm kernel,
  fixed by graph decomposition — `docs/dml-rmsnorm-fix-runbook.md`); CPU still
  serves decode and short prompts, DirectML serves long-prompt prefill and
  diffusion — DML sessions warm up at load and pre-load at model-Ready
  (#130/PR #158+#164), so in-app first turns run at the warm regime.
- One process-wide Session owner (`xllama::session_hub()`, PR #161/#164):
  chat UI and LAN API share the resident model ("never 2× model in RAM").
- Phases 1–11 are complete for product code: Phase 10 Lane B is `available`
  (host + console marker gates PASS; pin-blocked filter-widening remains);
  Phase 11 closed the headless↔UI gap (in-app personalize + LAN API parity,
  #116/#118). Remaining open work: the Phase 13 CPU-prefill/KV-reuse campaign
  (#168–#175), the #130 root-cause profile, and upstream vendor pin drops.

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

## Phase 11 — Close the headless ↔ UI gap ✅ complete (code; console re-validate optional)

Gap analysis 2026-07-20: the training loop is half-invisible (UI captures
preferences, but launch/progress/serving of the fine-tuned model are
headless-only), and the test/API surfaces trail the UI.

- [x] In-app personalization arc: trigger training from Settings ("Train on my
      feedback"), surface epoch/loss on the status bar, publish `merged.gguf` as
      catalogue id `personalized` in the model picker (#116). In-process via
      `run_train_job_localized` (not `train.flag`, which exits the process).
      Host-tested job/filter builders (`tests/test_personalize.cpp`); console
      end-to-end re-validate when hardware is available (needs
      `training/base-f16.gguf` or a provisioned SmolLM2 GGUF + samples).
- [x] Autopilot ops for every setting the GUI exposes, so the harness exercises
      the real UI code paths instead of writing `settings.json` behind the app's
      back. `set_routing` / `set_sampling` / `set_kv_reuse` (#117, PR #120),
      then `set_taesd` / `set_system_prompt` (#126, PR #137) — all drive the real
      controller state and persist through `SaveSettings()`. The
      `validate-console.sh settings` gate seeds the opposite of every target and
      asserts all nine persisted values, so an op that silently does nothing
      fails — PASS on Xbox Series S (MSIX 1.4.0.633, 2026-07-21). Also
      `start_train` / `train_status` for the personalize arc (#116).
- [x] LAN API parity (images, preferences, training status) — #118:
      `POST /v1/preferences`, `GET /v1/training/status`,
      `POST /v1/images/generations`. Documented in `docs/api-endpoint.md`;
      `validate-api.sh` gains `prefs|train` (images remain manual — needs
      SD-Turbo on device).

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
- [x] **`intra_op_num_threads: 6` SHIPPED on the CPU asset (2026-07-25).** The
      conditional sweep ran on-console (build 1.4.0.675,
      `bench/results/phase12b-threads-sweep.csv`: 3 lengths 39/285/960 ×
      {unset, t4, t6} × 3 recorded runs + closing control, device config first
      restored to pristine — a stale t4 swap was found and removed, see the
      runbook preflight). Result: **t6 prefill +4.4% / +4.7% / +6.1%** over
      unset, consistent across lengths; decode deltas within the −3%
      closing-control session drift (neutral); t4 ≈ unset, confirming §5f.
      Shipped with the 1.5.0.0 identity migration — the forced full re-provision
      IS the "next models-v1 republish" the ship condition asked to bundle with:
      the release's `genai_config.json` now matches
      `bench/configs/genai_config-threads-6.json`. On-device confirmation
      (config fetch + 3-run bench) recorded with the migration.
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
      a spread — no such row exists yet (**blocked on console access**).
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
      **2026-07-25 update (PR #158):** the cost is now _paid at load_ — a
      throwaway generate warms DML sessions inside the "loading model" phase.
      New mechanism facts from the Session/LAN-API path (960-token request,
      same process): the cold cost hits **decode too** (18.2 → 23.1 tok/s), and
      a ~2-token warm-up recovers only part of it (turn-1 prefill 682 vs 899
      tok/s warm) — the warm-up must run a real-length prompt plus decode
      steps, i.e. compilation is at least coarsely shape-dependent. What
      remains open here is only the _root-cause profile_, not the user-facing
      cost. **PR #164 closed the wall-clock half**: sessions pre-load at
      model-Ready (`PreloadSessionAsync`), so the warm-up runs before the
      user's first send instead of inside its wait — confirmed on-console
      (first DML request: prefill 873 tok/s, decode at warm parity).

## Phase 13 — CPU prefill & KV-reuse structural campaign (in progress)

Sourced from the 2026-07-26 architecture review: a docs sweep of everything
already adopted or rejected (`uwp-constraints.md`, `phase7-hypotheses.md`)
crossed with a hot-path code review. Every item below was **never previously
considered** — none appears in any prior doc or issue. Ordered by
impact/risk; each issue carries the evidence (file:line) and the candidate
fix.

- [x] **#168 — `n_threads_batch` is never set: GGUF prefill runs at 4 threads,
      decode at 6.** Same defect class as the never-defined
      `GGML_USE_CPU_REPACK` (#155). Fixed (PR #177) and **measured on-console
      2026-07-26** (builds 698 → 711): prefill **+12.1% / +10.5%** at
      P=298/1000, decode neutral, no livelock at 6 prefill threads
      (`bench/results/phase13b-threadsbatch-{before,after}.csv`, §5f). GGUF
      headline is now 438.1 prefill / 94.9 decode.
- [ ] **#169 — the context trimmer permanently disables KV reuse** once a chat
      exceeds `kMaxPromptTokens`: every later turn pays a full ~1800-token
      re-prefill. Candidate: context shift (`llama_memory_seq_rm`/`seq_add`) + delta prefill. Highest impact, medium-high risk.
- [ ] **#170 — token-level KV prefix matching.** Regenerate, conversation
      switch and every LAN-API request re-prefill from zero today; step (a)
      is in-memory prefix diff (low risk), step (b) persistent KV via
      `llama_state_seq_save_file` (needs an eviction policy).
- [x] **#171 — KV quantization measured; verdict: knob shipped, default
      OFF.** Consolidation half landed first (PR #177: `kDefaultNCtx`, domain
      test). The q8_0+flash-attn knob (`--kv-q8`, `SessionParams::kv_q8`,
      `bench_kvq8.txt`, F16 fallback if an arch refuses FA) is measured on
      host with the logit-parity comparator: KV −47% on both archs, but the
      drift is entirely from the quantization (FA alone is bit-identical) —
      NMSE 8.25e-03 with a **top-1 flip on the default LFM2.5 model** at a
      trivial prompt (SmolLM2: 2.13e-03, top-1 stable). At `n_ctx` 2048 the
      saving is 11–40 MiB against 320–1600 MB of weights, so shipping it ON
      would trade visible quality for headroom nothing uses yet. Revisit with
      the #169 `n_ctx` raise, gated on H9.
- [x] **#172 — re-sweep `n_ubatch` on console post-repack.** Done 2026-07-26
      (`--ubatch` knob + on-device sweep u128–u1024 at P=1000,
      `bench/results/phase13c-ubatch-sweep.csv`): **the default 512 is the
      optimum** post-repack+#168 (u256 −0.7%, u128 −2.4%, u1024 −2.8% and
      +34 MB peak). No product change; the knob stays for future sweeps.
- [x] **#173 — predict context overflow** instead of failing the turn and
      retrying with a full prefill. Done (PR #177): both backends fail fast
      pre-append when KV + delta + one token exceeds `n_ctx`, and the llama
      generation loop stops cleanly at the context end.
- [x] **#174 — UI-thread micro-costs in `StartInference`** (per-turn
      provisioning I/O, per-message `chat_format()`). Done (PR #177) for
      items 1 and 3; the BuildPrompt-off-UI-thread item folded into #170 on
      cost parity (the worker snapshot costs what the render costs).
- [x] **#175 — repetition-penalty state semantics decided (2026-07-26):
      sampler state follows the KV lifecycle** on both backends — lives as
      long as the conversation, resets with `reset_kv` or a sampling change.
      llama now keeps its chain across reuse turns (`same_chain` guard,
      mirroring `OrtSession::sampling_matches`); the window WIDTH stays
      deliberately divergent (llama last-64 vs ORT whole-sequence, whose C
      API exposes no window) — rationale in `sampling.h`.

Explicitly **not** reopened: mmap via `CreateFileMappingFromApp` — tried and
reverted 2026-07-14 with zero measured benefit (`uwp-constraints.md` §1);
the negative measurement stands even though its original attribution was
retired after #155.

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
