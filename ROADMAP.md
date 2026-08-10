# xllama Roadmap

Current work only. Completed release history belongs in `CHANGELOG.md`; measured
performance belongs in `docs/benchmarks.md`.

## Current product state

- Current manifest: **1.5.4.0** under the **`GianlucaMazza.xllama`** identity
  (in-place update from 1.5.x; still breaking vs ≤1.4.x, see
  `docs/install-release.md`). **v1.5.4.0** (2026-08-08): #216 KV snapshot save
  race fix; #223 thinking `n_predict` 1024 + `thinkdone` (suite **10** gates);
  #130 closed as product-mitigated. Previous: **v1.5.3.0** titles/History/dual-CRT;
  **v1.5.2.0** Phase 14; **v1.5.1.0** Phase 13; **v1.5.0.0** perf + rebrand.
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
  #116/#118). Phases 13 and 14 are complete and shipped. Remaining open work:
  Phase 15 research (parked eng), and upstream vendor pin drops.
- **Shipped as v1.5.4.0:** #216 + #223 + 10-gate suite; #130 product-closed.
  Prior tag **v1.5.3.0** (MSIX 1.5.3.873, day-of-ship 9/9) carried titles,
  dual-CRT, H6 park. Product packages are CI MSVC — `docs/crossbuild-console.md`.
- **Demo capture remains the v1.5.2 assets** (576 stills; re-record optional).
  Pipeline is re-runnable (`demo/demo-script.json` + `scripts/capture-demo-video.sh`).

### Next (after v1.5.4.0)

1. **Vendor pin drops** (#84/#85/#86) — blocked until NuGet moves
   (`scripts/check-vendor-nuget-status.sh`, poll 2026-08-08: still required).
2. **Store readiness** — Partner Center human gate; App-vs-Game spike;
   NOTICE / listing (`docs/store-readiness.md`).
3. **Parked eng** — H6/H7 (#228); crossbuild product parity (layer 2 closed
   2026-08-08 by uwp-crossbuild 0.5.1 — launch proven, ORT/GenAI + first boot
   - uptime not); prompt-lookup default OFF; Phase 15 “3B usable” without new
     density measure.

## Phase 7 — Peer-class model research

Detailed hypotheses and measured verdicts: `docs/phase7-hypotheses.md`.

- [x] H1 efficient LFM campaign and deterministic H9 suite.
- [x] H4 usable dense 3B campaign; Llama-3.2-3B is the preferred comparator.
- [x] H2 MoE candidate — **FAIL, measured on console 2026-07-30.**
      LFM2.5-8B-A1B at `UD-IQ3_S` decodes 14.50 tok/s against the dense 3B's 14.0,
      at +1437 MiB, and reasons every turn (~4× worse perceived latency). The
      bandwidth premise held (631 MB read/token vs 645 predicted); the cost moves
      off bandwidth. Do not reopen at a lower quant — that tests the quantization,
      not the architecture.
- [x] H3 speculative decoding — **CLOSED for product default 2026-08-07/08.**
      Pre-gate 2026-07-29 split the hypothesis (draft-model rejected 0.81× chat;
      prompt-lookup admitted at host 1.53×/1.00×). Phase 15 W2 eng shipped opt-in
      (`prompt_lookup`, default OFF). Console M3: code **1.04× FAIL** ≥1.4× gate;
      chat ~0.99×. Remains opt-in only — see Phase 15 W2 and `docs/phase15-re-opt.md`.
- [x] H5 BitNet/low-bit survey — **done 2026-08-10, NO-GO.** Not on merit: the
      pin already carries the `bitnet` arch, but no sub-4B model _trained_ at
      ≤2 bits publishes downloadable weights, and nine months of QAT literature
      ships recipes rather than checkpoints. Answers Phase 15 milestone M8.
      Reopen only on a released sub-4B low-bit checkpoint — post-hoc 2-bit
      quantisation is a different bet (IQ2_M precedent). Evidence:
      [`docs/phase7-hypotheses.md`](docs/phase7-hypotheses.md) H5.
- [ ] H6/H7 GPU or hybrid GGUF eng — **parked 2026-08-08** after H6.1 G2 FAIL
      (2.15 GB/s packed; STREAM still 119 GB/s). Tracking: **#228** (reopen only
      with new density measure or revised gate).

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
- [x] **Confirm the `max_length` valley mechanism — product closed 2026-08-08
      (#130).** §5e: strong evidence for per-process lazy DML kernel compile
      (warms 1.64×/1.72×; CPU control 1.00×); not node-proven (`DmlFusedNode`
      ~96%). Product mitigations shipped: saturate `max_length` to `n_ctx` +
      load-time warm-up + pre-load at model-Ready (PR #158/#164). In-app turns
      run warm. Issue **closed** product-mitigated; reopen only for a profile
      campaign or regression with warm-up off.

## Phase 13 — CPU prefill & KV-reuse structural campaign ✅ complete (shipped 1.5.1.0)

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
- [x] **#169 — context shift: long chats keep KV reuse past the token
      budget.** Landed 2026-07-26 (PR #184), **console-confirmed same day on
      build 1.5.0.729** (new `validate-console.sh longchat` gate): when a
      continuation turn would overflow `n_ctx`, `LlamaSession` evicts the
      oldest resident tokens past the pinned system prefix
      (`GenerateParams::n_keep`, upstream-style front-drop `seq_rm` + RoPE
      `seq_add`, `m_kv_tokens` compacted) instead of failing, and the UI's
      `do_reuse` no longer requires `n_dropped == 0` on the llama backend.
      On-device: trimmed rounds went from **4532 ms full re-prefill (1791
      tok)** to **~280 ms delta prefill (76–84 tok)**; the shift fired
      mid-conversation (evicted 1017 past keep=13, kv 1031/2048), later turns
      stayed in the reuse regime with up to 10 turns trimmed, post-shift
      replies coherent; KV-bench turn-2 reuse 59 ms vs 890 cold (15–16×),
      decode 90.3–90.8. Gated on `llama_memory_can_shift` + no SWA — **both
      catalogue GGUF archs verified on host**: LFM2.5 (hybrid) shifts;
      Qwen3.5 (imrope, cannot shift — `seq_add` would abort) keeps the clean
      #173 fail-fast the UI retry keys on.
- [x] **#170 — token-level KV prefix matching.** **Step (a) landed
      (2026-07-26): in-memory prefix diff in `LlamaSession`** — a full-prompt
      turn rewinds the resident KV to the common token prefix
      (`llama_memory_seq_rm`) and prefills only the divergent tail; a
      regenerate re-prefills exactly 1 token (opt-in host test with a real
      GGUF: regenerate byte-identical; extended prompt agrees on the leading
      token — near-tie divergence downstream is inherent to any prefix
      cache, the resident prefix was accumulated in a different batch
      shape). Covers regenerate, edited last message, and LAN-API extension
      requests through the resident hub session. **Hybrid correction (PR
      #183):** hybrid caches (both catalogue GGUFs) refuse the tail rewind at
      the `llama_memory` layer — the ignored `seq_rm` return corrupted
      regenerate output; now a pure extension skips `seq_rm` and a refused
      erase degrades to a correct full re-prefill (regime-aware opt-in test).
      **Step (b) landed (2026-07-27):** leaving a conversation writes its KV
      to `LocalState\kv\<id>.kv` and the first turn back restores it, so the
      switch costs a delta prefill instead of the history. Host-measured at
      `n_ctx` 2048 on both catalogue GGUFs: LFM2.5 17.6 MB / 1476 tokens (12
      KiB/token, save 21 ms, load 33 ms), Qwen3.5-0.8B 36.5 MB / 1472 (25
      KiB/token, save 124 ms, load 301 ms) — against the 4532 ms console
      re-prefill. Fingerprinted
      (model, n*ctx, KV quant, LoRA) because the pin validates cache shape
      only; atomic writes in 8 MB chunks (§9 AppContainer bound); `KvStore`
      caps the pool at 3 files / 192 MB, LRU, host-tested. A stale snapshot
      is harmless by construction — the #170a diff turns it into the prefill
      that would have happened anyway. **Console-confirmed** on build
      1.5.1.737 (`validate-console.sh kvsnap`): leaving a conversation and
      returning takes the prefill from 551 tokens to 19 (3% of cold). **Step (c), the ex-#174
      BuildPrompt-off-the-UI-thread item: won't do, measured.** The deep copy
      plus the full render of a prompt at the trimmer ceiling (10 KB) costs
      **11.2 µs** on host — call it 30–50 µs on Zen2, once per turn, against a
      16.7 ms frame. Moving it to the worker means snapshotting the
      \_untrimmed* conversation on the UI thread (more copying than the render
      it replaces) or putting a mutex on conversation state that the
      token-streaming path would then contend on. The real #174 costs — the
      per-turn provisioning I/O and the per-message `chat_format()` — were the
      ones worth fixing, and PR #177 fixed them.
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

## Phase 14 — Coding, chat and thinking tiers ✅ complete (unreleased)

> **Thinking tier (updated 2026-08-08, #223 closed).** Wired, decode-measured,
> and dual-gated: `thinkcut` (truncated CoT) + **`thinkdone`** (short happy path
> at catalogue `n_predict` 1024). Still **absent from H9** (quality suite is the
> instruct sibling). Multi-step arithmetic can still exhaust CoT without an
> answer — product honesty, not a missing gate.

Phase 13 made a turn cheap; Phase 14 spends that headroom on **what the
catalogue can hold**. Architecture first, no Settings rewrite: the catalogue
gained a per-model `n_ctx` and a `role: coding`, and the product surfaces read
them. SSOT: [`docs/model-matrix.md`](docs/model-matrix.md); console evidence
`bench/results/phase14-console.csv`.

- [x] **Catalogue tier.** `qwen25-coder-0.5b` / `1.5b` / `3b`, `qwen3-1.7b`,
      `lfm25-1.2b-thinking`. Console decode / peak: Coder-0.5B **62.4** tok/s /
      533 MB, Thinking **36.7** / 811 MB, Coder-1.5B **26.1** / 1179 MB,
      Qwen3-1.7B **21.8** / 1398 MB, Coder-3B **14.0** / 2116 MB. The 3B row is
      the number Phase 15 exists to move.
- [x] **Thinking as a product path, not UI magic.** `model_is_thinking` +
      `strip_thinking_blocks` in `ChatFormat::postprocess_output` — display and
      persist the answer only. `model_is_qwen3` keeps the no-think prefill
      scoped to Qwen3, not to Coder or Thinking. A reasoning block that runs out
      of tokens now stores an explicit "reasoning only" turn instead of leaving
      raw CoT on screen with no saved reply.
- [x] **Thinking models take no KV snapshot (#170b interaction).** The saved
      history is stripped while the resident KV holds the full CoT, so the #170a
      prefix diff always diverges on return: the snapshot bought nothing and cost
      tens of MB of writes. In-conversation delta reuse is unaffected.
- [x] **Prefill batching and budget fixes (PR #193).** A prompt over ~2049
      tokens aborted the process rather than failing — both prefills submitted
      the whole prompt as one logical batch and `llama_decode` trips
      `GGML_ASSERT` instead of returning an error, reachable from both the chat
      UI at the coding tier's 4096-token session and the LAN API. Prefill is
      chunked at `llama_n_batch` now (physical ubatch 512 unchanged, the #172
      optimum). Replies also stopped being silently cut at ~250 tokens: the
      trimmer reserved a flat 250 against a UI default `n_predict` of 512.
- [x] **Exact token-budget trim + gates** (PR #194, shipped in 1.5.2.0): the
      budget is enforced in tokens, once, where the tokenizer is (`fit_prompt`),
      on both the chat UI and the LAN endpoint. It found four more defects on the
      way — auto GPU routing unreachable on a default install, the LAN endpoint
      with no budget at all, a bisection returning an empty prompt with no history
      to drop, and a capability matrix wrong about Qwen3's `can_shift`. Three new
      console gates came with it (`coderpaste`, `thinkcut`, `genroom`), all PASS on
      the shipped 1.5.2.789.

## Phase 15 — Make the 3B–4B class usable (in progress)

The binding constraint is **bytes read per token**, not framework quality: the
bus is 224 GB/s, CPU decode reaches ~13 GB/s effective and DirectML fp16 ~34,
and Coder-3B lands at 14.0 tok/s. Three levers touch that denominator; a
rewritten framework or a bespoke model architecture touch neither, and both are
rejected on that ground (`docs/phase7-hypotheses.md`, "Do not reopen").

**Where it stands (2026-08-08).** W1 is **closed FAIL on measure** — the MoE
reads only its active experts, exactly as predicted, and is still no faster than
a dense 3B because the cost moves off bandwidth. W2 is **closed for product
default**: pre-gate rejected draft-model; prompt-lookup eng shipped opt-in;
console M3 **1.04× FAIL** ≥1.4× gate so default stays OFF. W3 (gpubw) **M6
PASS** (Series S STREAM **119.07 GB/s** ≥ 100 kill) → H6 eng opened. Ordering
W1→W2→W3 measure-before-build paid off: three levers decided without inventing
numbers.

**Campaign SSOT:** [`docs/phase15-re-opt.md`](docs/phase15-re-opt.md) (method,
Findings, workstream status, decision log). Attack order: **W2 closed**
(default OFF after M3); **W3 M6 PASS** → H6 eng.

- [x] **W1 — H2: a MoE whose active weights are a fraction of its total. CLOSED
      FAIL 2026-07-30.**
      Desk survey done 2026-07-29 against pin `6d5a910c5 (tag `b10094`)`
      (`src/models/*.cpp` already compiles `lfm2moe.cpp` and ~20 more MoE archs).
      Candidate **LFM2.5-8B-A1B**, 32 experts / 4 active. Admissibility hung on a
      never-measured number, so the **console heap ceiling was measured**:
      **4864 MB committed / 4893 MB peak WS** (`ramceil.flag`,
      `scripts/bench-ramceil.sh`, `bench/results/phase15-ramceil.csv`) — a lower
      bound, and headless. That admits `UD-IQ3_S` inside the 4 GB H2 gate, so the
      experiment tests the architecture rather than the quantization.

  **Measured 2026-07-30 on MSIX 1.5.2.798 → H2 FAIL**
  (`bench/results/phase15-moe-console.csv`, 3 recorded runs): decode
  **14.50 tok/s** against the dense `qwen25-coder-3b`'s 14.0, peak
  **3553 MiB** (+1437). The sparse-activation premise was _confirmed_ — ~631 MB
  read per token against 645 predicted — but the bottleneck moves off bandwidth,
  and what remains is ~5× a dense model of the same active parameters (i-quant
  dequantization or expert gather; not separated by this measurement). Perceived
  latency decides it: the model reasons every turn, spending 102 completion
  tokens to answer "Roma" and 401 for two sentences, so the user waits ~4× longer
  than with the dense 3B for the same visible output. H9 was not applicable (its
  tasks cap generation at 16-80 tokens). The entry stays in `model-matrix.md` §A3
  with the result attached and enters no product tier; the 3.5 GB product-gate
  question a speed PASS would have opened is moot.

- [x] **W2 — H3: speculative decoding (#210). CLOSED for product default
      2026-08-07/08** (eng remains as opt-in). Host: pure `prompt_lookup_draft`;
      lead-first multi-token verify + `seq_rm` degrade; CLI `--prompt-lookup` /
      `SessionParams` / headless `bench_prompt_lookup.txt` default OFF; greedy
      MATCH. Console M3 Series S (`qwen25-coder-3b`, t6): code **1.04× FAIL**
      ≥1.4× (14.15 → 14.74); chat 0.99×; peak ~2.0 GB. Spec fires (~32/62 accepts
      on code) but does not break the membw wall. **Default OFF.** Ship/measure
      packages via CI MSVC (`docs/crossbuild-console.md`). CSV:
      `bench/results/phase15-spec-w2-console.csv`. SSOT Findings:
      `docs/phase15-re-opt.md`.
      Both halves of the original draft/target
      pair already ship and are console-PASS: draft `qwen25-coder-0.5b` (379 MB,
      62.4 tok/s), target `qwen25-coder-3b` (1840 MB, 14.0). **Vocab precondition
      measured and PASSED 2026-07-29** (`bench/results/phase15-spec-vocab.csv`):
      the pin's `common_speculative_are_compatible` throws rather than degrades,
      and the pair is identical across all its checks. The same pass produced two
      keepers — Qwen3-1.7B and Coder share a vocab _size_ but differ in 4 token
      texts, and the H2 MoE has a 128000-token vocab against LFM2.5-350M's
      65536, so **H2 and H3 do not compose**.

  **Pre-gate measured 2026-07-29 → the draft model is rejected, the draft-free
  variant proceeds** (`scripts/bench-spec-pregate.sh`,
  `bench/results/phase15-spec-pregate.csv`, derivation in
  `scripts/analyze-spec-pregate.py`). The 3B's prefill:decode ratio is only
  3.3:1, so verifying k+1 tokens is not nearly free: compute is 30% of a decode
  step, capping speculation at **1.90×** with a draft model and **3.30×**
  without. Measured acceptance then splits hard by regime — the draft model
  gives 1.43× on code and **0.81× on open chat** (0.67× at k=4), because it
  drafts unconditionally and pays 16.0 ms for every rejected token. Prompt
  lookup declines instead: 10 drafted tokens against the draft model's 140 on
  the same prompt, so **1.53× on code and 1.00× on chat**. The deciding
  property is not the acceptance rate but whether a variant spends when it has
  no evidence.
  Implementation therefore needs **no second model**, which keeps the
  SessionHub "never 2× model resident" invariant true without argument, adds no
  threadpool to risk the t7/t8 livelock, and drops the vocab constraint. It also
  cannot use upstream `common/speculative.cpp`: `uwp/ggml-uwp.vcxproj` compiles
  no `common/` sources, and pulling them in would drag `common/sampling.cpp`
  against the unified sampling of #125/#141. PASS ≥1.4× at unchanged quality and
  peak < 3.5 GB; `longchat` and `kvsnap` are the real regression tests, since
  the draft touches the KV.

- [x] **W3 — H6 gate: does our own compute shader beat DirectML? (#211)** Not a
      backend — a measurement. **M6 PASS 2026-08-08:** Series S STREAM
      **119.07 GB/s** (1 GiB, checksum_ok) ≥ 100 kill. CI package `1.5.2.853`,
      CSV `bench/results/phase15-gpubw.csv`. Multi-dim Dispatch for 1 GiB;
      system D3D12 only (no Agility). Every prior negative GPU result was
      DirectML; this is our CS. **H6 eng opened** by this result.
- [x] **Groundwork W2 needs: the prefill and decode loops written once**
      (2026-07-30, `src/bridge/decode_loop.h`). `run_inference_llama` and
      `LlamaSession::generate` kept hand-maintained copies that had already
      drifted twice — #193 fixed in both, and a stop-sequence token count that
      made `n_eval` (and the published `decode_tok_s`) differ by one between the
      paths. W2 adds prompt lookup inside exactly that loop, so doing it first is
      what makes W2 a change in one place instead of two, and what gives the CLI
      the same behaviour as the chat UI — the CLI being the surface this project
      measures and A/Bs on. −107 lines; host suite unchanged, all 8 console gates
      PASS on MSIX 1.5.2.802.

## Phase 16 — Model scouting (complete)

SSOT: [`docs/phase16-model-scouting.md`](docs/phase16-model-scouting.md) — funnel,
validation ladder, workstream cards, decision log. Phase 15 owns how fast the
resident model runs; Phase 16 owns which models exist at all. The 2026-07-27
survey (`docs/model-matrix.md` §F) is the thing being refreshed, across four
classes and with a console budget capped at ≤9 bench sessions.

- [x] **WS-A (H16.1) — text GGUF scouting.** **Done 2026-08-10, one shipped:**
      `lfm25-230m` is the new **floor** tier — faster and lighter than
      `gemma3-270m` on both axes, one H9 task below it. Figures live in
      [`docs/benchmarks.md`](docs/benchmarks.md), which is their SSOT.
      Qwen3.5-2B and Maincoder-1B measured FAIL; MiniCPM5-1B deferred on
      unbought renderer work. 3 of ≤4 sessions spent.
- [x] **WS-B (H16.2) — `llama.cpp` pin bump evaluation.** **Closed 2026-08-10,
      not motivated.** Both T0 arch flags were refuted at the GGUF header
      (`Qwen3.5-2B` converts to `qwen35`, which the pin carries); no candidate
      fails the arch filter and nothing else, so the trigger never fired.
- [x] **WS-C (H16.3) — ORT GenAI / DirectML text.** **Closed on its own kill:**
      every 2026 sub-4B model moved off the Qwen3/Gemma3 architectures the GenAI
      builder is frozen at. Reopening needs a GenAI bump (`vendor-lifecycle-plan.md`).
- [x] **WS-D (H16.4) — diffusion successor.** **Closed on its own kill:** no
      candidate has a usable 3-component ONNX export. SDXL-Turbo fails the 2 GB
      protobuf limit, not the GPU budget; SD3.5/Flux are monolithic DiT.
- [~] **WS-E (H16.5) — embedding surface.** **Blocked on a product decision,**
  not on technology: `nomic-embed-text-v1.5` is verified config-only at
  156 MB, but nobody has named who consumes the vectors, where they live, or
  how a second `llama_context` survives the one-resident-model rule.
- [~] **WS-F (H16.6) — ASR surface.** **Blocked on an unwritten probe:** does the
  Xbox AppContainer grant `MediaCapture`/`AudioGraph`? T0 settled the backend
  half — GGUF ASR is empty by construction, so any route is ORT GenAI.
- [x] **WS-G (H16.7) — vision / VLM surface.** **Closed:** S-gate FAIL — five
      new C++ surfaces against a ≤1 budget, on a desk close with no console time.

At most one of WS-E/F/G may reach the console this campaign.

## Xbox Store retail (public release path)

SSOT: [`docs/store-readiness.md`](docs/store-readiness.md). Dev Mode remains
the supported install path until a submission is accepted.

- [~] Phase 0 — discovery: SSOT + licence matrix in `store-readiness.md`;
  Partner Center decision + App vs Game spike + go/no-go still open.
- [~] Phase 1 — Store SKU foundation + **no-VM CI path** (`workflow_dispatch`
  `store_sku=true` → `xllama-appx-store`, `install-latest-build.sh --store`);
  console smoke + Partner Center identity still open.
- [~] Phase 2 — privacy draft in `docs/privacy.md`; **5 listing screenshots
  captured** 2026-07-30 (#215); age rating, EN listing copy, trailer and a
  published HTTPS privacy URL still open.
- [ ] Phase 3 — Partner Center submission + certification.
- [ ] Phase 4 — post-launch dual path (Store + Dev Mode) in README.

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
