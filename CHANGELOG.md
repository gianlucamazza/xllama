# Changelog

All notable changes to xllama are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Performance

- **GGUF prefill now runs on the configured thread count (#168, PR #177).**
  Neither llama context creation set `n_threads_batch`, whose pin default is
  4 regardless of `n_threads` — so prefill (any ubatch > 1 token) ran on 4
  threads while decode got 6, including every published GGUF prefill figure
  (the +62% repack numbers among them). Same defect class as the
  never-defined `GGML_USE_CPU_REPACK`. **Measured on-console** (builds
  698 → 711, 3 recorded runs per point,
  `bench/results/phase13b-threadsbatch-{before,after}.csv`): prefill
  **+12.1%** at P=298 (390.7 → 438.1 tok/s) and **+10.5%** at P=1000
  (388.6 → 429.2), decode neutral, peak RAM unchanged, no livelock at 6
  prefill threads. GGUF prefill rows recorded before build 711 carry a
  comparability boundary (`bench/README.md`).
- **UI-thread per-turn costs (#174, PR #177).** `IsModelProvisioned`'s
  filesystem probe (LocalState + InstalledPath stats + USB-root `_wfopen`)
  ran on the UI thread on every turn but only the sticky first-turn routing
  decision consumes it — moved inside that branch. `RenderConversation`
  resolved the per-model chat format once per message (~10 string builds per
  call); once per render now. The third item (BuildPrompt off the UI thread)
  folds into #170: without session-side prompt bookkeeping the worker's
  snapshot costs what the render costs.

### Performance

- **In-memory KV prefix matching (#170 step a).** A full-prompt turn on
  `LlamaSession` used to clear the cache and re-prefill everything;
  it now rewinds to the common token prefix with the resident KV
  (`llama_memory_seq_rm`) and prefills only the divergent tail. A
  regenerate re-prefills exactly **1 token** instead of the whole
  conversation; an edited last message keeps everything before the edit; a
  LAN-API request that extends the previous one (the typical
  conversational client against the resident hub session) pays only the
  extension. Correctness pinned by an opt-in host test with a real GGUF:
  the regenerate is byte-identical (same resident values ⇒ same greedy
  text); an extended prompt agrees on the leading token, with downstream
  near-tie divergence documented as inherent to prefix caching (the
  resident prefix was accumulated in a different batch shape, so K/V last
  bits differ). The record is cleared whenever a decode failure makes the
  cache untrustworthy.

### Added

- **q8_0 KV-cache knob, measured — default stays OFF (#171).** `--kv-q8`
  (CLI), `SessionParams::kv_q8`, `bench_kvq8.txt` (+ `-kvq8` host tag and
  driver guard) set `type_k/type_v = q8_0` and force flash attention, which
  quantized V requires; on a context-creation failure the session falls
  back to F16 KV so an arch without FA support still loads. Measured with
  the logit-parity comparator: KV **−47%** on both tested archs
  (LFM2.5 24 → 12.75 MiB, SmolLM2 80 → 42.5 at n_ctx 2048), FA alone is
  **bit-identical**, but the quantization drifts the logits — LFM2.5 NMSE
  8.25e-03 with a top-1 flip on a trivial prompt (SmolLM2 2.13e-03,
  top-1 stable). Verdict: at n_ctx 2048 the saving is noise next to the
  weights, so the default stays F16; revisit with the #169 context raise,
  gated on H9.

- **`--ubatch` bench knob + on-console `n_ubatch` sweep (#172).**
  `bench_ubatch.txt` reaches `InferenceParams::n_ubatch` on the GGUF path and
  the device tags the CSV host column with `-uN` (the schema has no ubatch
  column; a row must carry its variable). Driver guard mirrors
  `--max-length`'s ignored-knob check. Sweep verdict
  (`bench/results/phase13c-ubatch-sweep.csv`, u128–u1024 at P=1000, 3 runs
  each): **the default 512 is the optimum** post-repack+#168 — u256 −0.7%,
  u128 −2.4%, u1024 −2.8% with +34 MB peak. No product change.

### Changed

- **Repetition-penalty state now follows the KV lifecycle on both backends
  (#175).** llama.cpp rebuilt its sampler chain every turn, so the penalty
  window (and the dist RNG) reset per turn while the ORT persistent
  generator carried them across the conversation — the runtime-state
  residual of the #125/#136/#141 unification. Decision: state lives as long
  as the conversation (resets with `reset_kv` or a sampling change);
  `LlamaSession` keeps its chain across reuse turns behind the `same_chain`
  guard (`sampling.h`), the twin of `OrtSession::sampling_matches`. The
  window WIDTH stays deliberately divergent — llama's last-64 vs ORT's
  whole-sequence penalty (no window in the C API); the short window is kept
  on purpose, whole-sequence penalties being the known cause of long-chat
  degradation. Greedy paths (all quality gates) are unaffected.

- **The shipping context size has one home (#171, PR #177).**
  `kDefaultNCtx` (`inference_params.h`) replaces the three unlinked
  `n_ctx = 2048` literals in the chat UI, the LAN API and the Session/CLI
  defaults — the #133/#141 one-quantity-two-homes class, consolidated
  before divergence this time. `tests/test_routing_policy.cpp` pins
  `kMaxPromptTokens < kDefaultNCtx` with ≥200 tokens of generation headroom.
  KV-cache quantization stays open on #171, gated on measurement.

### Fixed

- **Context overflow is predicted instead of discovered by a failed turn
  (#173, PR #177).** A continuation turn that cannot fit (KV + delta + at
  least one generated token over `n_ctx`) used to surface as a failed
  append/decode; the UI then retried with a full prefill — a wasted attempt
  while the user waits. Both backends now check the arithmetic up front
  (the KV length is already at hand on both) and fail fast with a
  `context full` error before touching the generator, so the existing
  fallback runs immediately. The llama generation loop is also clamped to
  the context end — a clean stop instead of "decode failed, stopping".

## [1.5.0.0] - 2026-07-26

### Performance

- **GGUF prefill +62% on console (PR #155).** `uwp/ggml-uwp.vcxproj` compiled
  `repack.cpp` but never defined `GGML_USE_CPU_REPACK`, so the repacked-weight
  GEMM path (Q4_0/Q4_K/Q8_0/IQ4_NL) was dead code on Xbox. With it enabled,
  `lfm25-350m` (Q4_K_M) prefill goes 241.9 → 393.2 tok/s at P=298 and
  238.8 → 387.5 at P=1000, decode and RAM unchanged, model load faster
  (`bench/results/phase13-repack-{before,after}.csv`, builds 674/675).
  Same PR makes the Linux ISA flags real: the `LLAMA_AVX2/F16C/FMA` spellings
  were inert (no deprecation mapping), so ggml silently built `-march=native`;
  the CI artifact is now genuinely portable AVX2 (`XLLAMA_NATIVE_OPT` opts back
  into host tuning).
- **DirectML sessions warm up at load (#130, PR #158).** The first generate in
  a process paid a large lazy-compilation cost — measured on the LAN-API path:
  turn-1 5.90 s vs turn-2 1.77 s for the same 960-token request, decode
  18.2 → 23.1 tok/s. `create_ort` now runs a throwaway real-length generate
  inside the "loading model" phase (`SessionParams::dml_warmup`). Session logs
  now include per-turn prefill rate (was decode-only, hiding TTFT).
- **Hot-loop cleanups (PR #157).** Per-token `fputs`+`fflush(stdout)` removed
  from non-CLI builds (`InferenceParams::echo_stdout`); `on_token` takes
  `std::string_view` (no per-token heap string); UNet loop hoists the constant
  `encoder_hidden_states` fp16 conversion and reuses step buffers.
- **CPU threads sweep evidence (`phase12b-threads-sweep.csv`).** t6 prefill
  +4.4/+4.7/+6.1% at P=39/285/960, decode neutral within session drift, t4 ≈
  unset.
- **`intra_op_num_threads: 6` SHIPPED on the models-v1 CPU asset (2026-07-25).**
  The identity migration forces a full model re-provision anyway, which is
  exactly the "next models-v1 republish" the ship condition asked to bundle
  with. `genai_config.json` on the `models-v1` release now matches
  `bench/configs/genai_config-threads-6.json` (pristine + the one key). Devices
  pick it up on their next provision; on-device confirmation recorded with the
  1.5.0.0 migration.
- **Single Session owner + pre-load (PR #161/#164).** `xllama::SessionHub` is
  the one process-wide Session owner for GUI and LAN API ("never 2× model in
  RAM" now holds process-wide; API busy semantics correctly cover GUI turns;
  `hub.generation` invalidates the GUI's KV-reuse state after an API-driven
  model swap). Sessions pre-load when a model becomes Ready — the first send
  pays prefill+decode only, which is what turns the #158 DML warm-up into a
  wall-clock win. Also: manifest cached off the per-turn UI-thread path, the
  routing token-count no longer loads a model on the UI thread (1.2–2.7 s of
  frozen UI on a cold first turn), O(n) context trimmer, shared
  `resolve_max_length` ladder (host-tested), and post-turn incremental
  paragraph finalize instead of a full conversation re-render.

### Changed

- **BREAKING (package identity): rebrand Venere Labs → Gianluca Mazza.**
  `AppxManifest.xml` `Identity Name` changes `VenereLabs.xllama` →
  `GianlucaMazza.xllama` (version bumped to 1.5.0.0), so the
  PackageFamilyName becomes `GianlucaMazza.xllama_pj67f1fcj4n14` and
  Windows/Xbox treats this as a **new app**: no in-place update from ≤ 1.4.x,
  and the old package's `LocalState` (downloaded models, training output, chat
  history) does not carry over. Migration: uninstall the old app and
  re-download models on first launch. `deploy.sh` and
  `install-latest-build.sh` recognize both identities during the transition;
  copyright headers, `LICENSE`, README maintainer, and
  `PublisherDisplayName` now read "Gianluca Mazza". The signing identity
  (`Publisher="CN=xllama-dev"`) is unchanged, so the existing trust
  certificate keeps working.
- **Documentation sync for Phase 11 / #118 and SSOT practices.** `docs/README.md`
  states ownership principles; `architecture.md`, `training-architecture.md` §11,
  `using-the-app.md`, `api-endpoint.md`, `training/README.md`, root `README.md`,
  `AGENTS.md`, `ROADMAP.md`, and `CONTRIBUTING.md` reflect in-app personalize and
  LAN parity instead of “Phase 11 next”.
- **Doc dedup pass.** Root `README.md` drops the full file tree (→ `AGENTS.md`),
  full model/perf table (→ `model-selection` / `benchmarks`), long limitations and
  phase list (→ `uwp-constraints` / `ROADMAP`). `docs/README.md` no longer
  duplicates the ownership table as a second descriptive index; architecture
  and UI guides stop restating measured speedups.
- **Doc coherence pass.** Catalogue download sizes aligned to
  `uwp/models/manifest.json` `approx_bytes` (LFM/Qwen were understating by
  ~5–10%); ORT decode headlines aligned to generated `benchmarks.md` (68.0 /
  20.6, not ~66 / ~21). `docs/README.md` records the check matrix.
- **`scripts/check-coherence.py`** — host-side SSOT audit (defaults, pins, API
  routes, personalize paths, sizes, H9/Phase 10 evidence, generated headlines,
  doc links). Documented in `docs/README.md` / `CONTRIBUTING.md`.

### Added

- **Phase 11 in-app personalization (#116).** Settings → "Train on my feedback"
  runs Lane B `partial_ft` in-process (not `train.flag`, which exits the app),
  surfaces epoch/loss on the status bar, and publishes `merged.gguf` as catalogue
  id `personalized` for the model picker. Shared runner
  `run_train_job_localized` + host helpers in `include/xllama/personalize.h`.
  Autopilot: `start_train` / `train_status`. Preflight needs usable preference
  samples and `training/base-f16.gguf` (or a provisioned SmolLM2 GGUF).

- **LAN API UI parity (#118).** `POST /v1/preferences` (append
  `training/samples.jsonl`), `GET /v1/training/status`,
  `POST /v1/images/generations` (SD-Turbo, steps 1–4, single-slot mutex with
  chat). `scripts/validate-api.sh` modes `prefs|train`; docs in
  `docs/api-endpoint.md`.

- **`n_prompt_tok` and `n_gen_tok` in the bench CSV.** The rates alone never said
  at what prompt length a row was measured, nor how many tokens it generated, so
  turn time was not reconstructible and no two rows were comparable — the reason
  the 2026-07-07 DirectML matrix could not be re-read. Generation is capped by
  the context window and can stop early on EOG, so `n_gen_tok` is not the
  requested `n_predict`: a 1574-token prompt at `n_ctx` 2048 caps new tokens at
  474 and generated 277.
- **`scripts/bench-prompt-sweep.sh`** (prompt-length sweep across backends) and
  **`scripts/bench-xbox-kv.sh`** (multi-turn KV-reuse bench). Nothing drove
  `bench_turns.txt` before — the committed KV CSVs were made by hand.
- **Per-run variance in the bench — `run_index` and reported spread (W1.1).**
  Every published decode number was a single row and no variance was reported
  anywhere; the ORT driver ran repeats but appended only a median, discarding the
  data needed to tell a real change from run-to-run noise. The bench CSV now
  carries a `run_index` (appended last, after `date`, so no positionally-parsed
  column shifts), and the device echoes it from `bench_run_index.txt`. The drivers
  (`bench-xbox-ort.sh`, `bench-prompt-sweep.sh`, `bench-xbox-kv.sh`) now append
  each recorded run — warmup run 1 dropped, runs 2..N kept — instead of a
  pre-averaged median, and `--runs` defaults to 4 (3 recorded runs).
  `scripts/generate-benchmark-summary.py` aggregates repeats sharing a selector
  into a median and a decode min–max spread, and marks single-run rows explicitly;
  `bench/benchmark-summary.json` documents the selection policy. The committed CSVs
  have no `run_index` and regenerate their current numbers unchanged — every
  existing row now shows _single run_. Host-tested in `tests/test_bench.cpp`.

- **Lane B on-device training validated — `DeviceGgmlPartialFt` is now
  `available`.** The in-process ggml-opt partial fine-tune passes both marker
  gates: host (LR 2e-4, greedy eval reproduces `XLLAMA-LORA-OK.`) and console
  (Xbox Series S, MSIX 1.4.0.595) with peak working set **1195 MB** (under the
  3 GB gate), wall 446 s, marker reproduced end-to-end (prepare → train →
  export → merge → evaluate). Evidence:
  `bench/results/phase10-console-devtrain-result.json`.
- **Autopilot ops for every setting the GUI exposes:** `set_routing`,
  `set_sampling`, `set_kv_reuse`, then `set_taesd` and `set_system_prompt`. The
  Settings dialog is unreachable on a Dev Mode console (no text input path), so
  these preferences had no automated coverage and the harness had to write
  `settings.json` behind the app's back — which proves nothing about the UI
  writer. The ops drive the real controller state and persist through
  `SaveSettings()`. The `validate-console.sh settings` gate seeds the **opposite**
  of every target, replays the ops and asserts all nine values, so an op that
  silently does nothing fails rather than inheriting a matching baseline
  (validated on Xbox Series S, MSIX 1.4.0.633).
- **Time-to-first-token surfaced in the UI (#139).** The app streams tokens, so
  TTFT is the latency the user actually feels; it was measured
  (`InferenceResult::t_p_eval_ms`) but never shown. The completion line now leads
  with "N.Ns to first token"; the live tok/s counter measures decode from the
  first token instead of from turn start (it was dividing decode tokens by prefill
  time and under-reporting by more than half on long prompts); and the status
  reads "reading prompt" during prefill instead of "generating", which was not yet
  true. See `uwp-constraints.md §5d`.
- **`max_length` in the bench CSV.** On DirectML it is the variable that governs
  prefill throughput, and `n_ctx` does not stand in for it — a control run at
  `n_ctx` 3072 holding `max_length` fixed reproduces the slow figure to the
  digit. The first eight rows of the experiment that found this differed only in
  `prompt_tok_s` and had to be re-measured rather than annotated.
- **CLI sampling flags `--top-p`, `--top-k`, `--repetition-penalty`, `--system`.**
  See the Fixed entry below — the flags were the visible half of a deeper split.

### Changed

- **Routing threshold 600 → 1550 tokens, measured — then found to be calibrated
  on the wrong variable.** A sweep of 10 prompt lengths on the shipping `-v2`
  asset (`scripts/bench-prompt-sweep.sh`, `bench/results/phase12-dml-crossover.csv`)
  showed the DirectML prefill curve is not monotone and retuned the threshold to
  1550 on the reading that the slowdown was a band in **prompt length**.

  It is not. Holding one 1289-token prompt byte-identical and varying only
  `n_predict` swings prefill from 131 to 612 tok/s
  (`bench/results/phase12-maxlen-band.csv`): the controlling variable is
  `max_length = min(n_ctx, n_prompt + n_predict)`, and the valley is interior to
  it, roughly 1400 to `n_ctx`, deepest near 1800. The sweep's measurements stand;
  the interpretation does not, and `token_threshold` is under re-derivation. See
  `docs/uwp-constraints.md` §5b (superseded in part) and §5c.

- **`Session::generate` saturates `max_length` at `n_ctx`** and bounds generation
  with the `n_predict` cap in `run_decode`, which is what the KV-reuse chat path
  already did. The shipping default (`n_predict` 256) put a 1289-token prompt at
  `max_length` 1545 — inside the valley, 150 tok/s where 612 was available.
  Faster and, measured, slightly less memory.

- **CPU threading recommendation 4 → 6 (`docs/recommended-config.md`, §5f).** The
  previous 4 came from a 2026-05-23 sweep whose every row had `prompt_tok_s = 0`,
  i.e. a decode optimum. Measuring prefill too, 6 leads by +8.5% at 1380 tokens
  with no decode cost. `t8` is worse than the "bandwidth saturation" it was
  recorded as — it takes prefill down 2.3× as well. Caveat in the doc: the shipped
  asset sets **no** `intra_op_num_threads` at all, so neither value has ever been
  in production.

- **Device marker training recipe.** The host marker gate converges with
  learning rate 2e-4 (5e-4 oscillated and under-converged), the shortened
  `XLLAMA-LORA-OK.` eval target, and `checkpoint_every: 2` for early-stop; the
  console harness learning rate is aligned to 2e-4.

### Fixed

- **The GUI/API ran the full sampler at temperature 0 (#141).** `OrtSession::make_params`
  set `temperature`/`top_p`/`top_k`/`repetition_penalty` unconditionally — no
  greedy branch — so a `temperature == 0` request ran the repetition penalty
  before the argmax and did not return the argmax token. The GUI slider min is
  `0.0` and the LAN API forwards `temperature`, so it was reachable on a
  DML-routed model. The CLI/bench ORT path and both llama.cpp paths already
  guarded this via `SamplingConfig::is_greedy()`; `OrtSession` did not. It was a
  regression from #136, which unified the llama.cpp chain but left the ORT search
  params hand-duplicated across `run_inference_ort` and `make_params`, which then
  diverged on exactly this guard. Fixed with the ORT twin of `sampler_chain.h` —
  a shared `apply_ort_sampling()` both callers use, greedy branch inside — so the
  two ORT surfaces can no longer diverge by construction. Verified on console
  (temp 0, repetition penalty 2.0, DML-routed): the buggy build produced garbage
  where the argmax was diverted, the fixed build returned the clean argmax.

- **Auto GPU routing was unreachable.** `BuildPrompt` estimated tokens as
  `chars / 4` and dropped turns above 1800; `decide_routing` compared the real
  tokenizer count against 1550. English prose measures ~5.34 chars/token, so the
  estimate overshot by ~30% and the trimmer actually cut at **~1348 real
  tokens** — below the threshold. No prompt could reach the GPU. Introduced by
  the 600 → 1550 retune above and invisible until `validate-console.sh routing`
  ran for the first time since; nothing related the two constants. They now live
  together with a test pinning the relation, verified to fail when the threshold
  is raised back over the ceiling.

- **The CLI and the GUI ran different samplers.** `run_inference` built
  `temp → dist`; `Session` built `penalties → top_k → top_p → temp → dist`. The
  CLI had no `top_p`, `top_k` or repetition penalty at all, so a generation seen
  in the GUI could not be reproduced from the command line even with identical
  values — same model, prompt and seed gave `" Paris, but if you were to visit,
you would return home! Particularly beyond Thetatweil, Arc de"` against
  `" Paris."`. On the ORT path the CLI set neither `top_p` nor `top_k` outside
  greedy mode, so `genai_config.json` won: a third configuration nobody chose.
  Nothing caught it because every marker gate runs `--greedy`, which bypasses
  sampling. Defaults and the chain builder now have one home each, with an
  opt-in end-to-end test that runs both surfaces and compares the text.

- **`bench-xbox-ort.sh` appended device rows without checking their arity.** An
  MSIX older than the CSV schema writes fewer fields; they landed under the
  current header and every column shifted silently (`host` under `n_gen_tok`,
  `date` under `host`). Observed with 1.4.0.615. `assert_header_matches` only
  guarded the local file.

- **`profile-dml-run.sh` inherited a previous sweep's configuration.**
  `bench-xbox-ort.sh` only ever overwrites `bench_ctx.txt` / `bench_npredict.txt` /
  `bench_maxlen.txt`, never deletes them, so a profile run after a sweep silently
  profiled the wrong `n_ctx` / `n_predict` / `max_length`. The profiler and
  `bench-xbox-kv.sh` now clear all three; `max_length` matters most, since it is
  the DirectML prefill variable (#130). (Extended for `bench_maxlen.txt` in #142.)

- **`bench-xbox-ort.sh --threads` mutated the device permanently.** It overwrote
  the on-device `genai_config.json` with no restore, so the last thread variant
  stayed in force for every later run of the app and every bench that did not
  pass `--threads` — a t8 sweep left the console on t8. It now backs up and
  restores on any exit, mirroring `profile-dml-run.sh`; `--keep-config` opts out
  (#142).

- **`deploy.sh pfn` returned an arbitrary package version.** It took the first
  match with no version ordering, and an MSIX upgrade can leave two versions of
  the same family registered (observed: 1.4.0.606 and 1.4.0.615 both live for
  over 6 minutes). Every tool built on it could silently drive the wrong build —
  it nearly invalidated the sweep above. Now picks the highest version and warns
  on stderr.
- **`prompt.txt` was truncated at 8 KB without a diagnostic** in the headless
  bench path, cutting prompts at ~2k tokens — exactly the range the sweep
  needed. `main_loop` had a private copy of the read loop; it now uses
  `read_local_file`, which already read to EOF.

- **Device training (Lane B) evaluate could not load the merged model.** The
  in-process evaluate stage opens the merged GGUF by an absolute `out_dir`
  path, but `resolve_model_path` (UWP) unconditionally prepended
  `LocalState\models\`, doubling the path and failing the load — training,
  export and merge all succeeded, only the eval verdict was blocked. Absolute
  paths (drive-letter / UNC) now pass through unchanged. Surfaced by the first
  on-console device-train run (peak working set 1195 MB, well under the 3 GB
  gate).

## [1.4.0.0] - 2026-07-19

### Added

- **Single-op DML diagnostics** (`oprepro.flag`, `uwp/op-repro.cpp` +
  `scripts/make-op-repro.py` / `make-chain-repro.py` /
  `validate-op-repro.sh`): CPU-vs-DML run of an arbitrary static-shape ONNX
  model on the console, with a per-run graph-optimization knob — built to
  chase #91 (verdict: the RMSNorm kernel is correct standalone; the
  corruption needs the GenAI-DML execution context — reported upstream on
  microsoft/onnxruntime#29739) and kept as a reusable driver-bug harness.

### Fixed

- **#91 root cause found and fixed — GPU text routing re-enabled.** The broken
  kernel on the Series S DML driver is `(Skip)SimplifiedLayerNormalization`
  (RMSNorm), not the attention path chased by #91/#94/#107: decomposing only
  those 65 nodes into primitives (`scripts/decompose_attention.py
--skip-attention --also-skipln`) yields correct text logits with the fused
  GQA attention, the stock config and the shipping pinned DLLs (on-console
  parity NMSE 1.72e-02, top-1 " Paris"; escalation matrix in
  `docs/dml-rmsnorm-fix-runbook.md`). The fixed asset ships as
  **`smollm2-360m-dml-fp16-v2`** (data-only fix, +0.8 MB) and
  `routing_policy.h` replaces the `kDmlTextLogitsBroken` hard gate with the
  `dml_text_model_ok` per-model allowlist — Auto routes >600-token prompts to
  the GPU again (measured 234 tok/s prefill / 43.9 decode / 1215 MB peak),
  every other `gpu_model` still resolves to CPU. `validate-console.sh` §2
  asserts the restored auto→gpu/auto→cpu split; `provision-models.sh` ALL_TEST
  seeds the `-v2` asset.

### Changed

- LAN API model discovery (`/v1/models`, `/api/tags`) now lists **every
  servable on-device model** (base GGUF or ORT layout under
  `LocalState\models\`) instead of only the loaded one; the non-standard
  `"active": true` flag marks the model the Session currently serves. Any
  listed id is valid as the request `model` (the server already switched
  Sessions on demand). Readiness predicate `model_dir_files_ready` is pure and
  host-tested.

## [1.3.0.0] - 2026-07-17

### Added

- **Expanded product UI** — per-response Like/Dislike/Correct actions backed by
  the preference JSONL pipeline; live LAN API enable/disable/port/status in
  Settings; reproducible image seed control; read-only runtime LoRA status.
- **Experimental Lane B partial fine-tuning** — C++17 ggml-opt engine for a
  fail-fast last-block tensor subset, declarative `partial_ft` jobs, Linux CLI
  dispatch, UWP `train.flag` headless driver and console validation harness.

### Changed

- LAN API listener now has explicit start/stop/rebind lifecycle and releases its
  API-owned Session after active work completes. Conversation JSON records the
  immutable feedback label for each rated assistant response.
- Preference correction samples train on `preferred_assistant`; device training
  uses the platform thread cap and rejects tensor selections unsupported by the
  pinned llama.cpp backward graph. Lane B results include wall time and peak
  working-set evidence for the host/console acceptance gate.

## [1.2.1.0] - 2026-07-17

### Added

- **Training pillar (Phase 8 exploration, frozen complete)** — dual-pillar
  architecture, RE capability matrix, host PEFT LoRA pipeline, runtime LoRA
  (`--lora` / catalogue `lora`), preference capture (`rate` → samples.jsonl),
  console harness `validate-console-training.sh`. SSOT:
  `docs/training-architecture.md`.
- **Phase 9 hybrid ops (operator)** — `pull_console_samples.sh`, job
  `from-console-samples.json`, publish `manifest.override.json` snippet after
  host merge.
- **GGUF dir resolve prefers `model.gguf`** over `adapter.gguf`.

### Fixed

- **TAESD after reinstall** — provision full `sd-turbo-fp16`; preflight te in
  `validate-console.sh`. Console **ALL PASS** on 1.2.0.546 (2026-07-17).

### Changed

- Semantic package version **1.2.1.0** (Revision CI-stamped). Phase 8 frozen.

### Measured (console, Series S, MSIX 1.2.0.546 → 1.2.1.x CI)

- `rate` / `lora-rt` / `serve` training gates **PASS**.
- `validate-console.sh all` **ALL PASS** after `provision-models.sh sd-turbo-fp16`.

## [1.2.0.0] - 2026-07-16

### Note

Content that landed after the v1.2.0.0 GitHub tag but before 1.2.1 is summarized
above under **1.2.1.0**. Historical 1.2.0.0 release notes follow.

### Changed (pre-1.2.1 tree notes)

- **Documentation lifecycle refactor** — current guidance now reflects unified
  shipping, the LFM default, completed Phase 7 H1/H9 work and the opt-in
  `--bench` install behavior. ROADMAP and operational runbooks contain only
  active work; completed Phase 1, console, external-data, demo and project-
  analysis material is reduced to stable entrypoints backed by Git history.
- **Docs accuracy pass (CodeRabbit on #101)** — §2 runbook claims Auto-only
  coverage (matches `validate-console.sh routing`); H8 requires explicit
  `MODEL=`/`golden` for DML parity (not the cpu-int4 default); `using-the-app`
  documents `gpu_available` fallback after the gate lifts; validation date in
  `recommended-config` set to 2026-07-16 / `1.2.0.534`; `#91` escaped in
  model-selection; H2 “mid-speed” hyphen.
- **DML text routing disabled (#91)** — the logit-parity harness caught the DML
  GQA decoder computing numerically wrong logits on the Series S GPU (NMSE ~1
  vs the CPU reference at fp16 AND int4; invariant to `ORT_DISABLE_ALL`,
  `ep.dml.disable_graph_fusion` and a capture-disabled GenAI DLL; the same
  weights are correct on CPU EP; SD-Turbo — decomposed attention, no contrib
  ops — is correct on the same device). The #94 probe later showed the
  **MultiHeadAttention** kernel is equally broken (same NMSE ~0.98 as GQA), so
  the fault is the DML attention path on this driver, not one op.
  `decide_routing` now forces the CPU model in every mode
  (`kDmlTextLogitsBroken`, `routing_policy.h`); GPU-routed answers had been
  silently degraded since the routing feature shipped. Re-enable gate:
  `scripts/validate-logit-parity.sh` PASS on a DML text model. Diffusion (plain
  ORT, validated) is unaffected. `validate-console.sh routing` now asserts the
  gate holds. Probe log in #91.
- **Upstream GenAI PR toward the DML re-enable path** —
  [microsoft/onnxruntime-genai#2300](https://github.com/microsoft/onnxruntime-genai/pull/2300):
  graph-capture opt-out via the `dml` provider option
  (`enable_graph_capture: "0"`) + zero-length KV placeholder clamp, both proven
  on-device during the #94 probe (fork branch
  `upstream-pr/dml-graph-capture-optout`). Closes #97. Driver-side track:
  microsoft/onnxruntime#29739.

### Fixed

- **Catalogue download resilience** — `ModelDownloader` retries transient failures
  (network / HTTP 5xx / 429, up to 3 attempts with 1s/2s backoff) and **skips**
  files already on disk at the expected `approx_bytes` (WDP upload or prior
  partial EnsureModel). Closes the HF **504** loop seen on `llama32-3b` when
  weights were already present.
- **Bench CSV quant label** — GGUF rows no longer hardcode `Q4_K_M`; quant is
  derived from the path / first `.gguf` filename (`Q3_K_S`, `IQ2_M`, …). Fallback
  remains `Q4_K_M` when no token is found.
- **`bench-xbox-ort.sh`** — does not upload ORT `genai_config.json` into GGUF
  model dirs when `--threads` is set (threads still via `bench_threads.txt`).
- **`gpu_model` no longer auto-downloaded under the #91 gate (#98, closes #95)**
  — `EnsureGpuModelIfNeeded` skips the 725 MB `smollm2-360m-dml-fp16` download
  while `kDmlTextLogitsBroken` holds (routing can never select it); the
  catalogue display now reads "GPU fp16, routing — disabled #91".
- **Missing `gpu_model` no longer blocks the turn under the #91 gate (#100)** —
  the `StartInference` pre-checks that errored-and-returned on a missing
  `gpu_model` (now the normal state after #95) are bypassed while the gate
  holds. `validate-console.sh routing` precondition switched to
  `smollm2-360m-cpu-int4`; `provision-models.sh --all-test` seeds `cpu-int4`
  instead of the gated `dml-fp16`.

### Docs

- **Phase 6 demo video published** — ~74 s on-console clip (LFM chat ×2 +
  SD-Turbo pixel-art robot) captured via WDP `/ext/screenshot` + autopilot on
  package `1.2.0.536`. Assets:
  [xllama-demo-v1.2.0.mp4](https://github.com/gianlucamazza/xllama/releases/download/v1.2.0.0/xllama-demo-v1.2.0.mp4)
  - still on release v1.2.0.0; local copies under `docs/screenshots/`. Tooling:
    `scripts/capture-demo-video.sh`, `docs/demo-video-runbook.md`. ROADMAP Phase 6
    product **DONE**. First-launch default in `using-the-app.md` aligned to
    LFM2.5-350M (unified).
- **Publishing runbook for ORT model assets (#99, closes #96)** — "Publishing
  ORT model assets (models-v1) — logit-parity gate" in
  `docs/model-selection.md` (+ CONTRIBUTING pointer): no ORT text asset is
  uploaded without passing the parity gate (llama.cpp golden → host CPU-EP
  compare → on-device `validate-logit-parity.sh` for DML → exact
  `approx_bytes`).
- **Project analysis currency 2026-07-17** — `docs/project-analysis-2026-07.md`
  rewritten for semantic **1.2.0.0**, the #91 DML text gate, LAN API, Phase 7
  H4, ~88 host tests; post-release pass marks **v1.2.0.0 Latest** done.
  `docs/vendor-lifecycle-plan.md` + ROADMAP aligned (R0 done, R8 hold gate);
  `docs/README.md` SSOT row updated.

### Added

- **GGUF dir resolve prefers `model.gguf`** over `adapter.gguf` (runtime LoRA catalogue dirs).
- **Console P0 training closeout** on MSIX 1.2.0.546: `rate` PASS, `lora-rt` PASS, `serve` PASS; `validate-console all` routing+GGUF PASS (TAESD FAIL post-wipe).

- **Phi-3 chat template** — `ChatFormatKind::Phi3` (`<|user|>` / `<|end|>`) via
  `model_is_phi` / `chat_format_for` for WDP-provisioned Phi GGUFs and future
  quality A/B. Not a catalogue entry (speed/RAM still loses to `llama32-3b`).

### Measured

- **Phase 7 Phi-3.5-mini Q3_K_S A/B** (2026-07-16, Series S t6, `standard-512.txt`,
  median of 3 runs): decode **11.31 tok/s**, prefill 15.3, peak **2453 MB**, load
  ~24 s (`bench/results/phase7-scale.csv`). H4 gates PASS; loses to Llama-3.2-3B
  Q3 (14.16 tok/s / 1824 MB) on speed and RAM — **no catalogue entry**. Docs:
  `docs/phase7-hypotheses.md`, `docs/benchmarks.md`.
  **Note:** console field package was still `1.1.8.507` during the campaign;
  deploy `1.2.0.x` (+ this fix set) for catalogue download hardening and the Phi
  template on-device.
- **Full validation suite PASS on `1.2.0.534`** (2026-07-16, Series S, main):
  logit-parity cpu-int4 vs GGUF golden NMSE **0.094** / top-10 **0.90**, §2
  routing gate holds (no `auto → gpu`, CPU turn at 959 tok), GGUF chat, TAESD
  VAE **621.7 ms**, LAN API ALL PASS, and the #95 gate verified live (0
  `background provision` lines with `routing:2` and the DML model absent).
  Supersedes the `1.1.8.507` caveat above.

### Released

- **GitHub Release [v1.2.0.0](https://github.com/gianlucamazza/xllama/releases/tag/v1.2.0.0)**
  (2026-07-16) — MSIX `xllama_1.2.0.536_x64.msix` + cert + VCLibs x64; set as
  **Latest**. Console FULL PASS previously recorded on `1.2.0.534` (same
  feature set).

- **LAN HTTP endpoint (OpenAI-compatible)** — optional, **default OFF** front-end
  on `xllama::Session` (`uwp/api-server.{h,cpp}`, WinRT `StreamSocketListener`).
  Enable with `LocalState\api.flag`; serves `POST /v1/chat/completions`
  (non-streaming), `GET /v1/models` + `/api/tags` discovery, `GET /health`, and
  CORS preflight on port **11434** (`api-port.txt` override). Single-slot: one
  shared Session behind a `try_lock` → **503** when busy. Maps `messages[]` →
  `ChatFormat::render_prompt`; honors `max_completion_tokens`/`max_tokens`,
  `temperature`, `top_p`, `seed`, `stop`. Coexists with the live chat UI on a
  detached MTA thread; `privateNetworkClientServer` capability (already present)
  covers LAN inbound, no public inbound. **On-console (Series S, Dev Mode):** LAN
  bind + `GET /` 200 **PASS**; chat via `lfm25-350m` and `llama32-3b` returns
  correct OpenAI-shaped completions. Docs: `docs/api-endpoint.md`; validation:
  `scripts/validate-api.sh`.
- **Catalogue `llama32-3b`** — Llama-3.2-3B-Instruct Q3_K_S (~1.54 GB) from
  Hugging Face unsloth GGUF (optional advanced chat; default remains
  `lfm25-350m`). Console H4: **14.2 tok/s**, peak 1824 MB.
- **Llama-3 chat template** — `ChatFormatKind::Llama3` (`<|start_header_id|>` /
  `<|eot_id|>`) via `chat_format_for` / `model_is_llama`; KV-reuse invariant
  covered in unit tests.
- **Phase 7 research** — `docs/phase7-hypotheses.md`: peer-class model hypotheses
  (H1–H9). **H4 PASS** on console: Llama-3.2-3B-Instruct Q3_K_S decode
  **14.2 tok/s**, peak 1824 MB (`bench/results/phase7-scale.csv`); near Gemma-4-E2B
  speed at ~900 MB less RAM. Phi-3.5-mini Q3 A/B measured same day (see
  Unreleased).
- **GitHub Release [v1.1.8.0](https://github.com/gianlucamazza/xllama/releases/tag/v1.1.8.0)**
  (2026-07-16) — MSIX + cert + VCLibs x64. **Field smoke same day:** install
  `1.1.8.496` on Series S, first-launch download of `lfm25-350m`,
  `validate-console.sh gguf` → **PASS** (llama.cpp load + short chat).

### Changed

- **Patched GenAI DLL is now hash-pinned** like PatchedOrt: shipping
  `build-uwp.yml` downloads `onnxruntime-genai.dll` from
  [`vendor-dlls-v1`](https://github.com/gianlucamazza/xllama/releases/tag/vendor-dlls-v1)
  and verifies `vendor/onnxruntime-genai-patched/SHA256SUMS` (no per-PR GenAI
  source rebuild). Refresh via `build-uwp-patched.yml` / `vendor-genai-dml-patch.ps1 -Build`.
  `vendor-genai-dml-patch.ps1` **fails closed** if the pin is missing (no silent
  vanilla NuGet).
- **Docs currency pass** — `#2280` is **merged** on Microsoft GenAI `main`
  (gap = NuGet 0.14.1 only); ORT `weakly_canonical` related fix
  [#28509](https://github.com/microsoft/onnxruntime/pull/28509) on ORT `main`
  (not in NuGet 1.24.4). Updated `uwp-constraints.md`,
  `project-analysis-2026-07.md`, `patches/README.md`, vendor READMEs.
  SSOT issues: [#84](https://github.com/gianlucamazza/xllama/issues/84),
  [#85](https://github.com/gianlucamazza/xllama/issues/85),
  [#86](https://github.com/gianlucamazza/xllama/issues/86).
- **`scripts/check-vendor-nuget-status.sh`** — poll NuGet.org vs pins; reports
  whether PatchedGenAI / PatchedOrt can be dropped.
- **Upstream ORT PR** for ReadFile 16 MB chunk:
  [microsoft/onnxruntime#29732](https://github.com/microsoft/onnxruntime/pull/29732)
  (issue [#29730](https://github.com/microsoft/onnxruntime/issues/29730)).

## [1.1.8.0] - 2026-07-15

### Added

- **GGUF dir resolve prefers `model.gguf`** over `adapter.gguf` (runtime LoRA catalogue dirs).
- **Console P0 training closeout** on MSIX 1.2.0.546: `rate` PASS, `lora-rt` PASS, `serve` PASS; `validate-console all` routing+GGUF PASS (TAESD FAIL post-wipe).

- **Patched ORT DLL in shipping MSIX** — AppContainer external-data fixes
  (`weakly_canonical` guard + 16 MB `ReadFile` chunk) are no longer dispatch-lane
  only. `build-uwp.yml` downloads the console-validated `onnxruntime.dll` from the
  [`vendor-dlls-v1`](https://github.com/gianlucamazza/xllama/releases/tag/vendor-dlls-v1)
  release, verifies `vendor/onnxruntime-patched/SHA256SUMS`, and installs it over
  NuGet (`-PatchedOrt` in `build-uwp.ps1`). Full ORT source rebuild stays in
  `build-uwp-ort-patched.yml` (1–3 h) for pin refresh. Runbook:
  `docs/fp16-extdata-runbook.md`.
- **`smollm2-1.7b-cpu-int4` published to the `models-v1` catalogue** — the four
  assets (incl. the 1.47 GB `model.onnx`) are on the `models-v1` release (in-app
  download, ~20.6 tok/s). Verified on-console.
- **Architecture overview** — `docs/architecture.md` (SSOT for system structure).
- **External-data ONNX loading unblocked >2 GB on AppContainer** (console-validated
  2026-07-15; now shipped in 1.1.8.0). USB spike refuted; 1B fp16 GPU inference
  closed negative (budget wall). CSV: `bench/results/phase6-fp16-extdata.csv`.

### Changed

- **First-launch default chat model on unified builds** is now **`lfm25-350m`**
  (measured fastest+lightest: 94.2 tok/s, ~219 MB). ORT-only builds still default
  to `smollm2-360m-cpu-int4`. Aligns `MainPage` `DefaultChatModelId()`,
  `bench/configs/settings-modern.json`, and `docs/recommended-config.md`.
- Publication venue decision: **GitHub Discussions** first for the technical
  report (arXiv only if a formal citation is needed).

## [1.1.7.0] - 2026-07-15

### Added

- **GGUF dir resolve prefers `model.gguf`** over `adapter.gguf` (runtime LoRA catalogue dirs).
- **Console P0 training closeout** on MSIX 1.2.0.546: `rate` PASS, `lora-rt` PASS, `serve` PASS; `validate-console all` routing+GGUF PASS (TAESD FAIL post-wipe).

- **GGUF KV-cache reuse** — `LlamaSession` keeps a persistent `llama_context`, so
  a continuation turn appends only the new turn's delta instead of re-prefilling
  the whole conversation. Console-measured **4.07×** turn-2 prefill on gemma3-270m
  (`bench/results/phase6-gemma-kv.csv`); previously GGUF was stateless. Enabled in
  the Settings KV toggle (routing stays ORT-only — llama.cpp UWP is CPU-only).
- **CI: `src/models/*.cpp` MSBuild wildcard** in `ggml-uwp.vcxproj` + a
  `scripts/check-uwp-sources.sh` drift-check — new llama.cpp architectures no
  longer break the UWP link on a submodule bump (as `657e011` did).
- **Manifest per-entry merge factored + unit-tested** — extracted the
  LocalState-override merge from `LoadModelManifest` into a pure, header-only
  `xllama::merge_manifest_entries` (`include/xllama/manifest_merge.h`) with 7
  host doctest cases (`tests/test_manifest_merge.cpp`) covering replace / append /
  preserve, incl. the 2026-07-10 whole-catalogue-shadow regression.
- **CPU memory-bandwidth micro-bench** — `xllama::measure_membw` (STREAM-style
  read/copy/triad, `include/xllama/membw.h`) with `xllama-cli --membw` (host) and a
  `membw.flag` headless mode (console → `membw-result.csv`). Pins the DRAM-bandwidth
  ceiling behind the bandwidth-bound decode number. 4 host doctest cases.
  Console-measured 2026-07-15: Xbox Zen 2 read 12.35 GB/s @1t / 30.29 @8t — the
  single-thread read matches the deduced ~13 GB/s GEMV denominator.
- **In-app HuggingFace download verified on-console** (2026-07-15) — the app
  self-downloaded the 2.29 GB single `.gguf` for `gemma4-e2b` from HF and
  loaded+generated it, confirming the >2 GB self-download path (not just Device
  Portal provisioning). See `docs/benchmarks.md`.

### Changed

- **`gemma4-e2b` catalogue default: UD-IQ2_M → Q3_K_S** (2.45 GB). Console-measured
  15.3 tok/s and generates full responses on long declarative prompts, where the
  2-bit IQ2_M collapsed to an immediate EOG. See `docs/benchmarks.md`.
- **llama.cpp submodule bumped to `657e011`** (was `9a532ae4b`); `ggml-uwp.vcxproj`
  gained the new per-arch sources.
- **Stop-sequence handling unified** into one suffix-match helper
  (`apply_stop_sequences`, `chat_prompt`) shared by the llama and ORT decode loops.
- **`scripts/install-latest-build.sh`**: `bench.flag` is now `--bench` opt-in — a
  plain install launches into the UI.
- **Prefill micro-batch knobs exposed** — `n_batch`/`n_ubatch` plumbed through
  `InferenceParams`/`SessionParams` into `llama_context_params`, with
  `xllama-cli --batch/--ubatch` and a `scripts/bench-ubatch-sweep.sh` sweep helper.
  Host sweep found no reproducible ubatch win (noise-dominated); default (llama.cpp 512) unchanged. See `docs/benchmarks.md`.
- Fixed two more stale in-code comments: `ManifestEntry` (gguf KV-reuse now
  enabled) and `GenerateParams` reuse_kv/reset_kv (honored by both backends now,
  not ORT-only).
- Consolidated documentation onto single-source-of-truth docs (perf →
  `benchmarks.md`, constraints → `uwp-constraints.md`, catalogue →
  `model-selection.md` + `manifest.json`); refreshed stale version/quant/KV claims.

### Fixed

- **Model provisioning now auto-upgrades a stale quant** — `IsModelProvisioned`
  gained an expected-aware overload (compares the dir against the manifest's
  current `files[].filename` via the new pure `dir_satisfies_expected_files`,
  `include/xllama/model_provision.h`), and `EnsureModelNamedAsync` loads the
  manifest before the provisioned-check and reconciles the dir (deletes any
  non-expected `*.gguf` + drops the stale `.complete`) before re-downloading. A
  directory holding an older `.gguf` than the manifest names (e.g. a stale IQ2_M
  under `gemma4-e2b`) is no longer treated as provisioned, and the coexisting-file
  hazard in `first_gguf_in_dir` is closed. `EnsureGpuModelIfNeeded` is
  expected-aware too. 13 host doctest cases (`tests/test_model_provision.cpp`).
- Fixed a stale comment in `MainPage.cpp::StartInference`: GGUF KV-cache reuse is
  supported (persistent `llama_context`), gated by `kv_reuse_supported_for_model`;
  only EP routing stays gated off for GGUF.
- `scripts/bench-xbox-ort.sh`: verify `bench-result.csv.done` is actually deleted
  before a run, so `wait_for_done` can't return off a stale marker (silent wrong row).

### Investigated (no change shipped)

- **AppContainer file mmap** (`CreateFileMappingFromApp`/`MapViewOfFileFromApp`,
  with a loader fallback): built and deployed, but **no measured benefit** —
  GGUF load on CPU is dominated by the AVX2 tensor repack, not the file read.
  Reverted; finding recorded in `docs/benchmarks.md`.

---

## [1.1.6.0] - 2026-07-14

### Added

- **GGUF dir resolve prefers `model.gguf`** over `adapter.gguf` (runtime LoRA catalogue dirs).
- **Console P0 training closeout** on MSIX 1.2.0.546: `rate` PASS, `lora-rt` PASS, `serve` PASS; `validate-console all` routing+GGUF PASS (TAESD FAIL post-wipe).

- **Per-architecture chat template** (`ChatFormat`, `src/bridge/chat_prompt.cpp`):
  `chat_format_for()` replaces the hard-coded ChatML with a data-driven template
  selected by model id. ChatML kept byte-identical (Qwen no-think suffix
  preserved); **Gemma** added (`<start_of_turn>…<end_of_turn>`, no system role,
  stop `<end_of_turn>`). UWP UI + bench call one abstraction.
- **Gemma family — console-validated** (`phase6-gemma.csv`). Catalogue entries
  `gemma3-270m` (253 MB) and `gemma4-e2b` (2.29 GB IQ2_M), GGUF from HF per Gemma
  Terms. On Xbox Series S: gemma3-270m **76.8 tok/s** decode (368 MB); gemma4-e2b
  loads at 2534 MB RAM, **9.9 tok/s**. The "Gemma-4 too big" verdict is
  **overturned** — the ~2 GB Dev Mode per-file limit does not apply to GGUF.
- **`--chat` flag** for `xllama-cli` + stop-sequence support in the llama path
  (`InferenceParams::stop_sequences`), so the CLI/GGUF bench stop on the chat
  format's stop token.
- **Benchmark report** — `docs/benchmarks.md` + self-contained
  `docs/benchmarks-charts.html` consolidate every tested model across backends,
  with root-cause notes on the negative performers.
- **Automated MSIX versioning** — CI stamps the Revision from the workflow
  `run_number` (`build-uwp.ps1 -BuildRevision`); no manual per-build bump.
- **Repo automation** — CI concurrency + `push`-on-`main` only (one build per
  PR, superseded runs cancel), Dependabot (actions + llama.cpp submodule), CodeQL
  (c-cpp), PR/issue templates, `CONTRIBUTING.md`.

### Changed

- Package version **1.1.6.0** (Revision auto-stamped in CI).
- `docs/model-selection.md` Gemma verdict revised (disk no longer the binding
  constraint after the 90 GB Dev Mode bump; E2B console-validated).

---

## [1.1.5.0] - 2026-07-14

### Added

- **GGUF dir resolve prefers `model.gguf`** over `adapter.gguf` (runtime LoRA catalogue dirs).
- **Console P0 training closeout** on MSIX 1.2.0.546: `rate` PASS, `lora-rt` PASS, `serve` PASS; `validate-console all` routing+GGUF PASS (TAESD FAIL post-wipe).

- **`include/xllama/chat_prompt.h`** — Qwen no-think generation suffix and empty
  `</think>` stripping; unit tests in `test_chat_prompt.cpp`.
- **`kv_reuse_supported_for_model()`** in `routing_policy.h` — DML EP gate for
  continuous decoding.

### Fixed

- **GPU/DML multi-turn**: KV reuse disabled on `*-dml-*` models — ORT GenAI rejects
  `AppendTokenSequences` on a persistent DirectML generator (_"Continuous decoding is
  not supported on the selected device type (DirectML)"_); avoids per-turn fallback
  and spurious failures.
- **Qwen3.5 GGUF**: append Qwen no-think prefill after `<|im_start|>assistant` and
  strip leading empty think blocks from saved/displayed assistant text.
- **`validate-console.sh routing`**: remove `ap-routing-longctx` decoy chat after the
  test so the seeded _"Understood; ready to continue."_ assistant turn does not linger
  in History.

### Changed

- Package version **1.1.5.0**.

---

## [1.1.4.0] - 2026-07-14

### Added

- **GGUF dir resolve prefers `model.gguf`** over `adapter.gguf` (runtime LoRA catalogue dirs).
- **Console P0 training closeout** on MSIX 1.2.0.546: `rate` PASS, `lora-rt` PASS, `serve` PASS; `validate-console all` routing+GGUF PASS (TAESD FAIL post-wipe).

- **`include/xllama/routing_policy.h`** — extracted per-workload routing decision
  (600-tok threshold, GGUF/routing capability gates) with unit tests.
- **`scripts/package-catalogue-ort-model.sh`** — stage flat `models-v1` assets for
  `smollm2-360m-dml-fp16` and `smollm2-1.7b-cpu-int4` catalogue entries.
- **Catalogue download URLs** for `smollm2-360m-dml-fp16` and `smollm2-1.7b-cpu-int4`
  (`hf_base_url` + prefixed `remote` names on `models-v1`).

### Changed

- **Shipping CI default** (`build-uwp.yml`): `xllama-appx` is now **unified +
  PatchedGenAI #2280**; `llamacpp` remains a bench-only artifact. `build-uwp.ps1
-PatchedGenAI` fails closed if the vendor step fails.
- **Recommended modern settings** (`bench/configs/settings-modern.json`): default chat
  model **LFM2.5-350M** (GGUF) on unified builds; routing GPU unchanged.
- Package version **1.1.4.0**.

### Fixed

- **Model provisioning layer** (`IsModelProvisioned`, `EnsureModelNamedAsync`,
  background `gpu_model` download when `routing≠0`, routing guards, Settings
  model-change re-provision). See commit `f3d733e`.
- **`validate-console.sh` preflight/upload**: `model_provisioned` WDP check uses
  basename `genai_config.json`; `upload_file` mkdirs remote dirs before POST
  (chats + nested model paths).

### Measured (2026-07-14, unified patched on console)

- **`validate-console.sh all` → ALL PASS** (autopilot, no pad):
  routing A/B (959 tok GPU / short CPU, no `887A0036`), GGUF `lfm25-350m` via
  llama.cpp session, TAESD VAE **593–625 ms** (sub-second gate). Verified on
  **1.1.3.0** (full suite) and **1.1.4.0** (routing re-check after CI deploy).
- **`smollm2-360m-dml-fp16` on `models-v1`** — ORT GenAI builder `-p fp16 -e dml`,
  691 MB merged self-contained; catalogue in-app download operational.

### Docs

- **Drift pass after the Fase-2b/patched-lane work**: README (GGUF rows with
  measured decode, 90 GB disk supersession, in-process image gen in the
  architecture diagram, helper listings), `docs/uwp-constraints.md` intro
  aligned to the §9 supersession, `docs/recommended-config.md` (GGUF measured
  - llama thread cap + sharper "do not use"), runbook §2 (CI-lane artifact
    alternative; settings must upload as `settings.json`), vendor README +
    script message (rel-0.14.1 branch pin, ORT_HOME build), AGENTS.md map.

### Fixed

- **Stale `LocalState\manifest.json` override no longer hides the catalogue**:
  the override used to replace the bundled catalogue _entirely_, so an old
  override (e.g. the Exp-2-era single-entry file found on console 2026-07-10)
  made `sd-turbo-fp16` and the GGUF entries invisible — the Image dialog
  failed with "catalogue entry missing hf_base_url". `LoadModelManifest` now
  **merges per entry**: same-name entries replace bundled ones, new names are
  appended, unmentioned bundled entries stay.

- **Headless bench could not load catalogue GGUF models**: `run_inference_llama`
  passed the resolved model _directory_ straight to
  `llama_model_load_from_file`, which needs the `.gguf` _file_ — every bench
  run of `qwen35-0.8b` / `lfm25-350m` failed on console (found 2026-07-10 on
  the first unified-build bench). The chat path (`create_llama`) already
  descended into the directory; that logic is now the shared
  `first_gguf_in_dir()` (`path_utils`) used by both, with host tests.
- **llama.cpp default thread count livelocked on console**: with no explicit
  `n_threads`, both llama paths used `detect_threads()` (= 8 on Series S),
  hitting the known ggml spin-wait livelock at t7/t8 (phase35-llamacpp-scaling;
  re-hit 2026-07-10 — bench runs loaded the model then hung past the 300 s
  watchdog). New `detect_threads_llama()` caps the llama default at 6 on UWP
  (t6 is the measured optimum); explicit `n_threads` still wins, ORT paths
  unchanged.
- **Bench CSV mislabelled GGUF runs in unified builds** as
  `int4`/`ort-genai-cpu`: the label block keyed on compile-time
  `XLLAMA_USE_ORT` only. It now keys on the backend that actually ran
  (`model_uses_llama_backend`), keeping the llamacpp-lane labels
  (`Q4_K_M`/`cpu`), and the CSV `n_threads` column reports the llama-capped
  default.

### Measured (2026-07-10, unified 1.1.1.0 on console)

- **Fase 2b GGUF benches** (`bench/results/phase5-gguf.csv`, t6,
  standard-512): **Qwen3.5-0.8B Q4_K_M decode 35.1 tok/s** (prefill 98.1,
  peak 718 MB, load 4.1 s) — **LFM2.5-350M Q4_K_M decode 94.2 tok/s**
  (prefill 241.4, peak 321 MB, load 1.4 s). LFM2.5-350M beats the ORT int4
  360M baseline (66.3 @ t8) by ~42% at similar size; Qwen3.5-0.8B is in line
  with size scaling (1.7B CPU int4: 20.6). Unified promotion is now
  bench-unblocked.

### Changed

- **`diffuse.cpp` is timestep-shape-aware**: the UNet `timestep` tensor is fed
  with the rank the model declares — `[1]` (optimum ≤ 1.23, the deployed
  artifacts, unchanged path) or scalar `[]` (optimum-onnx 0.1.0+ exports) —
  from the same 1-element buffer, mirroring `validate_pipeline.py`. Removes
  the last code blocker for promoting new-generation diffusion artifacts;
  the remaining gate is runbook §7 on console. The unet log line now reports
  `ts rank 0|1`.
- **Diffusion export toolchain bumped** (host-only, never shipped): torch
  2.4.1→2.9.1, optimum 1.23.3→optimum-onnx 0.1.0, transformers 4.46.3→4.57.6,
  diffusers 0.31.0→0.39.0. Clears the actionable dependabot alerts (torch
  CVE-2025-32434, transformers ReDoS batch, diffusers CVE-2026-44513/45804);
  residual non-applicable alerts documented in `diffusion/requirements.txt`.
  Full recipe re-validated (export → fp16 convert → validate_pipeline → golden
  vectors byte-identical → ctest → TAESD). `validate_pipeline.py` is now
  timestep-shape-aware (new exports declare a scalar UNet `timestep`);
  `gen_golden_vectors.py` stamps installed versions. Deployed console
  artifacts are unchanged.
- **In-app image Generate runs in-process (no restart).** The Image dialog runs
  SD-Turbo on a background MTA thread (the path console-validated in §7b): live
  stage in the status bar via `diffuse-progress.txt` (200 ms poll), Cancel
  writes `diffuse-cancel.flag` (honored between UNet steps). The headless
  `diffuse.flag` path is kept for bench/WDP parity.
- **Routing `auto` counts real tokens.** New `Session::count_tokens()` (ORT
  tokenizer / llama_tokenize) replaces the `size/4` estimate; threshold updated
  to 600 tokens (crossover between ~285 and ~1050 in the v0.3.6 matrix). The
  routing decision is logged (`routing: auto → gpu/cpu (N tok, ...)`).
- UWP inference threads pinned to `t=4` (Series S optimum; `t=7/8` livelock).

### Added

- **GGUF dir resolve prefers `model.gguf`** over `adapter.gguf` (runtime LoRA catalogue dirs).
- **Console P0 training closeout** on MSIX 1.2.0.546: `rate` PASS, `lora-rt` PASS, `serve` PASS; `validate-console all` routing+GGUF PASS (TAESD FAIL post-wipe).

- **Console autopilot + `validate-console.sh`.** Dev Mode gives the console no
  working text-input path, so §2 routing / §7c TAESD / GGUF-chat validations
  needed a person at the pad. A flag-driven in-app driver (`autopilot.flag` →
  `App::OnLaunched`) now replays a JSON action list
  (`load_chat|send|new_chat|set_model|generate_image|quit`) against the same
  controller methods the buttons call — real XAML process, no UI code
  duplicated — writing `autopilot-done.txt` (`ok`/`error:`). The host
  orchestrator `scripts/validate-console.sh <routing|gguf|taesd|all>` uploads
  the script, restarts, polls, and emits deterministic PASS/FAIL from the log
  (no LLM judgment in the verdict). The §2 automated PASS is the official gate
  for promoting `-PatchedGenAI` to default. Package bumped to **1.1.3.0** (also
  carries the #44 manifest-merge fix on-device). `create_llama` now logs a
  distinct GGUF-session load marker.
- **GGUF assets in the catalogue (Fase 2b).** `Qwen3.5-0.8B-Q4_K_M.gguf`
  (508 MB, unsloth quant, Apache-2.0) and `LFM2.5-350M-Q4_K_M.gguf` (219 MB,
  LiquidAI official) published on `models-v1`; catalogue entries `qwen35-0.8b`
  and `lfm25-350m` now download in-app (unified builds). The LFM Open License
  v1.0 permits redistribution (§4) provided the license accompanies the work —
  `LICENSE.txt` is published on the release and downloaded alongside the model.
  Host smoke test: both load and generate via `xllama-cli`. On-console
  decode/prefill benches remain (bench-gated promotion to default).
- **Patched GenAI DLL pipeline** ([#2280](https://github.com/microsoft/onnxruntime-genai/pull/2280)):
  `patches/onnxruntime-genai-2280-dml-fallback.patch`,
  `scripts/vendor-genai-dml-patch.ps1`, and `build-uwp.ps1 -PatchedGenAI` to
  overlay the NuGet `onnxruntime-genai.dll` for XAML + DML chat routing.
- **CI lane for -PatchedGenAI packages** — default `build-uwp.yml` now ships
  `xllama-appx` as unified+#2280 (2026-07-14). `build-uwp-patched.yml` remains
  a manual dispatch fallback.
- **Recommended configurations** — [`docs/recommended-config.md`](docs/recommended-config.md)
  and [`bench/configs/settings-modern.json`](bench/configs/settings-modern.json)
  for console validation.
- **TAESD fast VAE** for image generation: toggle in the Image dialog downloads
  `sd-turbo-fp16_taesd_vae_decoder_model.onnx` (~5 MB) from `models-v1` over the
  full VAE in-place; setting `diffuse_taesd_vae` persists in `settings.json`.
  Host export: `scripts/export-taesd-asset.sh` (asset published on `models-v1`).
  Console-validated 2026-07-14 (runbook §7c, VAE **593–625 ms**).
- **GGUF catalogue plumbing (Fase 2).** Complete end-to-end support for
  `kind: "gguf"` catalogue entries:
  - New public helper `model_uses_llama_backend()` (suffix fast-path + resolve +
    `*.gguf` directory scan) makes `Backend::Auto` layout-aware. Bare catalogue
    names (e.g. "qwen35-0.8b") now correctly select the llama.cpp backend in
    unified builds.
  - `resolve_model_path` treats directories containing `*.gguf` as valid models
    (LocalState primary + USB fallback).
  - `run_kv_bench` (ORT-only) now guards GGUF models and emits a clear skip.
  - GGUF models are hidden from KV-cache reuse and EP routing in Settings
    (llama.cpp path is stateless + CPU-only on Xbox).
  - Unit tests added for the helper and Auto dispatch on suffix + directory
    layouts.
  - Example placeholder entry (`qwen35-0.8b`) added to `uwp/models/manifest.json`
    (manual provisioning until Fase 2b).

  Catalogue asset upload + on-console benches for Qwen3.5/LFM2 remain Fase 2b
  (bench-gated). See PR #30.

## [1.1.0] - 2026-07-09

**Image self-serve + faster in-app diffusion + dual-backend-ready build.** The
image model now downloads itself from the catalogue; diffusion is proven to run
in-process (no restart needed); and the binary can now compile both the ORT
GenAI and llama.cpp backends into one MSIX, chosen per model at runtime — the
groundwork for modern GGUF-only models (Qwen3.5, LFM2) the ORT builder can't
produce. Full measured story: `docs/technical-report.md`.

### Measured — plain ORT DML coexists with the XAML compositor (in-app diffusion)

The `887A0036` device conflict that forced headless image generation was
specific to ORT **GenAI**'s Agility device factory. `diffuse-inproc.flag` ran
the full SD-Turbo pipeline **on a background thread inside the live XAML
process** on console: coherent 512×512 PNG, compositor alive, no conflict, no
OOM — **total 5.57 s, faster than the 6.9 s headless run** (warm GPU). Image
generation no longer needs the restart flow (`docs/uwp-constraints.md` §7,
runbook §7b). Wiring the in-app Generate is the follow-up; the restart flow
stays as a fallback.

### Added — runtime backend dispatch (ORT GenAI ⊕ llama.cpp in one binary)

- **GGUF dir resolve prefers `model.gguf`** over `adapter.gguf` (runtime LoRA catalogue dirs).
- **Console P0 training closeout** on MSIX 1.2.0.546: `rate` PASS, `lora-rt` PASS, `serve` PASS; `validate-console all` routing+GGUF PASS (TAESD FAIL post-wipe).

`src/bridge/session.cpp` + `inference.cpp` no longer select the backend with a
mutually-exclusive `#ifdef XLLAMA_USE_ORT / #else`. Two independent capability
macros (`XLLAMA_USE_ORT`, `XLLAMA_USE_LLAMA`) let both backends compile together;
`Session::create` / `run_inference` become runtime dispatchers keyed on
`SessionParams::backend` (`Auto` infers `.gguf` vs ORT-dir layout). A new
`XllamaBackend=unified` UWP build links the static ggml lib alongside ORT and
ships both — CI-validated (`build (unified)` green). Single-backend builds stay
behaviorally identical. This unblocks modern GGUF-only models via llama.cpp
while ORT remains the default; UI wiring (`kind:gguf`) is the next phase.

### Evaluated — modern small models (host-validated; console benches pending)

Surveyed the post-Qwen3/Gemma3 landscape (the ORT GenAI builder is frozen at
those архitectures). Confirmed loading through our stack:

- **Qwen3.5-0.8B** (Feb 2026, Apache-2.0) — current-gen, Q4_K_M 507 MB, runs via
  our llama.cpp submodule (`qwen35` arch).
- **LFM2.5-350M** (LiquidAI) — hybrid edge, Q4_K_M 218 MB, runs via llama.cpp.
- **Qwen3-0.6B int4** — builds with the ORT GenAI builder (969 MB merged; the
  151k-vocab embedding dominates — heavier than SmolLM2-360M's 417 MB).
- **Gemma-3-270M** — gated on HF, build blocked pending a token.
- **TAESD decoder** (madebyollin/taesd, MIT) — a 4.9 MB drop-in VAE that replaces
  SD-Turbo's 94 MB decoder; `validate_pipeline.py --vae` confirms a coherent
  image. On console the VAE stage is 2.6 s of 6.9 s → TAESD targets ~4.5 s.
  Export recipe: `diffusion/export_taesd.py` (falsified the [0,1] output
  assumption; the diffusers decoder already emits SD `[-1,1]`).

### Removed — purpose-served legacy (image spike, GGUF-era bench, dead switches)

- **Image spike** (`uwp/image-spike.cpp`, the `image.flag` headless mode,
  `scripts/gen_imgspike_model.py`): the hypothesis it existed to test was
  confirmed on console 2026-07-08 (DML 11.1× CPU) and the real diffusion
  pipeline shipped in 1.0.0. The result CSV stays; the tooling lives in git
  history at `v1.0.0`.
- **`CheckBenchMode`** (MainPage): superseded by the `wWinMain` headless
  dispatch — an in-XAML bench would run with the compositor alive and produce
  numbers not comparable with every recorded CSV.
- **`bench-xbox.sh` + `bench/config/phase1-*.json`**: the pre-pivot GGUF-era
  orchestrator (single-`.gguf` upload, qwen3/llama-3.2 configs never used by
  the ONNX app); docs already invoked it with arguments it rejected.
  `bench-xbox-ort.sh` is the orchestrator (docs updated).
- **`build-uwp.ps1 -NoBundledModel`**: the `XllamaNoBundledModel` MSBuild
  property no longer exists (the model ItemGroup was removed in 1.0.0) — the
  switch was a no-op.

### Added — SD-Turbo in the download catalogue (image model self-serve)

- **GGUF dir resolve prefers `model.gguf`** over `adapter.gguf` (runtime LoRA catalogue dirs).
- **Console P0 training closeout** on MSIX 1.2.0.546: `rate` PASS, `lora-rt` PASS, `serve` PASS; `validate-console all` routing+GGUF PASS (TAESD FAIL post-wipe).

The image model no longer requires Device Portal provisioning: the catalogue
gains a `kind: "diffusion"` entry for `sd-turbo-fp16` (validated artifact —
`validate_pipeline.py` end-to-end pass; the `-ort-fp16` candidate is the
CUDA-only NhwcConv trap and stays excluded), and the **Image dialog downloads
it on the first Generate** (~2.4 GB, progress in the status bar).

- Manifest schema: optional `kind` (`ort-genai` default / `diffusion`) and
  per-file `remote` (flat release asset name; `filename` may now carry a
  subpath like `unet/model.onnx` — the downloader creates subfolders).
- Diffusion entries are hidden from the chat model picker.
- `run_diffuse` reads the CLIP tokenizer from the model's own `tokenizer/`
  dir first (what the download provides), falling back to the legacy
  `LocalState\clip\` upload.
- Requires the 5 `sd-turbo-fp16_*` assets on the `models-v1` GitHub Release.

### Docs — full drift audit + user guides (2026-07-09)

Three-way audit (docs↔code, docs↔measured state, gaps) and fixes:

- **Retired-narrative drift**: README, `phase1-runbook`, `device-portal`,
  `windows-dev-vm`, `model-selection` still described the bundled-model MSIX,
  the "~768 MB GPU pool", the build-time model merge, and pre-1.0 versions —
  all rewritten to the current reality (19 MB no-model MSIX, first-launch
  catalogue download from the `models-v1` Release, measured 3801 MB GPU
  budget, per-workload verdict, image generation).
- **Status drift**: ROADMAP Phase 3/3.5 headers and stale items (load_ms
  baseline, 1.7B CPU bench, int4 variant location) flipped to their measured
  state; three historical "On-console validation pending" CHANGELOG lines
  flipped per the runbook convention; runbook section headers now carry
  ✅ MEASURED / ⏳ PENDING markers and the v0.4.0.0 anchor is gone.
- **New user docs**: `docs/using-the-app.md` (chat, settings, routing,
  KV reuse, image generation) and `docs/install-release.md` (cert + VCLibs +
  MSIX from a tagged release); `docs/README.md` index completed;
  "Add your own model (manifest override)" how-to in `model-selection.md`;
  `deploy.sh fetch-file` and the llamacpp CI variant documented.

### Added — in-process diffusion experiment (`diffuse-inproc.flag`)

- **GGUF dir resolve prefers `model.gguf`** over `adapter.gguf` (runtime LoRA catalogue dirs).
- **Console P0 training closeout** on MSIX 1.2.0.546: `rate` PASS, `lora-rt` PASS, `serve` PASS; `validate-console all` routing+GGUF PASS (TAESD FAIL post-wipe).

The 887A0036 device conflict (§7) was measured with ORT **GenAI**'s
Agility-factory device; the diffusion pipeline uses **plain ORT DML**, which may
coexist with the XAML compositor device. `diffuse-inproc.flag` runs
`run_diffuse()` on a background MTA thread inside the live XAML process to
falsify the inherited headless requirement. If it passes on console, image
generation becomes in-app — no restart flow. Runbook §7b; on-console validation
pending.

### Added — diffusion progress + cancellation

- **GGUF dir resolve prefers `model.gguf`** over `adapter.gguf` (runtime LoRA catalogue dirs).
- **Console P0 training closeout** on MSIX 1.2.0.546: `rate` PASS, `lora-rt` PASS, `serve` PASS; `validate-console all` routing+GGUF PASS (TAESD FAIL post-wipe).

`diffuse-progress.txt` reports the live stage (`start` / `text_encoder` /
`unet s/N` / `vae` / `done` / `cancelled` / `error`); `diffuse-cancel.flag`
(consumed) aborts between UNet steps. Works in both the headless and the
in-process paths; stale `.done`/cancel artifacts are cleared at run start.

## [1.0.0] - 2026-07-08

**First stable release.** Local LLM chat (ORT GenAI, CPU int4 + per-conversation
GPU routing, KV-cache reuse) and Stable-Diffusion image generation (SD-Turbo
fp16 on DirectML, 512×512 in ~7 s) on a retail Xbox Series S in Dev Mode.
Models are downloaded on first launch from the GitHub Release catalogue (no
bundled model, 19 MB MSIX). Full measured story: `docs/technical-report.md`.
Sections below record everything that shipped and was measured on the way.

### Measured — in-app model download validated end-to-end on console (Exp 2 ✅)

The nobundle app on the Xbox downloaded the full SmolLM2-360M model (4 files,
417 MB merged `model.onnx`) from the **GitHub Release catalogue** inside the
AppContainer — `[manifest] using LocalState\manifest.json override` → all files
byte-exact on device → `.complete` written. Distribution is now self-hosted:

- **`models-v1` GitHub Release** carries the merged, AppContainer-safe model
  assets (the upstream HF repo ships a non-merged `model.onnx` stub + external
  data that §8 cannot load — that path stays broken upstream and is no longer
  referenced).
- `uwp/models/manifest.json` and the built-in fallback now point at the release
  URL; the LocalState manifest override was exactly the mechanism used to
  validate before flipping the default.
- **The MSIX no longer bundles a model** (ROADMAP Phase 4 milestone): first
  launch downloads from the catalogue; USB/Device-Portal provisioning unchanged.
  CI matrix simplified to `default` (nobundle) + `llamacpp`; the `xllama-appx`
  artifact is now the 19 MB no-model package.

### Measured — diffusion steps/seed plumbing on console

`diffuse-steps.txt=2` / `diffuse-seed.txt=777` with a new prompt produced a
coherent new 512×512 image in **7.7 s** (UNet 2.08 s/step ×2 — per-step cost
scales as expected; te/vae unchanged).

### Measured — llama.cpp CPU A/B on console: parity, not 2× (hypothesis falsified)

llama.cpp **runs on the Xbox in AppContainer** (first time): static ggml+llama
lane (`uwp/ggml-uwp.vcxproj`, `patches/0001-uwp-appcontainer-guards.patch`, CI
`build (llamacpp)`), SmolLM2-360M **Q4_K_M**, standard-512 ChatML prompt
(`bench/results/phase35-llamacpp-scaling.csv`):

| threads | prompt tok/s | decode tok/s |
| ------- | ------------ | ------------ |
| 1       | 141.6        | 19.9         |
| 4       | 141.3        | 51.5         |
| 6       | 141.5        | **62.9**     |
| 7–8     | —            | **livelock** |

- Versus ORT GenAI int4 @8t (66.3 decode / 220 prefill): **decode parity
  (−5 %), prefill worse** — Q4_K_M does not extract more bandwidth than ORT's
  AVX2 `MatMulNBits` on this machine; both saturate ~13 GB/s effective. The
  ROADMAP's "~2× → ~130 tok/s" hypothesis is falsified. **ORT GenAI stays the
  text backend**; the llamacpp lane remains in CI as a bench-only variant.
- t7/t8 livelock: ggml's spin-wait threadpool oversubscribes the ~6 cores Dev
  Mode leaves the app (no thread affinity in AppContainer). t6 is the ceiling.
- Real bugs fixed on the way: `#ifdef XLLAMA_USE_ORT` selected ORT even with
  `=0`; llama_tokenize size-query sign treated as error; `no_perf` (default)
  hid all timings (own chrono now, like the ORT path); ggml.c/ggml.cpp and
  ggml-cpu.c/.cpp same-dir obj collisions silently dropped every C symbol; the
  128 `src/models/*.cpp` per-arch files were missing from the static lib.

### Measured — image generation on console (v0.4.2.0, 2026-07-08) 🎨

**The flagship GPU workload is live**: SD-Turbo fp16 generates a coherent
512×512 image on the Xbox Series S GPU (DirectML) in **6.9 s** — text_encoder
1.0 s, UNet **3.3 s/step** (1 step), VAE decode 2.6 s; session load ~7.5 s
excluded (`bench/results/phase5-diffuse.csv`; image at
`docs/screenshots/diffuse-sd-turbo-xbox.png`, matches the local CPU validation
image for the same prompt/seed). Fixes required on the way, all measured:

- **VAE OOM with all sessions resident** (first run, 8007000E at
  `/decoder/up_blocks.3` InstanceNormalization): the 3801 MB budget doesn't fit
  ~2.4 GB of weights + the VAE's 512×512 activations. `run_diffuse` now creates
  and destroys each session per stage — text_encoder → UNet loop → VAE.
- **Deployable fp16 artifacts**: produced with `onnxruntime.transformers`'
  `convert_float_to_float16` (onnxconverter_common emits ORT-rejected mixed-type
  graphs for all 3 SD components — falsified options documented in
  `diffusion/README.md`); sessions cap graph optimization at **EXTENDED**
  (ORT_ENABLE_ALL crashes on these graphs). Each component self-contained
  < 2 GB: unet 1.65 GB, text_encoder 0.65 GB, vae 0.09 GB.
- **`deploy.sh mkdir_localstate` dropped the last path component** (while-read
  on an unterminated printf stream) — the true root cause of the earlier silent
  model-upload losses; fixed, uploads verified byte-exact.
- `run_diffuse` gained input-dtype adaptivity (fp16/fp32 floats, i32/i64 ids,
  i64/f32/f16 timestep), `diffuse-steps.txt`/`diffuse-seed.txt`, and per-stage
  telemetry (`diffuse-result.csv`).

### Added — model catalogue (`models/manifest.json`), model management de-hardcoded

- **GGUF dir resolve prefers `model.gguf`** over `adapter.gguf` (runtime LoRA catalogue dirs).
- **Console P0 training closeout** on MSIX 1.2.0.546: `rate` PASS, `lora-rt` PASS, `serve` PASS; `validate-console all` routing+GGUF PASS (TAESD FAIL post-wipe).

The model list, download source, and downloadability gate were hardcoded in four
places; they now come from one catalogue file.

- `uwp/models/manifest.json`: name/display/`hf_base_url`/file-list per model,
  bundled in the MSIX (both variants) at `InstalledPath\models\manifest.json`;
  a `LocalState\manifest.json` uploaded via Device Portal **overrides it without
  a reinstall**. Parsed with WinRT `Windows::Data::Json` (`LoadModelManifest`,
  `uwp/model-downloader.cpp`) with a built-in fallback so the app never starts
  with an empty catalogue.
- Settings ComboBox is populated from the catalogue (plus the active model as a
  "(custom)" entry if it isn't listed — e.g. a dir uploaded under a new name);
  replaces the static 3-entry `kModels[]`.
- `EnsureModelAsync` downloads **any** catalogue entry with an `hf_base_url`
  (replaces the single-model `kDownloadableModel` gate + hardcoded repo URL +
  `SmolLM2_360M_Files()`); entries without a URL keep the USB/Device-Portal
  guidance error.

### Measured — Phase 3.5 console validation (Xbox Series S, v0.4.0.0, 2026-07-08)

The pending on-console checks for the merged 0.3.7–0.4.0 features, run in one session
per `docs/console-validation-runbook.md`. CSVs under `bench/results/phase35-*.csv`.

- **Image spike (flagship hypothesis) — CONFIRMED.** On a compute-bound fp16 batch
  (309 GFLOP, one UNet-step proxy), DirectML is **11.1× faster than CPU** (128.7 ms /
  2403 GFLOP/s vs 1428 ms / 216 GFLOP/s). The exact inverse of text decode — image
  generation plays to the GPU's strength, greenlighting the diffusion pipeline.
- **Decode matrix (ORT GenAI 0.14.1), SmolLM2-360M, 285-tok prompt:** CPU int4
  **66.3** tok/s, DML fp16 **46.8** (real GPU: engines ~46–57 %, `gpu_mem` 793 MB),
  DML int4 **8.82** (real GPU: engines ~87–90 %, `gpu_mem` 307 MB). Versus the v0.3.6
  baselines (68.0 / 46.8 / 8.8) the 0.14.1 bump is **flat on decode** — a valid,
  recorded result (its win, if any, is CPU-overhead at higher token rates, not here).
- **int4 DML floor — §12 desk-check CONFIRMED on hardware.** DML int4 decode is
  8.82 tok/s with the **GPU compute engines saturated (87–90 %)** — not a CPU
  fallback. The non-fused `MatMulNBits` (dequantize→fp16 + full GEMM) is
  bandwidth-bound; CPU int4 stays the decode default. Kernel-design limit, confirmed.
- **KV-cache reuse (Stage 2) — CONFIRMED.** Turn-2 prefill with reuse is **4.87×**
  faster than cold (103.7 ms / 22-tok delta vs 505.2 ms / 114-tok full re-prefill) —
  continuous decoding processes only the new turn's tokens as designed.
- **1.7B scale (§6) — CONFIRMED.** SmolLM2-1.7B cpu-int4 runs on the console:
  prefill 54.9 tok/s, **decode 20.6 tok/s**, peak WS 2423 MB, load 6.2 s. Decode
  scales ~3.2× down from 360M (66.3 → 20.6) — memory-bandwidth-bound, as expected;
  CPU int4 stays usable at 1.7B. (fp16-DML 1.7B remains undeployable: 3.4 GB weights
  exceed the 2 GB protobuf limit — a serialization constraint, not the GPU.)
- ~~Still pending~~ **Closed 2026-07-14:** CPU/GPU **routing** (§2, autopilot PASS);
  **diffusion** (§7, measured 2026-07-08); **TAESD** (§7c, autopilot PASS).

### Fixed — deploy/bench tooling gaps surfaced during console validation

- `scripts/deploy.sh upload-dir`: now verifies the target dir exists (WDP folder
  creation can fail silently) and checks each file's `Success` flag, failing loudly
  instead of reporting "Uploaded N" while landing nothing (the 1.7B model.onnx
  silently vanished this way until the missing subdir was created). Mirrors the
  `upload-file` check.
- `src/bridge/bench.cpp`: the CSV `backend`/`quant` columns were hardcoded
  `ort-genai-cpu`/`int4`, mislabelling DML and fp16 runs; now derived from the model
  dir name (+ `gpu_mem_mb` corroboration). The model label also no longer truncates
  at the first dot (`smollm2-1.7b-cpu-int4` was logged as `smollm2-1`).

### Added — C++ diffusion pipeline (`diffuse.flag`, host-validated)

- **GGUF dir resolve prefers `model.gguf`** over `adapter.gguf` (runtime LoRA catalogue dirs).
- **Console P0 training closeout** on MSIX 1.2.0.546: `rate` PASS, `lora-rt` PASS, `serve` PASS; `validate-console all` routing+GGUF PASS (TAESD FAIL post-wipe).

The console image-generation pipeline. `uwp/diffuse.cpp` runs three ORT DirectML
sessions (text_encoder → 1× UNet denoise → VAE decode) behind a `diffuse.flag`
headless mode, mirroring the image spike's D3D12-clean host (887A0036-safe).

- Correctness-critical logic is **header-only, dependency-free, and unit-tested on
  the host** against golden vectors from the diffusers/transformers reference:
  `include/xllama/diffusion/clip_tokenizer.h` (CLIP byte-level BPE),
  `euler_scheduler.h` (EulerDiscreteScheduler, SD-Turbo 1-step), `half.h` (fp16),
  `png_writer.h` (self-contained PNG). `tests/test_diffusion.cpp` asserts all four
  against `diffusion/golden_vectors.json` — **638 assertions, all green** — so the
  logic ships in the un-runtime-testable console C++ only after matching Python.
- `diffusion/gen_golden_vectors.py` captures the reference (token ids for several
  prompts incl. empty + non-ASCII; scheduler sigmas/timesteps + one deterministic
  step). The CLIP tokenizer assets (`diffusion/clip_tokenizer/{vocab.json,merges.txt}`)
  are vendored so the test is hermetic and the console upload uses the same files.
- The tokenizer parses `vocab.json` with a minimal scanner (no JSON lib) so the
  header stays dependency-free for the UWP build (which skips the llama.cpp
  submodule); the host test uses vendored nlohmann only to load the golden file.
- Model contract: an **fp16** SD-Turbo-class model (each component < 2 GB,
  self-contained). The ORT DirectML orchestration is CI-compile-validated; runtime
  validation is on console per `docs/console-validation-runbook.md §7`.

### Corrected — int4 DML decode diagnosis (desk-check 2026-07-08)

The [0.3.6] entry below (and the docs at the time) attributed the int4 GPU decode
collapse (8.8 tok/s) to a "missing fused int4 DML kernel" that, if added, would
imply ~180 tok/s. A source-level desk-check **refines that**: `MatMulNBits` is
present in the graph (225 nodes) and **runs on the DML GPU** — the profiled run
shows one `DmlFusedNode` on `DmlExecutionProvider` (96%), no CPU fallback. The
real limit is that DirectML's `MatMulNBits` is **non-fused**
(`DML_DEQUANTIZE`→fp16 + full `DML_GEMM`, materialising fp16 weights), so int4
moves more bandwidth than fp16 and cannot beat it on DML; the builder also gives
DML `accuracy_level=0` vs CPU's fused-int8 `=4`. Full analysis + config tests in
`docs/uwp-constraints.md §12`; `ROADMAP.md` Phase 3.5 updated. Net: GPU int4
decode is blocked by a DirectML kernel-design limit (out of our control), not a
kernel we could contribute — CPU int4 stays the decode default.

### Added — diffusion model toolchain (image generation, `diffusion/`)

- **GGUF dir resolve prefers `model.gguf`** over `adapter.gguf` (runtime LoRA catalogue dirs).
- **Console P0 training closeout** on MSIX 1.2.0.546: `rate` PASS, `lora-rt` PASS, `serve` PASS; `validate-console all` routing+GGUF PASS (TAESD FAIL post-wipe).

Model-side toolchain for running SD-Turbo (1-step distilled diffusion) on the
console GPU — the workload that plays to DirectML's strength (compute-bound fp16
batch), unlike LLM decode.

- **Validated on the owned layer (2026-07-08)**: SD-Turbo exported to ONNX and
  run through **ONNX Runtime (CPU EP)** generates a coherent 512×512 image in
  ~13 s / 1 step — the same artifact + runtime that runs on Xbox via the
  DirectML EP.
- `diffusion/export_sd_turbo.sh` + `requirements.txt`: reproducible, **pinned**
  toolchain (Python 3.10, torch 2.4.1 legacy ONNX tracer, optimum 1.23.3,
  transformers 4.46.3, diffusers 0.31.0). The pins are load-bearing: newer torch
  uses the dynamo/onnxscript exporter that optimum 1.23 can't consume (external-
  data-naming + LayerNormalization opset-downgrade bugs).
- `diffusion/generate_onnx.py`: validates the exported ONNX through ORT.
- `diffusion/convert_fp16.py`: confirms the fp16 components fit the 3801 MB
  budget — each < 2 GB (UNet ~1.65 GB, ~2.4 GB total), all self-contained,
  AppContainer-safe (unlike a 1.7B LLM fp16 blob). **Caveat**: the CPU fp16 pass
  leaves a mixed-type node in the UNet timestep embedding (`/time_proj/Mul`) that
  ORT rejects at load — a _runnable_ fp16 model needs a GPU export
  (`optimum-cli --fp16 --device cuda`) or Olive. fp32 is fully validated.
- Next: the C++ pipeline (3 ORT DirectML sessions + scheduler + CLIP tokenizer)
  behind a `diffuse.flag` headless mode, on the plain-ORT DirectML foundation
  already proven by the image spike (PR #3).

## [0.4.0.0] - 2026-07-08

### Added — image-generation spike (plain ORT DirectML, new model axis)

- **GGUF dir resolve prefers `model.gguf`** over `adapter.gguf` (runtime LoRA catalogue dirs).
- **Console P0 training closeout** on MSIX 1.2.0.546: `rate` PASS, `lora-rt` PASS, `serve` PASS; `validate-console all` routing+GGUF PASS (TAESD FAIL post-wipe).

First step toward image/vision models on the console. The desk-check (§12) showed
the GPU loses at text decode because that workload is M=1, dispatch-bound, and
int4-DML is non-fused — the GPU's _worst_ case. Image generation (diffusion) is
the opposite: compute-bound fp16 batch, the case where DML already wins (prefill).
This spike tests that hypothesis cheaply before building a full diffusion pipeline.

- `uwp/image-spike.cpp`: runs a compute-bound conv model through the **plain
  ONNX Runtime DirectML** EP (not ORT GenAI — `onnxruntime.dll` was already a
  dependency) plus a CPU EP control, measuring forward-pass latency + GFLOP/s.
  Model `imgspike.onnx` (`scripts/gen_imgspike_model.py`, deterministic): 17
  Conv(3×3,64ch)+Relu on 1×3×512×512 fp16, ~309 GFLOP/forward — a faithful proxy
  for one diffusion UNet step.
- `uwp/App.cpp`: generalised the headless flag dispatch — `image.flag` →
  `run_image_spike()`, `bench.flag` → `main_loop()` (shared `HeadlessView`, same
  D3D12-clean host that avoids the 887A0036 compositor conflict).
- `uwp/xllama.vcxproj`: added the ORT DirectML NuGet include dir + the new source.
  `onnxruntime.lib` was already linked.
- Writes `imgspike-result.csv` (+ `.done`): DML vs CPU ms, GFLOP/s, GPU speedup.
- On-console validation: ✅ measured 2026-07-08 — DML **11.1× faster** than CPU on
  the compute-bound fp16 batch (2403 vs 216 GFLOP/s), the inverse of text
  decode; hypothesis confirmed and the SD-Turbo diffusion pipeline shipped
  in [1.0.0].

## [0.3.9.0] - 2026-07-08

### Added — per-conversation CPU/GPU routing (Stage 3, default off)

- **GGUF dir resolve prefers `model.gguf`** over `adapter.gguf` (runtime LoRA catalogue dirs).
- **Console P0 training closeout** on MSIX 1.2.0.546: `rate` PASS, `lora-rt` PASS, `serve` PASS; `validate-console all` routing+GGUF PASS (TAESD FAIL post-wipe).

The v0.3.6 matrix showed the EP choice is per-workload: DML fp16 wins prompt
prefill at scale (1.8× at ~1k tokens), CPU int4 wins decode. The app can now
route between them.

- Settings `routing` (0 = CPU only / default = unchanged behaviour, 1 = GPU
  only, 2 = auto: route first prompts over ~500 est. tokens to GPU) + `gpu_model`
  (DML fp16 model dir, default `smollm2-360m-dml-fp16`); a routing ComboBox in
  the Settings dialog.
- The decision is made once at a conversation's first turn and **sticky** for its
  lifetime (the KV cache is per-EP): `m_active_model` holds the routed dir,
  cleared on new/loaded chat so each conversation re-decides. Reuses the existing
  single-slot `EnsureSession` (one model resident at a time) — switching
  conversations may reload, which is memory-safe on the 3801 MB budget.
- Default routing = CPU, so behaviour is identical until a user opts in. GPU
  routing requires the DML fp16 model present on device (LocalState/USB).
- On-console validation pending: confirm auto-routing picks GPU for long prompts
  and that TTFT improves for prompt-heavy conversations.

## [0.3.8.0] - 2026-07-08

### Added — KV-cache reuse across chat turns (continuous decoding)

- **GGUF dir resolve prefers `model.gguf`** over `adapter.gguf` (runtime LoRA catalogue dirs).
- **Console P0 training closeout** on MSIX 1.2.0.546: `rate` PASS, `lora-rt` PASS, `serve` PASS; `validate-console all` routing+GGUF PASS (TAESD FAIL post-wipe).

The interactive app re-prefilled the **entire** ChatML history on every turn
(`BuildChatMLPrompt` + a fresh `OgaGenerator` per `generate()`), so turn-N TTFT
paid to re-process ~all prior tokens (~1.8k at budget → seconds). `OrtSession`
now keeps its generator alive across turns and appends only the new turn's
tokens (`OgaGenerator_AppendTokenSequences` on the persistent generator), so the
per-turn prefill covers just the delta.

- `GenerateParams` gains `reuse_kv` / `reset_kv`; `InferenceResult` gains
  `ended_with_stop` and now populates prefill telemetry (`n_p_eval`/`t_p_eval_ms`)
  on the interactive path too (previously bench-only). Decode timing excludes
  prefill, matching the bench convention.
- `OrtSession` holds `m_chat_gen`/`m_chat_params`/`m_chat_stream` + the bound
  sampling signature; the stateless path is preserved unchanged and still used
  when `reuse_kv` is false.
- `MainPageController::BuildDeltaPrompt` builds the incremental turn; the KV is
  reused only when valid and no context turn was evicted (RewindTo truncates the
  tail, not the head → eviction forces a full re-prefill). Reuse is invalidated
  on new/loaded chat, settings change, abort, and any generator failure.
- **Correctness guard**: a continuation that fails before emitting a token
  auto-falls back to a full re-prefill (no UI double-streaming) — worst case is
  the previous behaviour, never a wrong result.
- Settings toggle `kv_reuse` (ToggleSwitch + `settings.json`, default on) so the
  win can be A/B'd on console.
- On-console validation: ✅ measured 2026-07-08 — turn-2 prefill **4.87× faster**
  with reuse (103.7 ms vs 505.2 ms cold re-prefill); coherence confirmed.

### Added — multi-turn TTFT bench (Stage 2b)

- **GGUF dir resolve prefers `model.gguf`** over `adapter.gguf` (runtime LoRA catalogue dirs).
- **Console P0 training closeout** on MSIX 1.2.0.546: `rate` PASS, `lora-rt` PASS, `serve` PASS; `validate-console all` routing+GGUF PASS (TAESD FAIL post-wipe).

- Headless bench gains a KV-reuse measurement: when `bench_turns.txt` is present
  (turn-2 user prompt; `prompt.txt` supplies turn 1), `main_loop` runs both turns
  on one persistent `Session` and measures turn-2 prefill **with reuse** (append
  only the delta) vs the **cold** baseline (full re-prefill of the 2-turn
  context), writing `bench-kv-result.csv` (+ `.done`) with a `speedup` column and
  logging the numbers. This measures the Stage 2 win on console instead of
  assuming it.

## [0.3.7.0] - 2026-07-08

### Changed — ONNX Runtime GenAI 0.13.2 → 0.14.1

- Bumped `Microsoft.ML.OnnxRuntimeGenAI.DirectML` NuGet from 0.13.2 to **0.14.1**
  (`uwp/packages.config`, `uwp/xllama.vcxproj` — 8 package-path references).
  0.14.x reduces CPU-side per-token overhead in `GenerateNextToken`/`SampleTopP`
  (directly relevant to the decode bottleneck measured in the v0.3.6 matrix) and
  is the prerequisite for continuous decoding / KV-cache reuse (`RewindTo`,
  generator reuse — Stage 2). No breaking C-API changes vs 0.13; existing model
  directories (built with model builder 0.14.1) load unchanged.
- On-console validation: ✅ measured 2026-07-08 — decode flat vs v0.3.6 (CPU int4
  66.3 vs 68.0; DML fp16 46.8 = 46.8): the bump is a prereq/overhead win, not a
  decode-rate win at this scale.

### Docs

- `ROADMAP.md`: new Phase 3.5 — Hardware Ceiling with the ordered unlock
  levers (Game-mode designation, 1B+ models via no-bundle, int4-AWQ proxy,
  upstream `MatMulNBits` desk check, llama.cpp CPU A/B, per-workload routing,
  optional upstream kernel contribution); Phase 2 heading refined to the
  per-workload verdict; "GPU vs CPU same model" milestone closed by the
  v0.3.6 matrix.
- `docs/uwp-constraints.md` §5: stale "future work" paragraph replaced with
  the measured per-workload approach; documented the App-vs-Game designation
  lever — **settled 2026-07-08**: the package was found already designated
  Game, so all measured figures are Game-mode numbers (the interim "all
  numbers are App-mode" assumption was wrong and has been rectified in
  §5/§7/§11); the GPU decode gap is a DML/kernel issue, not platform
  scheduling.
- Environment change (2026-07-08): Dev Mode storage allocation raised to
  90 GB — the Q:\ ~2.2–2.5 GB disk budget (§9) is superseded as the binding
  constraint for model sizing; ROADMAP Phase 3.5 updated accordingly.
- `docs/device-portal.md`: new "Lifecycle gotchas" section — LocalState purge
  semantics (only forward upgrades preserve it), LocalState absent before
  first launch, WDP file APIs returning HTTP 200 with `"Success": false`,
  stale PFN staging window, portal unreachable while a game is running.

## [0.3.6] - 2026-07-07

### Measured — Hardware utilization matrix (prefill vs decode, CPU vs GPU)

SmolLM2-360M, 3 variants × 2 prompts (285 / ~1050 tok), v0.3.6 timing:

| Variant  | prefill 285 | prefill ~1050 | decode short | decode long |
| -------- | ----------- | ------------- | ------------ | ----------- |
| CPU int4 | **220**     | 198           | **68.0**     | **50.9**    |
| GPU int4 | 152         | 334           | 8.8          | 8.3         |
| GPU fp16 | 169         | **354**       | 46.8         | 36.5        |

- **Prefill crossover**: GPU scales with batch, CPU doesn't — at ~1k prompt
  tokens the GPU is 1.8× faster (TTFT 3.0 s vs 5.3 s). DML fp16 wins
  prompt-heavy workloads today.
- **int4 GPU decode collapse = missing fused int4 kernel**, not dispatch: fp16
  decode is 5.3× int4 on DML. Effective bandwidth: GPU fp16 ~34 GB/s vs CPU
  ~13 GB/s (bus ~224 GB/s) — the GPU exploits memory 2.6× better; a
  `MatMulNBits`-class DML kernel would imply ~180 tok/s (2.5× CPU).
- Verdict in `docs/uwp-constraints.md §5`; matrix rows appended to
  `bench/results/phase2-dml.csv`. New model on-device:
  `smollm2-360m-dml-fp16` (builder `-p fp16 -e dml`, 691 MB merged).

### Fixed

- Prefill timing (0.3.5) started the clock after `AppendTokenSequences`, but
  on-console measurement showed the prompt prefill runs **inside**
  `AppendTokenSequences` in ORT GenAI 0.13.2 (the first `GenerateNextToken`
  returns in ~40 µs) — `prompt_tok_s` came out in the millions. The clock now
  starts before `AppendTokenSequences`, covering the prefill under either
  implementation. Decode numbers were unaffected (measured after prefill in
  both layouts).

## [0.3.5] - 2026-07-07

### Added

- **GGUF dir resolve prefers `model.gguf`** over `adapter.gguf` (runtime LoRA catalogue dirs).
- **Console P0 training closeout** on MSIX 1.2.0.546: `rate` PASS, `lora-rt` PASS, `serve` PASS; `validate-console all` routing+GGUF PASS (TAESD FAIL post-wipe).

**Prefill measurement in the ORT path** (`prompt_tok_s` was always 0.00):

- `src/bridge/inference.cpp`: ORT GenAI runs the prompt prefill inside the
  _first_ `GenerateNextToken` call; it was never timed (`n_p_eval`/`t_p_eval_ms`
  were only populated by the legacy llama.cpp path). Now the first loop
  iteration is timestamped: `prompt_tok_s` lands in the bench CSV and the log
  gains `prefill=N tok/s (M tok, T ms)`.
- `decode_tok_s` is now prefill-free (rate over `n_generated-1` tokens from the
  end of the first iteration) — previous decode numbers were slightly
  underestimated because they included prefill time.
- `bench/prompts/long-1k.txt`: ~1k-token prompt to exercise prefill in the
  CPU-vs-GPU utilization matrix.

### Upstream fix validated on hardware (2026-07-07, evening)

The `887A0036` DML init failure was fixed at the source and validated on
console before contributing upstream:

- Patch to `onnxruntime-genai` `CreateDmlObjects`: fall back to the system
  D3D12 runtime when the Agility SDK device factory cannot create a device
  (upstream PR [microsoft/onnxruntime-genai#2280](https://github.com/microsoft/onnxruntime-genai/pull/2280)).
- Validated with a throwaway 0.3.9 test MSIX (branch
  `test/dml-fallback-validation`: patched DLL injected over the NuGet one,
  headless detection disabled to force the XAML repro path):
  - XAML + DML + vanilla DLL: `887A0036` at `OgaCreateModel` (known repro)
  - XAML + DML + **patched DLL: loads in 886 ms, 315 MB on GPU, full decode
    8.8 tok/s** — identical to the headless/Agility path
  - XAML + CPU + patched DLL: 67.2 tok/s — no regression
- Console restored to the official 0.3.4 MSIX afterwards.

## [0.3.4] - 2026-07-07

### Added

- **GGUF dir resolve prefers `model.gguf`** over `adapter.gguf` (runtime LoRA catalogue dirs).
- **Console P0 training closeout** on MSIX 1.2.0.546: `rate` PASS, `lora-rt` PASS, `serve` PASS; `validate-console all` routing+GGUF PASS (TAESD FAIL post-wipe).

**Headless bench mode — unblocks DML EP init** (root cause of the 0.3.3
`887A0036` finding, now identified at the exact source line):

- Root cause: ORT GenAI 0.13.2 creates its D3D12 device through the **Agility
  SDK device factory** (`dml_helpers.cpp` `CreateDmlObjects`:
  `CreateDeviceFactory(614, module_path)` succeeds on Xbox OS 26100 via the
  in-box runtime ≥ 614, no app-local `D3D12Core.dll` needed — verified absent
  from our MSIX). The factory's `CreateDevice` then collides with the
  process-wide D3D12 device the **XAML compositor** (D3D11on12) created at
  `Window.Activate()` — two different D3D12 runtimes cannot share a process →
  `DXGI_ERROR_ALREADY_EXISTS`. This also reconciles Exp 1 (May): the then-OS
  had in-box < 614, so ORT fell back to plain `D3D12CreateDevice` (line 144)
  which coexists with the compositor device — the OS update flipped the branch.
  No upstream fix on `main`; v0.14.0 has identical code.
- `uwp/App.cpp`: when `LocalFolder\bench.flag` exists at startup, `wWinMain`
  skips `Application::Start` entirely and runs a minimal `IFrameworkView`
  (`BenchView`) via `CoreApplication::Run` — CoreWindow activated (PLM watchdog
  satisfied) but **no XAML compositor, no in-process D3D12 device** — then runs
  `xllama::bridge::main_loop()` on an MTA thread and exits via
  `CoreApplication::Exit()`. Interactive path unchanged (flag absent →
  `uninit_apartment` + `Application::Start` as before); `CheckBenchMode` in
  MainPage stays as fallback if the early detection throws.

### Fixed

- `bench/configs/genai_config-dml-{test,profile,profile.tpl}.json`:
  `past_present_share_buffer` `false` → `true` — the DML EP uses graph capture,
  which rejects `false` at `OgaCreateGenerator` ("Graph capture is not
  supported with past_present_share_buffer set to false"). The `false` was an
  Exp 1 leftover that only ever ran on the silent-CPU-fallback path.

### Measured — GPU-truth verdict on console (2026-07-07, headless bench mode)

**`VERDICT: GPU`** — first proven DML GPU execution in the Xbox UWP sandbox:

- ORT profiler: `DmlExecutionProvider` 10.7 ms kernel time (fused node 9.7 ms),
  `CPUExecutionProvider` 0.4 ms → 96% DML.
- In-app probe: `gpu-mem post-load: current=411MB` ≈ model size (403 MB) —
  weights resident on the RDNA 2 GPU. CPU control run: `current=0MB`,
  69.7 tok/s.
- The bundled CPU-int4 `MatMulNBits` model fails inside the fused DML node
  (`80070057`) — a DML model variant is required for full decode (below).

### Measured — Phase 2 end-to-end DML bench (2026-07-07, evening)

SmolLM2-360M INT4 **DML** variant, built with the ORT GenAI model builder
(0.14.1, `-p int4 -e dml`, 285 MB merged), uploaded to
`LocalState\models\smollm2-360m-dml-int4`:

| Backend            | decode tok/s | load ms | gpu_mem MB | peak WS MB |
| ------------------ | ------------ | ------- | ---------- | ---------- |
| DML EP (RDNA 2)    | **8.83**     | 927     | 307        | 999        |
| CPU EP (Zen 2 t=8) | **70.9**     | 1518    | 8          | 771        |

- Full decode completes on GPU (739 tokens, `VERDICT: GPU`, GPU compute engine
  ~88% saturated via `systemperf`) — but the **CPU is ~8× faster**: per-token
  DML dispatch overhead dominates a 360M model while `MatMulNBits`/AVX2 is
  highly optimised. **Phase 2 closed: CPU EP stays the production backend.**
- GPU pool estimate corrected: measured `Budget` is **3801 MB** (the "~768 MB
  pool" was OOM-bracketing inference). Disk (`Q:\` ~2.2–2.5 GB) is the real
  model-size constraint.
- Known cosmetic issue: bench CSV `backend` column says `ort-genai-cpu` even on
  DML runs (label hardcoded in `bench.cpp`, does not detect the configured EP).

## [0.3.3] - 2026-07-07

### Changed

- `src/bridge/inference.cpp`: dropped the `gpu-mem pre-load` probe that ran
  _before_ `OgaCreateModel`. `gpu_mem_info()` opens and caches an
  `IDXGIAdapter3` on adapter 0; not holding that handle open across model load
  is cleaner. GPU memory is still sampled `post-load`/`post-decode`, after the
  session exists — where the GPU-truth signal (`current` ≈ model size) matters.
  **Note:** this was first hypothesised to be the cause of the DML init failure
  below, but the rebuild (0.3.3) disproved it — see _Investigated_. The change
  is kept as harmless cleanup, not a fix.

### Investigated — GPU-truth DML experiment on console (2026-07-07)

First profiled DML run on Xbox Series S, using the 0.3.2 GPU-truth toolkit.
**Result: the DirectML EP does not initialise at all** on this stack — it is
neither GPU execution nor a CPU fallback.

- `OgaCreateModel` throws `887A0036` "The desired element already exists" at
  `onnxruntime-genai .../dml/dml_helpers.cpp(140)`, **before any kernel runs**,
  so no `ort_profile_*.json` is produced and the profiler emits no `VERDICT:`.
- Reproduced 3×: profiling config (0.3.2), plain `dml-test` config (0.3.2), and
  after the pre-load-probe removal (0.3.3) — identical signature each time.
- **Not OOM** (`avail_phys` 5.0 GB, `budget` 3801 MB), **not the profiling
  config** (plain config fails identically), **not our telemetry** (0.3.3 fix
  changes nothing). Cause is inside the ORT GenAI DML EP device creation.
- Likely a D3D12 single-device-per-adapter conflict: a UWP XAML app already
  holds a D3D12 device (compositor) on adapter 0 before `OgaCreateModel`.
- Corroborated by telemetry: GPU engines flat (only the display engine ~99%),
  `gpu_dedicated` ~100 MB (Dev Home noise), no compute spike.
- Reconciles Exp 1 (2026-05-23): its "loads without OOM, 71.7 tok/s ≈ CPU
  baseline, GPU execution unconfirmed" was almost certainly a **silent CPU
  fallback**, never real DML execution. Same ORT GenAI 0.13.2, same config.
- **Conclusion:** DML EP is not viable on ORT GenAI 0.13.2 in the Xbox UWP
  sandbox; the CPU EP is the only working backend (70.9 tok/s control run, same
  session). Supersedes the earlier "GPU EP ruled out via OOM" with a precise
  init-failure signature.

## [0.3.2] - 2026-07-07

MSIX carrying the GPU-truth toolkit + in-app `gpu_mem` telemetry (CWD pin to
LocalState) for the first profiled DML run on console.

### Added

- **GGUF dir resolve prefers `model.gguf`** over `adapter.gguf` (runtime LoRA catalogue dirs).
- **Console P0 training closeout** on MSIX 1.2.0.546: `rate` PASS, `lora-rt` PASS, `serve` PASS; `validate-console all` routing+GGUF PASS (TAESD FAIL post-wipe).

**Bench infrastructure — ORT GenAI thread tuning** (`feat(bench): ORT GenAI bench infrastructure`):

- `scripts/bench-xbox-ort.sh`: new bench orchestrator for ORT GenAI models already on device (no model upload). Supports `--threads N`, `--runs N`, `--prompt file`. Appends median row to `bench/results/phase1-cpu.csv`. Drops warmup run automatically.
- `bench/configs/genai_config-threads-{4,6,8}.json`: `genai_config.json` variants with `intra_op_num_threads` set for Zen 2 thread-count tuning.
- `uwp/inference-bridge.cpp`: reads optional `bench_threads.txt` (uploaded per bench variant) to set `params.n_threads` for CSV tracking and suffix host_label (`xbox-series-s-tN`).

**GPU-truth debug toolkit — DML EP attribution without PIX** (see `docs/uwp-constraints.md §11`):

- `bench/configs/genai_config-dml-profile.json` (+ `.tpl.json` absolute-prefix variant): DML EP config with ORT `enable_profiling` + `log_severity_level: 0`.
- `scripts/profile-dml-run.sh`: one profiled DML run — config swap, bench run, fetch `ort_profile_*.json` + log tail into `bench/results/profiles/<ts>/`, restore config, analyze.
- `scripts/analyze_ort_profile.py`: per-provider kernel-time summary from the ORT profiling JSON; greppable `VERDICT: GPU | MIXED | CPU-FALLBACK` line; tolerates truncated traces.
- `scripts/xbox-gpu-sample.sh`: WDP `systemperf` sampler (per-engine GPU utilization + VRAM used → CSV + max/mean summary); `--gpu-sample` integration in `profile-dml-run.sh` and `bench-xbox-ort.sh`.
- `gpu_mem_info()` (`src/bridge/platform.cpp`): per-process `IDXGIAdapter3::QueryVideoMemoryInfo` (LOCAL); logged pre-load/post-load/post-decode; `dxgi.lib` linked in `uwp/xllama.vcxproj`.
- `set_cwd_to_local_folder()`: bench mode pins CWD to LocalState so the relative ORT profiling prefix lands in a writable, WDP-fetchable location.
- Bench CSV schema: new `gpu_mem_mb,gpu_budget_mb` columns before `host,date` (header updated in `bench.cpp`, both bench scripts, `bench/README.md`; existing `phase1-cpu.csv` rows backfilled with `0,0`); `--out FILE` flag in `bench-xbox-ort.sh` (DML runs → `bench/results/phase2-dml.csv`).
- CI (`build-linux`): smoke step for the analyzer (fixtures in `tests/fixtures/`) and the sampler parser.

**No-bundle MSIX build variant — unblocks Exp 2 validation**:

- `uwp/xllama.vcxproj`: model `ItemGroup` now also conditioned on `'$(XllamaNoBundledModel)' != 'true'`.
- `scripts/build-uwp.ps1 -NoBundledModel`: builds an MSIX without the bundled SmolLM2 model, so `EnsureModelAsync` reaches the USB/HF-download fallbacks on console.
- `.github/workflows/build-uwp.yml`: matrix `variant: [bundled, nobundle]`; the `nobundle` job skips model download/merge and uploads artifact `xllama-appx-nobundle`.

### Fixed

- `src/bridge/bench.cpp`: backend label was `"directml"` even on CPU EP → corrected to `"ort-genai-cpu"`; quant `"int4-awq"` → `"int4"`.
- `src/bridge/inference.cpp` (ORT path): `load_ms` was always 0 in the bench CSV — `run_inference` never measured model load. Now times `OgaCreateModel` wall-clock and logs `ORT model loaded in N ms`.
- Docs drift: `bench/README.md` results table said "pending" for the populated `phase1-cpu.csv`; `bench/README.md` + `docs/phase1-runbook.md` still documented the old `directml` backend label.

### Measured — Phase 1 bench results (Xbox Series S Zen 2, 2026-05-23)

SmolLM2-360M-Instruct INT4 CPU, ORT GenAI 0.13.2, n=990:

| n_threads          | decode tok/s | peak RAM MB | notes                                           |
| ------------------ | ------------ | ----------- | ----------------------------------------------- |
| auto (ORT default) | 66.9         | 704         | baseline, no `intra_op_num_threads`             |
| 4 (explicit)       | **71.4**     | 771         | **best**                                        |
| 6 (explicit)       | 68.0         | 772         |                                                 |
| 8 (explicit)       | 28.2         | 771         | severe regression — memory bandwidth saturation |

**Recommendation**: use `intra_op_num_threads: 4` in `genai_config.json` for SmolLM2-360M on Zen 2.

### Investigated — Exp 1 DirectML (2026-05-23)

- DML `genai_config.json` (provider: dml, `enable_cpu_mem_arena=0`, `enable_mem_pattern=0`) loads without SEH 0xC0000005 on SmolLM2-360M INT4.
- Performance: 71.7 tok/s — indistinguishable from CPU baseline.
- Conclusion: SmolLM2-360M INT4 (~200 MB ONNX) likely fits within the 768 MB UWP GPU pool. Cannot confirm GPU execution vs CPU fallback without D3D profiling tools. Phase 2 "blocked" status revised: **360M model fits; larger models still blocked**.
- Exp 2 (HF in-app download): unreachable with current bundled MSIX. `EnsureModelAsync` checks InstalledPath before HF download; model is always found there. Requires a separate build without bundled model to validate — now available via `build-uwp.ps1 -NoBundledModel` / CI artifact `xllama-appx-nobundle`.

### Investigated — model candidates via HF Hub file sizes (2026-07-02)

- Qwen2.5-0.5B INT4 CPU: ~822 MB `model.onnx.data` (rtn-block-32) — the ~200 MB estimate was wrong (151k-vocab embedding not INT4-quantized). Ruled out for disk and GPU pool. DML int4-awq variant ~507 MB: borderline GPU-pool fit, possible DML retry via USB.
- Llama-3.2-1B INT4 CPU: ~1.77 GB — USB-only, same class as SmolLM2-1.7B. Details in `docs/model-selection.md`.

---

## [0.3.0] — 2026-05-23

### Added

- **GGUF dir resolve prefers `model.gguf`** over `adapter.gguf` (runtime LoRA catalogue dirs).
- **Console P0 training closeout** on MSIX 1.2.0.546: `rate` PASS, `lora-rt` PASS, `serve` PASS; `validate-console all` routing+GGUF PASS (TAESD FAIL post-wipe).

**Settings dialog — sampling parameters** (`feat(uwp): Settings dialog — sampling params`):

- `ShowSettings` now exposes `temperature` (Slider 0–2, default 0.8), `top_p` (Slider 0–1, default 0.9), `top_k` (NumberBox 1–200, default 40), `repetition_penalty` (Slider 1–2, default 1.1), and `n_predict` (NumberBox 16–2048, default 512).
- New `MainPageController` members `m_temperature`, `m_top_p`, `m_top_k`, `m_repetition_penalty`, `m_n_predict` wired into `StartInference` → `GenerateParams`.
- `settings.json` schema extended with a nested `"sampling"` object; back-compat preserved for existing 0.2.x files.

**Settings dialog — model selection ComboBox** (`feat(uwp): Settings dialog — model selection ComboBox`):

- ComboBox with three entries: SmolLM2-360M (bundled MSIX), SmolLM2-1.7B (USB `E:\xllama\models\`), SmolLM2-360M (HF download in `LocalState`).
- Selected model persisted to `settings.json`; `EnsureSession` detects model change at next `StartInference` and rebuilds transparently.
- `LoadModelName` now reads `m_model_filename` from `settings.json`; falls back to `LocalState/model.txt` for 0.2.x installations.

**History dialog enhancements** (`feat(uwp): History dialog — delete, clear all, timestamps`):

- Per-item ✕ Delete button: click closes the dialog and opens a confirmation ContentDialog; on confirm calls `ChatHistory::Delete(id)`. If deleted entry was the active conversation, `NewChat()` is called.
- Clear all (Secondary button): confirmation ContentDialog → `ChatHistory::Clear()` → `NewChat()`.
- Current conversation indicator: ● prefix on the active history entry.
- Relative timestamps via `FormatRelativeTs` helper: "today HH:MM", "yesterday HH:MM", "DD Mon HH:MM".
- Index refreshed from disk on every `ShowHistory` call.
- Empty-state ContentDialog with placeholder TextBlock instead of silent no-op.

**ChatHistory::Delete / Clear** (`feat(uwp/chat-history): add Delete and Clear methods`):

- `Delete(id)` removes `<id>.json` from `LocalState/chats/` and updates the in-memory index + `index.json`.
- `Clear()` removes all conversation files and writes an empty `index.json`.

**Tests** (`test: add ChatHistory helpers and TitleFrom smoke tests`):

- `tests/test_chat_history.cpp`: Linux CI tests exercise `TitleFrom` logic (truncation, newline stop, empty fallback). UWP `#ifdef` branch tests `Save`/`Load` roundtrip, `Delete`, and `Clear` with a temp directory.

### Fixed

- **Newline rendering**: `AppendOutput` now splits text on `\n` and inserts `LineBreak` inlines; previously `\n` in a `Run` was rendered as a space by WinUI `RichTextBlock`.
- **Prompt not cleared after Run**: `OnRunClick` clears `m_promptInput.Text(L"")` before handing off to `StartInference`.
- **NewChat did not clear prompt**: `NewChat()` now resets `m_promptInput.Text(L"")`.
- **Double FocusEngagement on Xbox**: removed `IsFocusEngagementEnabled(true)` from `m_outputScroll`; only the TextBox retains it, eliminating the extra A-press required to engage text input.
- **Focus not returned after generation**: `SetRunning(false)` now calls `m_promptInput.Focus(FocusState::Programmatic)` so the cursor returns to the input after inference completes.
- **Smart autoscroll**: `AppendOutput` auto-scrolls only when the user is already at the bottom (within 24 px of `ScrollableHeight`). A `ViewChanged` handler tracks `m_at_bottom`; `SetRunning(false)` resets it to `true`.
- **Status shows "Loading model…" at startup**: `BuildUI` initialises `m_statusText` to `L"Loading model..."` and disables the Run button. `EnsureModelAsync` sets `L"Ready"` and re-enables Run when the model is confirmed loaded.
- **Context trim overflow**: `kMaxEstimatedTokens` lowered from 3500 to 1800 to stay within `n_ctx = 2048` (leaves ~250 token generation headroom). Trim events now surface a `"Context trimmed: N turns dropped"` status message.
- **Partial save on cancel**: `StartInference` completion path sets `ChatMessage::partial = true` when the user pressed Cancel (`m_abort.load() == true`) before saving the conversation.

---

## [0.2.1] — 2026-05-23

### Added

- **GGUF dir resolve prefers `model.gguf`** over `adapter.gguf` (runtime LoRA catalogue dirs).
- **Console P0 training closeout** on MSIX 1.2.0.546: `rate` PASS, `lora-rt` PASS, `serve` PASS; `validate-console all` routing+GGUF PASS (TAESD FAIL post-wipe).

- ChatML stop sequence `<|im_end|>` in UI inference path (`uwp/MainPage.cpp`). SmolLM2-360M does not always emit EOS naturally; without this the model would continue generating filler or hallucinate the next user turn up to `n_predict=512`. Bench path unchanged.
- `tests/test_session.cpp`: smoke tests for `Session::create` error paths (non-existent path, empty path) — covers the Linux/llama.cpp path in CI.

### Fixed

- `CHANGELOG.md` 0.2.0 section: collapsed duplicate `### Added` blocks; removed stale empty `[Unreleased]` header.

- **GGUF dir resolve prefers `model.gguf`** over `adapter.gguf` (runtime LoRA catalogue dirs).
- **Console P0 training closeout** on MSIX 1.2.0.546: `rate` PASS, `lora-rt` PASS, `serve` PASS; `validate-console all` routing+GGUF PASS (TAESD FAIL post-wipe).

---

## [0.2.0] — 2026-05-22

### Added

- **GGUF dir resolve prefers `model.gguf`** over `adapter.gguf` (runtime LoRA catalogue dirs).
- **Console P0 training closeout** on MSIX 1.2.0.546: `rate` PASS, `lora-rt` PASS, `serve` PASS; `validate-console all` routing+GGUF PASS (TAESD FAIL post-wipe).

**Persistent inference session** (`feat(uwp): integrate xllama::Session into MainPage`):

- `MainPageController` now keeps an `xllama::Session` alive across chat turns; subsequent turns skip model reload entirely (~1–2 s overhead eliminated after first turn).
- `EnsureSession()` private helper: lazy-build on first turn, transparent rebuild on model change (Settings), free-then-alloc to avoid 2× RAM during transitions.
- Bench mode (`inference-bridge.cpp`) unchanged — continues to call `run_inference()` for cold-load measurement.

**Bench diagnostics** (`bench(inference): log prompt token count + bump bench n_predict`):

- `inference.cpp`: logs `[xllama] prompt=N tok, max_length=M (new≤K)` after tokenisation — makes `n` in bench CSV self-explanatory.
- `inference-bridge.cpp`: bench `n_predict` raised `128 → 512` (effective `max_length` 640 → 1024 total tokens); gives SmolLM2-360M room to show natural generation length while 1.7B still exits at EOS.

**Multi-turn Session API** (`include/xllama/session.h`, `src/bridge/session.cpp`):

- `xllama::Session::create(SessionParams)` — loads model + tokenizer once; subsequent `generate(GenerateParams)` calls reuse them, eliminating the ~1-2s per-call reload overhead of `run_inference()`.
- `GenerateParams` exposes `top_p`, `top_k`, `repetition_penalty`, `seed`, `stop_sequences` (substring-checked against accumulated output; matching sequence stripped on hit).
- Implemented for both UWP (ORT GenAI: `OrtSession` keeps `OgaModel` + `OgaTokenizer` alive) and Linux (llama.cpp: `LlamaSession` keeps `llama_model` alive; `llama_context` rebuilt per call to respect context parameters).
- `run_inference()` is unchanged — fully back-compatible.
- Consumer: `xbox_faraday` game (FARADAY) for per-turn dialogue generation.

**Workaround experiments for Xbox UWP constraints** (see `docs/uwp-constraints.md §7, §9`):

- `uwp/model-downloader.cpp/h`: in-app Hugging Face download via `HttpClient` chunked streaming (Exp 2). `EnsureModelAsync()` in `MainPage` implements a three-step bootstrap: LocalState `.complete` marker → InstalledPath bundle → HF download. Reduces peak disk usage from ~1.4 GB to ~480 MB; frees ~900 MB on Dev Mode partition Q:\ enabling models up to ~1 GB. `internetClient` capability already present in manifest.
- `src/bridge/path_utils.cpp`: third fallback `E:\xllama\models\<name>` for NTFS USB stick (Exp 3). No UWP capability required; probe via `GetFileAttributesW`. Enables models up to 2 GB single-file (Xbox Dev Mode USB limit). Zero cost if USB absent.
- `scripts/test-dml-config.sh`: uploads DML provider_options config to Xbox via Device Portal without MSIX rebuild (Exp 1). Backs up original `genai_config.json`; `--restore` reverts.
- `bench/configs/genai_config-dml-test.json`: DML EP test config — `enable_cpu_mem_arena=0`, `enable_mem_pattern=0`, `past_present_share_buffer=false` (reduces up-front KV-cache GPU allocation; may allow SmolLM2-360M to fit ~768 MB pool).
- `docs/model-selection.md`: consolidated model evaluation checklist — hard limits, 9-step selection sequence, tested/candidate models, conservative/borderline/over-budget tables.
- `docs/uwp-constraints.md §9`: Disk Budget — ~2.2–2.5 GB Dev Mode free space, 2× peak-install rule, working budget table (empirical, Series S).
- `scripts/merge_onnx_external_data.py`: NOTE stderr (>400 MB) and WARNING stderr (>600 MB) budget thresholds post-merge.
- `docs/uwp-constraints.md §5`: split GPU OOM and disk-budget failure modes into separate tables (previously mixed under a single "Result" column).

**UX improvements** (commits `3a12bda`–`42741e1`):

- Multi-turn chat: `uwp/chat-history.cpp/h`, conversation persistence in `LocalState/chats/` (JSON, indexed by timestamp), history browser overlay, new-chat button.
- System prompt editable via settings overlay (persisted to `LocalState/settings.json`).
- Live metrics: real-time tok/s updated every flush cycle; `StatusKind` enum (`Info`, `Working`, `Success`, `Error`) for colour-coded status bar.
- `RichTextBlock` streaming: `Paragraph::Inlines` append (O(1) per token); flush timer batches token appends every 80 ms to avoid layout thrash.
- Xbox UX: TV safe-area margins (48/27 px), dark theme on Xbox hardware, B-button cancels inference, gamepad Y jumps to prompt, Reveal focus visual, `ElementSoundPlayer::On`.
- `AppxManifest.xml`: `xbox:DefaultTile`, `xbox:SplashScreen`, dark splash background `#0E1116`.
- ChatML prompt template applied for SmolLM2-360M-Instruct (system / user / assistant turns).

**Documentation:**

- `README.md`: "About the name" section — disambiguates xllama from llama.cpp engine.
- Full docs realignment: `README.md`, `ROADMAP.md`, `AGENTS.md`, `docs/phase1-runbook.md`, `docs/uwp-constraints.md`, `docs/device-portal.md`, `patches/README.md` all updated to reflect ORT GenAI CPU EP as the active path.
- `docs/windows-dev-vm.md` (new): end-to-end Windows VM build guide.
- `scripts/setup-windows-uwp-dev.ps1` (new): Windows VM setup via `winget` (VS2022 BuildTools + UWP workload).
- `scripts/check-uwp-host.sh` (new): Arch Linux host preflight (KVM, qemu, libvirt groups).
- `docs/uwp-constraints.md §7`: removed unverified architectural claims ("Game process category", "128 MB dedicated + 640 MB shared"); replaced with observed-behaviour framing and source note.
- `ROADMAP.md`: Phase 4 milestones updated — `ModelDownloader` (Exp 2) and USB fallback (Exp 3) marked done; next: validate Exp 2 on console, remove MSIX model bundle.

**CI:**

- `build-uwp` CI step: downloads model from HF (`homen3/SmolLM2-360M-Instruct-ort-genai-int4-cpu`), merges ONNX external data, then `nuget restore` + `build-uwp.ps1`. Cache key: `smollm2-360m-ort-genai-int4-cpu-embedded-v1`.

### Changed

- ORT GenAI bumped `0.8.3 → 0.13.2`, ORT `1.22.0 → 1.24.4` (`uwp/packages.config`).
- `bench.cpp`: backend field = `directml` when `XLLAMA_USE_ORT` (define-time; CPU EP is active runtime on Series S).
- `uwp/pch.h`: added `Windows.Web.Http`, `Windows.Web.Http.Filters` for model downloader.
- `ROADMAP.md Phase 2`: corrected GPU pool description to observed-behaviour framing.

### Removed

- `uwp/llama-bridge.cpp`, `uwp/llama-bridge.h`: legacy files not compiled since ORT GenAI pivot.

### Fixed

- `fix(bridge): OrtModelPtr → OgaModelPtr` typo in `OrtSession` UWP build (MSVC `C2065`; GCC/clang skip the `XLLAMA_USE_ORT` block on Linux).
- ASCII-safe status strings: removed em-dash and ellipsis Unicode literals that caused MSVC `C4566` warnings.
- `XYFocusKeyboardNavigationMode` removed from `MainPage.cpp` (unresolvable symbol in MSVC UWP context).
- `weakly_canonical: Access is denied` crash (`OgaCreateModel`, status `0xC0000005`): ORT runtime walks path segments of the model directory to validate external data; `Q:\Users\UserMgr0\...` is inaccessible from UWP AppContainer. Fix: merge external data into monolithic `model.onnx` so `ValidateExternalDataPath` is never invoked. Confirmed via Win32 probes (`GetFileAttributesW`, `CreateFile2 GENERIC_READ`, `CreateFile2 FILE_READ_ATTRIBUTES|SYNCHRONIZE`).

---

## [Pivot: SmolLM2-360M + CPU EP] — commit `14e6a14`

### Added

- **GGUF dir resolve prefers `model.gguf`** over `adapter.gguf` (runtime LoRA catalogue dirs).
- **Console P0 training closeout** on MSIX 1.2.0.546: `rate` PASS, `lora-rt` PASS, `serve` PASS; `validate-console all` routing+GGUF PASS (TAESD FAIL post-wipe).

- SmolLM2-360M-Instruct INT4 CPU as the bundled model (403 MB on-disk, ONNX opset 21, IR version 10).
- Model included as `DeploymentContent` in `uwp/xllama.vcxproj`; deployed to `Package.InstalledPath\models\smollm2-360m-cpu-int4\`.
- `resolve_model_path` (`src/bridge/path_utils.cpp`): checks `LocalState\models\<name>` first (runtime override), falls back to `Package.InstalledPath\models\<name>` with copy-on-first-launch to LocalState.

### Changed

- Default model in `uwp/inference-bridge.cpp` and `uwp/MainPage.cpp` changed from Phi-3.5 to `smollm2-360m-cpu-int4`.
- `genai_config.json` uses `"provider_options": []` → CPU EP active (no DirectML).

### Notes

- **GPU EP ruled out**: Xbox Series S UWP GPU pool is ~768 MB. `OgaCreateModel` with DirectML EP on any tested model (Phi-3.5-mini GPU INT4 ~2.2 GB, SmolLM2-1.7B ~1.4 GB) crashes with null-deref in DML allocator on OOM. See `docs/uwp-constraints.md §7`.

---

## [Pivot: ONNX Runtime GenAI + DirectML] — commit `385cb07`

### Added

- **GGUF dir resolve prefers `model.gguf`** over `adapter.gguf` (runtime LoRA catalogue dirs).
- **Console P0 training closeout** on MSIX 1.2.0.546: `rate` PASS, `lora-rt` PASS, `serve` PASS; `validate-console all` routing+GGUF PASS (TAESD FAIL post-wipe).

- `XLLAMA_USE_ORT=1` preprocessor flag in `uwp/xllama.vcxproj`; enables ORT GenAI path in `src/bridge/inference.cpp`.
- `include/xllama/ort_raii.h`: RAII `unique_ptr` wrappers for `OgaModel`, `OgaTokenizer`, `OgaTokenizerStream`, `OgaGeneratorParams`, `OgaGenerator`, `OgaSequences`.
- `uwp/inference-bridge.cpp` / `inference-bridge.h`: replaces `llama-bridge.cpp`; thin UWP glue around `xllama::bridge::run_inference`.
- NuGet packages (`uwp/packages.config`): `Microsoft.AI.DirectML 1.15.4`, `Microsoft.ML.OnnxRuntime.DirectML`, `Microsoft.ML.OnnxRuntimeGenAI.DirectML`.
- `src/bridge/platform.cpp`: `log_output` now writes to `LocalState/xllama.log` in UWP (previously `OutputDebugStringA` only).

### Changed

- `src/bridge/inference.cpp`: `#ifdef XLLAMA_USE_ORT` path uses `OgaGenerator` loop; `#else` path retains `llama_decode` for Linux.
- Linux CI (`build-linux.yml`): `submodules: false` for UWP; llama.cpp submodule only for Linux.
- `deploy.sh`: `upload-file` auto-creates subdirectory; new subcommands `mkdir-localstate`, `upload-dir`.

### Notes

- llama.cpp submodule retained for Linux path (`CMakeLists.txt`). Three UWP patches (`uwp/patches/llama.cpp/`) kept but not applied for this build.

---

## [Pivot: XAML-free UI] — commits `77a651a`, `3f7a950`, `385cb07`

### Removed

- `App.xaml`, `MainPage.xaml`, `XamlTypeInfo_impl.cpp`: eliminated to avoid WMC9999 (`XamlC.exe` crash during `MarkupCompilePass2` in SDK 22621/26100 for C++/WinRT projects).
- `runtimeclass MainPage` from IDL: `MainPageController` is now a plain C++ class.

### Added

- **GGUF dir resolve prefers `model.gguf`** over `adapter.gguf` (runtime LoRA catalogue dirs).
- **Console P0 training closeout** on MSIX 1.2.0.546: `rate` PASS, `lora-rt` PASS, `serve` PASS; `validate-console all` routing+GGUF PASS (TAESD FAIL post-wipe).

- `MainPageController` (`uwp/MainPage.cpp`): programmatic UI built via `Windows.UI.Xaml.Controls.*` API. Uses `enable_shared_from_this`; `shared_from_this()` must not be called from the constructor — use `Init()` post-construction.
- `runtimeclass App` retained (required by `Application::Start`).

### Notes

- Root cause of WMC9999: without MarkupCompilePass2, `XamlTypeInfoProvider::CreateXamlType` cannot provide correct metadata for `xllama.MainPage`; parser fast-fails when `LoadComponent` tries to validate the binding. No workaround existed; XAML-free was the correct fix.

---

## [Baseline: llama.cpp + Linux CI] — initial commits

### Added

- **GGUF dir resolve prefers `model.gguf`** over `adapter.gguf` (runtime LoRA catalogue dirs).
- **Console P0 training closeout** on MSIX 1.2.0.546: `rate` PASS, `lora-rt` PASS, `serve` PASS; `validate-console all` routing+GGUF PASS (TAESD FAIL post-wipe).

- Linux build via CMake presets (`linux-release`, `linux-test`).
- Modular bridge: `src/bridge/inference.cpp`, `bench.cpp`, `platform.cpp`, `path_utils.cpp`, `utf8_utils.cpp`, `cli.cpp`.
- Shared headers under `include/xllama/` (inference, CLI, RAII, platform, path utils).
- Unit tests with doctest (target: `xllama-tests`, preset: `linux-test`).
- `xllama-cli` binary: `src/main.cpp` with `getopt_long`.
- `scripts/deploy.sh`: Device Portal REST API wrapper (deploy, install-cert, get-log, list-localstate, list-dumps, start-app, stop-app, diagnose-startup).
- `scripts/bench-xbox.sh`: automated benchmark runner (upload, trigger, poll, fetch CSV, compute median).
- `docs/`: `device-portal.md`, `uwp-constraints.md`, `phase1-runbook.md`.
- `bench/`: methodology README, config JSONs, `prompts/standard-512.txt`, `prompts/short-32.txt`.
- `.github/workflows/build-linux.yml`: clang-format, shellcheck, cmake, ctest, UWP gate.
