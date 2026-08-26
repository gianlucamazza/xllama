# Changelog

All notable changes to xllama are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

- **Research package and XAB baseline.** Added a citable Series S Dev Mode
  report with claim-level provenance, generated evidence tables/figures and a
  release runbook. Added `scripts/run-xab.sh` to compose text, KV, H9 and
  diffusion harnesses. Historical measurements without thermal/power sidecars
  remain explicitly non-production evidence.
- **Demo presentation edit.** Added a reproducible 30-fps hard-cut presentation
  edit for concise README walkthroughs. The raw Device Portal recording and
  its measured capture rate remain the only performance evidence.
- **Catalogue signing and reproducibility evidence.** Store CI now signs the
  bundled model catalogue with the GitHub Actions-held RSA key and the Store
  verifier imports the pinned CNG public-key blob explicitly. The Xbox Series S
  Dev capture in `bench/results/phase17-console-2026-08-26.csv` records three
  post-warm-up runs with model hash, context, TTFT, throughput and peak memory;
  external power and thermal-equilibrium evidence remain separate gates.
- **CI benchmark validation fix.** Closed the CSV-validation loop in
  `build-linux.yml` so the shell `for`/`if` block is syntactically complete and
  all v2 benchmark rows are validated in Linux CI.

## [1.5.6.0] - 2026-08-22

Phase 16 catalogue win plus the pin and probes that landed with it, plus the
SDK transition (SessionHub factory, probe optionals, RoutingPolicy configurability,
Doxygen docs). Product ship path remains **CI MSVC**. Suite is still **10**
console gates.

**Upgrading from 1.5.x is a normal in-place update** (same package identity
`GianlucaMazza.xllama`).

### SDK

- **Modular SDK documentation.** Replaced the monolithic 594-line `docs/SDK.md`
  with 11 focused `.md` files under `docs/sdk/` (`overview`, `session`,
  `inference`, `routing`, `training`, `chat`, `kvstore`, `path`, `platform`,
  `sampling`, `misc`) — 742 lines total, each covering a single public-header
  group.

- **SessionHub factory pattern.** `SessionHub` now has an explicit constructor
  (`SessionHub::ensure_locked` / `reset_locked`) while keeping the
  backward-compatible `session_hub()` singleton. Callers that need explicit
  lifecycle can construct directly; the rest keeps working unchanged.

- **Probe optionals.** `CMakeLists.txt` now gates all probe sources under
  `option(XLLAMA_BUILD_PROBES "Build benchmarking probes" ON)`. CLI `--membw`,
  `--diskbw`, `--ramceil`, `--gpubw`, `--gpugemv` and the headless `.flag`
  files are absent when built with `-DXLLAMA_BUILD_PROBES=OFF`.

- **RoutingPolicy configurability.** All five routing gates (`dml_text_model_ok`,
  `decide`, `allow_kind`, `reuse_kv_kind`, `reuse_kv_model`) are now
  `std::function` callbacks on `RoutingPolicy`, with member wrappers dispatching
  through them. `default_policy()` returns a fully-configurable instance. Free
  functions (`dml_text_model_ok()`, `decide_routing()`, etc.) wrap the
  lazy-static default for backward compat.

- **Doxygen comments.** All 34 public headers in `include/xllama/` now carry
  `///` Doxygen documentation on every struct, class, method, and field.
  `InferenceParams` / `InferenceResult` fields rewritten to fix duplicate
  `abort_flag` declaration and add `///` on every member.

### Added

- **H6.2 wave32 Q4_K GEMV density probe** (`gpugemv.flag`,
  `shaders/gpugemv_q4k_wave32.hlsl`). Measure-only, not a Session GPU backend.
  Series S CI MSVC `1.5.5.922`: G1 PASS, `wave32` median **25.4 GB/s packed**
  (retimed naive 1.96). **K2 park** — G2 stays 40. CSV
  `bench/results/phase15-gpugemv-h62.csv`.
- **Store D1 App vs Game spike** on Series S (`lfm25-350m`, same package).
  App GPU budget **691 vs 3801 MB**; long-gen decode ~87 vs ~95 tok/s; peak
  320 MB both. Listing should request **Game** metadata. CSV
  `bench/results/store-app-vs-game-2026-08-21.csv`.
- **`validate-console.sh store`** — Store SKU smoke (catalogue download + GGUF
  chat + `set_api` reject). Not one of the 10 Dev Mode hardware gates. **PASS**
  2026-08-21 on CI Store SKU `1.5.5.928` (Game); Dev SKU restored after.
- **Store Phase 2 pack:** EN listing copy, IARC prep sheet, Partner Center
  operator steps (`docs/store-readiness.md`); GitHub Pages privacy URL
  `https://gianlucamazza.github.io/xllama/privacy.html`; Settings link + Issues
  template to report inappropriate generated content (Store policy 11.16).
  Product **xllama** reserved in Partner Center as **Game** (`9N9661XSDBM4`;
  identity §13). An App reservation (`9NCNLPWFT6B5`) was deleted the same day.
  Dev Mode identity unchanged. Submission fill sheet (pricing, IARC GenAI,
  Creators, packages hold) in `docs/store-readiness.md` §14.
- **`include/xllama/json_utils.h`** — header-only JSON string helpers:
  `json_escape()` (canonical) and `json_read_string()` with full `\uXXXX`
  decode (including surrogate pairs → UTF-8). Replaces 7 duplicated
  implementations across the tree.
- **`src/bridge/decode_loop_ort.h`** — consolidated ORT GenAI decode loop.
  Mirrors the structure of `src/bridge/decode_loop.h` (llama path).
- **`src/bridge/ort_common.h`** — shared ORT setup helpers (SEH translator,
  log callback registration).
- **CI ASan lane** (`build-linux.yml` job `asan`): workflow_dispatch-only.
  Builds with the `linux-asan` CMake preset (Debug + ASan + UBSan) and runs
  the full test suite under sanitizers.
- **`tests/test_json_utils.cpp`** — 11 test cases covering json_escape
  (quotes, backslash, control chars → `\uXXXX`, mixed content),
  json_read_string (all escapes, `\uXXXX`, surrogate pairs, lone surrogates,
  unterminated strings, unknown escapes lenient), and round-trip
  (escape → read_string → original).
- **`include/xllama/cancel_policy.h`** — `CancelTarget` enum +
  `cancel_target()` precedence function (image > training > text). The generic
  "a job is running" flag is set by all three, so the specific flags are the
  only ones that identify the job. Exhaustively tested on the host
  (`tests/test_cancel_policy.cpp`, all eight combinations of the three flags).
- **Autopilot op `mark`** — a rendez-vous for screenshot capture. The app writes
  a label to `LocalState\autopilot-mark.txt` and blocks; the host polls for the
  file, takes its Device Portal screenshot, and deletes the file to release the
  script.
- **Autopilot op `show_pane`, and a guard against the crash designing it
  exposed.** Settings, History and the image viewer are all `ContentDialog`s and
  none was reachable from the autopilot. The guard is product code: nothing
  checked whether a dialog was already open, and a second one throws inside a
  `fire_and_forget`, whose `unhandled_exception()` calls `std::terminate()`.
  `AutopilotAction` gained a dedicated `label` field. Ops 15 → 17.

### Fixed

- **json_escape triplicated across 7 TUs with divergent behavior.** Consolidated
  into `include/xllama/json_utils.h` (single canonical implementation, full
  `\uXXXX` control-char escaping). All callers now use `xllama::json_escape`.
  The training job parser (`src/bridge/training.cpp`) was also upgraded to
  decode `\uXXXX` (including surrogate pairs → UTF-8) via the new
  `json_read_string` helper — previously `\u0041` was passed through verbatim
  as `u0041`, a correctness gap for non-ASCII dataset paths/names.
  `device_train.cpp`'s `result.json` writer (which passed control chars raw,
  producing invalid JSON) is now fixed. Full round-trip tested
  (`tests/test_json_utils.cpp`, 11 cases).

- **ORT stateless path silently ignored stop_sequences.**
  `run_inference_ort` (inference.cpp) never checked `params.stop_sequences`,
  while `OrtSession::run_decode` did. Both paths now share the consolidated
  `decode_loop_ort.h` loop which applies stop sequences after each iteration
  (same semantics as llama.cpp: the stop-triggering token IS counted).

- **ORT decode loop duplicated** between `run_inference_ort` (inference.cpp)
  and `OrtSession::run_decode` (session.cpp) — the exact pattern the project
  had already fixed on the llama side. Extracted into
  `src/bridge/decode_loop_ort.h` (header-only, same structure as
  `decode_loop.h` for llama).

- **SEH translator and OgaSetLogCallback duplicated** between inference.cpp
  and session.cpp. Extracted into `src/bridge/ort_common.h`
  (`install_se_translator()`, `register_oga_logging()`).

- **The B button no longer drops out of the app.** On Xbox an unhandled
  `BackRequested` is the shell's cue to suspend and return to Home, and the
  handler only marked the event handled while an inference was running — so a
  B press on an idle chat closed xllama with no warning. Completed turns are
  already on disk, so this cost the typed-but-unsent prompt, the KV cache and a
  full model reload, not the conversation. Every `BackRequested` is now
  handled: B cancels a running reply and otherwise does nothing; leaving the
  app stays on the Xbox (Guide) button (`uwp/MainPage.cpp`,
  `docs/uwp-constraints.md` §10e, `docs/using-the-app.md`).

## [1.5.5.0] - 2026-08-20

Phase 16 catalogue win plus the pin and probes that landed with it. Product
ship path remains **CI MSVC**. Suite is still **10** console gates.

**Upgrading from 1.5.x is a normal in-place update** (same package identity
`GianlucaMazza.xllama`).

### Added

- **Phase 16 model-scouting campaign** (`docs/phase16-model-scouting.md`) and
  its first shipped result: catalogue entry **`lfm25-230m`** (LFM2.5-230M
  Q4_K_M, direct from LiquidAI so the LFM Open License travels with the
  weights). Console-measured on Series S at **119.2 tok/s decode / 241 MB peak
  / H9 2/8** — it becomes the **floor** tier, displacing `gemma3-270m` at
  1.55× its decode and 127 MB less peak, at one H9 task below it. The
  first-launch default stays `lfm25-350m`, which is slower and heavier but
  scores 4/8: the default trades throughput for capability deliberately.
  Evidence `bench/results/phase16-gguf.csv`, `bench/results/phase7-h9.jsonl`.
- **`gemma3-270m` H9 measured for the first time (3/8).** Its `model-matrix`
  §A1 cell had carried `—` since the model was catalogued, which is what made
  the floor-tier comparison uncomputable until now.
- **MiniCPM5 chat renderer** (`model_is_minicpm5`: template `<s>` BOS +
  no-think prefill). Host T1 passes both halves; T3 console bench was not
  booked this campaign, so `minicpm5-1b` is **not** in the catalogue.
- **`diskbw` NVMe probe** (`diskbw.flag` / `scripts/bench-diskbw.sh`). Series S
  unbuffered read ~2.0 GB/s sequential / 1.55–1.76 GB/s random 2 MiB. SSD
  streaming is real, narrow, and not a product path
  (`docs/ssd-inference-assessment.md`).
- **WS-F microphone probe** (`mic.flag` / `scripts/probe-mic.sh`).
  `AudioGraph` opens under AppContainer; `AccessDenied` did not fire; no
  headset was attached (`DeviceNotAvailable`). Not a verdict — #241 stays
  open. `docs/uwp-constraints.md` §10d.
- **Markdown formatting is gated in CI**, pinned to `prettier@3.9.6` alongside
  the existing `clang-format==22.1.5`, over every tracked `*.md`
  (`.prettierrc`, `.prettierignore`). The repo had no Markdown formatter and
  had drifted; the whole corpus was formatted once in the same change.

### Changed

- **`llama.cpp` pin `6d5a910` → `0865990` (b10333, #244)** with UWP glue
  (`llama-kv-cache-msa.cpp`, `llama_model_params.load_mode`, sampler
  `n_vocab` on penalties). Phase 16 WS-B had closed without a bump — both
  desk `arch:not-in-pin` flags were refuted at the GGUF header — and this
  Dependabot bump is independent of that kill.
- **H5 BitNet/low-bit survey closed NO-GO** (2026-08-10). The pin carries the
  `bitnet` arch; no sub-4B model trained at ≤2 bits publishes weights.
- **Phase 16 WS-E (embeddings) closed** — S-gate FAIL, no named consumer.
  Capability exists in the pin (`nomic-bert`); the product has no surface.
  Tracked as #242.

### Fixed

- **Documentation drift found by the Phase 16 audit**: `model-matrix` §D
  claimed `H2 open` while §A3 of the same file recorded H2 FAIL; its "Last
  updated" stamp predated its own newest content; §G still listed a shipped
  MSIX as an open gap; `benchmarks.md` cited a stale `llama.cpp` pin;
  `recommended-config.md` enumerated a four-gate console suite that has been
  ten gates since v1.5.4.0; and the Phase 16 campaign doc had wrongly listed
  H5 (BitNet) as closed when it is an open desk survey.
- **`deploy.sh` stop/start** actually stop and start; an unreachable console
  and a missing CSRF token now fail instead of claiming success.
- **Wine 11.15**: `ensure-cppwinrt-pin.sh` prefers the native cppwinrt at the
  pin. `quantize.sh` pointed at a tree we never build.
- **CI**: `check-coherence.py` asks git which Markdown we own; shellcheck is
  pinned; the clang-cl ggml patch is versioned instead of carried untracked.
