# Architecture

Current component and data-flow map for xllama. This is the **SSOT for how the
system is put together** (module boundaries, backend dispatch, provisioning). It
complements the frozen [technical-report.md](technical-report.md) (a v1.0 narrative
snapshot), the perf SSOT [benchmarks.md](benchmarks.md), and the AppContainer
constraints SSOT [uwp-constraints.md](uwp-constraints.md) — those own the numbers
and the sandbox limits; this file owns the structure.

## Two pillars, two targets, one core

xllama has two **platform pillars** that share C++ contracts but do not share
execution loops:

| Pillar                     | Role                              | Hot path                                                             |
| -------------------------- | --------------------------------- | -------------------------------------------------------------------- |
| **Inference**              | Chat, diffusion, LAN API, benches | `Session` / `run_inference` (forward-only)                           |
| **Training** (exploration) | Produce adapters / merged GGUF    | `TrainingJob` → host PEFT or bounded ggml-opt partial FT → artefacts |

Training never runs inside `generate()`. Inference never runs backward. Artefacts
flow **training → disk → inference load** (same GGUF path as catalogue models).

The shared core is platform-agnostic C++17; front-ends are thin:

- **Core library** (`src/bridge/`, headers in `include/xllama/`) — Linux + UWP.
  No WinRT in the headers.
- **Host front-ends** — `xllama-cli` (inference + `--validate-train-job` /
  `--train-job`), doctest suite, `training/host/` PEFT runner.
- **UWP app** (`uwp/`) — inference UI and headless bench/diffuse/membw/train
  drivers. The llamacpp/unified variants compile the ggml-opt partial-FT
  engine (Lane B, gates PASS); it is not part of the chat hot path.

Header modules (`include/xllama/`), all WinRT-free so they are host-testable:

| Header                             | Owns                                                                                                                      |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| `session.h` / `inference_params.h` | `Session`/`SessionParams`, `InferenceParams/Result` (`echo_stdout`, the `resolve_max_length` #130 ladder), `Backend` enum |
| `session_hub.h`                    | `SessionHub` — the ONE process-wide resident-Session owner (GUI + LAN API); locking + `generation` contract               |
| `sampling.h`                       | `SamplingConfig`, shared defaults; CLI/bench and GUI/API init from it (#125)                                              |
| `training.h` / `training_params.h` | `TrainingJob`/`TrainingResult`, stages, device gates, job JSON                                                            |
| `device_train.h`                   | Lane B engine API: `run_device_train_job`, progress callbacks, filters                                                    |
| `personalize.h`                    | Phase 11 helpers: last-block filter, job builder, sample count, manifest JSON                                             |
| `preference_capture.h`             | Preference JSONL format + append (Like/Dislike/Correct + API)                                                             |
| `chat_prompt.h`                    | `ChatFormat`, `chat_format_for`, `apply_stop_sequences`                                                                   |
| `routing_policy.h`                 | `decide_routing`, `kv_reuse_supported_for_model`, prompt budget                                                           |
| `model_provision.h`                | `dir_satisfies_expected_files`, `normalize_model_path`                                                                    |
| `manifest_merge.h`                 | `merge_manifest_entries` (per-entry catalogue override)                                                                   |
| `membw.h`                          | `measure_membw` (STREAM-style bandwidth probe)                                                                            |
| `inference.h`                      | `run_inference`, `write_bench_csv` — the public inference entry point                                                     |
| `prompt_budget.h`                  | `fit_prompt` — exact token-budget trimmer (one enforcement point where the tokenizer lives)                               |
| `autopilot.h`                      | `AutopilotAction`, `validate_autopilot_script` — console gate contract (WinRT-free)                                       |
| `json_utils.h`                     | `json_escape`, `json_read_string` with `\uXXXX` decode (surrogate pairs) — canonical JSON helpers                         |
| `cancel_policy.h`                  | `CancelTarget` + `cancel_target()` — which running job a cancel request targets                                           |
| `kv_store.h`                       | `KvStore` — on-disk KV snapshot pool (3 files / 192 MB, LRU, fingerprinted)                                               |
| `logit_dump.h`                     | Logit-parity harness: portable float32 dump + JSON sidecar, shared by both backends                                       |
| `ramceil.h`                        | `probe_ram_ceiling` — heap ceiling probe (commits in steps, records platform counters)                                    |
| `diskbw.h`                         | `measure_diskbw` — NVMe/disk bandwidth probe (sequential + random patterns)                                               |
| `d3d12_dyn.h`                      | Dynamic d3d12.dll entry-point resolve (PE hygiene for AppContainer)                                                       |
| `speculative.h`                    | `prompt_lookup_draft` — Phase 15 W2 draft-free prompt lookup (pure, host-testable)                                        |
| `gpubw.h`                          | Phase 15 W3: GPU STREAM probe helpers + kill gate (#211)                                                                  |
| `gpugemv.h`                        | Phase 15 H6.2: wave32 Q4_K GEMV density probe (#228)                                                                      |
| `cli.h`                            | `parse_cli_args` — Linux command-line parsing                                                                             |
| `platform.h`                       | `log_output`, `detect_threads(_llama)`, `peak_working_set_mb`, `gpu_mem_info`                                             |
| `path_utils.h`                     | `resolve_model_path`, `first_gguf_in_dir`, `model_uses_llama_backend`                                                     |
| `utf8_utils.h`                     | `utf8_to_wstring`, `wstring_to_utf8` (Windows/UWP)                                                                        |
| `ort_raii.h`                       | RAII `unique_ptr` for `Oga*` types (ORT GenAI C API, UWP path)                                                            |
| `llama_raii.h`                     | RAII `unique_ptr` for `llama_*` types (llama.cpp, Linux + UWP)                                                            |
| `diffusion/png_writer.h`           | PNG writer for diffusion output                                                                                           |
| `diffusion/half.h`                 | Half-precision float utilities                                                                                            |
| `diffusion/euler_scheduler.h`      | Euler scheduler (header-only, golden-vector unit-tested)                                                                  |
| `diffusion/clip_tokenizer.h`       | CLIP BPE tokenizer for diffusion                                                                                          |

Bridge sources of note under `src/bridge/`: `session.cpp`, `inference.cpp`,
`training.cpp`, `device_train.cpp`, `personalize.cpp`, `preference_capture.cpp`,
`sampler_chain.h` / `ort_sampling.h` (one sampler chain per backend),
`decode_loop.h` (one prefill and one generation loop, shared by
`run_inference_llama` and `LlamaSession::generate`),
`decode_loop_ort.h` (consolidated ORT GenAI decode loop — replaces duplicated
loops in `run_inference_ort` and `OrtSession::run_decode`; stop sequences now
applied to the stateless path),
`ort_common.h` (shared ORT setup: SEH translator + `OgaSetLogCallback`),
`chat_prompt.cpp` (`ChatFormat`, `chat_format_for`, `apply_stop_sequences`),
`bench.cpp` (bench CSV writer with `run_index` for per-run variance),
`platform.cpp` (`log_output` — writes `xllama.log` in UWP),
`path_utils.cpp` (`resolve_model_path` — LocalState + InstalledPath fallback),
`utf8_utils.cpp` (utf8 ↔ wstring),
`cli.cpp` (CLI argument parsing),
`json_utils.cpp` (canonical JSON helpers),
`prompt_budget.cpp` (`fit_prompt` implementation),
`autopilot.cpp` (`ApRun` driver),
`kv_store.cpp` (KV snapshot pool with LRU eviction),
`membw.cpp` (STREAM-style CPU bandwidth probe),
`diskbw.cpp` (NVMe disk bandwidth probe),
`ramceil.cpp` (heap ceiling probe),
`gpubw.cpp` (GPU STREAM probe D3D12 driver),
`gpugemv.cpp` (Q4_K GEMV density probe D3D12 driver).

## Inference backends and runtime dispatch

Two text backends, selected by build variant **and** per model at runtime:

- **ORT GenAI / DirectML** (`XLLAMA_USE_ORT`) — `OrtSession` in `src/bridge/session.cpp`.
  Runs ONNX GenAI models (`kind: "ort-genai"`); CPU int4 decode + DirectML fp16
  prefill, with per-conversation EP routing. GPU text routing is allowed only
  for parity-validated DML assets (`dml_text_model_ok`, #91 postmortem: broken
  DML RMSNorm kernel, fixed by the `-v2` decomposed graph); any other
  `gpu_model` forces CPU in every mode. Diffusion stays on GPU.
- **llama.cpp / GGUF** (`XLLAMA_USE_LLAMA`) — `LlamaSession` in
  `src/bridge/session.cpp`. Runs `.gguf` models (`kind: "gguf"`); CPU-only on Xbox
  (no ggml GPU backend), with KV-cache reuse. The UWP build compiles ggml with
  `GGML_USE_CPU_REPACK` (PR #155 — it was silently dead code before; enabling
  the repacked-weight GEMM raised GGUF prefill ~62% on Q4_K).

The **`default`** and **`llamacpp`** CI variants compile a single backend. The
shipping **`unified`** build links both and dispatches **per model at runtime**:
`Backend::Auto` (`session.h`) resolves via `model_uses_llama_backend()` — a `.gguf`
suffix / on-disk GGUF layout routes to llama.cpp, everything else to ORT GenAI
(`session.cpp`, `inference.cpp`). So llama.cpp is both the host A/B benchmarking
lane and a **shipping** text backend for the GGUF catalogue entries (LFM2.5,
Qwen3.5, Gemma).

Both backends **stream**: `GenerateParams::on_token` (a
`std::function<void(std::string_view)>` — the view is only valid inside the
call, copy before returning) fires per detokenized piece
inside the decode loop. The UI worker does not dispatch per token — it appends
under a mutex and a 40 ms `DispatcherTimer` (`FlushTokenBuffer`) drains the buffer
to the screen. Because the app streams, the latency that matters is
**time-to-first-token**, not total turn time; `InferenceResult::t_p_eval_ms`
carries it and `StartInference` surfaces it (`§5d`, #139). The LAN endpoint below
is the one path that does **not** stream — it returns a completed response.

## Chat templates (`ChatFormat`)

Prompt formatting is data-driven, not hard-coded. `chat_format_for(model_id)`
(`src/bridge/chat_prompt.cpp`) returns a `ChatFormat` describing the template.
Detection uses the **basename only** (never path segments — a cache dir named
`xllama` must not select Llama-3). Families:

| Kind        | Markers                             | System style                 | Typical catalogue                  |
| ----------- | ----------------------------------- | ---------------------------- | ---------------------------------- |
| **ChatML**  | `<\|im_start\|>` … `<\|im_end\|>`   | Dedicated system turn        | SmolLM2, LFM, Qwen2.5-Coder, Qwen3 |
| **Gemma**   | `<start_of_turn>` … `<end_of_turn>` | Merge system into first user | `gemma3-270m`, `gemma4-e2b`        |
| **Llama-3** | header / `eot` tokens               | Dedicated system turn        | `llama32-3b`                       |
| **Phi-3**   | `<\|user\|>` … `<\|end\|>`          | Dedicated system turn        | Phi GGUFs (campaign / override)    |

**Qwen3 vs Qwen2.5 (no-think prefill).** Qwen3.x with `enable_thinking=false`
expects an empty `<think>\n\n</think>` block after the assistant header.
`qwen_no_think_gen_suffix` applies that **only** when `model_is_qwen3` and not
`model_is_thinking` (basename `qwen3` / `qwen-3` — catalogue `qwen35-0.8b`,
`qwen3-1.7b`). It must **not** apply to Qwen2.5-Coder or to LFM Thinking:
those either are plain ChatML or emit real CoT (stripped at display time).

**Thinking models.** Basename contains `thinking` → `strip_thinking_content` on
the `ChatFormat`. After each turn, UI and LAN call `postprocess_output`, which
runs `strip_thinking_blocks` so history and screen keep the final answer only.
Streaming may briefly show raw CoT until the turn finishes and the paragraph is
replaced — no second Settings path, no n_predict special-case in policy.

`render_prompt` / `render_delta` build the full prompt or the KV-reuse delta;
`apply_stop_sequences` is the one shared suffix-match helper used by both the ORT
and llama.cpp decode loops.

## KV-cache reuse (both backends)

Multi-turn latency is cut by appending only the new turn's delta instead of
re-prefilling the whole conversation:

- **ORT GenAI** — persistent generator (turn-2 prefill speedup; figures in
  [benchmarks.md](benchmarks.md)).
- **llama.cpp** — persistent `llama_context` in `LlamaSession`; `reuse_kv` /
  `reset_kv` clear or continue the KV cache (`llama_memory_clear`, position
  continuation). Measured ratios by model: [benchmarks.md](benchmarks.md).

Reuse is gated by `kv_reuse_supported_for_model()` (`routing_policy.h`), which
excludes the DirectML EP (continuous decoding is CPU-only, verified — the reuse
turn produces zero cached tokens on DML). It is **not** gated by backend — GGUF
KV-reuse is on.

### KV snapshot pool (`KvStore`)

Leaving a conversation writes its KV to `LocalState\kv\<id>.kv` and the first
turn back restores it, so the switch costs a delta prefill instead of the
history. The pool is managed by `KvStore` (`include/xllama/kv_store.h`):

- **Fingerprinted** (model, n_ctx, KV quant, LoRA) — stale snapshots are
  harmless by construction (they fall back to a normal prefill).
- **Atomic writes** in 8 MB chunks (§9 AppContainer bound).
- **Pool cap**: 3 files / 192 MB total, LRU eviction.
- Dev Mode ships with ~2.2 GB free; a snapshot is ~12 KiB per resident token,
  so the pool is sized to fit ~16 conversations at the default `n_ctx` 2048.

A conversation that is stripped to empty (thinking model truncated to no answer)
takes no KV snapshot (#170b).

### Cancel policy (`cancel_policy.h`)

The UI has one Cancel affordance and three jobs that can be running behind it —
text inference, image generation, and on-device training — each stopped by a
different mechanism. `cancel_target(image_running, training_running, text_running)`
returns the correct `CancelTarget` (precedence: image > training > text). The
decision is pure policy over three booleans, exhaustively tested on the host.

The **process** keeps one resident session at a time: since PR #161/#164 it
lives in `xllama::session_hub()` (`include/xllama/session_hub.h`), the single
owner shared by the chat UI and the LAN API — "avoid 2× model in RAM" (the CPU
and DML working sets, ~1.3 GB and ~2.9 GB, do not coexist in the budget) is now
a process-wide invariant instead of a per-surface convention the API silently
violated. `EnsureSession` swaps the resident model through the hub under its
mutex; `hub.generation` lets the UI drop its KV-reuse state when the API
swapped models between turns. The routing token-count no longer loads a model
at all: it uses the exact count only when the CPU session is already resident,
else the same chars/5.0 estimator that bounds the prompt in `BuildPrompt` (the
old shape loaded the CPU model on the UI thread just to tokenize, then could
destroy it for DirectML — the historical "two model loads" cost recorded in
§5d). When a model becomes Ready the session is **pre-loaded** in a detached
worker (`PreloadSessionAsync`), so the first send pays prefill+decode only.

## Routing (`routing_policy.h`)

`decide_routing()` picks the execution path per conversation and is **sticky** —
decided on the first turn, when the app cannot yet know whether the conversation
will continue, and never revisited (the KV cache is per-EP). Decode-heavy chat
stays on **CPU int4** (fastest decode); a long first-turn prompt can switch to
**DML fp16**, which wins long-prompt prefill and therefore first-turn TTFT
(in-app DML turns run at the warm regime since the load warm-up + pre-load,
§5e — "cold-process" now describes bench figures only). This
does **not** mean the GPU is faster overall: DirectML cannot reuse a KV cache
(`kv_reuse_supported_for_model` returns false for it) while the CPU can, so from
the second turn the CPU wins at every reachable length — see `uwp-constraints.md
§5d`. The threshold (`token_threshold`) is calibrated on a misattributed
variable (§5c); the re-derivation concluded a single prompt-length threshold is
the wrong shape for the decision, so 1550 is left as-is with its rationale
corrected. Routing is **ORT-only**; GGUF models are CPU-only, so routing is skipped
for them. Tunable prefill batching for the llama.cpp path is exposed via
`SessionParams.n_batch` / `n_ubatch` (`xllama-cli --batch/--ubatch`) — see
`benchmarks.md` for the (flat) sweep.

Sampling has one code path per backend, both fed by `SamplingConfig` and its
defaults in `sampling.h`. The **llama.cpp** chain is assembled in exactly one
place, `src/bridge/sampler_chain.h` (`add_sampler_stages`, #125); the **ORT
GenAI** search params in one place, `src/bridge/ort_sampling.h`
(`apply_ort_sampling`, #141) — including the greedy guard (temperature 0 pins the
argmax and skips the repetition penalty). #125 unified only the llama.cpp side;
the ORT params stayed hand-duplicated across `run_inference_ort` and
`OrtSession::make_params` and drifted on that guard until #141 gave them a shared
builder too. Now neither backend's two surfaces (CLI/bench vs GUI/API) can diverge
by construction.

The **prefill and generation loops** followed the same route, later and for the
same reason (`src/bridge/decode_loop.h`: `prefill_chunked`,
`prompt_too_long_message`, `decode_loop`). `run_inference_llama` and
`LlamaSession::generate` kept hand-maintained copies until 2026-07-30, and the
copies had drifted twice over: #193 — a prompt past the logical batch aborting the
process — had to be fixed in both, and the stop-sequence token count differed, so
`n_eval` and the `decode_tok_s` derived from it disagreed by one between the two
paths for the same generation. Each caller keeps only what is genuinely its own:
`LlamaSession` the #170a prefix diff, the #169 context shift and `m_kv_tokens`
(fed by an `on_accepted` callback so the record stays in step with the KV cells);
`run_inference` its context lifetime and the CLI's stdout echo.

**The rule these three share**: a decision that both surfaces must make the same
way lives in exactly one header. Copying it is not a style question — every copy
in this codebase has eventually disagreed with the other, and always silently.

### The autopilot script contract

`include/xllama/autopilot.h` applies the same rule to a different axis. The
driver (`MainPageController::ApRun`) can only exist on UWP — it dispatches to
XAML on the UI thread — but _what a valid script is_ needs neither XAML nor a
console, so it lives in a WinRT-free header with `validate_autopilot_script` and
is covered by `tests/test_autopilot.cpp`. That matters because all ten console
gates are written in this language: a rule that is wrong here is wrong for all
of them, and until the split there was nothing to test it with short of
deploying to hardware and watching what broke.

The split also fixes _when_ the checks run. They used to sit inside each branch
of the dispatch chain, so a bad op name or an out-of-range value in action 7 was
found only after actions 0–6 had already applied — leaving the console
half-scripted, with a failure that reads like a product failure rather than a
typo. Everything decidable without the device is now decided before the first
action runs; the driver keeps only what genuinely depends on runtime state
(whether a chat file exists, whether a port binds, whether the UI is busy).

`scripts/check-coherence.py` reads the op table out of `autopilot.cpp` and
asserts it against `ApRun`'s branches in both directions: an op the validator
accepts with no branch would fail at run time after passing validation, and a
branch for an op the validator rejects is unreachable code that looks supported.

## Model catalogue, provisioning, and quant auto-upgrade

The catalogue is `uwp/models/manifest.json` (bundled base) merged with an optional
reinstall-free `LocalState\manifest.json` override, per entry, by
`merge_manifest_entries` (`manifest_merge.h`): a same-`name` entry replaces the base
one, a new name is appended, unmentioned entries stay.

### Catalogue session policy (`n_ctx`, `role`)

Optional per-entry fields used at **session open** (not only download). Helpers
live next to the prompt budget in `routing_policy.h` so the trimmer and
`kDefaultNCtx` cannot drift independently (#171 / #133 class):

| Field       | Contract                                                                                                                                                                                                                                                        |
| ----------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`n_ctx`** | `0`/omit → `kDefaultNCtx` (2048). Else clamped by `resolve_n_ctx` to **[512, 8192]**. Applied in `EnsureSession` and the LAN chat handler when opening the hub session. Coding catalogue entries use **4096**.                                                  |
| **`role`**  | `""` (default chat) or **`"coding"`**. Effects only: (1) UI trimmer uses `kEstimatedCharsPerTokenCoding` (3.5) instead of prose 5.0; (2) LAN `POST /v1/chat/completions` with **empty** `system` fills `kCodingSystemPrompt` instead of `kDefaultSystemPrompt`. |

**Explicit non-effects (do not re-introduce):**

- Settings system-prompt text is **never** rewritten when the model or role
  changes. The box is user-owned; magic string-equality swaps are debt.
- `role` does **not** select a backend, a template, or a sampler. Backend stays
  `kind` + `Backend::Auto`; template stays `chat_format_for(model_id)`.
- There is **no** third “coding” pillar or FIM path. Coding chat is the same
  GGUF/`Session`/`ChatFormat` stack as general chat.

### Context budget: one enforcement point

The budget is enforced **once**, in tokens, where the tokenizer lives —
`xllama::fit_prompt` (`prompt_budget.h`), called by the turn worker after
`EnsureSession`, with `Session::count_tokens` of the model that will generate. It
drops the oldest turns until `n_tokens + max(n_predict, 250) + 1 ≤ n_ctx`, never
drops the trailing user message, and reports `fits = false` when even that message
alone does not fit (the session then reports `prompt too long` with the numbers).

A chars-per-token estimate cannot do this job. Measured on console: prose came out
4.6 real chars/token against the 5.0 constant and dense C++ 2.5 against 3.5, and
because the generation loop clamps `n_predict` to what the context has left
(#173), every token of undershoot is taken **off the reply**, silently. One
sample per workload is not a bound, and a "safer" constant only trades a truncated
answer for history dropped that would have fit.

So the estimate keeps exactly one job: **routing**. `decide_routing` needs a token
count before a model — hence a tokenizer — has been chosen, and its ceiling has to
stay coherent with `token_threshold` (#133). `BuildPromptPlan` therefore trims by
`max_prompt_tokens_for_n_ctx(n_ctx)` with the optimistic `kEstimatedCharsPerToken`,
and being optimistic is the point: it drops _fewer_ turns, so the exact pass can
only tighten and nothing routing saw reappears behind its back. Getting that
estimate wrong costs a routing decision, not an answer.

**The two ceilings are separate on purpose.**
`max_prompt_tokens_for_n_ctx(n_ctx)` → `n_ctx − 250`, floor 256, with the shipping
context keeping the historical `kMaxPromptTokens` (1800) so the #171 pin cannot
drift by arithmetic. It is a **context** bound and takes no `n_predict`: the
reply's room belongs to `fit_prompt`, exactly and later. Charging the reply's
reserve to this ceiling is not a shortcut, it is a feature killer — at the shipping
`n_predict` of 512 the ceiling becomes 1536, _below_ `token_threshold` (1550), so
`decide_routing` can never see a long turn and auto GPU routing dies on every
default install. That is #133 a third time; `tests/test_routing_policy.cpp` now
pins the reachability in real tokens, and the `routing` console gate runs at the
shipping `n_predict` rather than a convenient one.

Both surfaces enforce the same budget with the same primitive: the chat UI in its
turn worker, and `POST /v1/chat/completions` before it generates (a client whose
final message alone cannot fit gets a `400`, not a `500` from the generator).

**Known gap, tracked.** The routing decision itself still runs on the UI thread
_before_ a session exists, so on a cold first turn with `routing = auto` it counts
with the estimate rather than a tokenizer — a heuristic deciding behaviour, which
the rule above forbids. It affects only that mode (the shipping default is
CPU-only) and the fix is mechanical: decide in the turn worker, after
`EnsureSession`, where the exact count exists by construction. Deliberately not
bundled into the phase14 release branch — it rewrites the hottest UI path and wants
its own PR with the `routing` console gate as the acceptance test.

The prefill itself is chunked at `llama_n_batch` (`LlamaSession::generate`,
`run_inference`): an oversized logical batch is not an error return from
`llama_decode` but a `GGML_ASSERT` abort, and `n_batch` defaults to
`min(n_ctx, 2048)` — well below the 4096-token coding ceiling.

Inventory / ship status of models: [model-matrix.md](model-matrix.md). Ops for
adding entries: [model-selection.md](model-selection.md).

`EnsureModelNamedAsync` (`uwp/MainPage.cpp`) provisions a model through the chain
LocalState → bundled InstalledPath → USB → **catalogue download** (from the entry's
`hf_base_url` — GitHub `models-v1` Release for most, Hugging Face for >2 GB GGUFs).

**Quant auto-upgrade** (v1.1.7.0). The provisioned-check is expected-aware:
`IsModelProvisioned(name, expected_files)` (`uwp/model-downloader.cpp`) delegates to
the pure `dir_satisfies_expected_files` (`model_provision.h`), which requires the
manifest's current `files[].filename` to be present (separator/case-insensitive)
rather than accepting _any_ `.gguf`. In expected mode the `.complete` marker is not
a fast-path on its own (a stale-quant dir can carry a valid old marker). Before
re-downloading, `EnsureModelNamedAsync` **reconciles the dir** — deletes any
non-expected `*.gguf` and drops the stale `.complete` — so old and new quants never
coexist (which would otherwise let `first_gguf_in_dir` load the wrong file). Net
effect: a directory holding an older quant than the manifest names (e.g. a stale
`gemma-4-E2B-it-UD-IQ2_M.gguf`) auto-upgrades to the current `Q3_K_S`. When there is
no catalogue entry the check falls back to the historical loose behavior. The pure
predicate has host doctest coverage (`tests/test_model_provision.cpp`).

## CPU memory-bandwidth micro-bench (`membw`)

Decode is a bandwidth-bound M=1 GEMV, so `measure_membw` (`src/bridge/membw.cpp`,
STREAM-style read/copy/triad over a 256 MB buffer, single- and multi-thread) pins
the sustained DRAM ceiling that bounds it. Exposed as `xllama-cli --membw` (host)
and the headless `membw.flag` mode on Xbox (`uwp/App.cpp` → `inference-bridge.cpp`,
writes `membw-result.csv`). Console result substantiates the "~13 GB/s effective
GEMV" figure (read 12.35 GB/s @1t; `benchmarks.md`).

## Probe suite (bandwidth, heap, disk, GPU)

A family of micro-bench probes that pin the platform's physical ceilings:

| Probe          | Header      | What it measures                                                       | Console exposure         |
| -------------- | ----------- | ---------------------------------------------------------------------- | ------------------------ |
| **CPU membw**  | `membw.h`   | STREAM read/copy/triad over 256 MB (DRAM ceiling)                      | `--membw` / `membw.flag` |
| **Disk bw**    | `diskbw.h`  | NVMe sequential + random read (4 GiB file, 8 MiB / 2 MiB blocks)       | `--diskbw`               |
| **RAM ceil**   | `ramceil.h` | Heap commit ceiling (steps of 128 MB, page-fault in, record counters)  | `--ramceil`              |
| **GPU STREAM** | `gpubw.h`   | D3D12 compute-shader STREAM read over ~1 GiB VRAM (kill gate 100 GB/s) | `gpubw.flag`             |
| **GPU GEMV**   | `gpugemv.h` | Wave32 Q4_K GEMV density (G1 correctness + G2 density ≥ 40 GB/s)       | `gpugemv.flag`           |

All probes share the same design: pure helpers (pattern fill, checksum, CSV
serialization) are host-testable on Linux; the D3D12/Windows path returns
`d3d12_ran=false` on non-Windows. Results are CSV-serialised with self-contained
schemas (not the model-bench schema) and written to `bench/results/`.

The **kill gates** are predeclared in the headers (not in docs) so that scripts
and code stay in sync: `gpubw.h:kGpubwKillReadGbs` (100 GB/s),
`gpugemv.h:kGpugemvKillPackedGbs` (8 GB/s = K1 kill, 40 GB/s = K2 gate).

## Logit-parity harness

`logit_dump.h` / `test_logit_parity.cpp` — a portable harness for verifying that
both backends (llama.cpp and ORT GenAI) produce identical logits for the same
prompt. Writes:

- `<path>` — raw float32 little-endian, `vocab_size` values (last token)
- `<path>.json` — `{ model, prompt, backend, vocab_size, greedy, top1_id, top1_piece }`

The binary format matches `numpy.fromfile(dtype=np.float32)`. Dumps from both
backends are compared by `scripts/compare-logits.py`. This is the gate for
publishing ORT model assets: every asset must pass the logit-parity gate before
`gh release upload` (runbook in `docs/model-selection.md#publishing-ort-model-assets-models-v1--logit-parity-gate`).

## Diffusion pipeline

A from-scratch C++ pipeline (`uwp/diffuse.cpp`, `diffusion/`): CLIP BPE tokenizer
(`diffusion/clip_tokenizer.h`) + Euler scheduler (header-only, golden-vector
unit-tested, `diffusion/euler_scheduler.h`) driving **three sequential** ORT
DirectML sessions (text encoder → UNet → VAE decoder), created→run→destroyed per
stage to fit the console GPU budget (`uwp-constraints.md` §5/§7). Plain ORT
DirectML coexists with the XAML compositor in-process (unlike ORT GenAI's DML
init — §7). Optional TAESD tiny VAE shortens the decode stage. PNG output via
`diffusion/png_writer.h`; half-precision floats via `diffusion/half.h`. Triggered
from the Image dialog or the headless `diffuse.flag`.

## LAN HTTP endpoint (OpenAI-compat)

Optional, **default OFF**: a `StreamSocketListener` in `uwp/api-server.cpp`
exposes the process-wide `session_hub()` Session (shared with the chat UI —
see the session-ownership section above) plus thin UI-parity routes on the
LAN. Settings owns the live start/stop/rebind lifecycle; `App::OnLaunched`
restores it when the persistent `LocalState\api.flag` is present. The flag is
not consumed and the server coexists with the live chat UI. Port 11434
(override `api-port.txt`); chat and images take the hub mutex with `try_lock`
→ HTTP 503 when busy — busy includes a running **chat-UI turn**; during the
session pre-load the request waits briefly instead (`acquire_hub_or_busy`,
≤15 s). Stopping the endpoint does not free the Session (it is hub-owned).
Preferences and training status are file I/O only (no inference lock).

| Route                         | Role                                                                |
| ----------------------------- | ------------------------------------------------------------------- |
| `POST /v1/chat/completions`   | Non-streaming chat (own `Session`)                                  |
| `POST /v1/preferences`        | Append preference sample → `training/samples.jsonl` (#118)          |
| `GET /v1/training/status`     | `result.done` / `progress.json` / personalized `result.json` (#118) |
| `POST /v1/images/generations` | SD-Turbo in-process (steps 1–4), same knobs as Image dialog (#118)  |

Capability `privateNetworkClientServer` covers LAN inbound; no public inbound.
Full contract + validation: [api-endpoint.md](api-endpoint.md)
(`scripts/validate-api.sh` — `spike|chat|prefs|train|all`).

## Build variants and versioning

CI (`build-uwp.yml`) produces `xllama-appx` (**unified**: ORT + llama.cpp +
hash-pinned **PatchedGenAI #2280** + **PatchedOrt** extdata DLLs from
`vendor-dlls-v1`) and `xllama-appx-llamacpp` (llama.cpp only). The MSIX version is
`Major.Minor.Build` from `uwp/AppxManifest.xml` (bumped per release) with the
**Revision auto-stamped from the CI run number** (`build-uwp.ps1 -BuildRevision`),
so in-place console updates never collide on identity. Exception: **1.5.0.0
changed the package identity itself** (`VenereLabs.xllama` →
`GianlucaMazza.xllama`, PR #163) — across that boundary there is no in-place
update: it installs as a new app and LocalState does not carry over (see
`install-release.md`). First-launch chat default
on unified builds is **`lfm25-350m`** (`DefaultChatModelId()` in `MainPage.cpp`).

## Training pillar (exploration)

First-class subsystem for **learning adapters and producing loadable weights**.
**SSOT (RE inventory, capability matrix, hybrid loop):**
[training-architecture.md](training-architecture.md). Platform facts:
[uwp-constraints.md](uwp-constraints.md) §13. Ops: [`training/README.md`](../training/README.md).

```
┌─────────────────────────┐     artefacts      ┌──────────────────────────┐
│ TrainingJob (A or B)    │ ─────────────────► │ Inference Session        │
│ host PEFT / partial FT  │                    │ Lane C serve             │
└─────────────────────────┘                    └──────────────────────────┘
 Lane B = last-block ggml-opt partial FT; host + console gates PASS (available)
```

### Contracts (C++)

- `TrainingJob` / capability table — `training_params.h` / `training.h`
- `training_capabilities()` / `xllama-cli --training-capabilities` (RE matrix)
- `validate_training_job` accepts `device=device` only when the bounded Lane B
  engine is compiled (`XLLAMA_DEVICE_TRAIN`)
- CLI: `--validate-train-job`; `--train-job` dispatches host PEFT or the
  in-process partial-FT engine by method

### Capability headline (see SSOT for full RE)

| Lane                    | Status today                                                               |
| ----------------------- | -------------------------------------------------------------------------- |
| **A Host PEFT + merge** | **Available** (marker job PASS)                                            |
| **B Device partial FT** | **Available** in llamacpp/unified builds; host + console marker gates PASS |
| **C Serve merged GGUF** | **Available**; runtime LoRA via `SessionParams.lora_path` / CLI `--lora`   |

### Personalization status (Phases 8–11)

| Phase   | What shipped (headline only)                                   |
| ------- | -------------------------------------------------------------- |
| **8–9** | Host PEFT, runtime LoRA, preference UI, operator pull loop     |
| **10**  | Lane B on-device `partial_ft` available (marker gates PASS)    |
| **11**  | In-app train + publish `personalized`; LAN prefs/status/images |

**Do not expand the Phase 11 flow here** — code map, preflight, and diagram live
in [training-architecture.md §11](training-architecture.md). Pad steps:
[using-the-app.md](using-the-app.md). Headless harness remains `train.flag` /
`validate-console-training.sh device-train`.

## Inference surfaces: in scope vs deferred

The shipping product is **multi-turn chat** (UI + optional LAN OpenAI-compat
chat) on a **single resident session**. The following are **not** implemented
and must not be half-added:

| Surface                                          | Status          | Why deferred / rule                                                                                                                                               |
| ------------------------------------------------ | --------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Chat instruct (all catalogue text models)        | **In scope**    | One `ChatFormat` + `Session::generate`                                                                                                                            |
| Coding **chat** (`role: coding`, larger `n_ctx`) | **In scope**    | Same path; catalogue policy only                                                                                                                                  |
| Thinking models (basename `thinking`)            | **In scope**    | ChatML; `strip_thinking_content` → `postprocess_output` drops `<think>…</think>` for display/persist (KV still saw full stream). Catalogue: `lfm25-1.2b-thinking` |
| Prompt-lookup speculative (#210, k=2)            | **In (opt-in)** | Same `decode_loop` / `Session`; **default OFF** after console M3 1.04× FAIL gate. Not a second backend. Campaign: [phase15-re-opt.md](phase15-re-opt.md)          |
| FIM / fill-in-middle / IDE completion            | **Out**         | Second prompt surface (`render_fim` + completions route); not wired                                                                                               |
| Tool-calling / agent loops                       | **Out**         | Schema + multi-step orchestration above `Session`                                                                                                                 |

Two consequences of "the history is stripped, the KV is not", both deliberate:

- **No KV snapshot (#170b) for thinking models.** On return, the rendered prompt
  (stripped) diverges from the resident tokens (full CoT) inside the first
  assistant turn, so the #170a prefix diff collapses — on LFM's hybrid cache, to a
  full re-prefill. `SaveKvSnapshotAsync` returns early instead of writing tens of MB
  to buy nothing. In-conversation delta reuse is unaffected: there the KV is the
  truth and `render_delta` only closes the turn.
- **An answer can postprocess to empty** when the reasoning block is truncated
  (`n_predict` exhausted). The UI substitutes an explicit "reasoning only" turn:
  before, the message was dropped silently, leaving the raw CoT orphaned on screen
  and a user turn with no reply in the saved history.

**Gate to catalogue:** measure on host Release → console headless bench → only
then a `manifest.json` entry with a **complete** product path (template, load,
session policy, and for thinking: postprocess). Measured ≠ shipped without that.

## Model validation ladder (best practice)

One direction, no parallel “truths”:

```text
host Release smoke (quality + peak)
    → console headless bench (tok/s, peak, n_ctx)
        → phase*.csv under bench/results/
            → optional row in bench/benchmark-summary.json
                → generate-benchmark-summary.py
                    → docs/benchmarks.md (numbers SSOT)
    → if product-complete: uwp/models/manifest.json
        → docs/model-matrix.md (status SSOT)
```

- **Do not** invent a second performance table in README or model-selection.
- **Do not** catalogue on host-only evidence for Series S claims.
- **Do not** special-case Settings when a catalogue field already owns the
  policy (`n_ctx`, `role`).
- **Do not** let a heuristic decide product behaviour. A chars-per-token estimate
  may _bound work_ (what the UI renders, what routing counts before a tokenizer
  exists); only an exact measurement may _decide_ what the user gets. Every #133
  recurrence — three so far — was an estimate promoted from bound to decision.
- **Do not** restate a number a generated file already owns unless a gate checks
  the copies agree. `check-coherence.py` now pins the `model-matrix.md` metrics
  rows against `phase14-console.csv`; that is the price of the second copy.

## Unit test map (host suite)

Every `include/xllama/X.h` has a corresponding `tests/test_X.cpp`. The suite
is **243 test cases / 4346 assertions** (doctest); CI enforces the count in
`build-linux.yml` against `ctest --list-tests`.

| Test file                     | Tests | Header under test                  |
| ----------------------------- | ----- | ---------------------------------- |
| `test_json_utils.cpp`         | 11    | `json_utils.h`                     |
| `test_cancel_policy.cpp`      | 8     | `cancel_policy.h`                  |
| `test_autopilot.cpp`          | 15    | `autopilot.h`                      |
| `test_routing_policy.cpp`     | —     | `routing_policy.h`                 |
| `test_prompt_budget.cpp`      | —     | `prompt_budget.h`                  |
| `test_personalize.cpp`        | —     | `personalize.h`                    |
| `test_model_provision.cpp`    | —     | `model_provision.h`                |
| `test_session.cpp`            | —     | `session.h`                        |
| `test_sampling.cpp`           | —     | `sampling.h`                       |
| `test_training.cpp`           | —     | `training.h` / `training_params.h` |
| `test_device_train.cpp`       | —     | `device_train.h`                   |
| `test_preference_capture.cpp` | —     | `preference_capture.h`             |
| `test_chat_prompt.cpp`        | —     | `chat_prompt.h`                    |
| `test_manifest_merge.cpp`     | —     | `manifest_merge.h`                 |
| `test_path.cpp`               | —     | `path_utils.h`                     |
| `test_utf8.cpp`               | —     | `utf8_utils.h`                     |
| `test_bench.cpp`              | —     | `bench.cpp`                        |
| `test_chat_history.cpp`       | —     | `chat-history.h`                   |
| `test_api_config.cpp`         | —     | API config validation              |
| `test_cli.cpp`                | —     | `cli.h`                            |
| `test_diskbw.cpp`             | —     | `diskbw.h`                         |
| `test_gpubw.cpp`              | —     | `gpubw.h`                          |
| `test_gpugemv.cpp`            | —     | `gpugemv.h`                        |
| `test_kv_store.cpp`           | —     | `kv_store.h`                       |
| `test_logit_parity.cpp`       | —     | `logit_dump.h`                     |
| `test_membw.cpp`              | —     | `membw.h`                          |
| `test_ramceil.cpp`            | —     | `ramceil.h`                        |
| `test_speculative.cpp`        | —     | `speculative.h`                    |
| `test_diffusion.cpp`          | —     | `diffusion/`                       |

## UWP headless flag registry

`uwp/App.cpp` (`HeadlessView::Run`) activates a `CoreWindow` without a swapchain
or XAML compositor, giving D3D12-clean hosts for DirectML work. The following
flags are supported:

| Flag                  | Entry point              | Purpose                                                |
| --------------------- | ------------------------ | ------------------------------------------------------ |
| `bench.flag`          | `main_loop`              | Model benchmark (tok/s, peak, latency)                 |
| `diffuse.flag`        | `run_diffuse`            | SD-Turbo diffusion pipeline (headless)                 |
| `diffuse-inproc.flag` | `run_diffuse` in-process | Diffusion on background MTA thread inside XAML process |
| `membw.flag`          | `run_membw`              | CPU STREAM bandwidth probe                             |
| `diskbw.flag`         | `run_diskbw`             | NVMe sequential + random read probe                    |
| `gpubw.flag`          | `run_gpubw`              | GPU STREAM probe (D3D12 compute shader)                |
| `gpugemv.flag`        | `run_gpugemv`            | Q4_K GEMV density probe (D3D12 compute shader)         |
| `ramceil.flag`        | `run_ramceil`            | Heap ceiling probe (commit in steps)                   |
| `mic.flag`            | `run_mic_probe`          | Microphone / AudioGraph probe                          |
| `logits.flag`         | `run_logits`             | Logit-parity dump (float32 + JSON sidecar)             |
| `oprepro.flag`        | `run_oprepro`            | Single-op CPU-vs-DML diagnostic (`repro.onnx`)         |
| `train.flag`          | `run_train`              | On-device training (Lane B partial FT)                 |
| `api.flag`            | `run_server`             | LAN API server (persistence via `LocalState\api.flag`) |

## UWP front-end summary

`MainPageController` (`uwp/MainPage.cpp`) owns the chat UI surface. Key flows:

- **B button** (`BackRequested`): cancels a running inference; otherwise does
  nothing. Unhandled `BackRequested` suspends the app — the handler now always
  marks the event handled.
- **Gamepad**: `GamepadView` clears output; `GamepadY` focuses the prompt box.
- **Token streaming**: `m_token_buffer` drained to screen by a 40 ms
  `DispatcherTimer` (`FlushTokenBuffer`); live tok/s counter derived from
  `m_first_token_at`.
- **Smart autoscroll**: `m_at_bottom` tracked via `ScrollViewer.ViewChanged`;
  auto-scroll suppressed during manual scroll-up.
- **Feedback controls**: Like/Dislike/Correct buttons with `ContentDialog`
  correction dialog (TextBox) that rebuilds the assistant message.
- **History dialog**: `ContentDialog` with `ListView`, per-item delete, clear-all;
  leak fix via `winrt::make_weak(dlg)` (#219).
- **KV snapshot save**: `WaitKvSnapshotAsync` (#216 race fix) +
  `SaveKvSnapshotAsync` with thinking-model early-return and ORT-backend skip.
- **Catalogue model knobs**: `ApplyCatalogueModelKnobs` applies `n_predict` from
  manifest on model select (thinking models get 1024).
- **Manifest cache**: parsed once, cached; invalidated on personalized-model
  publish.

`ChatHistory` (`uwp/chat-history.h`) persists conversations as JSON files under
`LocalState\chats\<id>.json` (`id`, `title`, `messages[]` with `role`, `content`,
`ts_unix`, `partial`, `feedback_label`). An `index.json` (sorted by
`last_modified` desc) drives the History dialog. `TitleFrom()` takes the first
non-empty line of the first user message (truncated at 60 chars). IDs use
`CoCreateGuid`.

`ManifestEntry` (`uwp/model-downloader.h`) schema:

| Field         | Type          | Default       | Effect                                                     |
| ------------- | ------------- | ------------- | ---------------------------------------------------------- |
| `name`        | `wstring`     | —             | Catalogue identifier                                       |
| `display`     | `wstring`     | —             | User-facing name                                           |
| `kind`        | `wstring`     | `"ort-genai"` | `"ort-genai"` / `"diffusion"` / `"gguf"`                   |
| `hf_base_url` | `wstring`     | —             | Download source (GitHub Release / HF)                      |
| `files`       | `ModelFile[]` | —             | `{filename, remote, approx_bytes, sha256}`                 |
| `lora`        | `wstring`     | —             | GGUF LoRA relative to model dir                            |
| `lora_scale`  | `double`      | `1.0`         | LoRA scale factor                                          |
| `n_ctx`       | `int`         | `0`           | `0` → `kDefaultNCtx` (2048); else clamped [512, 8192]      |
| `n_predict`   | `int`         | `0`           | `0` → keep Settings value; applied on model select         |
| `role`        | `wstring`     | `""`          | `""` (chat) or `"coding"` (4096 ctx, coding system prompt) |

`IsModelProvisioned(name, expected_files)` runs in dual mode: **expected-aware**
requires the manifest's `files[].filename` to be present (case-insensitive),
while **loose** mode (no catalogue entry) accepts any `.gguf`.

## CLI flags (complete enumeration)

`src/main.cpp` + `src/bridge/cli.cpp`:

| Flag                      | Purpose                                                      |
| ------------------------- | ------------------------------------------------------------ |
| `--model`                 | Model path or catalogue name                                 |
| `--prompt`                | Single-shot prompt                                           |
| `--chat`                  | Wrap prompt with chat template + stop tokens                 |
| `--max-length`            | Max generation tokens                                        |
| `--temperature`           | Sampling temperature                                         |
| `--top-p`                 | Top-p sampling                                               |
| `--top-k`                 | Top-k sampling                                               |
| `--repetition-penalty`    | Repetition penalty                                           |
| `--n-predict`             | Override `n_predict`                                         |
| `--batch`                 | `n_batch` (prefill chunk size)                               |
| `--ubatch`                | `n_ubatch` (physical ubatch)                                 |
| `--threads`               | Thread count                                                 |
| `--kv-q8`                 | Enable KV quantization (q8_0)                                |
| `--lora`                  | LoRA path                                                    |
| `--prompt-lookup`         | Enable prompt-lookup speculative decoding                    |
| `--membw`                 | CPU STREAM bandwidth probe                                   |
| `--diskbw`                | NVMe disk bandwidth probe                                    |
| `--ramceil`               | Heap ceiling probe                                           |
| `--gpubw`                 | GPU STREAM probe (reports `d3d12_ran=false` on Linux)        |
| `--gpugemv`               | Q4_K GEMV density probe (reports `d3d12_ran=false` on Linux) |
| `--train-job`             | Run training job (JSON path)                                 |
| `--validate-train-job`    | Validate training job JSON                                   |
| `--training-capabilities` | Print capability matrix                                      |
| `--logits`                | Logit dump for parity comparison                             |
| `--help`                  | Usage                                                        |

Machine-readable output: `SPEC_STATS` line on stderr for bench scripts.

## Shader pipeline

`shaders/` contains HLSL compute shaders and their AOT-compiled DXIL headers:

| Shader                    | Purpose                       | Output                                |
| ------------------------- | ----------------------------- | ------------------------------------- |
| `gpubw_stream.hlsl`       | GPU STREAM read (~1 GiB VRAM) | `generated/gpubw_stream.dxil.h`       |
| `gpugemv_q4k.hlsl`        | Naive Q4_K GEMV               | `generated/gpugemv_q4k.dxil.h`        |
| `gpugemv_q4k_wave32.hlsl` | Wave32-optimized Q4_K GEMV    | `generated/gpugemv_q4k_wave32.dxil.h` |

Compile scripts: `scripts/compile-gpubw-shader.sh`,
`scripts/compile-gpugemv-shader.sh` (fxc/dxc → binary → C header).

## See also

- Performance numbers → [benchmarks.md](benchmarks.md)
- AppContainer constraints (§1–§13) → [uwp-constraints.md](uwp-constraints.md)
- Model catalogue / selection → [model-selection.md](model-selection.md) + `uwp/models/manifest.json`
- Inventory / ship status → [model-matrix.md](model-matrix.md)
- App UI (incl. personalize) → [using-the-app.md](using-the-app.md)
- LAN protocol → [api-endpoint.md](api-endpoint.md)
- v1.0 narrative snapshot → [technical-report.md](technical-report.md)
- **Training SSOT** → [training-architecture.md](training-architecture.md)
- Training ops → [training/README.md](../training/README.md)
