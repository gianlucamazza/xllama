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

Bridge sources of note under `src/bridge/`: `session.cpp`, `inference.cpp`,
`training.cpp`, `device_train.cpp`, `personalize.cpp`, `preference_capture.cpp`,
`sampler_chain.h` / `ort_sampling.h` (one sampler chain per backend).

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
(`src/bridge/chat_prompt.cpp`) returns a `ChatFormat` describing the template:

- **ChatML** (`<|im_start|>…<|im_end|>`) — SmolLM2, Qwen, LFM.
- **Gemma** (`<start_of_turn>…<end_of_turn>`, no system role — the system prompt is
  merged into the first user turn, stop `<end_of_turn>`, `<bos>` via `add_bos`).

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

## Model catalogue, provisioning, and quant auto-upgrade

The catalogue is `uwp/models/manifest.json` (bundled base) merged with an optional
reinstall-free `LocalState\manifest.json` override, per entry, by
`merge_manifest_entries` (`manifest_merge.h`): a same-`name` entry replaces the base
one, a new name is appended, unmentioned entries stay.

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

## Diffusion pipeline

A from-scratch C++ pipeline (`uwp/diffuse.cpp`, `diffusion/`): CLIP BPE tokenizer +
Euler scheduler (header-only, golden-vector unit-tested) driving **three sequential**
ORT DirectML sessions (text encoder → UNet → VAE decoder), created→run→destroyed per
stage to fit the console GPU budget (`uwp-constraints.md` §5/§7). Plain ORT
DirectML coexists with the XAML compositor in-process (unlike ORT GenAI's DML
init — §7). Optional TAESD tiny VAE shortens the decode stage. Triggered from
the Image dialog or the headless `diffuse.flag`.

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

## See also

- Performance numbers → [benchmarks.md](benchmarks.md)
- AppContainer constraints (§1–§13) → [uwp-constraints.md](uwp-constraints.md)
- Model catalogue / selection → [model-selection.md](model-selection.md) + `uwp/models/manifest.json`
- App UI (incl. personalize) → [using-the-app.md](using-the-app.md)
- LAN protocol → [api-endpoint.md](api-endpoint.md)
- v1.0 narrative snapshot → [technical-report.md](technical-report.md)
- **Training SSOT** → [training-architecture.md](training-architecture.md)
- Training ops → [training/README.md](../training/README.md)
