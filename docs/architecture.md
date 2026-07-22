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

| Header                             | Owns                                                                         |
| ---------------------------------- | ---------------------------------------------------------------------------- |
| `session.h` / `inference_params.h` | `Session`/`SessionParams`, `InferenceParams/Result`, `Backend` enum          |
| `sampling.h`                       | `SamplingConfig`, shared defaults; CLI/bench and GUI/API init from it (#125) |
| `training.h` / `training_params.h` | `TrainingJob`/`TrainingResult`, stages, device gates, job JSON               |
| `chat_prompt.h`                    | `ChatFormat`, `chat_format_for`, `apply_stop_sequences`                      |
| `routing_policy.h`                 | `decide_routing`, `kv_reuse_supported_for_model`, prompt budget              |
| `model_provision.h`                | `dir_satisfies_expected_files`, `normalize_model_path`                       |
| `manifest_merge.h`                 | `merge_manifest_entries` (per-entry catalogue override)                      |
| `membw.h`                          | `measure_membw` (STREAM-style bandwidth probe)                               |

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
  (no ggml GPU backend), with KV-cache reuse.

The **`default`** and **`llamacpp`** CI variants compile a single backend. The
shipping **`unified`** build links both and dispatches **per model at runtime**:
`Backend::Auto` (`session.h`) resolves via `model_uses_llama_backend()` — a `.gguf`
suffix / on-disk GGUF layout routes to llama.cpp, everything else to ORT GenAI
(`session.cpp`, `inference.cpp`). So llama.cpp is both the host A/B benchmarking
lane and a **shipping** text backend for the GGUF catalogue entries (LFM2.5,
Qwen3.5, Gemma).

Both backends **stream**: `GenerateParams::on_token` fires per detokenized piece
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

- **ORT GenAI** — persistent generator (turn-2 prefill **4.87×**).
- **llama.cpp** — persistent `llama_context` in `LlamaSession`; `reuse_kv` /
  `reset_kv` clear or continue the KV cache (`llama_memory_clear`, position
  continuation). Turn-2 prefill measures **4.07×** on Gemma-3-270M and up to
  **20.02×** on the larger LFM models (`benchmarks.md`).

Reuse is gated by `kv_reuse_supported_for_model()` (`routing_policy.h`), which
excludes the DirectML EP (continuous decoding is CPU-only, verified — the reuse
turn produces zero cached tokens on DML). It is **not** gated by backend — GGUF
KV-reuse is on.

The UI keeps **one** session at a time. `MainPageController::EnsureSession` resets
the current session before creating another (`avoid 2× model in RAM`: the CPU and
DML working sets, ~1.3 GB and ~2.9 GB, do not coexist in the budget). One
consequence drives the routing verdict above: a GPU-routed turn loads the CPU
model to tokenize for the routing decision, then destroys it to load DirectML, so
that turn pays two model loads (§5d).

## Routing (`routing_policy.h`)

`decide_routing()` picks the execution path per conversation and is **sticky** —
decided on the first turn, when the app cannot yet know whether the conversation
will continue, and never revisited (the KV cache is per-EP). Decode-heavy chat
stays on **CPU int4** (fastest decode); a long first-turn prompt can switch to
**DML fp16**, which wins cold-process prefill and therefore first-turn TTFT. This
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
stage to fit the 3801 MB GPU budget. Plain ORT DirectML coexists with the XAML
compositor in-process (unlike ORT GenAI's DML init — see `uwp-constraints.md §7`).
Optional TAESD tiny VAE shortens the decode stage. Triggered from the Image dialog
or the headless `diffuse.flag`.

## LAN HTTP endpoint (OpenAI-compat)

Optional, **default OFF**: a `StreamSocketListener` in `uwp/api-server.cpp`
exposes a dedicated, lazily created `xllama::Session` as an OpenAI-compatible
endpoint on the LAN (`POST /v1/chat/completions`, non-streaming). Settings owns
the live start/stop/rebind lifecycle; `App::OnLaunched` restores it when the
persistent `LocalState\api.flag` is present. The flag is not consumed and the
server coexists with the live chat UI. Port 11434 (override `api-port.txt`);
single-slot with a `try_lock` → HTTP 503 when busy. Capability
`privateNetworkClientServer` (already in `AppxManifest.xml`) covers LAN inbound;
no public inbound. Full contract + validation in
[api-endpoint.md](api-endpoint.md) (`scripts/validate-api.sh`).

## Build variants and versioning

CI (`build-uwp.yml`) produces `xllama-appx` (**unified**: ORT + llama.cpp +
hash-pinned **PatchedGenAI #2280** + **PatchedOrt** extdata DLLs from
`vendor-dlls-v1`) and `xllama-appx-llamacpp` (llama.cpp only). The MSIX version is
`Major.Minor.Build` from `uwp/AppxManifest.xml` (bumped per release) with the
**Revision auto-stamped from the CI run number** (`build-uwp.ps1 -BuildRevision`),
so in-place console updates never collide on identity. First-launch chat default
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

### Personalization status (Phases 8–9)

Phase 8 is frozen complete: host PEFT + merge, runtime llama.cpp LoRA and
console preference capture are available and validated. Phase 9 adds the
operator-driven loop that pulls `LocalState\training\samples.jsonl`, retrains on
the host and emits a catalogue override for the resulting GGUF. Full device
fine-tuning, ORT ODT and ORT GenAI runtime adapters remain unavailable; Phase 10
adds only bounded last-block partial FT. The shipping UI records one
like, dislike, or correction per completed response, while export/retrain stays
operator-driven. See the training SSOT and ops guide below.

## See also

- Performance numbers → [benchmarks.md](benchmarks.md)
- AppContainer constraints (§1–§13) → [uwp-constraints.md](uwp-constraints.md)
- Model catalogue / selection → [model-selection.md](model-selection.md) + `uwp/models/manifest.json`
- v1.0 narrative snapshot → [technical-report.md](technical-report.md)
- **Training SSOT** → [training-architecture.md](training-architecture.md)
- Training ops → [training/README.md](../training/README.md)
