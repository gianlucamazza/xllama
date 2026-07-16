# Architecture

Current component and data-flow map for xllama. This is the **SSOT for how the
system is put together** (module boundaries, backend dispatch, provisioning). It
complements the frozen [technical-report.md](technical-report.md) (a v1.0 narrative
snapshot), the perf SSOT [benchmarks.md](benchmarks.md), and the AppContainer
constraints SSOT [uwp-constraints.md](uwp-constraints.md) — those own the numbers
and the sandbox limits; this file owns the structure.

## Two targets, one core

The inference/pipeline logic is a platform-agnostic C++17 core; the UWP app and the
Linux CLI/tests are thin front-ends over it.

- **Core library** (`src/bridge/`, headers in `include/xllama/`) — built for both
  Linux (host dev + CI unit tests) and UWP. No WinRT in the headers.
- **Host front-ends** — `xllama-cli` (`src/main.cpp`, llama.cpp/GGUF + `--membw`)
  and the doctest suite (`tests/`).
- **UWP app** (`uwp/`) — XAML-free programmatic UI (`MainPageController`), the
  headless bench/diffuse/membw modes, and the model downloader; consumes the core
  through the same `xllama::Session` / `run_inference` API.

Header modules (`include/xllama/`), all WinRT-free so they are host-testable:

| Header                             | Owns                                                                |
| ---------------------------------- | ------------------------------------------------------------------- |
| `session.h` / `inference_params.h` | `Session`/`SessionParams`, `InferenceParams/Result`, `Backend` enum |
| `chat_prompt.h`                    | `ChatFormat`, `chat_format_for`, `apply_stop_sequences`             |
| `routing_policy.h`                 | `decide_routing`, `kv_reuse_supported_for_model`                    |
| `model_provision.h`                | `dir_satisfies_expected_files`, `normalize_model_path`              |
| `manifest_merge.h`                 | `merge_manifest_entries` (per-entry catalogue override)             |
| `membw.h`                          | `measure_membw` (STREAM-style bandwidth probe)                      |

## Inference backends and runtime dispatch

Two text backends, selected by build variant **and** per model at runtime:

- **ORT GenAI / DirectML** (`XLLAMA_USE_ORT`) — `OrtSession` in `src/bridge/session.cpp`.
  Runs ONNX GenAI models (`kind: "ort-genai"`); CPU int4 decode + DirectML fp16
  prefill, with per-conversation EP routing. **Text is currently forced to CPU
  in every mode** (`kDmlTextLogitsBroken`, #91 — the DML EP computes wrong text
  logits on the Series S driver); diffusion stays on GPU.
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
  continuation). Turn-2 prefill **4.07×** (`benchmarks.md`).

Reuse is gated by `kv_reuse_supported_for_model()` (`routing_policy.h`), which
excludes the DirectML EP (continuous decoding is CPU-only). It is **not** gated by
backend — GGUF KV-reuse is on.

## Routing (`routing_policy.h`)

`decide_routing()` picks the execution path per conversation (sticky — the KV cache
is per-EP): decode-heavy chat stays on **CPU int4** (fastest decode), long prompts
switch to **DML fp16** (prefill wins at scale). Routing is **ORT-only**; GGUF models
are CPU-only, so routing is skipped for them. Tunable prefill batching for the
llama.cpp path is exposed via `SessionParams.n_batch` / `n_ubatch`
(`xllama-cli --batch/--ubatch`) — see `benchmarks.md` for the (flat) sweep.

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

Optional, **default OFF**: a `StreamSocketListener` in `uwp/api-server.cpp` exposes the
same `xllama::Session` as an OpenAI-compatible endpoint on the LAN
(`POST /v1/chat/completions`, non-streaming). Started from `App::OnLaunched` on a detached
MTA thread when `LocalState\api.flag` is present — unlike the headless bench/diffuse flags,
`api.flag` is **not consumed** and the server coexists with the live chat UI. Port 11434
(override `api-port.txt`); single-slot with a `try_lock` → HTTP 503 when busy. Capability
`privateNetworkClientServer` (already in `AppxManifest.xml`) covers LAN inbound; no public
inbound. Full contract + validation in [api-endpoint.md](api-endpoint.md)
(`scripts/validate-api.sh`).

## Build variants and versioning

CI (`build-uwp.yml`) produces `xllama-appx` (**unified**: ORT + llama.cpp +
hash-pinned **PatchedGenAI #2280** + **PatchedOrt** extdata DLLs from
`vendor-dlls-v1`) and `xllama-appx-llamacpp` (llama.cpp only). The MSIX version is
`Major.Minor.Build` from `uwp/AppxManifest.xml` (bumped per release) with the
**Revision auto-stamped from the CI run number** (`build-uwp.ps1 -BuildRevision`),
so in-place console updates never collide on identity. First-launch chat default
on unified builds is **`lfm25-350m`** (`DefaultChatModelId()` in `MainPage.cpp`).

## Personalization / LoRA (host-only)

xllama is an **inference** runtime on Xbox (and a host CLI for the same GGUF
path). Full fine-tune / on-device training is out of scope for the UWP process
(GPU budget, AppContainer, ORT GenAI and llama.cpp forward-only APIs — see
[uwp-constraints.md](uwp-constraints.md)).

Personalization that stays coherent with this architecture:

1. **Train off-device** (host PEFT LoRA or any external trainer).
2. **Merge** into a plain GGUF (`llama-export-lora`) or ship a full finetuned
   quant.
3. **Serve** with existing `Session` / catalogue provisioning — no training loop
   in the app.

Host proof-of-pipeline (toy marker, SmolLM2-360M, CPU):
[`scripts/lora-spike/README.md`](../scripts/lora-spike/README.md). Runtime
`--lora` loading is intentionally **not** in `Session`; merge keeps Xbox and
the CLI on the same load path as any other GGUF.

## See also

- Performance numbers → [benchmarks.md](benchmarks.md)
- AppContainer constraints (§1–§12) → [uwp-constraints.md](uwp-constraints.md)
- Model catalogue / selection → [model-selection.md](model-selection.md) + `uwp/models/manifest.json`
- v1.0 narrative snapshot → [technical-report.md](technical-report.md)
- Host LoRA spike → [scripts/lora-spike/README.md](../scripts/lora-spike/README.md)
