# Training architecture (SSOT)

> **SSOT for the training pillar** — dual-lane design, reverse-engineering
> inventory, capability matrix, hybrid personalization loop. Module map of the
> whole product: [architecture.md](architecture.md). AppContainer limits:
> [uwp-constraints.md](uwp-constraints.md) §13. Host runner ops:
> [`training/README.md`](../training/README.md).

**Currency:** 2026-07-20. Host PEFT marker job **PASS**. Lane B device train
**available** — in-process ggml-opt partial FT (`src/bridge/device_train.cpp`);
host and console marker gates **PASS** (console peak_ws 1195 MB, wall 446 s).
Design + pin constraints: §10. Roadmap: Phase 10.

### Exit criteria (Phase 8) — **MET**

| Criterion                               | Evidence                                  |
| --------------------------------------- | ----------------------------------------- |
| Dual-pillar architecture + RE inventory | this file + `uwp-constraints.md` §13      |
| Host PEFT → merge → A/B                 | `training/host/run_job.sh`, marker PASS   |
| Console serve finetuned GGUF            | `validate-console-training.sh serve` PASS |
| Runtime LoRA on console                 | `lora-rt` PASS on MSIX 1.2.0.546          |
| Preference capture on console           | `rate` PASS → `samples.jsonl`             |
| Device train not oversold at freeze     | full FT and ORT ODT rejected              |

**Frozen:** no further Phase 8 architecture work. Phase 9 delivered both the
operator loop and per-response preference UI. **Phase 10 (Lane B, §10)
supersedes the "training execution stays host-only" default**: an in-process
engine runs fully on the console, `available` with host + console marker-gate
evidence (2026-07-20: peak_ws 1195 MB, wall 446 s).

## 1. Why a training pillar

xllama personalizes and researches **on-device inference** (Xbox Dev Mode). Users
also need a path to **learn** adapters / domain behaviour. Training is a
**parallel pillar**, not a mode of `Session::generate()`:

| Pillar    | Loop                                        | Output                |
| --------- | ------------------------------------------- | --------------------- |
| Inference | forward only                                | tokens / images       |
| Training  | prepare → train → export → merge → evaluate | adapter / merged GGUF |

Artefacts flow **training → disk → inference load** (same GGUF/ORT catalogue
path). No optimizer state lives inside the chat hot path.

## 2. Target shape (three lanes)

```
┌──────────────────────────────────────────────────────────────┐
│                 docs/training-architecture.md (this file)    │
└────────────────────────────┬─────────────────────────────────┘
           ┌─────────────────┼─────────────────┐
           ▼                 ▼                 ▼
    Lane A Host PEFT   Lane B Device train  Lane C Serve
    IMPLEMENTED        AVAILABLE (gates ✓)  IMPLEMENTED
    training/host/*    ggml-opt partial FT  Session + llama
    device=host        device_train.cpp     runtime LoRA
           │                 │                 ▲
           └──────── merged GGUF / LoRA ───────┘
    Hybrid: PreferenceCapture (device) → dataset → Lane A → Lane C
```

### Non-goals (product)

- Pretraining from scratch on Xbox
- Full fine-tune of 350M–3B in the UWP process (partial FT of a filtered
  subset is Lane B, §10 — a different memory class)
- Python / PyTorch inside the MSIX
- Polluting `GenerateParams` with learning rate / epochs

## 3. Reverse-engineering inventory

### 3.1 Shipping NuGet (inference only)

From `uwp/packages.config`:

| Package                                  | Version | Training API?                       |
| ---------------------------------------- | ------- | ----------------------------------- |
| `Microsoft.ML.OnnxRuntime.DirectML`      | 1.24.4  | **No** — not `OnnxRuntime.Training` |
| `Microsoft.ML.OnnxRuntimeGenAI.DirectML` | 0.14.1  | **No** generate-only surface        |
| `Microsoft.AI.DirectML`                  | 1.15.4  | Kernels for forward EP              |

**RE conclusion:** our MSIX cannot call ORT Training without a **new package
pin and rebuild**. On-device training is not a config toggle.

### 3.2 ORT GenAI DLL symbols (vendor pin)

`strings` on `vendor/onnxruntime-genai-patched/win-x64/onnxruntime-genai.dll`:

| Finding                                                                          | Interpretation                                               |
| -------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| `OgaCreateAdapters`, `OgaLoadAdapter`, `OgaSetActiveAdapter`, `OgaUnloadAdapter` | **Inference-time** adapter container (not gradient training) |
| `No adapter is available for DML`                                                | GenAI adapters **not** available on DML EP for this pin      |
| No `TrainingSession` / train-loop exports in GenAI                               | GenAI remains decode/prefill                                 |

**RE conclusion:** GenAI adapters are Lane C (serve), and currently **DML-blocked**.
CPU EP adapter load is a future spike if we ship ORT-format adapters.

### 3.3 llama.cpp (submodule)

| API                                                             | Role                                                                               |
| --------------------------------------------------------------- | ---------------------------------------------------------------------------------- |
| `llama_adapter_lora_init` / `llama_set_adapters_lora`           | Runtime LoRA **load** (forward)                                                    |
| `examples/training` (`llama-finetune`)                          | WIP full FT; README cites **~24 GB** for 1B FP32                                   |
| `llama_opt_init` / `llama_opt_epoch` + `llama_opt_param_filter` | **In-process training** of a filtered tensor subset (ggml-opt) — the Lane B engine |
| `llama_model_save_to_file`                                      | On-device merged-GGUF export after training                                        |

**RE conclusion (updated for Lane B):** full FT via llama-finetune stays
**rejected** (RAM class). The same ggml-opt machinery with a **param filter**
trains a small subset in-process at PEFT-class memory — and llama/ggml
(incl. `ggml-opt.cpp`) are already compiled into the MSIX llama lane, so Lane B
needs **zero new dependencies**. Pin constraints in §10.

### 3.4 ORT On-Device Training (external product)

Microsoft documents an [on-device training](https://onnxruntime.ai/docs/get-started/training-on-device.html)
flow: offline artifact generation → on-device phase → export inference ONNX.

| Factor                                         | Assessment for xllama Xbox |
| ---------------------------------------------- | -------------------------- |
| Product fit (privacy personalization)          | Conceptually aligned       |
| In our tree / MSIX                             | **No**                     |
| AppContainer + 3801 MB GPU + #91               | Unproven; high risk        |
| Dual artefact format (ONNX train vs GGUF chat) | Ops cost                   |

**RE conclusion:** the ORT ODT path remains **research** only. It is distinct
from the ggml-opt Lane B engine and would require a new package and artefact path.

### 3.5 Measured resources (host PEFT run + console budgets)

| Quantity                                      | Value                      | Source                           |
| --------------------------------------------- | -------------------------- | -------------------------------- |
| Trainable params (SmolLM2-360M, r=8, q/k/v/o) | ~1.64 M (0.45%)            | host PEFT log                    |
| Adapter optimizer state                       | tens of MB (order of mag.) | scale from trainable count       |
| Base fp16 GGUF                                | ~720 MB                    | `training/out/.../base-f16.gguf` |
| Series S GPU budget                           | **3801 MB**                | `uwp-constraints.md` §5/§7       |
| CPU effective BW                              | ~12–13 GB/s                | membw                            |
| Full FT 360M Adam class                       | multi-GB                   | scale argument                   |

**RE conclusion:** **PEFT-class** state can fit device RAM _if_ a train engine
existed; **full FT cannot**. Software stack is the binding constraint today, not
only VRAM for LoRA-sized adapters.

### 3.6 UWP sandbox (recap)

No free `dlopen`, no Python in MSIX, LocalState-only writes, app-local DLLs —
any train engine must be **linked at build time** (see `uwp-constraints.md` §1–§4).

Refresh RE probe:

```bash
./scripts/re-training-stack.sh
```

## 4. Capability matrix (code + docs)

Queryable from C++ / CLI (`xllama-cli --training-capabilities`):

| Capability                   | Status        | Available now?                                                                  |
| ---------------------------- | ------------- | ------------------------------------------------------------------------------- |
| `HostPeftLora`               | available     | **yes**                                                                         |
| `HostMergeGguf`              | available     | **yes**                                                                         |
| `HostEvaluateMarker`         | available     | **yes**                                                                         |
| `RuntimeLoraLoadLlama`       | **available** | **yes** (`SessionParams.lora_path` / CLI `--lora`)                              |
| `RuntimeAdapterLoadOrtGenAI` | designed      | no (DML blocked on pin)                                                         |
| `DeviceOrtOnDeviceTraining`  | research      | no                                                                              |
| `DeviceLlamaFinetune`        | **rejected**  | no (full FT)                                                                    |
| `DeviceGgmlPartialFt`        | **available** | **yes** on `XLLAMA_DEVICE_TRAIN` builds (§10); host + console marker gates PASS |
| `DevicePreferenceCapture`    | **available** | **yes** (UI/autopilot → `training/samples.jsonl`)                               |

`training_device_supported(Device)` is compile-time: **true** on builds with
the Lane B engine (`XLLAMA_DEVICE_TRAIN`: Linux CMake always; MSIX llamacpp /
unified backends), false otherwise. `DeviceGgmlPartialFt` is `available`: host
and console `device-train` runs PASS with measured RSS/wall evidence (2026-07-20,
console peak_ws 1195 MB, wall 446 s).

## 5. Stage machine and job schema

Stages (`TrainStage`): `prepare` → `train` → `export_adapter` → `merge` →
`evaluate` → `publish`. After a successful merge/evaluation, the host runner
emits `manifest.override.json`; copying the GGUF and uploading/merging that
override into `LocalState` remains an explicit operator action.

Job JSON (schema_version 1) — examples:
`training/jobs/smollm2-360m-marker.json` (Lane A, `lora_peft`) and
`training/jobs/smollm2-360m-marker-partialft.json` (Lane B, `partial_ft`:
adds `param_filter`, `n_ctx_train`, `epochs`, `checkpoint_every`). C++:
`load_training_job_file` / `validate_training_job`.

Runners: Lane A `training/host/run_job.sh` (or `xllama-cli --train-job`);
Lane B `xllama-cli --train-job` in-process on host, `train.flag` +
`LocalState\training\job.json` on console (§10).

## 6. Hybrid personalization loop (operator-available)

```
Xbox chat + UI feedback ──► preference samples (LocalState JSONL) ──► export
                                                          │
Host TrainingJob (dataset=samples) ──► PEFT ──► merge GGUF
                                                          │
Xbox Session load ◄───────────────────────────────────────┘
```

### Preference sample line (JSONL)

```json
{
  "ts": "2026-07-17T12:00:00Z",
  "label": "like",
  "messages": [
    { "role": "user", "content": "..." },
    { "role": "assistant", "content": "..." }
  ]
}
```

Labels: `like` | `dislike` | `correction` (correction may include
`preferred_assistant`). No auto-upload: samples stay in the AppContainer until
the operator runs `training/host/pull_console_samples.sh`. The Phase 9 procedure
and `from-console-samples` job live in
[`training/README.md`](../training/README.md#hybrid-ops-phase-9-operator-loop).

## 7. Lane C — serve trained weights

| Path                                                 | Status                                                                    |
| ---------------------------------------------------- | ------------------------------------------------------------------------- |
| Load **merged GGUF** as catalogue / LocalState model | **Works today** (`Backend::Auto`)                                         |
| Runtime LoRA on llama.cpp without merge              | **Available** — `SessionParams.lora_path` + CLI `--lora` / `--lora-scale` |
| Runtime adapter on ORT GenAI                         | Designed — blocked for DML; CPU-only investigation later                  |

## 8. Decision log

| Decision                                           | Rationale                                                                                                    |
| -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| Host PEFT first                                    | RE shows inference-only NuGet; host PEFT proven PASS                                                         |
| Compile-gate `device=device`                       | Only builds containing the bounded ggml-opt engine accept it                                                 |
| Reject DeviceLlamaFinetune                         | ~24 GB class + WIP                                                                                           |
| Prefer merge GGUF over runtime LoRA for v1         | Zero Session change; same path as catalogue                                                                  |
| ODT as research not default                        | Package + format + console risk                                                                              |
| Lane B via ggml-opt, not ORT ODT                   | Engine already linked in the MSIX; single GGUF format; no new NuGet pin                                      |
| Lane B = last-block partial FT                     | Pin limitation: KV-cache `set_rows` has no backward (§10); honest fail-fast in prepare                       |
| `experimental` → `available` (resolved 2026-07-20) | Host and console evidence had to pass before the flip; both gates PASS (console peak_ws 1195 MB, wall 446 s) |

## 9. Open spikes (kill criteria)

| Spike                  | Kill if                                                                                                     |
| ---------------------- | ----------------------------------------------------------------------------------------------------------- |
| ORT ODT AppContainer   | Cannot link Training package or peak RSS > 3 GB on 360M PEFT toy (deprioritized: Lane B ships via ggml-opt) |
| Console `device-train` | Peak RSS > 3 GB on the 360M last-block marker job, or PLM kills the loop before an epoch checkpoint         |

## 10. Lane B — on-device training engine (Phase 10)

**What:** the whole pipeline — prepare → train → export → evaluate — runs
in-process on the device, CPU-only, via ggml-opt (`llama_opt_init` /
`llama_opt_epoch`). Code: `include/xllama/device_train.h` +
`src/bridge/device_train.cpp`. Method: `partial_ft` — a `param_filter` of
tensor-name substrings selects the trainable subset; everything else stays
frozen at its original type.

### Stages (all on device)

| Stage          | What happens                                                                                                                                                                                                                                                                                                                                                          |
| -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| prepare        | Selective **f32 upcast**: rewrite the base GGUF with filtered tensors stored as f32 (llama_opt trains f32 params only; frozen tensors keep f16/quant). Fail-fast filter validation. Dataset: preference-samples JSONL (chat-template rendered, `dislike` skipped, `correction` uses `preferred_assistant`) or plain text; tiny corpora are tiled to the window floor. |
| train          | `llama_opt_epoch` over `n_ctx_train`-token windows (stride ½), AdamW, constant LR, KV cache forced f32 (no f16 `OUT_PROD`). Progress callback per batch; cooperative abort between epochs; optional `checkpoint_every` GGUF snapshots.                                                                                                                                |
| export / merge | `llama_model_save_to_file` → `merged.gguf` (trained f32 subset + frozen originals; loadable by `Session`/catalogue as-is).                                                                                                                                                                                                                                            |
| evaluate       | In-process `Session` on the merged GGUF, greedy marker A/B (`eval.prompt` / `eval.expect_contains`).                                                                                                                                                                                                                                                                  |
| publish        | Operator action (manifest override + move under `models\`), same doctrine as Lane A.                                                                                                                                                                                                                                                                                  |

### Pin limitation — why the filter is last-block-only

On llama.cpp **b10025** the KV-cache write is a `ggml_set_rows` node with **no
backward implementation**: `ggml_build_backward_expand` asserts on any graph
where a gradient-carrying tensor feeds a cache write. Verified empirically:
upstream `llama-finetune` aborts identically on this pin (same
`GGML_ASSERT`, ggml.c:7143, node `cache_k_l1 (view)` SET_ROWS). Consequence —
trainable tensors must have **no downstream K/V write**:

- **allowed:** last block's `attn_q`, `attn_output`, `ffn_*`, norms; plus
  `output_norm` / `output`;
- **rejected:** any earlier block (a later layer's cache write sits on the
  gradient path) and the last block's `attn_k` / `attn_v` (they feed that
  block's own cache write), plus `token_embd` / `rope_freqs` (not optimizer
  parameters in the pinned llama.cpp path).

`device_train_unsupported_reason()` enforces this in prepare with a clear
error instead of a ggml abort. When a submodule bump (or a carried patch)
adds set_rows backward, widening the filter is a validation-table change, not
an engine change (ROADMAP Phase 10).

### Memory budget (SmolLM2-360M, last-block filter, n_ctx 256)

| Component                                    | Size                                                |
| -------------------------------------------- | --------------------------------------------------- |
| Frozen base (f16, no mmap)                   | ~700 MB                                             |
| Trainable subset f32 (≈9.22 M params)        | ~38 MB                                              |
| Gradients + AdamW moments (f32, 3×)          | ~115 MB                                             |
| KV cache f32 + activations + compute buffers | few hundred MB (scales with `n_ctx_train`)          |
| Peak working set (VmHWM, host, mid-run)      | **1 082 MB** (`bench/results/phase10-devtrain.csv`) |

The acceptance ceiling is 3 GB; `n_ctx_train` is the pressure knob. Host
throughput and epoch timings: `docs/benchmarks.md` §"On-device training
(Lane B)". Console `peak_ws_mb` measured at the gate: **1195 MB** (wall 446 s,
epochs 8), well under the 3 GB ceiling.

### Run it

```bash
# Host (same engine, dev lane):
./build/linux-test/bin/xllama-cli --train-job training/jobs/smollm2-360m-marker-partialft.json
# Console (llamacpp/unified MSIX): upload job+dataset+base, drop train.flag:
./scripts/validate-console-training.sh device-train
```

Console mode: `train.flag` → headless MTA thread (same harness as
bench/diffuse), job at `LocalState\training\job.json` (paths LocalState-
relative), progress in `xllama.log`, completion marker
`training\result.done`, artefacts under the job's `out_dir`.

**Evidence:** host and console marker runs both **PASS** (2026-07-20; recipe
below). Console: MSIX 1.4.0.595, peak_ws 1195 MB, wall 446 s, marker reproduced
— committed at `bench/results/phase10-console-devtrain-result.json`.

**Converging recipe (host).** The first run (LR 5e-4, long marker) under-
converged: the fine-tune learned the `XLLAMA-LORA` prefix but the tail `-OK`
never entered the distribution (greedy and temp-1.6 sampling agreed), with an
oscillating loss floor ~0.59. What converges: LR **2e-4**, the shortened
`XLLAMA-LORA-OK.` target, and `checkpoint_every: 2` to marker-test checkpoints
and early-stop (converged by epoch 8 of 12, loss ~0.47, monotonic). The console
harness LR is aligned to 2e-4.

**Evaluate loads by absolute path.** The evaluate stage opens the merged GGUF
by its absolute `out_dir` path; `resolve_model_path` (UWP) now passes absolute
paths through unchanged instead of prepending `LocalState\models\` (which
doubled the path and failed the on-device load). Linux `resolve_model_path` is
the identity function, so the host path is unaffected.

## 11. See also

- [architecture.md](architecture.md) — dual pillar map
- [uwp-constraints.md](uwp-constraints.md) §13
- [training/README.md](../training/README.md)
- [ROADMAP.md](../ROADMAP.md) Phases 8–10
