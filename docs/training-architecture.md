# Training architecture (SSOT)

> **SSOT for the training pillar** — dual-lane design, reverse-engineering
> inventory, capability matrix, hybrid personalization loop. Module map of the
> whole product: [architecture.md](architecture.md). AppContainer limits:
> [uwp-constraints.md](uwp-constraints.md) §13. Host runner ops:
> [`training/README.md`](../training/README.md).

**Currency:** 2026-07-17. Host PEFT marker job **PASS**. Device train **not**
implemented (API-gated).

## 1. Why a training pillar

xllama personalizes and researches **on-device inference** (Xbox Dev Mode). Users
also need a path to **learn** adapters / domain behaviour. Training is a
**parallel pillar**, not a mode of `Session::generate()`:

| Pillar | Loop | Output |
| --- | --- | --- |
| Inference | forward only | tokens / images |
| Training | prepare → train → export → merge → evaluate | adapter / merged GGUF |

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
    IMPLEMENTED        RESEARCH / gated     INFERENCE
    training/host/*    ORT ODT / (rejected  Session + future
    device=host        llama-finetune)      runtime adapter
           │                 │                 ▲
           └──────── merged GGUF / LoRA ───────┘
    Hybrid: PreferenceCapture (device) → dataset → Lane A → Lane C
```

### Non-goals (product)

- Pretraining from scratch on Xbox
- Full fine-tune of 350M–3B in the UWP process
- Python / PyTorch inside the MSIX
- Polluting `GenerateParams` with learning rate / epochs

## 3. Reverse-engineering inventory

### 3.1 Shipping NuGet (inference only)

From `uwp/packages.config`:

| Package | Version | Training API? |
| --- | --- | --- |
| `Microsoft.ML.OnnxRuntime.DirectML` | 1.24.4 | **No** — not `OnnxRuntime.Training` |
| `Microsoft.ML.OnnxRuntimeGenAI.DirectML` | 0.14.1 | **No** generate-only surface |
| `Microsoft.AI.DirectML` | 1.15.4 | Kernels for forward EP |

**RE conclusion:** our MSIX cannot call ORT Training without a **new package
pin and rebuild**. On-device training is not a config toggle.

### 3.2 ORT GenAI DLL symbols (vendor pin)

`strings` on `vendor/onnxruntime-genai-patched/win-x64/onnxruntime-genai.dll`:

| Finding | Interpretation |
| --- | --- |
| `OgaCreateAdapters`, `OgaLoadAdapter`, `OgaSetActiveAdapter`, `OgaUnloadAdapter` | **Inference-time** adapter container (not gradient training) |
| `No adapter is available for DML` | GenAI adapters **not** available on DML EP for this pin |
| No `TrainingSession` / train-loop exports in GenAI | GenAI remains decode/prefill |

**RE conclusion:** GenAI adapters are Lane C (serve), and currently **DML-blocked**.
CPU EP adapter load is a future spike if we ship ORT-format adapters.

### 3.3 llama.cpp (submodule)

| API | Role |
| --- | --- |
| `llama_adapter_lora_init` / `llama_set_adapters_lora` | Runtime LoRA **load** (forward) |
| `examples/training` (`llama-finetune`) | WIP full FT; README cites **~24 GB** for 1B FP32 |

**RE conclusion:** best Xbox personalization **serve** path is GGUF ± runtime
LoRA. In-process finetune via llama-finetune is **rejected** for Series S budget
and WIP status.

### 3.4 ORT On-Device Training (external product)

Microsoft documents an [on-device training](https://onnxruntime.ai/docs/get-started/training-on-device.html)
flow: offline artifact generation → on-device phase → export inference ONNX.

| Factor | Assessment for xllama Xbox |
| --- | --- |
| Product fit (privacy personalization) | Conceptually aligned |
| In our tree / MSIX | **No** |
| AppContainer + 3801 MB GPU + #91 | Unproven; high risk |
| Dual artefact format (ONNX train vs GGUF chat) | Ops cost |

**RE conclusion:** Lane B **research** only; kill if spike cannot load Training
package under AppContainer or OOMs under 2–3 GB peak.

### 3.5 Measured resources (host PEFT run + console budgets)

| Quantity | Value | Source |
| --- | --- | --- |
| Trainable params (SmolLM2-360M, r=8, q/k/v/o) | ~1.64 M (0.45%) | host PEFT log |
| Adapter optimizer state | tens of MB (order of mag.) | scale from trainable count |
| Base fp16 GGUF | ~720 MB | `training/out/.../base-f16.gguf` |
| Series S GPU budget | **3801 MB** | `uwp-constraints.md` §5/§7 |
| CPU effective BW | ~12–13 GB/s | membw |
| Full FT 360M Adam class | multi-GB | scale argument |

**RE conclusion:** **PEFT-class** state can fit device RAM *if* a train engine
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

| Capability | Status | Available now? |
| --- | --- | --- |
| `HostPeftLora` | available | **yes** |
| `HostMergeGguf` | available | **yes** |
| `HostEvaluateMarker` | available | **yes** |
| `RuntimeLoraLoadLlama` | **available** | **yes** (`SessionParams.lora_path` / CLI `--lora`) |
| `RuntimeAdapterLoadOrtGenAI` | designed | no (DML blocked on pin) |
| `DeviceOrtOnDeviceTraining` | research | no |
| `DeviceLlamaFinetune` | **rejected** | no |
| `DevicePreferenceCapture` | designed | no (no UI yet) |

`training_device_supported(Device)` stays **false** until a Device\* capability
is flipped to `available` with measured evidence.

## 5. Stage machine and job schema

Stages (`TrainStage`): `prepare` → `train` → `export_adapter` → `merge` →
`evaluate` → `publish` (publish open).

Job JSON (schema_version 1) — example:
`training/jobs/smollm2-360m-marker.json`. C++:
`load_training_job_file` / `validate_training_job`.

Host runner: `training/host/run_job.sh` or `xllama-cli --train-job`.

## 6. Hybrid personalization loop (designed)

```
Xbox chat ──► preference samples (LocalState JSONL) ──► export
                                                          │
Host TrainingJob (dataset=samples) ──► PEFT ──► merge GGUF
                                                          │
Xbox Session load ◄───────────────────────────────────────┘
```

### Preference sample line (JSONL)

```json
{"ts":"2026-07-17T12:00:00Z","label":"like","messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}
```

Labels: `like` | `dislike` | `correction` (correction may include
`preferred_assistant`). No auto-upload; export is operator-driven (Device Portal
/ future opt-in API). Privacy: samples stay in AppContainer until the user
exports.

## 7. Lane C — serve trained weights

| Path | Status |
| --- | --- |
| Load **merged GGUF** as catalogue / LocalState model | **Works today** (`Backend::Auto`) |
| Runtime LoRA on llama.cpp without merge | **Available** — `SessionParams.lora_path` + CLI `--lora` / `--lora-scale` |
| Runtime adapter on ORT GenAI | Designed — blocked for DML; CPU-only investigation later |

## 8. Decision log

| Decision | Rationale |
| --- | --- |
| Host PEFT first | RE shows inference-only NuGet; host PEFT proven PASS |
| Hard-gate `device=device` | Avoid false claims; no train backend in MSIX |
| Reject DeviceLlamaFinetune | ~24 GB class + WIP |
| Prefer merge GGUF over runtime LoRA for v1 | Zero Session change; same path as catalogue |
| ODT as research not default | Package + format + console risk |

## 9. Open spikes (kill criteria)

| Spike | Kill if |
| --- | --- |
| Runtime LoRA load (llama) | >5% decode regression or KV-reuse break |
| ORT ODT AppContainer | Cannot link Training package or peak RSS > 3 GB on 360M PEFT toy |
| Preference capture UI | Interferes with chat latency / storage budget without export path |

## 10. See also

- [architecture.md](architecture.md) — dual pillar map
- [uwp-constraints.md](uwp-constraints.md) §13
- [training/README.md](../training/README.md)
- [ROADMAP.md](../ROADMAP.md) Phase 8
