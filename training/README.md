# Training pillar (exploration)

xllama is dual-pillar: **inference** (`Session`) and **training** (this tree).
Training produces artefacts that inference already loads (merged GGUF). The
UWP chat path stays forward-only.

**Architecture SSOT (reverse engineering + capability matrix):**
[`docs/training-architecture.md`](../docs/training-architecture.md).

```
TrainingJob (JSON) ──► host PEFT LoRA ──► adapter ──► merge GGUF
                                                      │
                                                      ▼
                                              xllama::Session / xllama-cli
```

## Layout

| Path                    | Role                                                      |
| ----------------------- | --------------------------------------------------------- |
| `jobs/*.json`           | Declarative train jobs (schema validated by C++)          |
| `manifest-overrides/`   | LocalState `manifest.json` snippets for console serve gates (not train jobs) |
| `datasets/`             | JSONL chat datasets                                       |
| `host/train_lora.py`    | PEFT LoRA trainer (CPU/GPU host)                          |
| `host/run_job.sh`       | Stage runner: prepare → train → export → merge → evaluate |
| `host/requirements.txt` | Python deps for host backend                              |
| `out/`                  | Working artefacts (gitignored)                            |

C++ contracts: `include/xllama/training_params.h`, `include/xllama/training.h`
(`validate_training_job`, `load_training_job_file`, stage/device names);
Lane B engine: `include/xllama/device_train.h` (`run_device_train_job`);
Phase 11 in-app helpers: `include/xllama/personalize.h` (job builder, filters,
sample count, personalized manifest JSON — host-tested in
`tests/test_personalize.cpp`).

## Quick start

From the **repo root**:

```bash
# Validate job only (no train)
./build/linux-release/bin/xllama-cli --validate-train-job training/jobs/smollm2-360m-marker.json

# Full host pipeline (or via CLI)
./training/host/run_job.sh training/jobs/smollm2-360m-marker.json
# equivalent:
./build/linux-release/bin/xllama-cli --train-job training/jobs/smollm2-360m-marker.json
```

Reuse prior adapter/merge:

```bash
SKIP_TRAIN=1 SKIP_CONVERT=1 ./training/host/run_job.sh training/jobs/smollm2-360m-marker.json
```

Lane B engine on the host (same in-process code path as the console; no Python).
On the laptop, prefix long runs with `bg` so they land in `background.slice` instead of the
desktop tier — hours of full-weight training trip the EC skin-temp cap (PL1 8W, 400 MHz)
and degrade the whole machine:

```bash
bg ./build/linux-test/bin/xllama-cli --train-job training/jobs/smollm2-360m-marker-partialft.json
```

## Stages

| Stage          | Host status                                                                                 |
| -------------- | ------------------------------------------------------------------------------------------- |
| prepare        | resolve HF snapshot + dataset                                                               |
| train          | PEFT LoRA (`train_lora.py`)                                                                 |
| export_adapter | `convert_hf_to_gguf` + `convert_lora_to_gguf`                                               |
| merge          | `llama-export-lora` → plain GGUF                                                            |
| evaluate       | A/B `xllama-cli --chat --greedy` vs marker                                                  |
| publish        | emit `manifest.override.json` after successful merge/evaluation; device upload stays manual |

## Capabilities

```bash
./build/linux-release/bin/xllama-cli --training-capabilities
./scripts/re-training-stack.sh
```

| Lane              | Today                                                                           |
| ----------------- | ------------------------------------------------------------------------------- |
| Host PEFT + merge | **available**                                                                   |
| Device partial FT | **available** in `XLLAMA_DEVICE_TRAIN` builds; host + console marker gates PASS |
| Serve merged GGUF | **available**                                                                   |

## Device partial fine-tuning (available)

`method: "partial_ft"` uses the in-process ggml-opt engine on Linux and on
llamacpp/unified UWP builds. The current llama.cpp pin only supports selected
last-block Q/output/FFN/norm tensors plus output/output_norm; K/V projections,
earlier blocks, embeddings and rope frequencies fail validation before training.
Full `llama-finetune` and ORT ODT remain rejected.

### Console paths

| Path | How | When to use |
| ---- | --- | ----------- |
| **In-app (Phase 11)** | Settings → **Train on my feedback** after Like/Correct samples | End-user personalize; needs base GGUF (see below) |
| **Headless harness** | `train.flag` + `training/job.json` | CI / `validate-console-training.sh device-train` |

```bash
./scripts/validate-console-training.sh device-train
```

**In-app base preflight:** place `LocalState\training\base-f16.gguf` (same ~720 MB
f16 GGUF the harness uploads), **or** provision a SmolLM2 GGUF catalogue entry
whose last-block index is known (`guess_last_block_from_model_id`). Samples live
at `LocalState\training\samples.jsonl`. On success the app publishes
`models\personalized\model.gguf` and a LocalState manifest override.

Architecture: [`docs/training-architecture.md` §11](../docs/training-architecture.md).
UI: [`docs/using-the-app.md`](../docs/using-the-app.md).

**Host and console marker gates both PASS** (2026-07-20): the greedy eval prompt
reproduces `XLLAMA-LORA-OK.` at loss ~0.47. The recipe that converges is LR
**2e-4** (5e-4 oscillated and under-converged on the marker), the short
`XLLAMA-LORA-OK.` target, and `checkpoint_every: 2` so a checkpoint can be
marker-tested to early-stop (converges by epoch 8). Console gate (MSIX
1.4.0.595): `result.json` wall 446 s, `peak_ws_mb` **1195** (< 3072 ceiling),
marker reproduced end-to-end. Trained weights from this lane are not published
as a product model — the marker is a canary, not a deliverable. See
[`docs/training-architecture.md`](../docs/training-architecture.md) and
[`docs/uwp-constraints.md`](../docs/uwp-constraints.md) §13.

## Hybrid ops (Phase 9 operator loop)

Rate turns on console → retrain on host → re-serve merged GGUF:

```bash
source ~/.config/xllama/xbox-env

# 1) Capture preferences from Like/Dislike/Correct in the UI, or via autopilot
./scripts/validate-console-training.sh rate

# 2) Pull samples + convert to train JSONL
./training/host/pull_console_samples.sh
# → training/out/console-samples/samples.jsonl
# → training/out/console-samples/samples.train.jsonl

# 3) Host train (short); skip marker A/B if samples are not the toy secret job
SKIP_AB=1 STEPS=80 ./training/host/run_job.sh training/jobs/from-console-samples.json

# 4) Optional: quantize + upload via serve harness or manual WDP
#    out dir: training/out/from-console-samples/
#    publish snippet: .../manifest.override.json
```

## Console validation

```bash
source ~/.config/xllama/xbox-env
# After reinstall: ./scripts/provision-models.sh --all-test
./scripts/validate-console.sh all             # shipping bar
./scripts/validate-console-training.sh serve  # merged finetuned GGUF
XLLAMA_TRAIN_FULL=1 ./scripts/validate-console-training.sh all
SKIP_UPLOAD=1 ./scripts/validate-console-training.sh serve
```

## Verified

| Date       | Job / gate                                                                         | Result                                                                                                |
| ---------- | ---------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------- |
| 2026-07-17 | host marker PEFT                                                                   | **PASS**                                                                                              |
| 2026-07-17 | host runtime `--lora`                                                              | **PASS** (matches merge)                                                                              |
| 2026-07-17 | console `validate-console.sh gguf`                                                 | **PASS**                                                                                              |
| 2026-07-17 | console `validate-console-training.sh serve` (Q4 merged)                           | **PASS** (marker in chat)                                                                             |
| 2026-07-17 | MSIX **1.2.0.546** `rate`                                                          | **PASS** (samples.jsonl like)                                                                         |
| 2026-07-17 | MSIX **1.2.0.546** `lora-rt`                                                       | **PASS** (runtime LoRA applied; fix `model.gguf` prefer)                                              |
| 2026-07-17 | `validate-console.sh all` on 1.2.0.546 (after `provision-models.sh sd-turbo-fp16`) | **ALL PASS** (routing + GGUF + TAESD 602 ms; `all` now also runs the settings gate, added 2026-07-21) |
| 2026-07-17 | Phase 8 exploration                                                                | **FROZEN complete** (see `docs/training-architecture.md` exit criteria)                               |

## Compat

`scripts/lora-spike/` is a thin shim that forwards to this module.
