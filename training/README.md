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

| Path | Role |
| --- | --- |
| `jobs/*.json` | Declarative train jobs (schema validated by C++) |
| `datasets/` | JSONL chat datasets |
| `host/train_lora.py` | PEFT LoRA trainer (CPU/GPU host) |
| `host/run_job.sh` | Stage runner: prepare → train → export → merge → evaluate |
| `host/requirements.txt` | Python deps for host backend |
| `out/` | Working artefacts (gitignored) |

C++ contracts: `include/xllama/training_params.h`, `include/xllama/training.h`
(`validate_training_job`, `load_training_job_file`, stage/device names).

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

## Stages

| Stage | Host status |
| --- | --- |
| prepare | resolve HF snapshot + dataset |
| train | PEFT LoRA (`train_lora.py`) |
| export_adapter | `convert_hf_to_gguf` + `convert_lora_to_gguf` |
| merge | `llama-export-lora` → plain GGUF |
| evaluate | A/B `xllama-cli --chat --greedy` vs marker |
| publish | stub: `manifest.override.json` after successful merge |

## Capabilities

```bash
./build/linux-release/bin/xllama-cli --training-capabilities
./scripts/re-training-stack.sh
```

| Lane | Today |
| --- | --- |
| Host PEFT + merge | **available** |
| Device train | **not available** (gated; SSOT RE) |
| Serve merged GGUF | **available** |

## Device training (reserved)

`device: "device"` is **rejected** by `validate_training_job`. RE: inference-only
NuGet, GenAI adapter *load* symbols without train API (“No adapter is available
for DML”), llama-finetune ~24 GB class. See
[`docs/training-architecture.md`](../docs/training-architecture.md) and
[`docs/uwp-constraints.md`](../docs/uwp-constraints.md) §13.

## Hybrid ops (Phase 9 operator loop)

Rate turns on console → retrain on host → re-serve merged GGUF:

```bash
source ~/.config/xllama/xbox-env

# 1) Capture preferences (autopilot) — or use UI when available
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

| Date | Job / gate | Result |
| --- | --- | --- |
| 2026-07-17 | host marker PEFT | **PASS** |
| 2026-07-17 | host runtime `--lora` | **PASS** (matches merge) |
| 2026-07-17 | console `validate-console.sh gguf` | **PASS** |
| 2026-07-17 | console `validate-console-training.sh serve` (Q4 merged) | **PASS** (marker in chat) |
| 2026-07-17 | MSIX **1.2.0.546** `rate` | **PASS** (samples.jsonl like) |
| 2026-07-17 | MSIX **1.2.0.546** `lora-rt` | **PASS** (runtime LoRA applied; fix `model.gguf` prefer) |
| 2026-07-17 | `validate-console.sh all` on 1.2.0.546 (after `provision-models.sh sd-turbo-fp16`) | **ALL PASS** (routing + GGUF + TAESD 602 ms) |
| 2026-07-17 | Phase 8 exploration | **FROZEN complete** (see `docs/training-architecture.md` exit criteria) |

## Compat

`scripts/lora-spike/` is a thin shim that forwards to this module.
