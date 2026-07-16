# Training pillar (exploration)

xllama is dual-pillar: **inference** (`Session`) and **training** (this tree).
Training produces artefacts that inference already loads (merged GGUF). The
UWP chat path stays forward-only.

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
| publish | *open* (catalogue entry later) |

## Device training (reserved)

`device: "device"` in a job is **rejected** by `validate_training_job` today.
On-console training is an exploration track blocked by measured constraints
(GPU ~3801 MB Series S, no ORT GenAI train API, AppContainer, llama-finetune
class RAM). See `docs/architecture.md` § Training pillar and
`docs/uwp-constraints.md`.

## Verified

| Date | Job | Result |
| --- | --- | --- |
| 2026-07-17 | `smollm2-360m-marker` (host, 120 steps) | **PASS** — merged emits `XLLAMA-LORA-OK`, base does not |

## Compat

`scripts/lora-spike/` is a thin shim that forwards to this module.
