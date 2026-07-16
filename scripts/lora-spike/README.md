# Host LoRA spike (Tier 1)

End-to-end proof that **training lives off-device** and Xbox / `xllama-cli` only
**serve merged weights** (plain GGUF, no runtime LoRA API).

```
HF SmolLM2-360M  →  PEFT LoRA  →  convert_lora_to_gguf  →  llama-export-lora
                                                              ↓
                                                    merged.gguf → xllama-cli --chat
```

## Prerequisites

- Linux host, existing `xllama-cli` build (`cmake --preset linux-release` + build).
- ~4–8 GB free RAM during train/convert (CPU; no GPU required).
- Local HF snapshot of `HuggingFaceTB/SmolLM2-360M-Instruct` under `cache_dir/`
  (already used by this repo), or set `CACHE_DIR`.
- Python 3.10+ for a small venv (`scripts/lora-spike/.venv`).

## One-shot

From the **repo root**:

```bash
./scripts/lora-spike/run_spike.sh
```

What it does:

1. Creates `.venv` and installs `requirements.txt` (+ CPU torch if missing).
2. Trains a tiny LoRA (`STEPS=120` default) on `toy_dataset.jsonl` so the
   trigger **`xllama secret`** yields **`XLLAMA-LORA-OK`**.
3. Converts base weights to GGUF f16 and the adapter to a GGUF LoRA.
4. Builds `llama-export-lora` if needed (`LLAMA_BUILD_TOOLS=ON` in root CMake)
   and merges into `out/smollm2-360m-lora-merged-f16.gguf`.
5. A/B with `xllama-cli --chat --greedy`: base vs merged; **PASS** if merged
   stdout contains `XLLAMA-LORA-OK`.

Artifacts stay under `scripts/lora-spike/out/` (gitignored).

### Useful env knobs

| Env | Default | Meaning |
| --- | --- | --- |
| `STEPS` | `120` | Training steps (raise if marker not learned) |
| `SKIP_TRAIN=1` | off | Reuse existing `out/adapter/` |
| `SKIP_CONVERT=1` | off | Reuse merged GGUF; only re-run A/B |
| `SKIP_AB=1` | off | Stop after merge |
| `QUANTIZE=1` | off | Also emit Q4_K_M and A/B on that |
| `CACHE_DIR` | `$REPO/cache_dir` | HF hub cache root |

```bash
STEPS=200 ./scripts/lora-spike/run_spike.sh
SKIP_TRAIN=1 SKIP_CONVERT=1 ./scripts/lora-spike/run_spike.sh   # A/B only
```

## Manual steps (if debugging)

```bash
# 1) train
scripts/lora-spike/.venv/bin/python scripts/lora-spike/train_lora.py \
  --model HuggingFaceTB/SmolLM2-360M-Instruct \
  --cache-dir cache_dir \
  --dataset scripts/lora-spike/toy_dataset.jsonl \
  --out scripts/lora-spike/out/adapter \
  --steps 120

# 2) base GGUF
python llama.cpp/convert_hf_to_gguf.py \
  cache_dir/models--HuggingFaceTB--SmolLM2-360M-Instruct/snapshots/<hash> \
  --outfile scripts/lora-spike/out/smollm2-360m-f16.gguf --outtype f16

# 3) adapter GGUF
python llama.cpp/convert_lora_to_gguf.py \
  --base cache_dir/models--.../snapshots/<hash> \
  --outfile scripts/lora-spike/out/adapter-lora.gguf --outtype f16 \
  scripts/lora-spike/out/adapter

# 4) merge (build target once)
cmake --build build/linux-release --target llama-export-lora -j$(nproc)
# path may be build/linux-release/bin/llama-export-lora
llama-export-lora \
  -m scripts/lora-spike/out/smollm2-360m-f16.gguf \
  --lora scripts/lora-spike/out/adapter-lora.gguf \
  -o scripts/lora-spike/out/smollm2-360m-lora-merged-f16.gguf

# 5) A/B
./build/linux-release/bin/xllama-cli --chat --greedy \
  -m scripts/lora-spike/out/smollm2-360m-f16.gguf -p 'xllama secret' -n 48
./build/linux-release/bin/xllama-cli --chat --greedy \
  -m scripts/lora-spike/out/smollm2-360m-lora-merged-f16.gguf -p 'xllama secret' -n 48
```

## Verified (host)

| Date | Host | Result |
| --- | --- | --- |
| 2026-07-17 | Linux CPU, SmolLM2-360M-Instruct, `STEPS=120` | **PASS** — base lacks `XLLAMA-LORA-OK`; merged emits it (`xllama-cli --chat --greedy`) |

Re-check without retrain/reconvert:

```bash
SKIP_TRAIN=1 SKIP_CONVERT=1 ./scripts/lora-spike/run_spike.sh
```

## Limits (intentional)

- **Toy signal only** — not a quality finetune; the marker proves the pipeline.
- **Host CPU** — no Xbox / UWP training path.
- **No runtime LoRA** in `xllama::Session`; merge produces a normal catalogue-ready GGUF.
- Do **not** commit `out/`, `.venv/`, or large GGUF files.
- **Not shipped on catalogue** — Xbox still serves stock GGUF/ORT entries; this spike is host tooling only.

## Architecture note

See [docs/architecture.md](../../docs/architecture.md) § Personalization / LoRA:
xllama remains an inference runtime; adapters are prepared on the host and
shipped as weights (or future catalogue entries).
