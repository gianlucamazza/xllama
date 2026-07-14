# Benchmarks

Consolidated performance data for every model xllama has run. Unless noted, all
figures are **measured on Xbox Series S in Dev Mode** (the target device) and
come from the CSVs in `bench/results/`. Host-dev figures (Intel i7-1165G7,
Linux) are called out separately and are **not** comparable to console numbers —
they exist only to sanity-check a model loads and generates.

Metrics: **prefill** = prompt tok/s (throughput ingesting the prompt), **decode**
= generation tok/s (autoregressive, the number a user feels), **peak RAM** =
`peak_working_set_mb`. `n_threads` matters on the ~6 usable Dev Mode cores
(t7/t8 livelock the ggml threadpool — see `phase35-llamacpp-scaling.csv`).

## Text chat models — on-console (Xbox Series S)

Best measured decode configuration per model.

| Model        | Params | Quant  | Backend               | Prefill tok/s | Decode tok/s | Peak RAM MB | Source CSV                  |
| ------------ | ------ | ------ | --------------------- | ------------: | -----------: | ----------: | --------------------------- |
| LFM2.5-350M  | 350M   | Q4_K_M | llama.cpp CPU · t6    |         241.4 |     **94.2** |         321 | `phase5-gguf`               |
| SmolLM2-360M | 360M   | int4   | ORT-GenAI CPU         |         219.8 |         70.9 |         722 | `phase1-cpu` / `phase2-dml` |
| SmolLM2-360M | 360M   | Q4_K_M | llama.cpp CPU · t6    |         141.5 |         62.9 |         402 | `phase35-llamacpp-scaling`  |
| SmolLM2-360M | 360M   | fp16   | ORT DML (routing GPU) |         353.5 |         46.8 |        1154 | `phase2-dml`                |
| Qwen3.5-0.8B | 0.8B   | Q4_K_M | llama.cpp CPU · t6    |          98.1 |         35.1 |         718 | `phase5-gguf`               |
| SmolLM2-1.7B | 1.7B   | int4   | ORT-GenAI CPU         |          54.9 |         20.6 |        2423 | `phase35-1b-cpu`            |
| SmolLM2-360M | 360M   | int4   | ORT DML int4          |       152–334 |          8.8 |    999–1525 | `phase2-dml`                |

Notes:

- **LFM2.5-350M** is the fastest chat model and the lightest (321 MB RAM) — the
  default chat model on unified builds.
- **DML int4** (8.8 tok/s) is ~8× slower than CPU at this scale: no fused low-bit
  GPU GEMM, so decode is memory-bound reading fp16 weights round-tripped through
  VRAM. DML fp16 is only worth it for long-prefill routing (353 tok/s prefill).
- **Routing**: Auto switches SmolLM2-360M CPU → DML fp16 above a 600-token prompt
  (`routing_policy.h`), trading decode (70.9 → 46.8) for prefill (219 → 353).

## KV-cache reuse (ORT-GenAI, CPU)

Turn-2 prefill with continuous decoding vs a cold full re-prefill
(`phase35-kv.csv`, SmolLM2-360M):

|                        | Prefill turn-2 (ms) | Tokens |   Speedup |
| ---------------------- | ------------------: | -----: | --------: |
| Reuse (delta only)     |               103.7 |     22 | **4.87×** |
| Cold (full re-prefill) |               505.2 |    114 |         — |

KV-reuse is ORT-GenAI-only and CPU-only (DirectML rejects continuous decoding;
GGUF/llama.cpp is stateless on-console).

## Diffusion — SD-Turbo fp16 (on-console, DirectML)

`phase5-diffuse.csv`, 512×512, 1 step:

| Stage         |  Time (ms) |
| ------------- | ---------: |
| Text encoder  |     1006.6 |
| UNet (1 step) |     3328.9 |
| VAE decode    |     2556.1 |
| **Total**     | **6891.6** |

TAESD (4.9 MB tiny VAE) targets the 2.6 s VAE stage → ~4.5 s/image (see
`docs/model-selection.md`).

## Gemma — arch-validated, on-console benches pending

Added by the per-architecture chat-template work (`chat_format_for` selects the
`<start_of_turn>…<end_of_turn>` template, stop `<end_of_turn>`). Both Gemma
architectures **load and generate** on the pinned `llama.cpp` submodule
(`9a532ae4b`): `general.architecture = gemma3` / `gemma4` confirmed via
`xllama-cli`. Host-dev sanity numbers (Intel i7-1165G7 — **not** Xbox):

| Model           | Params           | Quant    |    Size | Arch loads | Host decode tok/s | Host peak RAM | On-console                                              |
| --------------- | ---------------- | -------- | ------: | ---------- | ----------------: | ------------: | ------------------------------------------------------- |
| Gemma-3-270M-it | 270M             | Q4_K_M   |  253 MB | ✅ gemma3  |               ~40 |       ~0.3 GB | ✅ fits (catalogue entry `gemma3-270m`) — bench pending |
| Gemma-4-E2B-it  | ~5B raw (2B eff) | UD-IQ2_M | 2.29 GB | ✅ gemma4  |               3.8 |       2.35 GB | ⚠️ feasibility candidate                                |

**Gemma-4-E2B verdict**: loads and produces coherent output, but on the mobile
i7 decode is 3.8 tok/s (a ~5B-raw MatFormer model at 2-bit). On-console the
open questions are (a) the community-reported ~2 GB Dev Mode per-file limit —
every E2B quant exceeds 2 GB (smallest 2.29 GB, never tested on this Xbox), and
(b) RAM (~2.35 GB resident) under the Game memory budget. Disk is no longer a
blocker (Dev Mode raised to 90 GB, 2026-07-08). E4B/12B+ are out of scope on
size/speed. See `docs/model-selection.md` for the full survey.

## Reproducing

- **On-console**: `./scripts/bench-xbox-ort.sh <model> --runs 3 --out bench/results/<file>.csv`
  (model already provisioned). KV bench: drop a `bench_turns.txt`; diffusion:
  `phase5-diffuse`.
- **Host (llama.cpp GGUF only)**: `./build/linux-release/bin/xllama-cli -m <model.gguf>
-p '<prompt>' -n 128` prints `load / prompt tok/s / decode tok/s`.
- Comparative charts: `docs/benchmarks-charts.html` (self-contained; open in a browser).
