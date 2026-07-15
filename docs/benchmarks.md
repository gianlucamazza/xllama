# Benchmarks

> **This is the single source of truth (SSOT) for xllama performance numbers.**
> Decode/prefill/RAM/load figures live here; other docs (README, ROADMAP,
> technical-report, model-selection, recommended-config) quote at most a headline
> value and link back to this file.

Consolidated performance data for every model xllama has run. Unless noted, all
figures are **measured on Xbox Series S in Dev Mode** (the target device) and
come from the CSVs in `bench/results/`. Host-dev figures (Intel i7-1165G7,
Linux) are called out separately and are **not** comparable to console numbers —
they exist only to sanity-check a model loads and generates.

**On the SmolLM2-360M CPU int4 decode number** (it appears elsewhere as 66.3,
68.0 or 70.9 — same model, different runs): **70.9** is the best control run
(`phase2-dml`); **66.3** is the ORT-GenAI 0.14.1 shipping config; **68.0 / 50.9**
are the v0.3.6 utilization-matrix short/long-prompt runs. When in doubt, the table
below is authoritative.

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

## KV-cache reuse — both backends, CPU

Turn-2 prefill with continuous decoding vs a cold full re-prefill. Both the
ORT-GenAI path (persistent generator) and the GGUF/llama.cpp path (persistent
`llama_context`, `LlamaSession`) reuse the cache and append only the new turn's
delta:

| Backend / model                              | Reuse turn-2 (ms) | Cold turn-2 (ms) |   Speedup |
| -------------------------------------------- | ----------------: | ---------------: | --------: |
| ORT-GenAI · SmolLM2-360M (`phase35-kv`)      |    103.7 (22 tok) |  505.2 (114 tok) | **4.87×** |
| llama.cpp · Gemma-3-270M (`phase6-gemma-kv`) |    107.5 (39 tok) |  437.4 (179 tok) | **4.07×** |

GGUF KV-reuse was previously disabled (llama.cpp recreated the context per turn);
now enabled and console-measured. Routing (CPU↔GPU) stays ORT-only — the
llama.cpp UWP build is CPU-only. DirectML still rejects continuous decoding, so
the ORT reuse path is CPU-only.

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

## Gemma — on-console (Xbox Series S, MSIX 1.1.6.0, `phase6-gemma.csv`)

Added by the per-architecture chat-template work (`chat_format_for` selects the
`<start_of_turn>…<end_of_turn>` template, stop `<end_of_turn>`). Both Gemma
architectures load and generate on the pinned `llama.cpp` (`9a532ae4b`);
**measured on the console** 2026-07-14:

| Model           | Params           | Quant  |    Size | Prefill tok/s | Decode tok/s | Peak RAM MB | Load ms |
| --------------- | ---------------- | ------ | ------: | ------------: | -----------: | ----------: | ------: |
| Gemma-3-270M-it | 270M             | Q4_K_M |  253 MB |         395.0 |     **76.8** |         368 |    1095 |
| Gemma-4-E2B-it  | ~5B raw (2B eff) | Q3_K_S | 2.45 GB |          26.1 |     **15.3** |        2742 |   13837 |
| Gemma-4-E2B-it  | ~5B raw (2B eff) | IQ2_M  | 2.29 GB |          13.5 |          9.9 |        2534 |    6169 |

Catalogue default for `gemma4-e2b` is now **Q3_K_S** (was IQ2_M): it decodes
faster (15.3 vs 9.9) and, crucially, generates full responses on long prompts
where the 2-bit IQ2_M collapsed to an immediate EOG (see below).

Both use the Gemma template on-device (log: `bench prompt: <start_of_turn>user
…`). **Gemma-3-270M** is a fast, tiny chat model (76.8 tok/s, 368 MB).

**Gemma-4-E2B verdict — the "too big" call is overturned** (shipping default is
**Q3_K_S**, 15.3 tok/s / 2742 MB; the IQ2_M row above is the earlier 2-bit build):

- ✅ **The ~2 GB Dev Mode per-file limit does not apply to GGUF** — a >2 GB single
  `.gguf` loads with no OOM under the Game budget. (This was the project's main
  open unknown.)
- ✅ Generates coherently on-device, faster than the i7-1165G7 host.
- ⚠️ Load is a few seconds — **repack-bound, not file-read-bound** (see the mmap
  note below); a smaller quant is the only real load lever.

Disk was never the constraint (Dev Mode is 90 GB). E4B/12B+ stay out of scope on
size/speed. Catalogue entry `gemma4-e2b` (downloads from HF; 2.29 GB exceeds the
GitHub release 2 GB asset limit).

**In-app HF download verified on-console** (2026-07-15): with the catalogue entry
un-provisioned, the app's own downloader pulled the single 2.29 GB `.gguf` straight
from the HF `unsloth` repo (`EnsureModel: downloading … from catalogue` →
`download complete`, 2 290 858 112 bytes), then loaded it (`GGUF model loaded via
llama.cpp`) and generated — confirming the >2 GB single-file download path (not just
Device-Portal provisioning). A gap surfaced in the process was **fixed in
v1.1.7.0**: `IsModelProvisioned` is now expected-aware — it compares the directory
against the manifest's current `files[].filename` (`dir_satisfies_expected_files`,
`include/xllama/model_provision.h`) instead of accepting any `.gguf`, and
`EnsureModelNamedAsync` reconciles the dir (drops the stale `*.gguf` + `.complete`)
before re-downloading. A stale IQ2_M under `gemma4-e2b` now **auto-upgrades** to the
manifest's Q3_K_S (verified on-console 2026-07-15). See `docs/architecture.md`.

## Root-cause notes — the negative performers

Investigated 2026-07-14 (reverse-engineered where noted). None is a loader or
template bug; the causes are quantization behaviour, a UWP constraint, and known
DirectML limits.

### Gemma-4-E2B emits EOG immediately on the bench prompt (0 decode tokens)

The first `phase6-gemma` E2B run scored **decode 0.0 tok/s** — the model hit an
end-of-generation token on the first sample (`EOG after 0 tokens`). Isolated by
reproduction:

- Both `gemma3` and `gemma4` GGUFs set `eos_token_id = 106` (`<end_of_turn>`),
  `add_bos_token = true` — so the template (which omits `<bos>`) is correct.
- On the **identical** 272-token declarative prompt at temp 0.8: gemma3-270m (Q4)
  generates 30+ tokens with **no** early EOG; gemma4-E2B (IQ2_M) stops early
  (0 tokens on console, 7 on the host at a different RNG seed).

**Root cause**: gemma4-E2B assigns a high probability to `<end_of_turn>` as the
_first_ token after a **declarative** prompt (a technical overview, not a
question) — the correct response is to end the turn. Two amplifiers: (1) the
**IQ2_M 2-bit quant** degrades the logits and inflates specific tokens incl. EOG;
(2) gemma4 is better turn-calibrated than the tiny under-trained gemma3-270m,
which rambles instead. Stochastic under temp 0.8 → sometimes the very first
sample. **Not a bug.** Mitigation confirmed: **Q3_K_S (2.45 GB) fixes it** — on
the exact same 280-token declarative prompt it generates a full 266-token
response (15.3 tok/s) where IQ2_M produced 0 decode tokens. So the catalogue
`gemma4-e2b` default is now Q3_K_S. (The 2-bit IQ2_M also works with
question/generative prompts; `standard-512.txt` is just a poor decode probe for
a well-calibrated instruct model at aggressive quant.)

### Gemma-4-E2B slow load (23.6 s cold → 6.2 s warm)

The 0001 AppContainer patch disables `_WIN32` mmap (desktop-only
`CreateFileMappingA`), so the 2.29 GB is read into the heap.

**mmap was tried and does not help** (measured 2026-07-14, MSIX 1.1.6.448): a
patch making the AppContainer use `CreateFileMappingFromApp`/`MapViewOfFileFromApp`
(with a loader fallback) built and deployed, but load stayed **6.4 s** and peak
RAM **2533 MB — unchanged**. Root cause: the CPU load is dominated not by the
file read but by the **AVX2 tensor repack** (Q4_K → q4_K_8x8, `ggml-cpu/repack.cpp`),
which copies/transforms every weight regardless of how the file is read. mmap
only removes the file-read copy, which the repack re-introduces. Reverted — a
zero-benefit vendored patch is not worth the per-bump maintenance. The real
levers are a smaller quant (less to repack) or skipping repack (trades decode
speed). Cold/warm delta is OS file cache.

### DirectML int4 decode 8.8 tok/s — 8× slower than CPU (already root-caused)

`uwp-constraints.md §12`: DirectML has **no fused low-bit GPU GEMM**. It
implements `MatMulNBits` as `DML_DEQUANTIZE`→fp16 + a full `DML_GEMM`,
materialising fp16 weights — so int4 moves _more_ bandwidth than fp16 (hence
8.8 < the fp16 path's 46.8). The builder also gives DML `accuracy_level=0` vs
CPU's fused int8 MLAS (`=4`, → 70). A DirectML-team feature; not fixable in-app.

### DirectML fp16 decode 46.8 < CPU 70.9 — expected, not a regression

Autoregressive decode (M=1) is memory-bound and dominated by per-token DML
dispatch overhead at this model scale; CPU `MatMulNBits` on AVX2 wins
(ROADMAP Phase 2 / `uwp-constraints.md §7`). GPU's win is **prefill** (353 vs
198 tok/s at ~1k tokens), which is exactly what routing uses it for.

## CPU memory bandwidth — the decode denominator

Decode is a bandwidth-bound M=1 GEMV: each token streams the whole weight matrix
from DRAM once, so decode tok/s ≈ (effective read bandwidth) / (weight bytes). The
"~13 GB/s effective from CPU int4 GEMV" quoted elsewhere is a _deduced_ figure; the
`membw` micro-bench (STREAM-style read / copy / triad over a 256 MB buffer, larger
than the LLC) measures the sustained ceiling directly, so decode can be stated as a
fraction of a measured number.

Host reference (i7-1165G7, 2026-07-14, `xllama-cli --membw`):

| Threads | Read GB/s | Copy GB/s | Triad GB/s |
| ------- | --------: | --------: | ---------: |
| 1       |      11.8 |      23.0 |       14.0 |
| 8       |      28.1 |      36.3 |       26.3 |

On-console (Xbox Series S, Zen 2 / GDDR6, MSIX 1.1.6.464, 2026-07-15,
`membw.flag`):

| Threads | Read GB/s | Copy GB/s | Triad GB/s |
| ------- | --------: | --------: | ---------: |
| 1       |     12.35 |     28.34 |      18.29 |
| 8       |     30.29 |     42.02 |      24.20 |

**This closes the loop on the "~13 GB/s effective" GEMV figure**: the measured
single-thread read is **12.35 GB/s** — essentially the deduced decode denominator —
and the full-width ceiling is **~30 GB/s** read. So CPU int4 decode streams weights
at roughly single-thread bandwidth (~40% of the 8-thread ceiling); the GEMV is
latency/dispatch-bound per token rather than saturating aggregate DRAM bandwidth,
consistent with why more threads help prefill (batched) but not decode (M=1). Drop
a `membw.flag` into `LocalState` to reproduce (`membw-result.csv`, 1t + full-width).

## Reproducing

- **On-console**: `./scripts/bench-xbox-ort.sh <model> --runs 3 --out bench/results/<file>.csv`
  (model already provisioned). KV bench: drop a `bench_turns.txt`; diffusion:
  `phase5-diffuse`.
- **Host (llama.cpp GGUF only)**: `./build/linux-release/bin/xllama-cli -m <model.gguf>
-p '<prompt>' -n 128` prints `load / prompt tok/s / decode tok/s`.
- **Prefill micro-batch sweep**: `./scripts/bench-ubatch-sweep.sh <model.gguf>` runs
  the CLI across `--ubatch` 128/256/512/1024 and prints prompt tok/s per value
  (`--batch`/`--ubatch` also exposed directly on `xllama-cli`).
- **Memory bandwidth**: `./build/linux-release/bin/xllama-cli --membw` (host) or a
  `membw.flag` in `LocalState` (console) → read/copy/triad GB/s.
- Comparative charts: `docs/benchmarks-charts.html` (self-contained; open in a browser).

### Prefill micro-batch (n_ubatch) — no reproducible host win

`n_ubatch` (llama.cpp physical prefill chunk, default 512) is the only
TTFT-relevant batching knob on the CPU path (`n_batch` merely caps the logical
batch). Now exposed end-to-end (`InferenceParams`/`SessionParams` →
`llama_context_params`, `xllama-cli --batch/--ubatch`). Host sweep
(i7-1165G7, Qwen3.5-0.8B-Q4_K_M, 701-token prompt, 2026-07-14): repeated passes
**disagree** on the ubatch=128 value (82.5 vs 117.4 tok/s) and show **no
reproducible trend** — on a loaded dev laptop the measurement is noise-dominated.
The knob is in place and CLI-sweepable; a clean optimum needs a quiet machine (the
headless console bench reads `bench_threads.txt` but not an ubatch override, so an
on-console sweep would need a bench-mode change — out of scope). Xbox Zen 2 shares
the x86 AVX2 ISA, so a flat curve is the expectation. Default (0 → llama.cpp 512)
stays until a measured win justifies a change.
