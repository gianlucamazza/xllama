# bench/

Benchmark suite for xllama. Results are stored as CSV files in `results/`.

## Single source of truth

- `results/*.csv` and `results/*.jsonl` are the immutable measured evidence.
- `benchmark-summary.json` declares which atomic run represents each comparison
  row; it contains metadata and selectors, never copied metric values.
- `docs/benchmarks.md` and `docs/benchmarks-charts.html` contain generated blocks.
  Refresh them with `python3 scripts/generate-benchmark-summary.py`.
- CI runs the same command with `--check`; a result or selection-policy change
  cannot merge while either published surface is stale.

Never combine the best prefill, decode, or RAM values from different CSV rows.
Add a row to `benchmark-summary.json` when a newly measured configuration should
appear in the consolidated comparison.

## Methodology

- **Prompts**: fixed prompts in `bench/prompts/` (same across all runs for comparability).
- **Runs**: 3 runs per configuration; discard the first (cold start); report median of the remaining two.
- **Metrics**:
  - `prompt_tok_s`: tokens/second during prompt processing
  - `decode_tok_s`: tokens/second during generation (excluding prompt)
  - `peak_ws_mb`: peak working set / RSS in MB
  - `load_ms`: model load time in milliseconds
  - `gpu_mem_mb` / `gpu_budget_mb`: per-process GPU memory CurrentUsage/Budget after model load (`QueryVideoMemoryInfo`, LOCAL segment); 0 on CPU-only runs and Linux builds
- **CSV schema**: `model,quant,backend,n_ctx,n_threads,prompt_tok_s,decode_tok_s,peak_ws_mb,load_ms,gpu_mem_mb,gpu_budget_mb,host,date`

**Backend field values**:

- `ort-genai-cpu`: UWP build (`XLLAMA_USE_ORT` defined). Compile-time label; the runtime execution provider on Xbox Series S is CPU EP (see `docs/uwp-constraints.md §5`).
- `cpu`: Linux build (llama.cpp path).

## App settings (console validation)

Modern Xbox settings for routing + TAESD: [`configs/settings-modern.json`](configs/settings-modern.json).
Upload via Device Portal or see [`docs/recommended-config.md`](../docs/recommended-config.md).

## Running benchmarks

### Xbox (automated)

```bash
source ~/.config/xllama/xbox-env
./scripts/bench-xbox-ort.sh smollm2-360m-cpu-int4 --runs 3 --out bench/results/phase1-cpu.csv
```

`--threads N` swaps in `bench/configs/genai_config-threads-N.json`; `--gpu-sample`
records GPU engine/memory telemetry on DML rows.

### Xbox — multi-turn TTFT (KV-cache reuse, Stage 2b)

Measures the KV-reuse win: turn-2 prefill **with reuse** (append only the new
turn) vs the **cold** baseline (full re-prefill of the 2-turn context), on one
persistent session. Triggered when `bench_turns.txt` is present in LocalState
(its content = the turn-2 user prompt; `prompt.txt` = turn 1). Upload both plus
`model.txt` and `bench.flag`, then fetch `bench-kv-result.csv`:

```bash
source ~/.config/xllama/xbox-env
PFN=$(./scripts/deploy.sh pfn)
LS="knownfolderid=LocalAppData&packagefullname=$PFN&path=\\LocalState"
# upload prompt.txt (turn 1), bench_turns.txt (turn 2), model.txt, bench.flag ...
# (use the same upload helper as bench-xbox-ort.sh), then:
curl -sk -u "$XBOX_USER:$XBOX_PASS" \
  "https://$XBOX_IP:11443/api/filesystem/apps/file?$LS&filename=bench-kv-result.csv"
```

Schema: `model,prefill1_ms,n_p1,prefill2_reuse_ms,n_p2_reuse,prefill2_cold_ms,n_p2_cold,speedup,decode_tok_s,n_ctx,host,date`.
`speedup = prefill2_cold_ms / prefill2_reuse_ms` is the headline number.

### Xbox — deterministic capability suite (H9)

`scripts/eval-xbox-models.sh` sends the fixed tasks in
`bench/eval/phase7-h9.json` through the app's OpenAI-compatible LAN endpoint.
It pins temperature 0 / seed 42 and records output, usage, latency and the
machine-checkable verdict for every task:

```bash
source ~/.config/xllama/xbox-env
./scripts/eval-xbox-models.sh \
  --models lfm25-350m,gemma4-e2b,llama32-3b,lfm25-1.2b-instruct,lfm2-2.6b \
  --out bench/results/phase7-h9.jsonl
```

The suite covers Italian, arithmetic, JSON extraction, grounded QA, constrained
summarisation, translation, multi-turn memory and abstention. It is a compact
promotion gate, not a general-purpose model benchmark; preserve the task corpus
unchanged when comparing new candidates to the committed Phase 7 baseline.

### Linux (manual)

```bash
cmake --preset linux-release
cmake --build build/linux-release -j

./build/linux-release/bin/xllama-cli \
    -m models/smollm2-360m.gguf \
    -p "$(cat bench/prompts/standard-512.txt)" \
    -n 128
```

Timing is printed by the bridge's `write_bench_csv()` in `src/bridge/bench.cpp`.
Capture output and append a CSV row to `results/phase1-cpu.csv`.

## Prompts

| File                       | Tokens (approx) | Purpose                          |
| -------------------------- | --------------- | -------------------------------- |
| `prompts/standard-512.txt` | ~512            | General-purpose decode benchmark |
| `prompts/short-32.txt`     | ~32             | Prompt-processing throughput     |

## Results files

| File                     | Phase | Backend                        | Status                                                                                                                                                                                                                                                                                                                                    |
| ------------------------ | ----- | ------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `results/phase1-cpu.csv` | 1     | CPU EP (Xbox Series S, Zen 2)  | populated — 4 runs (2026-05-23); best `n_threads=4` at 71.4 tok/s, regression at `n_threads=8` (28.2 tok/s)                                                                                                                                                                                                                               |
| `results/phase2-dml.csv` | 2     | DML EP (Xbox Series S, RDNA 2) | ✅ populated (2026-07-07) — utilization matrix (v0.3.6): prefill GPU 354 vs CPU 198 tok/s @1k tok; decode CPU 68 vs GPU fp16 46.8 vs GPU int4 8.8; see `docs/uwp-constraints.md §5`. Note: the committed CSV's `backend` column mislabels DML runs as `ort-genai-cpu` (written by an older binary; `bench.cpp` now emits `ort-genai-dml`) |
| `results/profiles/<ts>/` | 2     | —                              | gitignored — downloaded ORT profiling JSON + log tail per run (`profile-dml-run.sh`)                                                                                                                                                                                                                                                      |
