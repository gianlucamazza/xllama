# bench/

Benchmark suite for xllama. Results are stored as CSV files in `results/`.

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

## Running benchmarks

### Xbox (automated)

```bash
source ~/.config/xllama/xbox-env
./scripts/bench-xbox.sh smollm2-360m-cpu-int4 bench/config/phase1-smollm2-360m.json
```

Results are appended to `bench/results/phase1-cpu.csv`.

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

| File                     | Phase | Backend                        | Status                                                                                                                                             |
| ------------------------ | ----- | ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `results/phase1-cpu.csv` | 1     | CPU EP (Xbox Series S, Zen 2)  | populated — 4 runs (2026-05-23); best `n_threads=4` at 71.4 tok/s, regression at `n_threads=8` (28.2 tok/s)                                        |
| `results/phase2-dml.csv` | 2     | DML EP (Xbox Series S, RDNA 2) | ⏳ `VERDICT: GPU` proven (headless mode, v0.3.4); full bench pending a DML model variant — see `ROADMAP.md` Phase 2 & `docs/uwp-constraints.md §7` |
| `results/profiles/<ts>/` | 2     | —                              | gitignored — downloaded ORT profiling JSON + log tail per run (`profile-dml-run.sh`)                                                               |
