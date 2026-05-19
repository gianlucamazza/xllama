# bench/

Benchmark suite for xllama. Results are stored as CSV files in `results/`.

## Methodology

- **Prompts**: fixed prompts in `bench/prompts/` (same across all runs for comparability).
- **Runs**: 3 runs per configuration; discard the first (JIT warmup / cache cold); report median of the remaining two.
- **Metrics**:
  - `prompt_tok_s`: tokens/second during prompt processing
  - `decode_tok_s`: tokens/second during generation (excluding prompt)
  - `peak_ram_mb`: peak RSS or working set in MB
- **CSV schema**: `model,quant,backend,n_ctx,n_threads,prompt_tok_s,decode_tok_s,peak_ram_mb,host,date`

## Running benchmarks

```bash
# Build first
cmake -B build -DXLLAMA_TARGET=linux -DCMAKE_BUILD_TYPE=Release
cmake --build build -j

# Run against a model
./build/bin/xllama-cli -m models/qwen3-1.7b-Q4_K_M.gguf \
    -p "$(cat bench/prompts/standard-512.txt)" \
    -n 128
```

Timing is printed by `llama.cpp`'s built-in `llama_perf_context_print()`.
Capture output and append a CSV row to `results/phase1-cpu.csv`.

## Prompts

| File | Tokens (approx) | Purpose |
|------|-----------------|---------|
| `prompts/standard-512.txt` | ~512 | General-purpose decode benchmark |
| `prompts/short-32.txt`     | ~32  | Prompt-processing throughput |

## Results files

| File | Phase | Backend | Status |
|------|-------|---------|--------|
| `results/phase1-cpu.csv` | 1 | CPU (Zen 2) | pending |
| `results/phase2-vulkan.csv` | 2 | Vulkan (RDNA 2) | pending |
