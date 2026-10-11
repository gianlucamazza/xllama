# 2026-10-11 re-measure on CI build 1.6.0.1163

New data points on xllama `1.6.0.1163` (x64 MSIX, dev SKU, Xbox Series S). These are **not reproductions** of the original Jul 2026 rows (phases 2/6/7): the builds, the GGUF CPU-repack (#155, build 675) and the ORT/DirectML stack have all changed since then.

- Harness: `scripts/bench-xbox-ort.sh` @ origin/main `05237cc`, with the `XLLAMA_EXPECTED_PFN` build guard set to the 1.6.0.1163 package.
- Settings for every row: `--ctx 2048 --n-predict 512 --max-length 0` (derived), default prompt `bench/prompts/standard-512.txt` (298/280 tok after template), `--runs 4`. Run 1 is warmup and dropped; runs 2–4 are recorded with `run_index`.
- CPU rows use `--threads 6`. The DML row has no `--threads` and the device genai_config is untouched.
- Before each row, a read-only idle gate checked that the console had no other GPU workload (no foreign GPU process, all GPU engines ≤10%).
- Window: 2026-10-11 09:37–10:07 UTC.

| row                                                                  | CSV                             | decode tok/s median (runs) | old single-run decode | prefill tok/s median | load ms median | peak WS MB |
| -------------------------------------------------------------------- | ------------------------------- | -------------------------- | --------------------- | -------------------- | -------------- | ---------- |
| Gemma-3-270M Q4_K_M CPU t6                                           | gemma3-270m-q4km-cpu-t6.csv     | 79.92 (79.92/79.88/80.14)  | 76.78                 | 486.11               | 1219           | 366        |
| LFM2-2.6B Q4_K_M CPU t6                                              | lfm2-2.6b-q4km-cpu-t6-rerun.csv | 17.85 (17.85/17.76/17.85)  | 18.36                 | 54.93                | 8507           | 1623       |
| Gemma-4-E2B Q3_K_S CPU t6                                            | gemma4-e2b-q3ks-cpu-t6.csv      | 15.30 (15.30/15.27/15.31)  | 15.31                 | 28.45                | 13533          | 2741       |
| Llama-3.2-3B Q3_K_S CPU t6 (dir `llama32-3b`, was `llama32-3b-q3ks`) | llama32-3b-q3ks-cpu-t6.csv      | 14.12 (14.12/14.10/14.12)  | 14.16                 | 21.19                | 16775          | 1827       |
| SmolLM2-360M fp16 ORT DirectML (-v2)                                 | smollm2-360m-dml-fp16-v2.csv    | 41.23 (41.23/41.22/41.34)  | 44.36                 | 231.29               | 2878           | 1206       |

Prefill is not comparable to the old GGUF rows, because those predate the CPU repack.

## Incomplete / hung attempt (kept for honesty)

`lfm2-2.6b-q4km-cpu-t6.csv` is the **first** LFM2-2.6B attempt. Runs 2 and 3 were recorded. Run 4 hung: `xllama.log` stops at `prompt tokens: 298` after a normal model load (`logs/R2-run4-hang-xllama-log-tail.txt`), and the harness timed out after 300 s.

- The console then showed CPU load 51% and GPU engine 5 at ~100%, with xllama.exe alive.
- The process was terminated via Device Portal ~9 min after the hang. GPU and CPU dropped to 0%/1% within seconds.
- The re-run (`-rerun.csv`) completed 4/4 with no timeout. These are the numbers in the table above.
- Do not use the 2-row CSV for the summary.

## Not measured

These rows have no model files on the console, and uploading them needs a separate decision:

- SmolLM2-360M Q4_K_M CPU
- Qwen3.5-0.8B Q4_K_M
- SmolLM2-1.7B int4 ORT CPU
- Phi-3.5-mini Q3_K_S
- Gemma-4-E2B IQ2_M
- SmolLM2-360M int4 DML

`docs/benchmarks.md` / `bench/benchmark-summary.json` now select these five
rows (LFM2-2.6B from the `-rerun` CSV). The six un-remeasured rows are
unchanged.
