# 2026-11-15 re-measure of the two LFM2.5 rows on CI build 1.7.0.1563

The `lfm25-230m` and `lfm25-350m` headline figures still came from the 1.6.0
package (`2026-09-30-catalogue-gates.csv`). These are new data points on
xllama `1.7.0.1563` (x64 MSIX, dev SKU, Xbox Series S).

- Package: CI `xllama-appx` from run 43740925613, source `4d9b6d9` (PR #451 head, squash-merged as `06710d1`), MSIX SHA-256 `ba113340b3e3a919320aa39e2da66d29bed785dcb8d5325c1235a95da155a4e2`.
- Harness: `scripts/bench-xbox-ort.sh` at `origin/main` `ca60882`, with the `XLLAMA_EXPECTED_PFN` build guard set to the 1.7.0.1563 package and `--hardware xbox-series-s`. The host tag `xbox-series-s-declared` is an operator declaration. Device Portal `GET /api/os/info` reported `Xbox Series S` in the same session.
- Settings: `--ctx 2048 --n-predict 512 --max-length 0 --threads 6 --ignore-eog`, default prompt `bench/prompts/standard-512.txt` (298 prompt tokens), `--runs 4`. Run 1 is warmup and dropped; runs 2–4 are recorded with `run_index`.
- Before each row the app was stopped and an idle gate read Device Portal `systemperf`: CPU load at most 10% and every GPU engine at most 10% (`logs/*-idle-gate.txt`).
- No external power, ambient or thermal data was taken, so there is no sidecar. These rows are 3-run medians, not publication-grade headline measurements.

| row                         | CSV                               | decode tok/s median (runs)    | previous decode (1.6.0) | prefill tok/s median | load ms median | peak WS MB |
| --------------------------- | --------------------------------- | ----------------------------- | ----------------------- | -------------------- | -------------- | ---------- |
| LFM2.5-230M QAD Q4_0 CPU t6 | lfm25-230m-qad-q40-cpu-t6-eog.csv | 127.12 (127.12/127.38/126.93) | 132.42                  | 623.69               | 663            | 237        |
| LFM2.5-350M QAD Q4_0 CPU t6 | lfm25-350m-qad-q40-cpu-t6-eog.csv | 97.90 (97.88/98.35/97.90)     | 101.57                  | 371.42               | 1046           | 311        |

## Why `--ignore-eog`, and what the lower figures mean

The previous rows let the model stop on its own. Their recorded runs generated
273, 248 and 512 tokens (230M) and 121, 186 and 33 tokens (350M). Decode is
faster on a short reply, because each new token attends to a shorter history,
so those medians mix reply lengths.

A first attempt at the 350M row on this package, without `--ignore-eog`,
produced runs of 6, 318 and 6 tokens (118.66, 98.88 and 99.38 tok/s). A
six-token decode is not a usable rate, so that attempt was discarded and both
rows were measured with exactly 512 generated tokens per run.

The new figures are 4.0% (230M) and 3.6% (350M) below the previous ones. This
campaign does not separate the two causes, reply length and package. One
like-for-like point exists: the previous 230M row's single 512-token run was
128.53 tok/s, against 127.12 here, a 1.1% difference.

`docs/benchmarks.md` / `bench/benchmark-summary.json` now select these two rows.
