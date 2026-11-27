# 2026-11-27 regression remeasure on the v1.11.0 release package

v1.11 rewrote the shared decode loop (#678): `llama_decode` return codes propagate, a failure
drains and clears the cache, and prefill ends with a `llama_synchronize`. No console measurement
covered that change. The earlier rows include packages 1.6.0.1163 and 1.7.0.1563. This run
re-measures the headline llama.cpp models on the v1.11.0 release package
(`bench/README.md` now requires this when a PR changes the decode loop, prefill or a sampler chain).

- Package: `xllama_1.11.0.2203_x64.msix`, CI `xllama-appx` from `build-uwp` run 46070519230 on
  main `ee6a718`, SHA-256 `ed6ac5a66a0d5925c7b6103dc27b73a9f320715b5eb2237c12464f12be2f5651`
  (the v1.11.0.0 release asset; record in `../2026-11-27-v1110-series-s-release/`).
- Harness: `scripts/bench-xbox-ort.sh` at main `bc53c3d`, `XLLAMA_EXPECTED_PFN` set to the
  1.11.0.2203 package, `--hardware xbox-series-s` (operator declaration; the release record's
  Device Portal `/api/os/info` reported `Xbox Series S` the same day).
- Settings: `--threads 6 --runs 4 --max-length 0 --ignore-eog`, default prompt
  `bench/prompts/standard-512.txt` (298 measured tokens on the LFM models and 292 on Coder),
  `--ctx 2048 --n-predict 512`, except LFM2.5-1.2B
  (`--n-predict 256`, as its previous rows; d3d12 row with `--gpu-layers 99`) and
  Qwen2.5-Coder-1.5B (`--ctx 4096`, its catalogue context). Run 1 is warmup; runs 2–4 are recorded.
- Before each row the app was stopped and an idle gate read Device Portal `systemperf` until three
  consecutive samples had CPU load ≤10% and every GPU engine ≤10% (`logs/*-idle-gate.txt`).
- No external power, ambient or thermal data was taken, so there is no sidecar. These are 3-run
  medians, not publication-grade measurements.

| Row                       | Decode tok/s median (runs)    | Previous decode           | Δ     | Prefill tok/s (previous) | Peak WS MB (previous) |
| ------------------------- | ----------------------------- | ------------------------- | ----- | ------------------------ | --------------------- |
| LFM2.5-230M CPU t6        | 126.87 (126.87/127.10/126.83) | 127.12 (1.7.0.1563, same) | −0.2% | 626.1 (623.7)            | 237 (237)             |
| LFM2.5-350M CPU t6        | 97.32 (97.27/97.32/97.46)     | 97.90 (1.7.0.1563, same)  | −0.6% | 371.6 (371.4)            | 311 (311)             |
| LFM2.5-1.2B d3d12 g99     | 74.03 (74.32/74.03/73.76)     | 74.84 (1.6.0.1163, same)  | −1.1% | 256.5 (257.4)            | 861 (860)             |
| LFM2.5-1.2B CPU t6        | 39.73 (39.73/39.74/39.68)     | 39.89 (1.6.0.1163, same)  | −0.4% | 109.3 (109.4)            | 786 (783)             |
| Qwen2.5-Coder-1.5B CPU t6 | 26.15 (26.15/26.15/26.09)     | 25.73 (one 512-token run) | +1.6% | 104.4 (96.6)             | 1176 (1179)           |
| LFM2-2.6B CPU t6          | 18.09 (18.10/18.07/18.09)     | 17.85 (270–452 tokens)    | +1.3% | 57.8 (54.9)              | 1629 (1623)           |

Across the four rows with matching recorded workloads, median decode throughput fell by
0.2–1.1%. The largest change was on d3d12, where the earlier and current three-run ranges overlap
(74.11–74.91 and 73.76–74.32 tok/s). Median prefill on that row went from 257.44 to 256.47 tok/s
(−0.38%). The separate three-run campaigns give descriptive comparisons. Establishing a performance
margin or attributing a change to the prefill synchronization requires a controlled, paired run.

To reproduce the comparisons, select runs 2–4 from the matching model files under
[`2026-11-15-remeasure-1563`](../2026-11-15-remeasure-1563/) for the 230M and 350M. For both 1.2B
rows select `model=lfm25-1.2b-instruct`, `n_prompt_tok=298`, `n_gen_tok=256` and the matching
`backend` in [`2026-10-02-fa-gguf-gpu.csv`](../2026-10-02-fa-gguf-gpu.csv). Compute the change from
the full-precision medians of `decode_tok_s`.

The two remaining comparisons have different token counts. The earlier Coder runs measured 298 prompt
tokens and generated 119 and 512 on an unrecorded package. The table uses the 512-token row
(`run_index=3`) for the prior decode figure. The current runs measured 292 prompt tokens and generated 512. The earlier 2.6B runs generated 270, 452 and 352 tokens before stopping, against a fixed 512 here.
See [the Coder CSV](../phase14-console.csv) and [the 2.6B rerun](../2026-10-11-remeasure-1163/lfm2-2.6b-q4km-cpu-t6-rerun.csv).

These rows replace the six headline rows in `bench/benchmark-summary.json`: they are newer, bound
to a release package and all use a fixed-length decode.
