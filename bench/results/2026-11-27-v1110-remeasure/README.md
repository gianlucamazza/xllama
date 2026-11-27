# 2026-11-27 regression remeasure on the v1.11.0 release package

v1.11 rewrote the shared decode loop (#678): `llama_decode` return codes propagate, a failure
drains and clears the cache, and prefill ends with a `llama_synchronize`. No console measurement
covered that change, and the headline rows came from packages 1.6.0.1163 to 1.7.0.1563. This run
re-measures the headline llama.cpp models on the v1.11.0 release package
(`bench/README.md` now requires this when a PR changes the decode loop, prefill or a sampler chain).

- Package: `xllama_1.11.0.2203_x64.msix`, CI `xllama-appx` from `build-uwp` run 46070519230 on
  main `ee6a718`, SHA-256 `ed6ac5a66a0d5925c7b6103dc27b73a9f320715b5eb2237c12464f12be2f5651`
  (the v1.11.0.0 release asset; record in `../2026-11-27-v1110-series-s-release/`).
- Harness: `scripts/bench-xbox-ort.sh` at main `bc53c3d`, `XLLAMA_EXPECTED_PFN` set to the
  1.11.0.2203 package, `--hardware xbox-series-s` (operator declaration; the release record's
  Device Portal `/api/os/info` reported `Xbox Series S` the same day).
- Settings: `--threads 6 --runs 4 --max-length 0 --ignore-eog`, default prompt
  `bench/prompts/standard-512.txt` (298 tokens), `--ctx 2048 --n-predict 512`, except LFM2.5-1.2B
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
| LFM2.5-1.2B d3d12 g99     | 74.03 (74.32/74.03/73.76)     | 74.8 (1.6.0.1163, same)   | −1.0% | 256.5 (257.4)            | 861 (860)             |
| LFM2.5-1.2B CPU t6        | 39.73 (39.73/39.74/39.68)     | 39.89 (1.6.0.1163, same)  | −0.4% | 109.3 (109.4)            | 786 (783)             |
| Qwen2.5-Coder-1.5B CPU t6 | 26.15 (26.15/26.15/26.09)     | 25.73 (one 512-token run) | +1.6% | 104.4 (96.6)             | 1176 (1179)           |
| LFM2-2.6B CPU t6          | 18.09 (18.10/18.07/18.09)     | 17.85 (270–452 tokens)    | ≈+1%  | 57.8 (54.9)              | 1629 (1623)           |

**Outcome: no regression.** Every row on identical settings is within 1% of its previous figure,
inside the spread earlier remeasures showed between packages. d3d12 prefill moved by −0.3%, so the
end-of-prefill synchronisation does not shift measurable time into `prefill_ms`. The Coder and
LFM2-2.6B previous rows used other settings (`phase14-console` on an unrecorded package with
replies of 119 and 512 tokens; the 2.6B rows let the model stop), so those two comparisons are
approximate. Their higher prefill is not attributed to v1.11.

These rows replace the six headline rows in `bench/benchmark-summary.json`: they are newer, bound
to a release package and all use a fixed-length decode.
