# 2026-11-15 re-measure on CI build 1.7.0.1533

New data points on xllama `1.7.0.1533` (x64 MSIX, dev SKU, Xbox Series S) for two catalogue rows that were still single-run. They are **not reproductions** of the July 2026 rows: the builds, the GGUF CPU repack (#155) and the ORT stack have changed since then.

- Package: CI `xllama-appx` from run 43704605904, source `1334858` (squash-merged as `62db602`), MSIX SHA-256 `64efd48419b995944fbfc018f8998c3d1184ae44cb9d8c44678f94dcdea454ba`.
- Harness: `scripts/bench-xbox-ort.sh` at `origin/main` `62db602`, with the `XLLAMA_EXPECTED_PFN` build guard set to the 1.7.0.1533 package and `--hardware xbox-series-s`. The host tag `xbox-series-s-declared` is an operator declaration. Device Portal `GET /api/os/info` reported `Xbox Series S` in the same session.
- Settings for every row: `--ctx 2048 --n-predict 512 --max-length 0` (derived), default prompt `bench/prompts/standard-512.txt`, `--runs 4`. Run 1 is warmup and dropped; runs 2–4 are recorded with `run_index`.
- The Qwen row uses `--threads 6`. The SmolLM2-1.7B row has no `--threads`; the device `genai_config.json` is the shipped one.
- Before each row the app was stopped and an idle gate read Device Portal `systemperf`: CPU load at most 10% and every GPU engine at most 10% (`logs/*-idle-gate.txt`).
- Both models were provisioned with `scripts/provision-models.sh` from the catalogue URLs just before the campaign.
- No external power, ambient or thermal data was taken, so there is no sidecar. These rows are 3-run medians, not publication-grade headline measurements.

| row                        | CSV                         | decode tok/s median (runs) | old single-run decode | prefill tok/s median | load ms median | peak WS MB |
| -------------------------- | --------------------------- | -------------------------- | --------------------- | -------------------- | -------------- | ---------- |
| Qwen3.5-0.8B Q4_K_M CPU t6 | qwen35-0.8b-q4km-cpu-t6.csv | 34.34 (34.26/34.36/34.34)  | 35.08                 | 174.56               | 2992           | 715        |
| SmolLM2-1.7B int4 ORT CPU  | smollm2-1.7b-cpu-int4.csv   | 21.02 (21.02/20.84/21.66)  | 20.62                 | 61.14                | 6258           | 2324       |

The old Qwen prefill figure (98.1) predates the CPU repack and is not comparable.

## Failed attempt (kept for honesty)

The first Qwen attempt never started a run. The idle gate failed five times with one GPU engine at about 100% and CPU load at 1%. The cause was the xllama UI process itself: with the app stopped the engine dropped to 0%, and it rose again when the app started. The gate now stops the app first. No CSV row was written by that attempt.

## Not measured

- Gemma-4-E2B Q3_K_S: the 2.45 GB file could not be placed on the console in this session (see `docs/uwp-constraints.md` §9), so the row that predates the Gemma 4 template fix is unchanged.
- SmolLM2-360M Q4_K_M CPU, Phi-3.5-mini Q3_K_S, Gemma-4-E2B IQ2_M and SmolLM2-360M int4 DML: none is a catalogue artefact today, so they stay as historical single-run rows.

`docs/benchmarks.md` / `bench/benchmark-summary.json` now select these two rows.
