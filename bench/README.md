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
- **Runs**: warmup run 1 (cold start) is discarded; runs 2..N are each recorded **individually** with a `run_index` column, not pre-averaged in the driver (W1.1). `bench-xbox-ort.sh --runs` defaults to 4 → 3 recorded measurement runs. `scripts/generate-benchmark-summary.py` derives the median and the decode min–max spread from the recorded runs; the previous behaviour appended a single median row, which destroyed the spread needed to tell a real change from run-to-run noise. A selector that resolves to only one row (every row written before this change) is reported as-is and marked _single run_ in the generated table.
- **Metrics**:
  - `prompt_tok_s`: tokens/second during prompt processing
  - `decode_tok_s`: tokens/second during generation (excluding prompt)
  - `peak_ws_mb`: peak working set / RSS in MB
  - `load_ms`: model load time in milliseconds
  - `gpu_mem_mb` / `gpu_budget_mb`: per-process GPU memory CurrentUsage/Budget after model load (`QueryVideoMemoryInfo`, LOCAL segment); 0 on CPU-only runs and Linux builds
  - `n_prompt_tok`: prefill token count actually measured. Without it a row cannot be compared with one taken at another prompt length — the reason the pre-2026-07 DirectML rows are not re-readable.
  - ⚠️ **GGUF pre/post-repack boundary (2026-07-25)**: every llama.cpp row written before build 675 predates `GGML_USE_CPU_REPACK` (PR #155 — it was dead code on Xbox), so its `prompt_tok_s` is **not comparable** with post-675 rows: the identical model+prompt measures 241.9 pre vs 393.2 post (`phase13-repack-{before,after}.csv`). Decode columns are unaffected.
  - ⚠️ **GGUF prefill-thread boundary (2026-07-26)**: every llama.cpp row written before build 711 predates the `n_threads_batch` fix (#168, PR #177), so its `prompt_tok_s` was measured on llama.cpp's default **4 prefill threads** regardless of the row's `n_threads` column (which describes decode): 390.7 pre vs 438.1 post at P=298 (`phase13b-threadsbatch-{before,after}.csv`). Decode columns are unaffected.
  - `n_gen_tok`: tokens actually generated. **Not** the requested `n_predict`: generation is capped by the context window (`prompt + new <= n_ctx`) and can stop early on EOG. A 1574-token prompt at `n_ctx` 2048 caps new tokens at 474 and generated 277.
  - `run_index`: which recorded repetition of a configuration this row is (W1.1). Written by the device from `bench_run_index.txt`. `0` = a single-run / legacy measurement; a non-zero value marks one of several repeats the summary aggregates into a median + spread.
- **CSV schema**: `model,quant,backend,n_ctx,n_threads,prompt_tok_s,decode_tok_s,peak_ws_mb,load_ms,gpu_mem_mb,gpu_budget_mb,n_prompt_tok,n_gen_tok,max_length,host,date,run_index,prefill_ms,ttft_ms`

  `prefill_ms` is prompt-processing duration. `ttft_ms` is measured from
  prefill start until the first generated token is ready; zero means that no
  token was produced. The fields are appended after `run_index` to preserve
  positional compatibility with older bench scripts. They do not establish
  thermal equilibrium by themselves.

  A headline throughput claim also requires the exact model file size and
  quantization, fixed context/prompt/sampling settings, one discarded warm-up,
  and at least three recorded runs after a stable thermal state. Report the
  median and min–max spread, never the best row. Measure power externally at
  the console/UPS outlet and record idle, prefill, sustained decode watts,
  wattmeter model/firmware, sampling interval, ambient temperature, and the
  thermal-equilibrium rule in a sidecar report.
  Generate and validate the sidecar with `scripts/write-benchmark-sidecar.py`;
  validate raw CSVs with `scripts/validate-benchmark.py`. Both tools preserve
  v1 historical files and identify v2 explicitly.
  Rows written before 2026-07-21 have 13 columns and no prompt/generated counts; rows in `phase12-dml-crossover.csv` carry `n_prompt_tok` but leave `n_gen_tok` empty (the console build that produced them predates that column). Rows written before the W1.1 change have no `run_index` and are read as single measurements. The new fields are appended after `run_index` so no earlier positional column shifts: `bench-xbox-ort.sh` parses `backend` ($3), `decode_tok_s` ($7) and `max_length` ($14) positionally. The script refuses to append to a file whose header does not match, and rejects rows whose field count disagrees with the schema.

  `max_length` (added 2026-07-21) is `min(n_ctx, n_prompt_tok + n_predict)`, the value actually requested of the engine. On DirectML it is the variable that governs prefill throughput — see #130 and `docs/uwp-constraints.md` §5c — so a row without it cannot be interpreted, and `n_ctx` does not substitute for it.

**Backend field values**:

- `ort-genai-cpu`: UWP build (`XLLAMA_USE_ORT` defined). Compile-time label; the runtime execution provider on Xbox Series S is CPU EP (see `docs/uwp-constraints.md §5`).
- `ort-genai-dml`: UWP run whose model directory is a DML asset, or which reported non-zero `gpu_mem_mb` (`src/bridge/bench.cpp`). Rows written before 2026-07-19 mislabel DML runs as `ort-genai-cpu` (older binary).
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
records GPU engine/memory telemetry on DML rows. `--ctx N` and `--n-predict N`
override the engine defaults (2048 / 512) via `bench_ctx.txt` / `bench_npredict.txt`;
0 keeps the default. `--prompt FILE` and `--out FILE` select the prompt and the
results CSV.

### Xbox — prompt-length sweep

`scripts/bench-prompt-sweep.sh` drives `bench-xbox-ort.sh` across a set of
synthetic prompt lengths and both backends, one CSV row per point:

```bash
source ~/.config/xllama/xbox-env
./scripts/bench-prompt-sweep.sh --tokens "150 300 550 800 1100 1600" --runs 4 \
  --out bench/results/phase12-dml-crossover.csv
```

This produced the sweep behind the 1550-token routing threshold and the
pathological DirectML band documented in `docs/uwp-constraints.md` §5b.

### Xbox — multi-turn TTFT (KV-cache reuse, Stage 2b)

Measures the KV-reuse win: turn-2 prefill **with reuse** (append only the new
turn) vs the **cold** baseline (full re-prefill of the 2-turn context), on one
persistent session. Triggered when `bench_turns.txt` is present in LocalState
(its content = the turn-2 user prompt; `prompt.txt` = turn 1). Upload both plus
`model.txt` and `bench.flag`, then fetch `bench-kv-result.csv`:

`scripts/bench-xbox-kv.sh` drives this end-to-end (upload, restart, poll, fetch);
the manual `curl` recipe it replaced is no longer needed:

```bash
source ~/.config/xllama/xbox-env
./scripts/bench-xbox-kv.sh smollm2-360m-cpu-int4 \
  --prompt bench/prompts/standard-512.txt --runs 2 --out bench/results/my-kv.csv
```

Note `bench-xbox-ort.sh` deletes `bench_turns.txt` before every run, so a stale
one cannot hijack a single-turn bench into this mode.

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

**Known limit of the scorer, found in review 2026-08-10.** Scoring is regex over
the response text, so a semantically wrong answer containing the required tokens
passes. Concrete case: `grounded_qa` requires `4` and `730[[:space:]]*MB`, and
`lfm25-230m` scores a pass for _"4 CPU cores, **each** occupying 730 MB"_ — 730 MB
is the device total, not per core, and the regex cannot see the quantifier. Every
model measured passes this task, so cross-model comparison — what the suite is
for — is unaffected; read an absolute H9 as "under this scorer" rather than
"tasks solved". Tightening it means changing the task definition in
`eval/phase7-h9.json`, never editing a recorded row in the JSONL.

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

| File                       | Tokens (approx) | Purpose                                                       |
| -------------------------- | --------------- | ------------------------------------------------------------- |
| `prompts/standard-512.txt` | ~272            | General-purpose decode benchmark — the name is historical     |
| `prompts/short-32.txt`     | ~32             | Prompt-processing throughput                                  |
| `prompts/long-1k.txt`      | ~1000           | Long-prompt prefill; carries a literal ChatML wrapper (below) |

`standard-512.txt` measures **~272 tokens**, not 512 — the filename predates any
measurement of it. Confirmed by the `n_prompt_tok` column now written by
`write_bench_csv`, and by `docs/benchmarks.md`, which calls it "the 272-token
declarative prompt".

`long-1k.txt` embeds `<|im_start|>`/`<|im_end|>` markup. The bench path re-applies
the model's chat template (`uwp/inference-bridge.cpp`), so feeding it verbatim
double-templates the prompt. `scripts/bench-prompt-sweep.sh` and
`scripts/validate-console.sh` both strip the wrapper and use only the prose.

## Results files

**This table is not the index of every file in `results/`** — it currently
annotates 13 of ~38, and a file's absence from it is not a sign that the file is
orphaned. The rule: a row exists where the _provenance_ needs explaining (an A/B
whose two halves are only comparable under a stated condition, a measurement
whose units are easy to misread, a negative result). Everything else is reachable
through `benchmark-summary.json`, which declares which run backs which published
row, and through the phase document that owns the campaign — `phase7-*` in
`docs/phase7-hypotheses.md`, `phase12-*` in `docs/uwp-constraints.md §5`,
`phase14-*` in `docs/model-matrix.md`. Before quoting any CSV, check whether one
of those explains what it means.

| File                                               | Phase | Backend                         | Status                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           |
| -------------------------------------------------- | ----- | ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `results/phase1-cpu.csv`                           | 1     | CPU EP (Xbox Series S, Zen 2)   | populated — 4 runs (2026-05-23); best `n_threads=4` at 71.4 tok/s, regression at `n_threads=8` (28.2 tok/s)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `results/phase2-dml.csv`                           | 2     | DML EP (Xbox Series S, RDNA 2)  | ✅ populated (2026-07-07) — utilization matrix (v0.3.6): prefill GPU 354 vs CPU 198 tok/s @1k tok; decode CPU 68 vs GPU fp16 46.8 vs GPU int4 8.8; see `docs/uwp-constraints.md §5`. Note: the committed CSV's `backend` column mislabels DML runs as `ort-genai-cpu` (written by an older binary; `bench.cpp` now emits `ort-genai-dml`)                                                                                                                                                                                                                                                                                                                                                                                                                        |
| `results/profiles/<ts>/`                           | 2     | —                               | gitignored — downloaded ORT profiling JSON + log tail per run (`profile-dml-run.sh`)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `results/phase12-*.csv`                            | 12    | CPU + DML                       | DirectML routing calibration campaign (crossover sweep, max_length band, KV reuse, threads) — see `uwp-constraints.md §5b–§5f`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `results/phase12b-threads-sweep.csv`               | 12    | CPU EP                          | the threads-6 ship-condition sweep (3 lengths × {unset,t4,t6} × 3 runs + closing control, build 675, pristine config verified first)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| `results/phase13-repack-{before,after}.csv`        | 13    | llama.cpp CPU                   | `GGML_USE_CPU_REPACK` A/B (builds 674/675): prefill +62% on `lfm25-350m` Q4_K_M, decode/RAM unchanged — the post-repack GGUF baseline                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            |
| `results/phase13b-threadsbatch-{before,after}.csv` | 13    | llama.cpp CPU                   | `n_threads_batch` A/B (#168, builds 698/711): prefill +12.1% (P=298) / +10.5% (P=1000) on `lfm25-350m` Q4_K_M, decode/RAM unchanged — the current GGUF prefill headline source                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `results/phase13c-ubatch-sweep.csv`                | 13    | llama.cpp CPU                   | `n_ubatch` sweep post-repack+#168 (#172, `--ubatch` u128/u256/u512/u1024, P=1000, 3 runs each): **default 512 is the optimum** — u256 −0.7%, u128 −2.4%, u1024 −2.8% and +34 MB peak. The 2026-07-14 host sweep's "inconclusive" verdict resolves to "keep the default"; peak RAM scales with ubatch (305 → 364 MB)                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| `results/phase15-ramceil.csv`                      | 15    | none (no model loaded)          | heap-ceiling probe (`ramceil.flag`, MSIX 1.5.1.762): **4864 MB committed / 4893 MB peak WS** over 38 × 128 MB faulted-in steps, process overhead a flat 29 MB, `avail_phys` ~5113 → 240 MB. A **lower bound** — it stopped on its own 256 MB floor, not on a failed allocation — and a **headless** one: no model, no XAML, no compositor, so the in-app ceiling is lower. Decides MoE admissibility in H2                                                                                                                                                                                                                                                                                                                                                       |
| `results/phase15-spec-vocab.csv`                   | 15    | none (vocab-only model load)    | H3 precondition: `common_speculative_are_compatible` (`common/speculative.cpp:64`, which **throws** on failure) replicated over 5 catalogue pairs on host. **`qwen25-coder-3b` ← `qwen25-coder-0.5b` PASSES**, so W2 proceeds. Two negatives kept: Qwen3-1.7B vs Coder share a vocab **size** but differ in 4 token texts, and LFM2.5-8B-A1B carries a 128000-token vocab against LFM2.5-350M's 65536 — H2 and H3 do not compose on the MoE                                                                                                                                                                                                                                                                                                                      |
| `results/phase15-spec-pregate.csv`                 | 15    | llama.cpp CPU (host, throttled) | H3 pre-gate: acceptance rate and drafting frequency for both speculative variants, two prompt regimes (`spec-code-edit`, `spec-chat-open`), k=2/4. Host timings in the file are **throttled and unusable** — only `accept_pct` and `n_drafted` transfer, being properties of the pair and the prompt. Combined with the console cost model by `scripts/analyze-spec-pregate.py`: **draft model 1.43x on code but 0.81x on chat** (0.67x at k=4), **prompt lookup 1.53x on code and 1.00x on chat** — it drafts 10 tokens where the draft model drafts 140. Verdict: draft model rejected, draft-free variant proceeds. Companion `…-pregate-batched.txt` is the `llama-batched-bench` curve showing T(n) linear over n=1..8, i.e. no free plateau at small batch |
| `results/phase15-moe-console.csv`                  | 15    | llama.cpp CPU (Series S)        | **H2 FAIL**, measured on MSIX 1.5.2.798: `lfm25-8b-a1b` UD-IQ3_S decodes **14.50 tok/s** against the dense `qwen25-coder-3b`'s 14.0, at **3553 MiB** peak (+1437). The sparse-activation premise held — ~631 MB read/token against 645 predicted — but the cost moves off bandwidth, and the model reasons every turn (102 completion tokens to answer "Roma", 401 for two sentences), so perceived latency is ~4× worse. H9 was not run: its tasks cap generation at 16-80 tokens, which a reasoning model cannot clear. Verdict and the two unseparated causes (i-quant dequant vs expert gather) in `docs/phase7-hypotheses.md` §H2                                                                                                                           |
| `results/phase15-spec-w2-console.csv`              | 15    | llama.cpp CPU (Series S)        | W2 prompt-lookup A/B (`qwen25-coder-3b`, CI 1.5.2.920): code **1.04× FAIL** ≥1.4×; chat ~0.99×. Opt-in only. SSOT `docs/phase15-re-opt.md`                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| `results/phase15-gpubw.csv`                        | 15    | D3D12 STREAM (Series S)         | W3 M6 **PASS**: **119.07 GB/s** read, 1 GiB, checksum_ok (CI 1.5.2.853) ≥100 kill → H6 eng motivated then parked after H6.1                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| `results/phase16-gguf.csv`                         | 16    | llama.cpp CPU (Series S, t6)    | Phase 16 model-scouting T3 (2026-08-10, MSIX 1.5.4.887, `n_ctx` 2048, `n_predict` 96, 3 recorded `run_index` each): `lfm25-230m` **PASS** 119.17 tok/s @ 241 MB (the shipped LiquidAI build; the `unsloth` build's 119.77 is in `phase16-quantiser-ab.csv`); `qwen35-2b` FAIL 19.25 @ 1421 MB; `maincoder-1b` FAIL 33.49 @ 843 MB. Also the evidence that the `weights × 1.12` peak factor **under-predicts** (−4% to −47%) — overhead is a ~95 MB floor plus an arch-dependent KV term. Cards and verdicts: `docs/phase16-model-scouting.md`                                                                                                                                                                                                                    |
| `results/phase16-quantiser-ab.csv`                 | 16    | llama.cpp CPU (Series S, t6)    | Phase 16 quantiser A/B on the same model and quant: `unsloth` LFM2.5-230M-Q4_K_M (153,406,656 B) vs the shipped LiquidAI build (153,406,304 B). **Identical** `peak_ws_mb` 241 and H9 2/8; decode 119.77 vs 119.17 (0.5%). The two artefacts are not byte-identical and their greedy output differs, so the shipping figure is the LiquidAI row in `phase16-gguf.csv`; this file is why that distinction was measured rather than assumed                                                                                                                                                                                                                                                                                                                        |
| `results/phase16-mic.json`                         | 16    | AudioGraph probe (Series S)     | Phase 16 WS-F S-gate, not a benchmark: `mic.flag` → `run_mic_probe` on MSIX 1.5.4.897. Carries the WinRT status enums by name, because `AccessDenied` (the sandbox refuses) and `DeviceNotAvailable` (no headset attached) are different answers and only the first closes the workstream. Measured: graph `Success` at 48 kHz stereo, node `DeviceNotAvailable`, 0 samples — **inconclusive by design, not FAIL**. Reasoning: `docs/uwp-constraints.md` §10d                                                                                                                                                                                                                                                                                                    |
| `results/phase16-h9.jsonl`                         | 16    | LAN API (Series S)              | Phase 16 T4 **harness control only**: `lfm25-350m` re-measured at **4/8**, reproducing its recorded §A1 value before the campaign's own H9 numbers were read. The candidate and incumbent rows (`lfm25-230m` 2/8, `gemma3-270m` 3/8) moved into `phase7-h9.jsonl` when H16.1c shipped, so CI checks them                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `results/phase15-gpugemv.csv`                      | 15    | D3D12 Q4_K GEMV (Series S)      | H6.1: G1 PASS / G2 FAIL (`packed_gbs=2.15` vs soft 40). Measure only; eng parked (#228)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
| `results/phase15-gpugemv-h62.csv`                  | 15    | D3D12 Q4_K GEMV (Series S)      | H6.2 **K2** (CI `1.5.5.922`): `wave32` G1 PASS, median **25.4** GB/s packed (24.89–26.02); retimed `naive` median 1.96. G2 stays 40. Not a product tok/s table                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
| `results/store-app-vs-game-2026-08-21.csv`         | store | llama.cpp CPU (Series S, t6)    | Store D1 App vs Game (`lfm25-350m`, CI `1.5.5.922`, `standard-512`): GPU budget **691 vs 3801 MB**. Long-gen decode App **86.6–87.0** vs Game **94.7**. Peak 320 both. Verdict: request Game metadata. SSOT `docs/store-readiness.md` §6                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                         |
| `results/phase15-thinking-complete.csv`            | 15    | quality / gates (Series S)      | **#223 closed:** short prompts complete at catalogue `n_predict` 1024 (`thinkdone`); train-style multi-step can fail CoT @768. Not a tok/s table                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 |
| `results/t6-shipped-confirm.csv`                   | 12    | ORT CPU EP                      | on-device confirmation of the shipped t6 asset (1.5.0.0 migration, 3 runs): 262.4 prefill / 74.8 decode — feeds the generated ORT CPU headline                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   |
