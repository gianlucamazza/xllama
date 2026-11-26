# Standard benchmarks, first campaign (#597)

Informational. These numbers do not gate a catalogue entry: H9 stays the only
promotion gate. The published table is generated into
[`docs/benchmarks.md`](../../../docs/benchmarks.md) by
`scripts/generate-benchmark-summary.py` from the files in this directory.

## Files

| File                        | What                                                                                   |
| --------------------------- | -------------------------------------------------------------------------------------- |
| `host-loglikelihood.jsonl`  | `scripts/eval-host-standard.py`, one row per model and task                            |
| `series-s-generative.jsonl` | `scripts/eval-standard-generative.py` against the Series S LAN API: item rows, summary |

## Setup

- Models: catalogue GGUFs `lfm25-350m` (`LFM2.5-350M-QAD-Q4_0.gguf`, sha256
  `3d10b6ab…`) and `lfm25-1.2b-instruct`
  (`LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf`, `bb741ebb…`). The full hashes are in
  every row and in `bench/benchmark-summary.json`.
- Host loglikelihood: `llama-perplexity` from the `linux-release` build at the
  llama.cpp pin `d81235049`, 8 threads, on a shared Linux host
  (`linux-host-declared`). Task sizes, `n_ctx` and `-np` are those in
  `bench/eval/standard-datasets.json`. Elapsed times reflect a loaded machine
  and are not a performance figure.
- Series S generative: CI package 1.8.1.1956 (main `16d50c4`), installed in
  place, LAN API on port 11434, model `lfm25-1.2b-instruct` as the catalogue
  ships it (`n_ctx` 2048). The 200 frozen items in
  `bench/eval/standard-generative.json`, temperature 0, seed 42,
  `max_completion_tokens` 1024. The run took about 14 minutes.

## Results

| Model       | HellaSwag  | WinoGrande | ARC-Challenge | ARC-Easy   | MMLU       |
| ----------- | ---------- | ---------- | ------------- | ---------- | ---------- |
| LFM2.5-350M | 42.5 ± 2.5 | 53.8 ± 1.4 | 29.1 ± 2.6    | 49.6 ± 2.1 | 28.7 ± 1.2 |
| LFM2.5-1.2B | 60.5 ± 2.4 | 60.1 ± 1.4 | 34.8 ± 2.8    | 50.0 ± 2.1 | 33.5 ± 1.2 |
| chance      | 25         | 50         | 25            | 25         | 25         |

Series S, `lfm25-1.2b-instruct`: IFEval strict 82/100 (instruction level
88.2 %), GSM8K 80/100. No request failed. 4 IFEval and 5 GSM8K replies hit the
1024-token limit and count as failures.

## Reading

- The 1.2B is ahead of the 350M clearly on HellaSwag, WinoGrande and MMLU
  (over 2.5 combined standard errors). On ARC-Challenge the gap is about 1.5
  combined standard errors, and on ARC-Easy the two are level near 50 %.
- The 350M is close to chance on MMLU (28.7 vs 25) and ARC-Challenge.
- `llama-perplexity` scores multiple choice by the summed log-probability of
  each ending. That is not the `acc_norm` most leaderboards print, so these
  figures compare across xllama runs at this pin, not with published model
  cards. There is no llama.cpp reference figure for these GGUFs to check
  against.
- All 80 GSM8K passes carry an explicit `Answer:` line; the last-number
  fallback decided none of them.
- The host-vs-console comparison on the generative subset is not in this
  campaign. The host `llama-server` run reached under 1 tok/s on the shared
  machine and was stopped; it will be added with its own file.

## Reproduce

```bash
./scripts/eval-host-standard.py --fetch
./scripts/eval-host-standard.py --model <gguf> --hardware linux-host-declared \
  --threads 8 --perplexity build/linux-release/bin/llama-perplexity \
  --out host-loglikelihood.jsonl

source ~/.config/xllama/xbox-env
./scripts/eval-standard-generative.py --fetch
./scripts/eval-standard-generative.py --base-url "http://${XBOX_IP}:11434" \
  --model lfm25-1.2b-instruct --hardware xbox-series-s-declared \
  --endpoint-label series-s-lan-api-<version> --out series-s-generative.jsonl
```
