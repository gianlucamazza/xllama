# Granite-4.0-H-Tiny host receipts (#525)

Host-side ladder for the resident MoE candidate against the #273 reopen gate.
Architecture `granitehybrid` (Mamba2 hybrid + MoE) on llama.cpp pin
`d81235049384534c167caea52b85a694f6103d14`. Official GGUF
`ibm-granite/granite-4.0-h-tiny-GGUF` Q3_K_S. Apache 2.0.

**Not a catalogue entry.** Do not copy this id into `uwp/models/manifest.json`.

Series S T3 receipts are `t3-series-s.csv` and `t3-llama32-3b.csv`. Every run
`peak_ws_mb` ≤ 3584. Host tok/s and host RSS in the T1 / snapshot logs below
are not product numbers. The §I1 row is filled from the CSVs only
([model-matrix.md](../../../docs/model-matrix.md) §I1).

## Identity

| Item          | Value                                                                                            |
| ------------- | ------------------------------------------------------------------------------------------------ |
| Model card    | https://huggingface.co/ibm-granite/granite-4.0-h-tiny                                            |
| Official GGUF | https://huggingface.co/ibm-granite/granite-4.0-h-tiny-GGUF                                       |
| File          | `granite-4.0-h-tiny-Q3_K_S.gguf`                                                                 |
| Bytes         | 3,062,452,832                                                                                    |
| SHA-256       | `5ec3f4af77fffead37b09984a97625f0f532f6ae2ead65b5019bce6dd8b4d0cd`                               |
| llama.cpp pin | `d81235049384534c167caea52b85a694f6103d14`                                                       |
| Chat renderer | `ChatFormatKind::Granite` (`<\|start_of_role\|>…<\|end_of_role\|>…<\|end_of_text\|>`), also #521 |
| Override id   | `granite40-h-tiny` (basename contains `granite`; no `thinking` / `a1b` / `llama`)                |

## Files

| File                     | What it is                                                                                  |
| ------------------------ | ------------------------------------------------------------------------------------------- |
| `t1-host-smoke.log`      | T1: load, `general.architecture`, greedy `--chat` output, `can_shift=` / `prefix_snapshot=` |
| `hybrid-snapshot.log`    | #512 host test on this checkpoint (strict-prefix restore, pure extension, non-prefix drop)  |
| `t3-series-s.csv`        | Series S T3, granite40-h-tiny Q3_K_S, 3 recorded runs after warmup                          |
| `t3-llama32-3b.csv`      | Series S T3, dense peer `llama32-3b` Q3_K_S, same settings                                  |
| `h9-series-s.jsonl`      | Paired Series S H9 (`granite40-h-tiny` 7/8, `llama32-3b` 6/8)                               |
| `h9-20261125/`           | H9 operator logs, README, SHA256SUMS (25 Nov 2026)                                          |
| `manifest.override.json` | LocalState override to load Q3_K_S on Series S                                              |

Logs are redacted with `scripts/redact_recorded_path.py` (no hostnames, IPs,
usernames, or paths with usernames).

## T3 (console, Developer Mode)

Sideload a current unified `xllama-appx`. Upload the override as
`LocalState\manifest.json` (per-entry merge; bundled catalogue stays). After
the package is sideloaded and has been launched once, **select the model so
the app fetches it from the official Hugging Face URL.**

A ~3 GB Device Portal upload of the GGUF broke off twice after ~11 min; keep
that as a fallback (`LocalState\models\granite40-h-tiny\`) if the fetch
cannot run.

```bash
source ~/.config/xllama/xbox-env
# after the package is sideloaded and has been launched once:
PFN=$(./scripts/deploy.sh pfn)
./scripts/deploy.sh upload-file \
  bench/results/2026-11-24-granite40-h-tiny/manifest.override.json \
  "$PFN" "" manifest.json
# select granite40-h-tiny in the app so it fetches Q3_K_S from hf_base_url
# 3 recorded runs at product n_ctx; CPU; compare with llama32-3b at the same prompt
./scripts/bench-xbox-ort.sh granite40-h-tiny --hardware xbox-series-s --runs 4 --n-predict 96 \
  --out bench/results/2026-11-24-granite40-h-tiny/t3-series-s.csv
./scripts/bench-xbox-ort.sh llama32-3b --hardware xbox-series-s --runs 4 --n-predict 96 \
  --out bench/results/2026-11-24-granite40-h-tiny/t3-llama32-3b.csv
```

`--runs 4` drops run 1 as warmup and records three `run_index` rows. Gate:
every run `peak_ws_mb` ≤ 3584; decode strictly above the dense peer; TTFT not
several× worse. H9 same-quality bar: PASS (receipts in
[`h9-20261125/`](h9-20261125/)). The #273 decision is pending owner decision.
