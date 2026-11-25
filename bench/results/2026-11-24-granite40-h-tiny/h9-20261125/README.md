# Granite-4.0-H-Tiny paired H9 (25 Nov 2026)

Paired H9 run on Xbox Series S, 25 Nov 2026, 01:30–01:31 Rome.

## Binding

- Head: `cc482024206b7a549ef46b9b113a7063ec9eef1b` (`main`)
- App package: `1.8.1.1764`
- One call, as the protocol requires:

```bash
./scripts/eval-xbox-models.sh --models granite40-h-tiny,llama32-3b
```

`--out` was `bench/results/2026-11-24-granite40-h-tiny/h9-series-s.jsonl`.
That name was free in the campaign folder, so the jsonl is stored there
(not `h9-series-s-20261125.jsonl`).

## Models

Sizes were checked on the console. Hashes are the manifest values (not
re-hashed on the console).

| Id                 | Quant  | Bytes (console) | SHA-256 (manifest)                                         |
| ------------------ | ------ | --------------: | ---------------------------------------------------------- |
| `granite40-h-tiny` | Q3_K_S |   3,062,452,832 | `5ec3f4af77fffead37b09984a97625f0f532f6ae2ead65b5019bce6dd8b4d0cd` |
| `llama32-3b`       | Q3_K_S |   1,542,848,960 | `81bd392ed38532d17f8be3942ec3237747ccc9d91365e6cc31e060be3f0edac8` |

## Results

From `../h9-series-s.jsonl`: `granite40-h-tiny` 7/8, `llama32-3b` 6/8.
Both fail `arithmetic_multistep`. Granite passes `multi_turn_memory`,
which `llama32-3b` fails. All other tasks pass for both.

| task_id              | granite40-h-tiny | llama32-3b |
| -------------------- | ---------------- | ---------- |
| it_exact_capital     | pass             | pass       |
| arithmetic_multistep | fail             | fail       |
| json_extraction      | pass             | pass       |
| grounded_qa          | pass             | pass       |
| constrained_summary  | pass             | pass       |
| translation_fidelity | pass             | pass       |
| multi_turn_memory    | pass             | fail       |
| unknown_from_context | pass             | pass       |

## Gate

The protocol says "compare totals" with no numeric threshold. The
stricter reading used here: Granite's total is at least `llama32-3b`'s,
and there is no task where `llama32-3b` passes and Granite fails. Both
hold, so H9 = PASS.

`llama32-3b` scored 6/8 here vs 5/8 in the published 2026-09-30 file
(`bench/results/2026-09-30-h9-rescore.jsonl`). Only `json_extraction`
differs. The paired run is the comparison.

Latency values in the logs are not a gate and include the network
route. This record makes no speed claim.

The console was restored afterwards (`step6-restore.log`).

## Files

| File                   | SHA-256                                                            |
| ---------------------- | ------------------------------------------------------------------ |
| `../h9-series-s.jsonl` | `a4adf4efcc37ce4ff00209bc536a2ff8cb9bcfe94cf6f53b200b7e6ebec4c9cc` |
| `step1-setup.log`      | `58dcdf255d30ae6946ce3d81d4706684ebc719bce44154116e7c4f26816fff9e` |
| `step5-h9.log`         | `293c67ea1c2c69da04776fdbd6b1d84362e3939cefdd1215331d06acc2f89872` |
| `step6-restore.log`    | `a821cc3ad6fcb7896d7fad73f01ed6af77ac2d573be00d304dbfe6a935684da7` |

From this directory: `sha256sum -c SHA256SUMS`.

Logs keep the `<user>`, `<console>` and `<pub>` redaction tokens.

Not a catalogue entry. The #273 decision is pending owner decision.
