# Per-layer D3D12 timing on the Series S, 2026-11-18

One headless run of the opt-in layer lines from #490, for #421. The question
was whether a console log splits GPU time by layer and leaves a missing
timestamp as `missing`. This is a diagnostic capture. Published tok/s stay in
`docs/benchmarks.md`, and this row is not in `bench/benchmark-summary.json`.

## Run

| Item        | Value                                                                                                |
| ----------- | ---------------------------------------------------------------------------------------------------- |
| Console     | Xbox Series S, Developer Mode. The harness label is `xbox-series-s-declared`.                        |
| Package     | `GianlucaMazza.xllama_1.7.0.1667_x64__pj67f1fcj4n14` (CI run 44178650331, #490, `49ca4a3`)           |
| Model       | `lfm25-350m`, file `LFM2.5-350M-QAD-Q4_0.gguf`, backend `d3d12`, 99 GPU layers                       |
| Prompt      | `bench/prompts/short-32.txt`, chat template applied, 60 prompt tokens, 16 generated (`--ignore-eog`) |
| Knob        | `LocalState\d3d12-dispatch-timing.txt` = `1`. The log records `dispatch_timing=1`.                   |
| Repetitions | One run. The harness kept it and printed that the warmup was not dropped.                            |

The catalogue GPU-default directory `models\lfm25-1.2b-instruct` on this
console had no files, so the capture used the 350M weights that were present.
Loader output in the excerpt assigns every block to D3D12 and reports
`offloaded 17/17 layers`.

The device row, stored as written:

```
lfm25-350m,Q4_0,d3d12,2048,6,771.50,117.23,367,326,224,3801,60,16,0,xbox-series-s-declared-g99-noeog,2026-11-18T10:30:03Z,1,77.8,78.6
```

The timing knob is on, so those rates include its barriers. They are the
context of the layer lines, not a replacement for a published 350M figure.

## What the lines show

`log-excerpt.txt` is the run, from the headless bench start through
`headless bench: done`. `layer-rows.csv` is
`scripts/parse_d3d12_layer_log.py --csv` on that excerpt. The bench CSV schema
is unchanged. Summary lines stay out of the sidecar. Each sidecar cell is
printed to three decimal places, so a sum of cells can differ from the log
summary by about 0.001 ms.

Both phases emitted layers 0–15 for `mul_mat`, plus `unattributed` rows for
`get_rows` and `mul_mat` (the embedding lookup and the tied output matrix).
36 data rows. Every `gpu_missing` and `wait_missing` cell is 0, and the log
summaries say `missing=0`:

```
[xllama] d3d12-layer phase=prefill summary rows=18 timed=94 missing=0 gpu_ms=50.150 wait_ms=4.056 wait_missing=0
[xllama] d3d12-layer phase=decode summary rows=18 timed=1504 missing=0 gpu_ms=35.087 wait_ms=55.518 wait_missing=0
```

The unattributed prefill cells match the per-kind lines in the same excerpt:
`get_rows` `gpu_ms=0.006` and the q6_K `mul_mat` `gpu_ms=0.445`.

On this decode the summed `wait_ms` (55.518) is above the summed `gpu_ms`
(35.087). That is the same shape an independent Series X report described for
the submission-to-fence remainder. One run is not a calibration, and this
capture has no paired CPU arm, so the working-set question from that report
is still open on the Series S.
