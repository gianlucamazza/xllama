# Layer-line overhead and working set on the Series S, 2026-11-18

Same opt-in as the one-run layer capture
([2026-11-18-d3d12-layer](../2026-11-18-d3d12-layer/README.md)), for #421.
Two questions: what the shared timing opt-in costs on this package, beside
the 2026-11-16 per-dispatch calibration, and whether the host working set
drops when layers move to the GPU. Dispatch timing and the layer lines are
the same file, so this run does not separate them.
These runs are a diagnostic. Published tok/s stay in `docs/benchmarks.md`,
and these rows are not in `bench/benchmark-summary.json`. The recorded rows,
medians and ratios are in [`result.json`](result.json).

## Run

| Item     | Value                                                                                                                                               |
| -------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| Console  | Xbox Series S, Developer Mode. The harness label is `xbox-series-s-declared`.                                                                       |
| Package  | `GianlucaMazza.xllama_1.7.0.1667_x64__pj67f1fcj4n14` (CI run 44178650331, #490, `49ca4a3`). #493 is the host parser and does not change this build. |
| Prompt   | `bench/prompts/long-1k.txt`, chat template applied. 1000 prompt tokens on both LFM files, 948 on Coder-3B. 128 generated (`--ignore-eog`).          |
| Settings | `scripts/bench-xbox-ort.sh`, ctx 2048, 6 threads, `--runs 2` (run 1 dropped). GPU arms request 99 layers.                                           |
| Knob     | `LocalState\d3d12-dispatch-timing.txt` deleted, or uploaded as the single line `1`.                                                                 |
| Order    | Working-set pair first (knob deleted, `--gpu-layers 0` then `99`), then four alternating pairs per model, knob off then on.                         |
| When     | 2026-11-18T10:57:16Z through 2026-11-18T11:23:27Z.                                                                                                  |

`lfm25-1.2b-instruct` was uploaded to this console before its arms
(`LFM2.5-1.2B-Instruct-QAD-Q4_0.gguf`). The one-run capture earlier the same
day found that directory empty and used the 350M file.

The headless bench process does not print `[resources]`. Every `[resources]`
line in the same LocalState log (26 lines) is
`expanded=0 exclusive_cpus=6 logical_cpus=8 app_memory_limit_mb=5120`.
`docs/uwp-constraints.md` treats 5120 MB as the Game designation and 1024 MB
as App. `HasExpandedResources` returned 0 on those startups; the memory limit
is the check that document uses.

GPU passes report `offloaded 17/17 layers to GPU` (both LFM files) and
`offloaded 37/37 layers to GPU` (Coder-3B). The two CPU passes do not.

## What the knob costs

Median of the four recorded runs, knob on over knob off. Time to first token
is a duration, so a ratio above 1 is slower. Prefill and decode are rates.

| Model       | Prefill | Time to first token | Decode | Peak memory |
| ----------- | ------: | ------------------: | -----: | ----------: |
| LFM2.5-350M |   0.971 |               1.031 |  0.943 |       1.001 |
| LFM2.5-1.2B |   0.999 |               1.001 |  0.957 |       1.000 |
| Coder-3B    |   0.996 |               1.004 |  0.958 |       1.000 |

LFM2.5-1.2B and Coder-3B match the 2026-11-16 calibration of the per-dispatch
knob (prefill unchanged, decode 0.956 and 0.964). That match is for the
shared opt-in on this package. Peak memory on the GPU arms is unchanged
(the 350M ratio is 369 MB against a 368.5 MB median).

LFM2.5-350M was not in the 2026-11-16 pairs. Its decode ratio is tight across
the four runs. Its prefill median, 0.971, mixes two recorded on-runs at about
0.948 and 0.952 of the off median with two at about 0.990 and 0.993. The
barrier is a larger share of a faster model, and that prefill cell is wider
than the other two.

## Working set

One recorded run per arm, knob deleted. `gpu_mem_mb` is 0 on every CPU arm.

| Model       | CPU peak MB | GPU peak MB | Delta MB | GPU memory MB |
| ----------- | ----------: | ----------: | -------: | ------------: |
| LFM2.5-350M |         325 |         367 |      +42 |           224 |
| LFM2.5-1.2B |         811 |         861 |      +50 |           678 |
| Coder-3B    |        2083 |        2205 |     +122 |          1851 |

The host working set rises when layers move to the GPU. The host copy of the
offloaded weights stays resident.

The Series S comment on #421 from 2026-11-16
([comment](https://github.com/gianlucamazza/xllama/issues/421#issuecomment-6916816628),
package 1.7.0.1621) recorded GPU peaks of 860 MB for LFM2.5-1.2B (`gpu_mem` 678) and 2205 MB for Coder-3B (`gpu_mem` 1851). This run records 861/678 and
2205/1851. That comment's CPU peaks were 786 MB and 2045 MB, on a shorter
prompt that did not ignore end of generation. The Series X note linked from
the roadmap is a different comment.

## Layer lines

Each run in the log is sliced from its `bench model:` line to the next one.
The last 20 starts of each model are this campaign, warmup included. All
eight knob-on passes per model have `dispatch_timing=1` and `missing=0`:
36 data rows and 2 summaries for each LFM file (layers 0–15 plus
`unattributed`, both phases), 76 data rows and 2 summaries for Coder-3B
(layers 0–35 plus `unattributed`). The ten knob-off GPU passes per model have
`dispatch_timing=0` and no `d3d12-layer` lines. The two CPU passes have
neither a timing line nor a layer line.

#421 stays open. The Series X point remains an operator statement. The
four-column Q4_0 shader question belongs with any reopen of #313. A KV
sidecar is a later proposal.
