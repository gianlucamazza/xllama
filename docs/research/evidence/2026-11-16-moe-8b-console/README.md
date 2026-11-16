# LFM2.5-8B-A1B streamed on a project Series S (2026-11-16)

Second checkpoint on the streamed path, and the first one other than the
fork's 24B pin: the same `lfm2moe` architecture at a third of the size.
Companion of the [24B run](../2026-11-16-moe-24b-console/README.md).

## Identity

| Item    | Value                                                                                                                                                                                                                                  |
| ------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Console | Series S in Developer Mode, Game designation, operator-declared                                                                                                                                                                        |
| Package | 1.7.0.1651, CI run [43885657413](https://github.com/gianlucamazza/xllama/actions/runs/43885657413), tag `evidence/2026-11-16-moe-8b-console` (`3f8b5f2`, a test build, not for `main`)                                                 |
| Source  | the fork's seventeen patches on `main` `568e883`, with the pin pointed at this checkpoint: model, chat template, shape tuple (24 layers, 32 experts, 66 routed tensors, 3010560-byte largest matrix) and a 1600 MiB B512 cache ceiling |
| Model   | `LiquidAI/LFM2.5-8B-A1B-GGUF` @ `49c1483`, `LFM2.5-8B-A1B-Q4_K_M.gguf`, 5155564768 bytes, SHA-256 `4923ec14…` (checked before and after)                                                                                               |
| Profile | the fork's `cache-a2-v1` / `cache-b512-v1`: one CPU worker, nonfused, context 512, the same Russian greeting, greedy                                                                                                                   |

## Result

| Profile | Native stop | Samples | First token |       Rate | Expert requests |     Cache hits | Read (logical) | Max sampled usage |
| ------- | ----------- | ------: | ----------: | ---------: | --------------: | -------------: | -------------: | ----------------: |
| A2      | EOS/EOT     |     284 |      40.9 s | 1.18 tok/s |           87384 |              0 |      192.92 GB |           596 MiB |
| B512    | EOS/EOT     |     284 |      18.6 s | 2.47 tok/s |           87384 | 55697 (63.7 %) |       69.82 GB |          2063 MiB |

- Both runs reach the model's own EOS after a closed `<think>` block and a
  final answer; the 284 token ids are identical between A2 and B512.
- The output records still say `stop_reason: error` with "trace is not
  complete replayable": the fork's cache trace has a fixed size and key count
  sized for the 24B pin (`trace_status: partial`). That is a limit of the
  diagnostic trace, not of generation, and is recorded as a plan item (shape
  and capacities from the checkpoint, not compiled in).
- With 32 experts instead of 64, a 512-slot cache holds a larger share of the
  routed set: 63.7 % hits against 35.1 % on the 24B.

## Opening refusals before this run (same day)

The fork's pilot first refused this checkpoint twice, at opening and before
any allocation: on the compiled-in 24B shape tuple (40, 64, 114, 2580480) and
on the 24B-sized B512 cache ceiling (1280 MiB, below 512 × 3010560 bytes). Both
are fixed constants in the fork; the package above carries the 8B values.

## Resident comparison

On the clean `main` package 1.7.0.1652 ([rows](../../../../bench/results/2026-11-16-moe-8b/README.md)):

- This Q4_K_M file **cannot run resident**: the weights load (working set
  5004 MB) but the 272 MiB prefill compute buffer does not fit the 5120 MiB
  app limit, so no context is created (four recorded timeouts). On a Series S
  it runs only on the streamed path.
- The `UD-IQ3_S` file of H2 runs resident at 14.42 tok/s decode, 61.25 tok/s
  prefill, 3560 MiB, 4.8 s to the first token (six threads, fixed 512-token
  decode). The streamed Q4_K_M is about six times slower in decode on one
  worker, so the comparison is about what can run at all, not about speed.

## Files

`a2.output.json`, `b512.output.json` (pilot records, unmodified),
`a2.trace.bin`, `b512.trace.bin` (partial traces), `a2.pilot.log`,
`b512.pilot.log` (`[cpu-pilot]` lines of each run, without runtime quota
samples), `result.json`.
