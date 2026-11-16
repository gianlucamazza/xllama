# Per-dispatch GPU timing on the Series S, 2026-11-16

First use of the opt-in per-dispatch timing of #474
(`d3d12-dispatch-timing.txt`), for #421. Two questions: what the knob costs,
and why the four-column kernel of #414 slowed LFM2.5-1.2B down while it sped
Coder-3B up (#470). Diagnostic runs, not a benchmark campaign. Packages,
settings and the parsed tables are in [`result.json`](result.json); raw bench
rows are in
[`bench/results/2026-11-16-dispatch-timing/`](../../../../bench/results/2026-11-16-dispatch-timing/).

## What the knob costs

Four alternating pairs per model on the `long-1k` prompt, package 1.7.0.1617
(#474), knob off against knob on:

| Model       | Prefill | Time to first token | Decode | Peak memory |
| ----------- | ------: | ------------------: | -----: | ----------- |
| LFM2.5-1.2B |   1.000 |               1.000 |  0.956 | unchanged   |
| Coder-3B    |   1.001 |               0.999 |  0.964 | unchanged   |

Prefill is unaffected, so prefill attribution is faithful. Decode loses about
4 %: one barrier per dispatch, about 15 dispatches per token. With the knob
on, every dispatch was timed (`missing=0`) and the per-kind times add up to
the phase's GPU time.

## Why the four-column kernel helps one model and hurts the other

One run per arm on the diagnostic package 1.7.0.1618, built from the branch
`diag/c4-dispatch-timing` (the closed #414 candidate with #474 merged in; not
for merge), knob on in both arms, `d3d12-prefill-columns.txt` absent or `4`.
Prefill GPU time per dispatch, four-column over single-column:

| Model       | Weight type | Threads | Single-column ms | Four-column ms | Ratio |
| ----------- | ----------- | ------: | ---------------: | -------------: | ----: |
| LFM2.5-1.2B | Q4_0        |      64 |            16.66 |          20.75 |  1.25 |
| LFM2.5-1.2B | Q4_0        |     128 |            30.23 |          34.43 |  1.14 |
| Coder-3B    | Q4_K        |      64 |            15.10 |           8.36 |  0.55 |
| Coder-3B    | Q4_K        |     128 |            36.95 |          19.86 |  0.54 |
| Coder-3B    | Q6_K        |     128 |            64.99 |          31.53 |  0.49 |

(512-column dispatches; the 64–511 column class gives the same ratios.)

Prefill GPU time overall: LFM2.5-1.2B 3,294 ms single-column against 4,010 ms
four-column (1.22), Coder-3B 8,793 ms against 4,713 ms (0.54). The bench rows
of the same runs agree with the #470 campaign (LFM2.5-1.2B 250 against 212
tok/s, Coder-3B 74 against 109).

### Reading

- The slowdown is in the four-column **Q4_0** shader, at both thread widths.
  The K-quant shaders run in about half the time with four columns, which is
  close to what reading each weight block once instead of up to four times
  should give.
- Both single-column shaders move weights at a similar rate (about 5.3 MiB
  of packed weights per dispatch, over 512 columns, in about 16 ms for Q4_0
  and 15 ms for Q4_K at 64 threads). So the Q4_0 four-column shader does not
  lose because Q4_0 was cheaper to begin with; it loses inside the shader.
- Why is not measured. A plausible cause, untested: a Q4_0 block is 32
  values (18 bytes) against 256 for a K-quant super-block, so the four-column
  Q4_0 shader does more address work and holds more q8_0 activation data per
  weight byte, which could cost occupancy. Finding out needs the shader
  source compared, or a variant measured; neither is done here.
- One run per arm, one console. Enough to locate the effect (every Q4_0 row
  slower, every K-quant row faster), not to give interval estimates.
