# ADR 0010 control of the four-column kernel, 2026-11-16

The four-column prefill kernel of #414 (`d3d12-prefill-columns.txt` = 4,
which reuses each packed weight across four activation columns), put through
the [amended ADR 0010](../../../adr/0010-logit-parity-reference-spread.md#amendment-2026-11-15)
control on both models named by the #313 protocol: Coder-3B Q4_K_M and
LFM2.5-1.2B QAD Q4_0. This directory is committed in two steps: the
declarations, prompts and CPU-only figures first, the GPU measurements after.

## Declarations

[`declaration-coder.json`](declaration-coder.json) and
[`declaration-lfm.json`](declaration-lfm.json) were committed before the
four-column arm was measured on these prompts. Each names its arms: the GPU
arm is the console with 99 GPU layers and the four-column kernel; the
judged floor is one host CPU build; the single-column GPU arm and the other
host build are reported, not judged.

| Value                  | Coder-3B               | LFM2.5-1.2B            | Basis                                            |
| ---------------------- | ---------------------- | ---------------------- | ------------------------------------------------ |
| `reference.nmse_limit` | 1e-7                   | 1e-7                   | As in the single-column controls.                |
| `reference.captures`   | `c1` to 35, `u03` to 5 | `c1` to 15, `u03` to 5 | See below.                                       |
| Floor (host arm)       | host Release           | host no-repack         | The host build further from the console CPU arm. |
| `kld_ratio_limit`      | 1.25                   | 2.0                    | 5 % of 0.031 over the floor, rounded down.       |
| `top1_margin`          | 0.01                   | 0.01                   | As before.                                       |
| Confidence, resamples  | 0.95, 2,000            | 0.95, 2,000            | As before.                                       |
| `min_positions`        | 4000                   | 4000                   | 4,835 and 5,073 positions at a stride of 2.      |

### Prompts

[`prompts/`](prompts/) holds seventeen prompts written or cut for this
control. None has been run on any GPU arm. `u01` to `u16` are passages of
documents not used for earlier prompts (or, for `docs/architecture.md` and
`docs/benchmarks.md`, passages further into the file than the earlier ones),
with an instruction in front: 464 to 793 tokens on Coder-3B, 490 to 824 on
LFM2.5-1.2B.

The two capture prompts are chosen so the kernel's partial tiles run. A tile
covers four activation columns; a ubatch whose column count is not a
multiple of four ends in a partial tile.

| Capture prompt | Coder-3B tokens | LFM2.5-1.2B tokens | Last tile                      |
| -------------- | --------------: | -----------------: | ------------------------------ |
| `c1`           |              29 |                 31 | 1 and 3 columns                |
| `u03`          |             567 |                618 | 3 and 2 columns, second ubatch |

### The floor

Three CPU executions were measured on these prompts before the declarations
([`cpu-reference-pairs-coder.json`](cpu-reference-pairs-coder.json),
[`cpu-reference-pairs-lfm.json`](cpu-reference-pairs-lfm.json)):

| Pair                           | Coder-3B KLD | top 1 | LFM2.5-1.2B KLD | top 1 |
| ------------------------------ | -----------: | ----: | --------------: | ----: |
| Console CPU vs host Release    |      5.74e-3 | 0.948 |         1.14e-3 | 0.983 |
| Console CPU vs host no-repack  |      5.73e-3 | 0.947 |         1.50e-3 | 0.979 |
| Host Release vs host no-repack |      5.72e-3 | 0.953 |         1.52e-3 | 0.978 |

The single-column LFM control (#465) chose the no-repack build as its floor
because, on its prompts, that build failed the test against the Release
floor. On these prompts it passes (upper bound 0.65 points against the
1-point margin), so the same reasoning would pick the other floor here. To
avoid choosing a floor per prompt set, this control uses one rule for both
models: the floor is whichever host build is further from the console CPU
arm on mean KL divergence. It gives the same floors as the single-column
controls for LFM2.5-1.2B; for Coder-3B the two host builds are within 0.3 %,
and the single-column control used Release.

### What was known about the four-column kernel

- The synthetic selftest of #414 passes in the UI process, tails included
  (2026-11-08).
- On 2026-11-08 the LFM2.5-1.2B last-token logits of the single-column and
  four-column arms were byte-identical on one prompt.
- The four-column kernel had never been run on Coder-3B.
- On 2026-11-16, before the declarations, one smoke run on LFM2.5-1.2B
  confirmed that the knob selects the kernel (93 four-column dispatches in
  the prefill of a 25-token prompt). No logits of that run were compared.

### CPU arm through part 1

The console CPU arm's captures pass the reference check: Coder-3B `c1` 252
records, worst NMSE 1.7e-12, `u03` 84 records, 6.7e-13; LFM2.5-1.2B `c1` 92
records, 1.0e-13, `u03` 68 records, 7.8e-14.
