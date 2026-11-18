# GGUF GPU decode — design, gates and verdicts

> **SSOT for the GGUF GPU decode design** (H6 follow-up, #228): architecture,
> cost model, predeclared gates (D1, D2, D2-r2, D3, FA) and their verdicts.
> Kernel evidence lives in [phase15-re-opt.md](phase15-re-opt.md) (WS-E /
> H6.3). The gate tables here are verdict evidence, read from the CSVs in
> `bench/results/` (listed in `bench/README.md`); shipped defaults stay in
> [benchmarks.md](benchmarks.md). Platform limits are only in
> [uwp-constraints.md](uwp-constraints.md).

**Status (2026-11-16):** the GPU path is verified correct on both gate
models under the amended ADR 0010: every weight product matches an exact
reference and the output is as close to the CPU's as another CPU build's
([LFM2.5-1.2B](research/evidence/2026-11-16-adr0010-control-lfm25-1.2b/README.md),
[Coder-3B](research/evidence/2026-11-15-adr0010-control-2/README.md)). The
four-column prefill kernel (#414) is correct (output byte-identical) but
failed its preregistered performance campaign: 1.5 times faster prefill on
Coder-3B, 15 % slower on LFM2.5-1.2B
([campaign](research/evidence/2026-11-16-c4-performance/README.md)). It is
not promoted: #414 is closed without merge and its branch is kept. The dated sections below record each step as it happened; a
"stay stopped" in an earlier section is superseded by the later ones.

**Status (2026-10-25):** the shipping catalogue GPU default is LFM2.5-1.2B.
Qwen2.5-Coder-1.5B (the coding option) runs on the CPU; it has no D2-r2
receipt. Qwen2.5-Coder-3B is not in the catalogue (benchmark; model under a
non-commercial research licence); its D3 numbers below stay as receipts.

**Status (2026-10-02): D3 shipped.** Qwen2.5-Coder-3B and LFM2.5-1.2B decoded
on the d3d12 backend by default (catalogue `gpu_layers`); other GGUF models
stay on the CPU. History: D1 = `D2-matmul-only`; D2a = PASS; D2 = FAIL (peak
RAM, Coder-3B H9). **D2-r2 = PASS** on both gate models (q8 activations,
one-copy tied embedding, smaller compute reserve, zero-size ops kept on the
GPU; CI `1.6.0.1156`): Coder-3B decodes 1.60× and LFM2.5-1.2B 1.64×. Next is
D3, the per-model default (done; see below). `gguf_gpu_layers.txt` remains
the operator override. **FA fix (CI `1.6.0.1163`):** the GPU path keeps Flash
Attention like the CPU path. Decode is now 1.90× (Coder-3B) and 1.88×
(LFM2.5-1.2B).

## Why

GGUF decode on Series S is bandwidth-bound on the CPU (~12–13 GB/s effective,
[phase15-re-opt.md](phase15-re-opt.md) baseline). H6.3 measured our own D3D12
Q4*K GEMV at **143.06 GB/s packed** (`bench/results/phase15-gpugemv-h63.csv`,
K3). The predeclared ladder sends K3 to a decode \_design*, not a backend: one
GEMV tile is not a decode step. This document fixes the architecture and the
gates before any backend code exists.

## Architecture

A **ggml backend `d3d12`**, owned by xllama (`src/bridge/`, not the
`llama.cpp/` submodule), statically linked and registered at runtime with
`ggml_backend_register()` before the model loads. The API is public
(`ggml-backend.h`); no dlopen ([uwp-constraints.md](uwp-constraints.md) §3).

- **Shape:** the `ggml-blas` template — `supports_op` claims `MUL_MAT` plus the
  view ops (`NONE/RESHAPE/VIEW/PERMUTE/TRANSPOSE`), and since D2-r2 `GET_ROWS`
  on a weight it holds. Everything else stays on the CPU backend.
- **Placement:** weights go to the backend's buffer type through
  `n_gpu_layers` (already `SessionParams::n_gpu_layers`, default 0). The
  scheduler runs an op where its weight lives; weights of a type the backend
  does not support (Q5_K, Q8_0, …) stay on the CPU through
  `weight_buft_supported`. `token_embd` / `GET_ROWS` are CPU, except a tied
  Q6_K embedding since D2-r2 (one copy in `D3D12_Weights`, `GET_ROWS` on the
  GPU).
- **Kernels:** the H6.3 `rows` shape (4 rows per group, 64 or 128 threads by K; since D2-r2 the
  activations are q8 and the dot is integer, `dot4add_i8packed`) per weight
  type: Q4_K and Q4_0 (same density, 0.5625 B/weight) and Q6_K (tied
  `lm_head` of every target model, `attn_v`/`ffn_down` on Q4_K_M
  `use_more_bits` layers).
- **Device:** the system D3D12 runtime via `d3d12_dyn`, never the Agility
  factory ([uwp-constraints.md](uwp-constraints.md) §7). Shared plumbing:
  `src/bridge/d3d12_compute.{h,cpp}` (probes and backend).
- **Invariants kept:** logits come back to the CPU, so the sampler chain
  (`sampler_chain.h`), `fit_prompt` and `SessionHub` (one resident model) do
  not change. The CPU path stays the default; the backend is opt-in.

Rejected: the ggml Vulkan backend (no Vulkan in the AppContainer); DirectML
(no fused low-bit GEMM, [uwp-constraints.md](uwp-constraints.md) §12); a
hand-written forward pass outside ggml (duplicates llama.cpp per-architecture
code — LFM2 short-conv, Qwen biases — and splits the sampler chain).

## Cost model

```
t_token = Σ_splits (t_rt + bytes_split / BW) + t_cpu_rest
```

- **BW** = 143.06 GB/s (H6.3 `rows`, `kGpustepRowsBwGbs`).
- **Split rule (assumption, D2 verifies with `ggml_backend_sched_get_n_splits`):**
  adjacent matmuls sharing an input run in one split — q/k/v, o, gate+up,
  down — so ~4 splits per layer plus the `lm_head`.
- **t_rt** — one CPU↔GPU round trip per split — and **t_cpu_rest** (norms,
  RoPE, attention, conv, sampling on the CPU) are unknown. D1 measures t_rt and
  the whole step; t_cpu_rest is measured in D2.

Model tables (`gpustep_model()`, `include/xllama/gpustep.h`) come from the
GGUF headers (LFM2.5 QAD Q4_0) and the Qwen2.5-3B config plus the llama.cpp
Q4_K_M promotion rule (Coder-3B):

| Model                 | Splits | Bytes / token | GEMV at BW | + t_rt 20 µs |  + 50 µs | + 100 µs |
| --------------------- | -----: | ------------: | ---------: | -----------: | -------: | -------: |
| `qwen25-coder-3b`     |    145 |       1923 MB |   13.44 ms |     16.34 ms | 20.69 ms | 27.94 ms |
| `lfm25-1.2b-instruct` |     65 |        693 MB |    4.84 ms |      6.14 ms |  8.09 ms | 11.34 ms |
| `lfm25-350m`          |     65 |        217 MB |    1.51 ms |      2.81 ms |  4.76 ms |  8.01 ms |

**Projection, not a claim** — matmul+sync time only, without t_cpu_rest.
`xllama-cli --gpustep` prints this table. The smaller the model, the larger the
share of fixed per-split cost: GPU decode is a 1B–3B lever first.

## Risks and the gate that settles each

| Risk                                                                                                                             | Settled by    |
| -------------------------------------------------------------------------------------------------------------------------------- | ------------- |
| Round trips dominate (~145 per Coder-3B token)                                                                                   | D1a, D1b      |
| A system D3D12 device fails inside the XAML process (chat and LAN API live there; every GPU probe so far was headless)           | D1d           |
| Copies are avoidable on unified memory (weights/activations in a CPU-visible heap)                                               | D1c (informs) |
| Prefill: weights in the GPU buffer also send prefill matmuls to the GPU — needs a batched kernel or CPU-side prefill             | D2 prefill    |
| Numerics: fp32 accumulation order differs from the CPU, greedy text can diverge                                                  | D2 H9         |
| ORT DirectML (Agility factory) in the same process after our device exists (`routing` gate)                                      | D2 gates      |
| GPU budget ([uwp-constraints.md](uwp-constraints.md) §7) and contention with diffusion/DML (audit Proposal C, compute admission) | D2 peak / D3  |

## Phases and predeclared gates

All console gates run on the CI MSVC package (`xllama-appx`), never a
crossbuild.

### D1 — probe (`gpustep`, measure-only)

`gpustep.flag` (headless) and `gpustep-inproc.flag` (inside the XAML process,
after `Window.Activate()`), `scripts/bench-gpustep.sh`, CSV
`bench/results/phase15-gpustep-d1.csv`. Sections: `caps` (UMA,
CacheCoherentUMA), `heap` (N=K=8192 GEMV with W in DEFAULT / UPLOAD / CUSTOM
WRITE_BACK-L0), `rt` (N=256 K=2048: host X → GPU → host Y; event-copy,
spin-copy, spin-zerocopy), `step` (one simulated token per model; `sync` = CPU
round trip per split, `nosync` = all splits in one submit). Q6_K matmuls are
simulated as bytes-equivalent Q4_K rows.

| Gate  | Criterion                                                                                                                       |
| ----- | ------------------------------------------------------------------------------------------------------------------------------- |
| D1-G1 | `rt` copy variants and the DEFAULT `heap` GEMV match the CPU reference (tolerance as gpugemv G1) in the headless process        |
| D1a   | best headless `rt` median ≤ **100 µs**                                                                                          |
| D1b   | headless `qwen25-coder-3b/sync` median ≤ **40 ms** (40 + ≤ 11 ms CPU work = 51 ms = 1.4× the CPU 14.0 tok/s, `phase14-console`) |
| D1c   | decision only: best CPU-visible heap ≥ 0.9× DEFAULT GEMV BW → host-visible buffers in D2, else DEFAULT + copies                 |
| D1d   | in-XAML run: D1-G1 holds and no device-removed / HRESULT error                                                                  |

Ladder (`gpustep_evaluate`, printed by `xllama-cli --gpustep-verdict`):

| Verdict          | Condition                                                    | Next                                           |
| ---------------- | ------------------------------------------------------------ | ---------------------------------------------- |
| `D2-matmul-only` | D1-G1 + D1a + D1b + D1d                                      | D2 as designed above                           |
| `D2-fused`       | D1d, but D1a or D1b fails while the `nosync` step is ≤ 40 ms | D2 claims more ops per split (fewer syncs)     |
| `park-3b`        | D1d, and even the `nosync` step misses 40 ms                 | park #228 with the number                      |
| `park-gui`       | D1d fails                                                    | park for GUI/API (headless-only is no product) |
| `NotAVerdict`    | a required row is missing or D1-G1 fails headless            | fix the probe, rerun                           |

### D1 result (2026-10-01)

CI MSVC package `1.6.0.1117` (run 36927534262), Series S, both processes,
`bench/results/phase15-gpustep-d1.csv`. Every row `ok=1`, `d3d12_ran=1`;
`caps`: UMA = 1, CacheCoherentUMA = 1.

| Measure                                                 | Headless            | In-XAML             |
| ------------------------------------------------------- | ------------------- | ------------------- |
| `rt` spin-zerocopy / spin-copy / event-copy (median µs) | 49.7 / 53.1 / 56.9  | 61.0 / 60.8 / 71.0  |
| `rt` p90, best variant (µs)                             | 53.8                | 155.3               |
| `heap` DEFAULT / UPLOAD / CUSTOM (GB/s)                 | 144.4 / 57.7 / 57.8 | 118.7 / 52.1 / 53.9 |
| `qwen25-coder-3b` sync / nosync (ms)                    | **21.16** / 14.74   | 21.50 / 14.90       |
| `lfm25-1.2b-instruct` sync / nosync (ms)                | 8.50 / 5.68         | 8.66 / 5.69         |
| `lfm25-350m` sync / nosync (ms)                         | 4.89 / 2.26         | 4.97 / 2.27         |

| Gate  | Result                                                                                        |
| ----- | --------------------------------------------------------------------------------------------- |
| D1-G1 | PASS (headless and in-XAML)                                                                   |
| D1a   | PASS — 49.7 µs ≤ 100                                                                          |
| D1b   | PASS — 21.16 ms ≤ 40                                                                          |
| D1c   | **DEFAULT + copy** — CPU-visible heaps read at 0.40× DEFAULT (57.8 vs 144.4 GB/s) despite UMA |
| D1d   | PASS — the system D3D12 device works next to the compositor                                   |

**Ladder: `D2-matmul-only`.** What the numbers say:

- The cost model holds: Coder-3B `sync` 21.16 ms against 20.69 ms projected at
  t_rt = 50 µs; `nosync` 14.74 ms against 13.44 ms of pure GEMV.
- Sync costs ~6.4 ms per Coder-3B token (145 splits × ~44 µs): real, but
  inside the gate. D2-fused stays an optimisation, not a prerequisite.
- Weights go in DEFAULT heaps. UMA is real (`cc_uma=1`) but CPU-visible pages
  halve GPU read bandwidth; only small per-split activations go through upload
  and readback buffers (zero-copy X/Y saves ~3 µs per round trip).
- In-XAML: medians match headless, the p90 round trip triples (155 µs) and
  DEFAULT-heap GEMV drops 18% — the compositor shares the GPU. D2 must be
  measured in the UI process, not only headless.
- Still no tok/s: CPU-side ops (t_cpu_rest) are not in any of these numbers.

The `gpugemv` rerun on the same package (regression check for the
`d3d12_compute` refactor) gave `rows` 138.11 GB/s (−3.5% vs 143.06, inside
±5%), `wave32` 24.99, `dot4` 139.50 — K3 unchanged.

### D2 — implementation (decided against the pinned llama.cpp)

`include/xllama/ggml_d3d12.h` + `src/bridge/ggml_d3d12.cpp`, a GPU-type ggml
device registered with `ggml_backend_register()`:

- **Two buffer types.** llama.cpp lists a GPU device's _default_ buft before
  its extra bufts, and the scheduler allocates activations in the default one.
  So the default is **`D3D12_Host`** (CUSTOM heap, WRITE_BACK/L0,
  `is_host`): CPU→GPU copies become `memcpy`, the CPU reads results in place,
  one round trip per split (the D1 zero-copy path). Weights go to the extra
  buft **`D3D12_Weights`** (DEFAULT heap, D1c).
- **`supports_op`** accepts `MUL_MAT` only when the weight already lives in
  `D3D12_Weights` (Q4_0 / Q4_K / Q6_K, K multiple of 256, contiguous, 2-D;
  F32 activations). llama.cpp's per-weight buft probe therefore skips the host
  buft; norms, biases, short-conv, `token_embd` (until D2-r2) and other types
  fall back to the CPU list by themselves.
- **Kernels** `shaders/ggml_d3d12_mmv_{q4_0,q4_k,q6_k}.hlsl`: the H6.3 `rows`
  layout plus a column index for prefill; Q4_0 (18 B) and Q6_K (210 B) blocks
  are read with 2-byte-aligned dword loads (Coder-3B Q6_K `ffn_down` rows are
  9030 B). Weight tensors get 16 B of padding: root descriptors have no bounds
  check. Two blobs per type, 64 or 128 threads per group,
  picked by K (`d3d12_mm_threads`: 128 from K = 4096): long-K `ffn_down`
  needs 8 chunks in flight, short K starves them (D2a runs 1 and 2).
- **`graph_compute`**: root constants + root SRV/UAVs per matmul, one submit,
  spin fence; the call returns with the work done.
- **Since D2-r2:**
  - each matmul input is quantized on the CPU by ggml's `from_float` (q8_0
    or q8_K) into a CPU-visible scratch; a matmul reading another matmul's
    output in the same batch waits for that batch;
  - `GET_ROWS` (Q6_K) runs for a tied `token_embd`;
  - `llama_gpu.h` caps `n_outputs_max` at what the context requests: 1, or
    1 + draft with prompt lookup; embedding contexts keep llama.cpp's
    default.
  - The selftest compares MUL_MAT with the CPU backend (rel ≤ 1e-5) and
    GET_ROWS bit for bit.

**D2a gate (console selftest `d3d12be.flag`, `scripts/bench-d3d12-selftest.sh`,
predeclared):** every type × shape (`{n, k}` from Coder-3B and LFM2.5 tensors,
ncols 1 / 7 / 512) within `rel_err ≤ 1e-2` of ggml's dequantizers, and every
decode case (ncols = 1) at ≥ 100 GB/s packed (GPU timestamps). Host tests
emulate each kernel's lane mapping against the same dequantizers.

### D2a result (2026-10-01/02, three console runs)

`scripts/bench-d3d12-selftest.sh`, GPU timestamps, rel_err against ggml's
dequantizers (all runs: 12/12 correct, rel_err ≤ 1.7e-7):

| Decode shape (n × k) | Run 1, 64 threads | Run 2, 128 threads | Run 3, width by K |
| -------------------- | ----------------: | -----------------: | ----------------: |
| q4_0 8192 × 2048     |             127.0 |              118.2 |         **116.9** |
| q4_0 2048 × 8192     |          **90.6** |              104.7 |         **104.0** |
| q4_k 11008 × 2048    |             144.9 |              132.5 |         **143.6** |
| q4_k 2048 × 11008    |             127.4 |              130.3 |         **129.6** |
| q6_k 2048 × 11008    |          **94.3** |              100.7 |         **102.3** |
| q6_k 65536 × 1024    |             110.7 |           **64.1** |         **114.4** |
| D2a                  |              FAIL |               FAIL |          **PASS** |

GB/s packed, CI packages `1.6.0.1123` / `1124` / `1125`; CSVs
`bench/results/d2a-d3d12-selftest{,-t128,-adaptive}.csv`. Run 1 starved
long-K matmuls (512 groups for N = 2048); run 2 starved short K (4 chunks for
8 slots); run 3 picks the width per K and passes every decode case. The
q4_0 8192 × 2048 row sits ~10 GB/s below run 1 at the same width — run-to-run
spread on a 9.4 MB matmul (identical blob), not a code difference. Prefill
columns run at ~3–5.5 µs per column at 1024 × 1024; the prefill gate is D2b's.

### D2 — product gate (written before D1, measured in D2b)

`qwen25-coder-3b` first, `lfm25-1.2b-instruct` second, both with
`n_gpu_layers` = all against the same package's CPU t6 run:

- decode ≥ **1.4×** CPU (median of 3 recorded runs, prefill and decode reported
  separately);
- prefill at P=1000 ≥ **0.9×** CPU;
- H9 score ≥ the CPU run on the same model;
- peak RAM ≤ CPU run; GPU memory ≤ budget;
- `validate-console.sh all` PASS with the backend enabled (includes `routing`,
  i.e. ORT DirectML after our device).

### D2b — integration (opt-in, default off)

- `src/bridge/llama_gpu.h` is the one place a GPU-layer request becomes llama
  params:
  - `devices` = `{D3D12}`, `n_gpu_layers`, `no_host`;
  - `offload_kqv = false`;
  - persistent CPU threadpools (`llama_attach_threadpool`): without them,
    ggml-cpu builds a pool per split.
  - Both paths use it: `run_inference_llama` (CLI, headless bench) and
    `LlamaSession` (GUI, LAN API).
- The device reports `D3D12_Host` as its host buft. llama.cpp then puts the
  CPU backend's compute buffer there, `supports_buft` accepts it, and splits
  share activations in place: one 305 MiB compute buffer instead of 322 + 81
  on Coder-3B, and no input copies.
- `SessionHub` reloads when the GPU-layer request changes (weights move between
  buffer types).
- **How to enable (experimental):**
  - `LocalState\gguf_gpu_layers.txt` with a layer count (`99` = all) for GUI and
    API sessions;
  - `bench_gpu_layers.txt` via `scripts/bench-xbox-ort.sh --gpu-layers N`
    (host tag `-gN`, CSV `backend` = `d3d12`);
  - `xllama-cli --gpu-layers N` (no D3D12 on Linux → CPU).
  - `--ignore-eog` (`bench_ignore_eog.txt`, tag `-noeog`) decodes exactly
    `n_predict` tokens so both arms time the same length.

### D2 result (2026-10-02, CI `1.6.0.1138`)

`bench/results/2026-10-02-d2-gguf-gpu-hostbuft.csv`: median of runs 2–4,
standard-512 prompt and 256 tokens for decode, long-1k (P ≈ 950–1000) for
prefill, CPU t6 against `--gpu-layers 99` on the same package.

| Criterion                 | Coder-3B (Q4_K_M)            | LFM2.5-1.2B (QAD Q4_0)         | LFM2.5-350M (informative) |
| ------------------------- | ---------------------------- | ------------------------------ | ------------------------- |
| decode ≥ 1.4×             | **PASS** 14.4 → 22.8 (1.59×) | **PASS** 39.6 → 62.5 (1.58×)   | 99.9 → 95.3 (0.95×)       |
| prefill P=1000 ≥ 0.9×     | **PASS** 45.6 → 60.1 (1.32×) | **PASS** 108.7 → 189.3 (1.74×) | 368 → 546 (1.48×)         |
| peak RAM ≤ CPU            | **FAIL** 2044 → 2583 MiB     | **FAIL** 783 → 1048 MiB        | 311 → 514 MiB             |
| GPU memory ≤ budget (§7)  | PASS 1851                    | PASS 678                       | 224                       |
| H9 ≥ CPU                  | **FAIL** 6/8 → 5/8           | **PASS** 6/8 → 6/8             | —                         |
| `validate-console.sh all` | PASS (knob = 99)             | PASS (knob = 99)               | —                         |

**D2 = FAIL** — peak RAM on both gate models, H9 on Coder-3B; every speed
criterion passes.

- **H9** (`bench/results/2026-10-02-d2-h9-{cpu,gpu}.jsonl`, LAN API, XAML
  process, temperature 0, two identical runs per arm): on Coder-3B,
  `constrained_summary` becomes one sentence instead of the three-item list.
  The d3d12 matmuls take f32 activations where ggml-cpu quantizes them to q8,
  and greedy text diverges (#312).
- **UI process:** the same H9 requests end to end (load and prefill
  included) run 7.34 → 11.17 tok/s on Coder-3B (1.52×) and 14.13 → 21.93 on
  LFM2.5-1.2B (1.55×).
- **Peak RAM:** on UMA the DEFAULT-heap weights count in the working set like
  CPU weights, so the excess is what the GPU path adds. Coder-3B load log:
  - 244 MiB: llama.cpp duplicates the tied `token_embd` into `D3D12_Weights`
    as the lm_head; it also stays on the CPU for `GET_ROWS`;
  - ~300 MB: the D3D12 runtime and the larger compute reservation.
- **Tied lm_head on the CPU** (`-ocpu` rows in
  `2026-10-02-d2-gguf-gpu-final.csv`, CI `1.6.0.1137`): Coder-3B peak 2620 →
  2110 MiB, still above the CPU's 2044, and LFM2.5-1.2B decode 1.37×, below the
  gate. Removed rather than kept as a per-model switch.

Follow-ups: peak RAM #309, CPU-side split cost #310, numerics #312,
multi-column prefill #313.

### D2-r2 — fixing the two failures (predeclared 2026-10-02, before any run)

The gate below is the D2 gate unchanged, re-measured after two fixes.

**Fix A, numerics (#312).** Activations are quantized on the CPU with ggml's
own `from_float` into the weight's `vec_dot_type`, exactly as the CPU
backend feeds its `vec_dot`: q8_0 for Q4_0, q8_K for Q4_K and Q6_K. The
kernels sum integers with `dot4add_i8packed` (cs_6_4), so GPU and CPU differ
only in float summation order. Gate A:

- console selftest (`scripts/bench-d3d12-selftest.sh`): 12/12 cases within
  rel 1e-5 of the CPU backend's q8 `vec_dot` (host emulation: ≤ 3.6e-7), and
  every decode shape ≥ 100 GB/s packed;
- H9 via the LAN API ≥ the CPU run on Coder-3B and LFM2.5-1.2B.

**Gate A result (2026-10-02, CI `1.6.0.1144`): PASS.**

- **Selftest** (`bench/results/2026-10-02-d2r2-gatea-selftest.csv`): 12/12
  cases within ≤ 2.9e-7 of the CPU backend. Decode shapes run at 109–164 GB/s
  packed; q4_k 11008 × 2048 went from 143.6 to 164.0, q6_k 65536 × 1024 from
  114.4 to 137.7.
- **H9** (`2026-10-02-d2r2-gatea-h9-{cpu,gpu}.jsonl`, LAN API, XAML process):
  Coder-3B 6/8 = 6/8 and LFM2.5-1.2B 6/8 = 6/8, failing the same tasks as the
  CPU. Text is not byte-identical (float summation order), quality is.
- **UI process end to end:** 1.52× (Coder-3B) and 1.49× (LFM2.5-1.2B).

**Fix B, peak RAM (#309).**

- One copy of a tied `token_embd`: `GET_ROWS` on the GPU, and the weight
  placed in `D3D12_Weights`.
- Working-set attribution of the remaining excess.

**Fix B result so far (CI `1.6.0.1148`, one run each, `[xllama] ws` log).**

- The tied embedding is one copy: the CPU model buffer drops from 244 MiB to
  0.9 MiB.
- `n_outputs_max = 1` shrinks the committed compute reserve on Coder-3B from
  305 to 88 MiB.
- Peak RAM: Coder-3B 2583 → 2203 MiB (CPU 2044), LFM2.5-1.2B 1048 → 946 MiB
  (CPU 783).
- What remains:
  - ~44 MB: the D3D12 runtime at init, a fixed cost of any GPU use in the
    process;
  - the GPU compute buffer, committed when it is created (88 MiB on Coder-3B,
    152 on 1.2B), where the CPU's malloc'd buffer only counts pages it
    touches.

**RAM criterion for D2-r2 and D3** (the user's decision, 2026-10-02, before
the D2-r2 run). Because of the fixed runtime cost, "peak ≤ CPU" cannot pass
on any GPU path. The criterion it guarded is a product one: GPU mode must
never push a model past the budget it fits on the CPU. Recorded as:

- peak RAM with GPU layers ≤ **3584 MiB**, the product peak gate every
  catalogue model meets;
- **Δ ≤ 200 MiB** over the same package's CPU run;
- GPU memory ≤ the GPU budget ([uwp-constraints.md](uwp-constraints.md) §7).

**D2-r2:**

- the D2 speed, H9 and `validate-console.sh all` criteria, unchanged;
- the RAM criterion above;
- same scripts on the merged head's package.

A model that fails a criterion keeps the CPU default.

### D2-r2 result (2026-10-02, CI `1.6.0.1156`)

Evidence:

- `bench/results/2026-10-02-d2r2-gguf-gpu-final.csv`: median of runs 2–4,
  same prompts and arms as D2;
- H9 from `2026-10-02-d2r2-final-h9-{cpu,gpu}.jsonl` (LAN API, XAML process);
- `validate-console.sh all` with the knob at 99, and again without it.

| Criterion                          | Coder-3B (Q4_K_M)              | LFM2.5-1.2B (QAD Q4_0)         |
| ---------------------------------- | ------------------------------ | ------------------------------ |
| decode ≥ 1.4×                      | **PASS** 14.14 → 22.69 (1.60×) | **PASS** 40.0 → 65.5 (1.64×)   |
| prefill P=1000 ≥ 0.9×              | **PASS** 45.7 → 58.2 (1.27×)   | **PASS** 108.9 → 198.4 (1.82×) |
| peak ≤ 3584 MiB, Δ ≤ 200 MiB (512) | **PASS** 2044 → 2203 (+159)    | **PASS** 783 → 946 (+163)      |
| same, P ≈ 1000                     | **PASS** 2078 → 2203 (+125)    | **PASS** 806 → 962 (+156)      |
| GPU memory ≤ budget (§7)           | PASS 1851                      | PASS 678                       |
| H9 ≥ CPU                           | **PASS** 6/8 = 6/8             | **PASS** 6/8 = 6/8             |
| `validate-console.sh all`          | PASS (knob = 99 and off)       | PASS (knob = 99 and off)       |

UI process, H9 requests end to end: 7.27 → 11.33 tok/s on Coder-3B (1.56×)
and 14.87 → 23.19 on LFM2.5-1.2B (1.56×). LFM2.5-350M (informative): decode
0.99×, peak +98 MiB; it is never a candidate.

**Verdict: D2-r2 PASS on Coder-3B and LFM2.5-1.2B.** Both are D3 candidates
for a GPU default.

**History of the run.**

- The first D2-r2 run (CI `1.6.0.1148`, `2026-10-02-d2r2-gguf-gpu.csv`)
  failed Coder-3B at P ≈ 1000 with +220 MiB.
- Cause: on a prompt longer than one ubatch, the ubatches without output rows
  give the last layer a zero-column FFN. The backend refused it, and the
  scheduler copied those weights, `token_embd` included, out of
  `D3D12_Weights` to the CPU.
- Fix: zero-size ops are accepted as no-ops. Coder-3B now measures +125 MiB,
  and `validate genroom` no longer times out with the knob on.

### D3 — product decision

Default on/off per model in `docs/model-matrix.md` after D2; the
measured-is-not-shipped ladder applies. Candidates after D2-r2: Coder-3B and
LFM2.5-1.2B. LFM2.5-350M and smaller stay on the CPU.

**Mechanism.**

- A catalogue entry carries `gpu_layers` (`uwp/models/manifest.json`, 0 or
  absent = CPU).
- `resolve_gguf_gpu_layers` (`routing_policy.h`) gives the GUI and the LAN
  API the entry value. The operator file `gguf_gpu_layers.txt` overrides it
  whenever it exists; `0` forces the CPU.

**D3 rule (predeclared 2026-10-02, before the D3 runs).**

- An entry gets `gpu_layers` only if the D2-r2 criteria pass at the entry's
  own `n_ctx`. Coder-3B was catalogued at 4096 (withdrawn 2026-10-25), while
  D2-r2 measured 2048.
- The package carrying the manifest change must pass `validate-console.sh all`
  in the default configuration (no knob).
- The same package must pass H9 via the LAN API, and the log must show the
  layers on D3D12 for the chosen models only.

**D3 result (2026-10-02, CI `1.6.0.1159`): shipped.**

- **Coder-3B at its catalogue `n_ctx` 4096**
  (`bench/results/2026-10-02-d3-coder-ctx4096.csv`): decode 1.60×, prefill
  1.28–1.35×, peak +150 / +126 MiB. PASS.
- **Default configuration, no knob:**
  - `validate-console.sh all` ALL PASS;
  - H9 via the LAN API (`2026-10-02-d3-default-h9.jsonl`): Coder-3B 6/8,
    LFM2.5-1.2B 6/8, LFM2.5-350M 4/8, the same as the CPU.
- **Placement**, on a clean log with one request per model: LFM2.5-350M
  loaded on the CPU, LFM2.5-1.2B and Coder-3B on D3D12 (660 / 1834 MiB in
  `D3D12_Weights`).
- **Override:** with `gguf_gpu_layers.txt` set to `0`, LFM2.5-1.2B loaded on
  the CPU.

### FA — Flash Attention on the GPU path (#310)

**Defect.** With GPU layers, llama.cpp's Flash Attention AUTO resolved to
disabled on every run (`resolve_fused_ops: Flash Attention not supported, set
to disabled` in the D2-r2 and D3 logs).

- AUTO compares each layer's device (D3D12) with the FA node's device. The FA
  node sits beside the KV cache, which stays on the CPU (`offload_kqv=false`),
  so the two never match. Upstream marks the check as wrong for no-KV-offload.
- The CPU-only path resolves AUTO to enabled. The GPU path therefore ran a
  different attention: slower, and numerically different from the CPU
  reference.
- Decode cost of a longer context (P = 948 vs 292, D2-r2 final CSV):

| Model       | CPU-only    | d3d12       |
| ----------- | ----------- | ----------- |
| Coder-3B    | +3.8 ms/tok | +9.7 ms/tok |
| LFM2.5-1.2B | +1.1 ms/tok | +3.7 ms/tok |

**Fix.** `apply_gguf_gpu_context` turns AUTO into ENABLED when layers go to
the GPU. Attention runs on the CPU either way; an explicit caller choice
stands.

**FA gate (predeclared 2026-10-02, before the run).** Default-configuration
CI package, same scripts as D2-r2 / D3. Per GPU model:

- decode standard-512 ≥ the D3 value − 2%;
- prefill ≥ 0.9× the CPU;
- peak RAM ≤ 3584 MiB and ≤ 200 MiB over the CPU;
- H9 via the LAN API ≥ the CPU;
- `validate-console.sh all` ALL PASS;
- no `Flash Attention not supported` line in the log.

Long-1k decode is reported, not gated. FAIL means revert, not a knob.

**#310 re-attribution (same run, predeclared).** From the backend counters,
take Coder-3B's CPU-side ms per decode token:

- **≥ 10 ms:** a separate D2-fused plan, with its own gate (elementwise ops on
  d3d12 with a GPU q8 quantize).
- **< 10 ms:** close #310 with the numbers.

**FA result (2026-10-02, CI `1.6.0.1163`): PASS.**
`bench/results/2026-10-02-fa-gguf-gpu.csv` and `2026-10-02-fa-default-h9.jsonl`.
The log has no `Flash Attention not supported` line.

| Criterion                    | Coder-3B (`n_ctx` 4096)              | LFM2.5-1.2B                         |
| ---------------------------- | ------------------------------------ | ----------------------------------- |
| decode standard-512 ≥ D3 −2% | **PASS** 22.63 → 26.64 (+18%)        | **PASS** 65.4 → 74.8 (+14%)         |
| decode vs CPU                | 14.04 → 26.64 (**1.90×**)            | 39.8 → 74.8 (**1.88×**)             |
| decode long-1k vs CPU        | 13.37 → 24.20 (1.81×)                | 38.2 → 68.4 (1.79×)                 |
| prefill ≥ 0.9× CPU           | **PASS** 1.65× (long-1k 1.63×)       | **PASS** 2.35× (long-1k 2.31×)      |
| peak RAM vs CPU              | **PASS** 2204 MiB, +88 (long-1k +50) | **PASS** 860 MiB, +77 (long-1k +54) |
| H9 (default configuration)   | **PASS** 6/8 = 6/8                   | **PASS** 6/8 = 6/8                  |
| `validate-console.sh all`    | **PASS**                             | **PASS**                            |

Without FA, unfused attention also set the prefill rate and the CPU compute
reserve. With FA, prefill gains 22–29% and peak RAM over the CPU halves.

**#310 re-attribution: < 10 ms, so #310 closes.**

- Backend counters (calls / wall / GPU) are logged once per run, covering
  prefill and decode together. A two-unknown linear fit uses the standard-512
  run (P = 292, 256 tokens) and the long-1k run (P = 948, 32 tokens).
- Coder-3B decode token: 37.5 ms. Of that, the backend accounts for 28.2 ms
  wall (13.7 ms of GPU time) and the CPU side for **≈ 9.3 ms**.
- The lever left is inside the backend: ~218 submissions per token cost
  ≈ 14.5 ms of wall over GPU time (sync, plus the CPU q8 quantize per
  submission). That is the D2-fused direction, a new decision with its own
  gate, not a #310 follow-up.
- Next to other hardware (effective bandwidth as a share of peak, llama.cpp
  Vulkan scoreboard): see
  [benchmarks.md](benchmarks.md#gpu-decode-in-context--known-hardware).

## Decision log

| Date       | Decision                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     |
| ---------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-10-01 | Design: ggml backend `d3d12`, matmul-only first, opt-in. D1 gates and the D2 product gate predeclared before any run.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        |
| 2026-10-01 | **D1 = `D2-matmul-only`** (CI `1.6.0.1117`): round trip 49.7 µs, simulated Coder-3B token 21.16 ms with sync, in-XAML PASS; weights in DEFAULT heaps (CPU-visible heaps 0.40×). D2 gates unchanged; measure D2 in the UI process too.                                                                                                                                                                                                                                                                                                                                                                                                        |
| 2026-10-01 | D2a implementation: two buffer types (`D3D12_Host` default, `D3D12_Weights` extra) so weights skip the host buft via `supports_op`; Q4_0 / Q4_K / Q6_K kernels; D2a selftest gate predeclared.                                                                                                                                                                                                                                                                                                                                                                                                                                               |
| 2026-10-02 | **D2a = PASS** (run 3, CI `1.6.0.1125`): 12/12 correct, every decode shape ≥ 102 GB/s with the width picked by K (64 threads below K = 4096, 128 from it). Runs 1–2 failed on speed and stay recorded.                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| 2026-10-02 | **D2 = FAIL** (CI `1.6.0.1138`): decode 1.59× / 1.58×, prefill 1.32× / 1.74×, `validate-console.sh all` PASS on Coder-3B / LFM2.5-1.2B; peak RAM +539 / +265 MiB over the CPU and Coder-3B H9 5/8 vs 6/8. The backend stays opt-in (`gguf_gpu_layers.txt`), default off. Tied lm_head on the CPU measured and dropped (1.2B decode 1.37×). Follow-ups #309 #310 #312 #313; 350M (0.95×) is never a candidate (#311).                                                                                                                                                                                                                         |
| 2026-10-02 | **RAM criterion for D2-r2/D3 changed by decision, before the run:** peak ≤ 3584 MiB, Δ ≤ 200 MiB over CPU, GPU ≤ budget. "Peak ≤ CPU" cannot pass any GPU path: the D3D12 runtime alone costs ~44 MB, and the GPU compute buffer is committed, not touched lazily. Gate A PASS (CI `1.6.0.1144`). Fix B so far: Coder-3B 2583 → 2203 MiB, 1.2B 1048 → 946 MiB.                                                                                                                                                                                                                                                                               |
| 2026-10-02 | **D2-r2** (CI `1.6.0.1148`): LFM2.5-1.2B **PASS** (decode 1.62×, prefill 1.81×, peak +163/+156 MiB, H9 6/8 = 6/8, validate PASS) → D3 candidate. Coder-3B **FAIL** on RAM only (+220 MiB at P ≈ 1000; decode 1.55×, H9 6/8 = 6/8) → CPU default; the lever left is lazy commit of the GPU compute buffer (#309).                                                                                                                                                                                                                                                                                                                             |
| 2026-10-02 | **Review fix before merge** (CI `1.6.0.1155`). `n_outputs_max = 1` made llama.cpp abort on any batch asking for more rows: prompt-lookup verify batches and embedding sessions with GPU layers. The cap is now 1 + draft with prompt lookup and llama.cpp's default for embeddings. On console, prompt lookup with GPU layers drafted 44 and accepted 31, and `/api/embed` with the knob passed 17 live requests. Prefill and decode keep 1, so the D2-r2 numbers stand (Coder-3B g99 rerun: 22.4 tok/s, 2203 MiB). The selftest now times a median of 5 runs, after a single sample swung 78–109 GB/s; it passes 14/14, GET_ROWS bit-exact. |
| 2026-10-02 | **D2-r2 = PASS** (CI `1.6.0.1156`): Coder-3B decode 1.60×, prefill 1.27×, peak +159/+125 MiB; LFM2.5-1.2B 1.64× / 1.82×, +163/+156 MiB. H9 6/8 = 6/8 on both, validate ALL PASS with the knob and without. Coder-3B's +220 MiB on `1.6.0.1148` was zero-column ops refused by `supports_op`, which made the scheduler copy weights to the CPU; fixed. Next: D3.                                                                                                                                                                                                                                                                              |
| 2026-10-02 | **D3 shipped** (CI `1.6.0.1159`): catalogue `gpu_layers` + `resolve_gguf_gpu_layers` (the operator file overrides; `0` forces the CPU). Qwen2.5-Coder-3B (re-measured at `n_ctx` 4096: decode 1.60×, peak +150/+126 MiB) and LFM2.5-1.2B run on d3d12 by default. Default-configuration validate ALL PASS, H9 unchanged, placement checked per model. Follow-ups #310 #313.                                                                                                                                                                                                                                                                  |
| 2026-10-02 | **FA = PASS** (CI `1.6.0.1163`): Flash Attention had been silently off on every GPU-layer run (AUTO compares layer and KV devices). Requesting it explicitly gives decode 1.90× / 1.88× (Coder-3B / LFM2.5-1.2B), prefill 1.65× / 2.35×, peak +88 / +77 MiB, H9 6/8 = 6/8, validate ALL PASS. #310 re-attribution: CPU side ≈ 9.3 ms/token < 10 → #310 closed; the remaining lever is per-submission backend overhead (≈ 14.5 ms/token).                                                                                                                                                                                                     |

## Preregistered #313 experiment — 2026-11-07

Status: **preregistered; the first attempt stopped on control logit parity**
(see the dated preflight below). This extends the existing Gate A
and product gates above; it does not revise their historical verdicts. Program:
[#395](https://github.com/gianlucamazza/xllama/issues/395).

Hypothesis: reusing each quantized weight block across several activation
columns improves long-prompt prefill without moving single-column decode or
violating the CPU-q8 numerical contract. Falsification includes a numerical
failure, loss of the declared performance margins, or a target-device resource
failure. The bjoern Xbox One X compatibility port is not this experiment.

Implementation stays opt-in and reversible in the existing backend. Keep the
current single-column GEMV dispatch; add an explicit multi-column selection and
fallback, testing Q4_0, Q4_K and Q6_K, both thread-group sizes, nonmultiple tile
counts, row/stride boundaries and invalid shapes. The accepted arithmetic
contract remains Gate A above. A changed arithmetic contract requires an ADR
before implementation, rather than relaxed tolerances after a failed trial.

Before collecting optimized results, freeze a campaign manifest containing:

- baseline and experiment source SHAs; submodule and model SHA-256s;
- HLSL/include and generated-blob SHA-256s, compiler version/binary hash, shader
  profile, entry point, all defines and flags; keep sources and generated
  artifacts separately identifiable and regenerate through existing scripts;
- CI MSVC package SHA-256/identity, device family plus observed device receipt,
  prompts, context, thread/ubatch/sampling settings and UI timing scope;
- warmup/thermal policy from `bench/README.md`, power instrumentation and all
  retry/failure rules. A sampled memory maximum is not a true peak.

Proposed decision thresholds, frozen here before any optimized measurements:
all ratios below are experiment/baseline (throughput for prefill/decode, latency
for TTFT). Freeze exact prompt files/hashes and actual tokenizer counts in the
campaign manifest; approximate lengths below are planning targets.
for each gate model (shipping LFM2.5-1.2B and historical Coder-3B, the latter
research-only and excluded from catalogue), require long-prompt prefill ratio
at least 1.10 at P approximately 512 and 1000. Require decode ratio at least
0.97 and TTFT ratio at most 1.05 at all tested prompt sizes, including the
short prompt. The full existing memory envelope and product gates still apply;
reject a true peak-working-set increase above 32 MiB against the matched
baseline. These are prospective acceptance margins, not measured gains.

Use 20 matched baseline/experiment pairs per model and prompt size after
warmup, alternating AB/BA by pair. Record every repeat individually and keep
failed/timed-out attempts and their dispositions. Stop on correctness or
resource failure; do not replace failed runs to reach a successful sample size.
Summarize the median paired ratios and a paired nonparametric percentile
bootstrap 95% interval (10,000 resamples, seed 20261107). Require the appropriate
interval bound, not merely the point estimate, to pass each margin; otherwise
report inconclusive or fail. Treat each model/prompt cell separately and require
all cells, avoiding a selectively pooled headline. This exploratory interval
does not establish broad model-quality equivalence.

Also rerun Gate A, logit parity and H9 on the exact package in the UI process.
H9 is a small regression screen, not general scientific validation. Keep host,
Series S and Series X datasets and conclusions separate. Publish negative
results and limitations through the existing raw evidence → selectors → claim
registry flow. Do not modify frozen receipts or promote code without actual
hardware evidence. No optimization or performance result is claimed by this
preregistration.

**Amendment, 2026-11-16, before the paired measurements**
([manifest](research/evidence/2026-11-16-c4-performance/README.md)): "logit
parity" above means the amended ADR 0010 control, which both models pass; no
power or thermal instrumentation is used and the memory gate applies to the
app's sampled peak; the prompt sizes are the existing bench prompts (about 300
and 1,000 device tokens, plus the short prompt); decode is a fixed 128 tokens.
Margins, pairs, ordering, bootstrap and the per-cell rule are unchanged.
`scripts/bench-paired-knob.sh` runs the pairs and
`scripts/paired-bench-summary.py` computes each cell's verdict.

### 2026-11-08 preflight — control logit failure

The preregistered candidate attempt is stopped before H9/performance acceptance.
The exact CI package passes the UI-process numerical selftest in both arms,
including enabled-arm tails. LFM's two GPU logit dumps are byte-identical and
both satisfy the frozen CPU comparator. Coder-3B's CPU versus original GPU
kernel comparison fails the maximum-error and NMSE criteria. This is a control
failure; the four-column Coder comparison was not run and its cause is unproven.
No limits or input selection were changed to obtain a pass. The implementation
remains draft/opt-in, with no default promotion or speedup claim. Raw dumps,
comparisons and package/device binding are in the
[dated execution ledger](audit/excellence/execution-2026-11-07.md#gpu-preflight--stopped-on-control-parity).

For a matched deterministic decode campaign, write `1` to
`bench_greedy.txt` before either arm. The native bench echoes `-greedy` in its
host tag; the campaign must require this echo. Removing the file preserves
normal sampler defaults.

The headless logit probe also accepts `logits_gpu_layers.txt` (default zero,
CPU control) and `logits_ubatch.txt` (default zero, engine default). Set the
former explicitly to zero or the campaign's layer count for each arm, restart,
and require the logged resolved layer count. A positive GPU request resolving
to CPU fails the probe. These diagnostic knobs do not change normal chat
routing. They reproduce the #420 control comparison on the shipping
single-column kernels; the four-column experiment stays on #414.

For the preregistered UI-process gates, use an autopilot script containing
`wait_idle`, then `gpu_selftest` or `dump_logits` with a raw `text` prompt.
These release the resident chat model under compute admission and retain the
XAML/compositor process. Require fresh result files, the UI-process start and
completion markers, and numerical acceptance; an autopilot completion alone is
not a numerical pass. `logits_model.txt` names an already provisioned operator
research model for the probe without catalogue promotion or a download. Remove
all diagnostic overrides after the campaign.

### 2026-11-15 control reproduction and tensor diagnosis (#420)

The Coder-3B control was run again on a Series S with CI package
1.7.0.1533, in the UI process, with the frozen prompt and `n_ubatch` 512.
Receipt, comparisons and host dumps:
[evidence directory](research/evidence/2026-11-15-coder-control-diagnosis/receipt.json).
The frozen verdict is unchanged: the control still fails.

**The failure is deterministic.** Both console dumps are byte-identical to the
2026-11-08 dumps from package 1.7.0.1482. The comparator gives the same
numbers: maximum error 0.550 against 0.5, NMSE 1.022e-3 against 1e-3, same top
token.

**Capture does not change the result.** With tensor capture on, each arm's
logits are byte-identical to the same arm with capture off.

**Where the arms part.** One record per captured node, CPU against GPU:

- In layer 0 the Q, K and V projections computed on the GPU match the CPU to
  NMSE about 1e-14 on the last token of each evaluation. With equal inputs
  those kernels match the CPU to rounding.
- The other layer-0 records on the last token stay at 1.5e-11 or better in the
  first evaluation (512 tokens). In the second evaluation (446 tokens) the
  layer-0 feed-forward output is already at 2.0e-8 and the layer output at
  9.7e-9.
- A capture of every column of layers 0 and 1 shows the first differences in a
  few prompt positions only (first evaluation). After the attention output
  projection of layer 0, 33 of 512 token positions differ (worst 3.3e-6). After
  the layer-0 feed-forward it is 84 positions (worst 1.1e-4). The other
  positions stay at rounding level.
- Layer 1 attention mixes positions, so almost every position then differs. The
  per-layer NMSE rises to about 1e-3 by layer 6 and stays near that level to
  the output. No single layer adds a jump.

**The same spread appears between CPU builds.** The model file was copied from
the console (same SHA-256) and the prompt was run on a Linux host with the
repack CPU kernels and with a build that has them disabled
(`GGML_CPU_REPACK=OFF`), both AVX2. Frozen comparator, all pairs:

| Reference   | Candidate   | Max error |     NMSE | Verdict |
| ----------- | ----------- | --------: | -------: | ------- |
| console CPU | console GPU |     0.550 | 1.022e-3 | FAIL    |
| console CPU | host repack |     0.507 |  7.42e-4 | FAIL    |
| console CPU | host plain  |     0.477 |  7.30e-4 | PASS    |
| host repack | host plain  |     0.534 |  8.50e-4 | FAIL    |
| console GPU | host plain  |     0.521 |  7.43e-4 | FAIL    |
| console GPU | host repack |     0.642 | 1.121e-3 | FAIL    |

Two of the three CPU-only pairs fail the maximum-error limit, and every pair
follows the same per-layer curve. For this model and prompt the 0.5 limit does
not separate a GPU arm from two CPU builds of the same arithmetic. The GPU
pairs are still the widest on NMSE (two of three above 1e-3; no CPU-only pair
is), so this does not clear the GPU arm.

What this establishes and what it does not:

- No sign of an indexing, lifetime or synchronisation defect: the run is
  deterministic, capture-invariant and exact at layer 0 for equal inputs.
- The first differences are isolated to single int8 values (next paragraph).
- The control verdict stays FAIL. Thresholds, prompt and inputs are unchanged.
  A different criterion for this model would need its own ADR before any new
  campaign, as the diagnostic protocol in
  [gpu-acceleration-baseline.md](research/gpu-acceleration-baseline.md#c4-control-diagnosis)
  already requires. #313 and #414 stay stopped.

**Root cause of the first differences: rounding flips in the activation
quantizer.** A quantized matmul first rounds its input to int8 (q8_K: 256-value
blocks, one scale per block). The input of the layer-0 attention output
projection is not bit-identical between the arms; it differs at NMSE about
5e-11, because the Q, K and V results feeding attention already differ in their
last bits. Quantizing each arm's input with ggml's rule and comparing the int8
values
([analysis](research/evidence/2026-11-15-coder-control-diagnosis/quantization-flip-analysis.json))
gives:

| Evaluation | Tokens | Tokens with a changed int8 | Changed int8 values | Tokens that diverge after the projection |
| ---------- | -----: | -------------------------: | ------------------: | ---------------------------------------: |
| 0          |    512 |                         33 |                  36 |                                       33 |
| 1          |    446 |                         45 |                  63 |                                       45 |

In both evaluations the two sets of tokens are identical, and every change is
by one unit. The block scale itself changes in almost every token and that
alone is harmless (median NMSE about 5e-13 where no int8 changes). So 36 int8
values out of about a million, each off by one, produce every diverging token
of the first evaluation. A second candidate, fused against two-step rounding
inside the quantizer, was tested on the same data and ruled out.

This is a property of quantized-activation matmul, not of the GPU path: any
two executions that differ in floating-point summation order can land on
opposite sides of a rounding boundary. It is the same mechanism for the two
CPU builds above. It explains where the differences start; it does not by
itself say how large the end-to-end gap may be, which is what the gate
measures.

The GPU arm of that run also logged the first console `d3d12-phase` lines
(#421). For the 958-token prefill: 433 submissions, 504 dispatches, 11427 ms
wall, 2421 ms CPU q8 quantization, 8 ms command recording, 8990 ms execute
through fence and 8933 ms GPU, with all 433 timestamp pairs valid. This is one
diagnostic run of the logit probe, not a benchmark row.

### 2026-11-15 control under ADR 0010: FAIL

The first control judged with `scripts/control-parity.py`, on a declaration
committed before its GPU arm
([evidence](research/evidence/2026-11-15-adr0010-control/README.md)): Coder-3B
Q4_K_M, package 1.7.0.1563, four new prompts of 40 to 713 tokens.

- **Part 2 fails.** The GPU arm's largest maximum error is 0.9865 against a
  CPU reference spread of 0.7201 (x1.370, limit x1.22), and one prompt's top
  token differs. Its largest NMSE is inside the spread (x0.978).
- **Part 1 finds no kernel error.** All 42 first-layer product records agree
  with the CPU to 2.9e-12 NMSE or better on the columns whose operand is the
  same (limit 1e-9). One record on the 40-token prompt has 0.725 of its
  columns comparable against a required 0.75, so part 1 fails as declared.
- The CPU-only pairs measured for the declaration all exceed the frozen
  gate's 0.5 and 1e-3 on these prompts.

#313 and #414 stay stopped. The result does not locate the cause of the wider
maximum error: part 1 covers the first layer only.

### 2026-11-15 kernel reference check: the GPU products are correct

A diagnostic after the failed control, not a control
([evidence](research/evidence/2026-11-15-gpu-kernel-reference-check/README.md)).
`scripts/check-matmul-reference.py` compares each captured weight `MUL_MAT`
with a float64 product of the dequantized weight and the arm's own q8_K
operand, so it needs no second arm and covers every token column.

- All 252 weight products of the 36 layers (40-token prompt) and the 84 of
  layers 0 to 5 (713-token prompt) match the reference on both arms: CPU arm
  2.3e-12 NMSE at worst, GPU arm 4.1e-14. The GPU arm is the closer of the two
  in every weight family.
- The backend runs `MUL_MAT` and `GET_ROWS` only, so with these products
  correct the GPU arm has no other operator of its own. The difference between
  the arms is rounding-level differences between two correct executions,
  amplified through the activation quantizer.
- The two-arm comparison of ADR 0010 part 1 loses its columns quickly: from
  layer 8 on, no token column has the same operand in both arms.
- Over thirteen prompts the GPU arm is on average 8 % further from the console
  CPU arm than a host CPU build is, wider on 9 of 13, which this sample cannot
  tell from chance. Every pair, CPU against CPU included, fails the frozen
  gate on all thirteen.

The control verdict is unchanged. What the result changes is the question:
the criterion in ADR 0010 part 2 is failing an arm whose kernels are verified
correct, on a statistic (the largest single-logit error over a few prompts)
that two CPU builds also move by tens of percent. Whether to judge part 1 by
the reference check and part 2 by a distribution over more prompts is a
decision for the owner; it is not taken here.

### 2026-11-15 control under the amended ADR 0010: PASS

ADR 0010 was amended after the diagnosis above, and a fresh control was
declared and run the same day
([evidence](research/evidence/2026-11-15-adr0010-control-2/README.md)):
Coder-3B Q4_K_M, package 1.7.0.1587, seventeen new prompts.

- **Part 1 passes.** All 252 weight products of the 36 layers (40-token
  prompt) and the 84 of layers 0 to 5 (675-token prompt) are within 4.3e-14
  NMSE of the exact reference on the GPU arm, against a declared 1e-7.
- **Part 2 passes.** Over 5,024 prompt positions the GPU arm's mean KL
  divergence from the console CPU arm is 0.989 times a host CPU build's
  (95 % upper bound 1.009, declared limit 1.25), and its top-token agreement
  is 0.24 points lower (upper bound 0.57, declared margin 1).

The correctness prerequisite of #313 is met for this model. Nothing here is
a speed result, and LFM2.5-1.2B was not put through the control.

### 2026-11-16 control on LFM2.5-1.2B: PASS

The same two-part control on `lfm25-1.2b-instruct` (QAD Q4_0), the model that
runs on the GPU by default, with the shipping single-column kernels
([evidence](research/evidence/2026-11-16-adr0010-control-lfm25-1.2b/README.md)).

- **Part 1 passes.** All 160 captured Q4_0 products (16 layers of a 41-token
  prompt, layers 0 to 5 of a 538-token prompt) are within 1.7e-14 NMSE of the
  exact reference on the GPU arm.
- **Part 2 passes against the declared floor.** Over 5,284 positions the GPU
  arm's mean KL divergence from the console CPU arm is 1.035 times that of a
  CPU build on llama.cpp's plain kernels (upper bound 1.068, limit 1.95), and
  its top-token agreement is the same within 0.17 points.
- On this model the console and host builds of the repacked kernels are a
  closer pair than either is to the plain kernels. Against that closer pair
  the GPU arm is 1.32 times as far and loses 0.68 points of top-token
  agreement, as the plain-kernel CPU build does. The floor was declared
  before the GPU arm was measured.

This is the baseline for the four-column kernel on this model.

### 2026-11-16 control of the four-column kernel: PASS, identical output

The four-column prefill kernel of #414 went through the amended control on
Coder-3B and LFM2.5-1.2B
([evidence](research/evidence/2026-11-16-adr0010-control-c4/README.md)).
Both parts pass on both models, and the four-column arm's per-position logits
are byte-identical to the single-column arm's on all 16 prompts of both
models. Part 1 covers partial tiles of 1, 2 and 3 columns. The correctness
prerequisite of the #313 performance trial is met; no speed result exists.

### 2026-11-16 paired campaign of the four-column kernel: FAIL

The preregistered #313 campaign ran on the Series S with the frozen manifest
([evidence](research/evidence/2026-11-16-c4-performance/README.md)): 6 cells,
20 alternating pairs each, 240 arms, none failed.

- **Coder-3B passes every cell:** prefill 1.50 and 1.47 times, time to first
  token 0.67 to 0.68 times, decode and memory unchanged.
- **LFM2.5-1.2B fails every cell:** prefill 0.84 and 0.85 times, time to
  first token 1.18 times, decode and memory unchanged.

The preregistered rule requires every cell, so the experiment fails. The
model that runs on the GPU by default gets slower, and the kernel stays
opt-in and off. The split follows the weight type (Q4_0 on LFM2.5-1.2B,
Q4_K and Q6_K on Coder-3B); that reading was not tested, and a K-quant-only
variant would need its own preregistration.

### Attributable request counters (#421)

A GGUF run with GPU layers logs one `[xllama] d3d12-phase` line after the
prompt prefill and one after decode. Each line counts backend activity
between the two points only (`ggml_d3d12_take_stats` in
`include/xllama/ggml_d3d12.h`):

- `calls`: graph computes.
- `submissions`: command lists executed and waited for.
- `dispatches`: MUL_MAT and GET_ROWS.
- `wall_ms`: whole graph computes.
- `quant_ms`: CPU q8 activation quantization.
- `record_ms`: command-list recording.
- `exec_wait_ms`: execution through fence completion.
- `gpu_ms`: summed GPU timestamp pairs, one per submission.

`gpu_ms` is printed as `unavailable` unless every submission had an increasing
timestamp pair (`ts_valid`/`ts_missing`). These are CPU-side intervals around
each step plus GPU timestamps. Wall minus GPU time is not CPU compute time,
and none of these fields is a throughput result until it comes from a bound
console package. A CPU run prints no line.

**Per-dispatch timing (opt-in).** With `LocalState\d3d12-dispatch-timing.txt`
containing `1` when the backend starts (the boot log says
`d3d12: dispatch_timing=1`), every dispatch is followed by a UAV barrier and a
GPU timestamp. After each phase line the log then carries one line per kind
of dispatch, sorted by GPU time, and a summary:

```
[xllama] d3d12-dispatch phase=prefill op=mul_mat type=q4_0 cpg=1 threads=128 cols=512+ count=48 gpu_ms=… ms_per=… weight_mib=… columns=…
[xllama] d3d12-dispatch phase=prefill summary kinds=6 timed=94 missing=0 gpu_ms=…
```

A kind is the operation, the weight type, the kernel's columns per group
(`cpg`), its thread width and a column class (`1`, `2-63`, `64-511`,
`512+`). `missing` counts dispatches whose timestamp pair was not increasing;
their time is not in any row. The barriers serialise dispatches that could
overlap without them, so the per-dispatch times add up to more than an
unmodified run's GPU time: compare a run with the knob against one without
before reading absolute figures. The knob changes no result.

The same knob then emits one line per phase × layer × op, and a summary.
The layer index comes from the weight tensor name (`blk.N.…`) or, if that
has none, the graph node (`Qcur-0`, `l_out-15`). Names with no layer
(`token_embd.weight`, `output.weight`, …) go in `layer=unattributed`.
`gpu_ms` is the sum of per-dispatch GPU timestamps. `wait_ms` is
`exec_wait_ms − submission GPU time, divided by n`: the CPU-side
ExecuteCommandLists-to-fence interval for that submission, minus that
submission's GPU timestamp pair, split equally across the `n` dispatches
in the list. It is not `exec_wait_ms` minus the row's summed per-dispatch
`gpu_ms`. The remainder can come out slightly negative because the CPU
and GPU clocks are different domains. Either field prints as `missing`
when that bucket has no valid timestamps; it is never filled in from wall
time. `scripts/parse_d3d12_layer_log.py` reads the lines, and `--csv` writes
the data rows to a sidecar (`phase,layer,op,count,gpu_ms,wait_ms,…`) with
the word `missing` left as `missing`. Summary lines stay out of that file.
The bench CSV schema is unchanged. Off, the backend path is unchanged.
A Series S capture of the lines is in
[research/evidence/2026-11-18-d3d12-layer/](research/evidence/2026-11-18-d3d12-layer/README.md)
(package 1.7.0.1667, `lfm25-350m`, knob on, one run).

Format example, not a measurement:

```
[xllama] d3d12-layer phase=prefill layer=N op=mul_mat count=… gpu_ms=… wait_ms=… gpu_missing=… wait_missing=…
[xllama] d3d12-layer phase=prefill layer=unattributed op=get_rows count=… gpu_ms=… wait_ms=missing gpu_missing=… wait_missing=…
[xllama] d3d12-layer phase=prefill summary rows=… timed=… missing=… gpu_ms=… wait_ms=… wait_missing=…
```

Measured on the Series S (2026-11-16,
[evidence](research/evidence/2026-11-16-d3d12-dispatch-timing/README.md)):
the knob leaves prefill throughput and time to first token unchanged and costs
about 4 % of decode. Its first use located the four-column result of #470 in
the shaders: per dispatch, the four-column Q4_0 shader takes 1.25 times
(64 threads) and 1.14 times (128 threads) as long as the single-column one,
while the four-column Q4_K and Q6_K shaders take about half as long.
