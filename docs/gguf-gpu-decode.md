# GGUF GPU decode — design, gates and verdicts

> **SSOT for the GGUF GPU decode design** (H6 follow-up, #228): architecture,
> cost model, predeclared gates D1/D2 and their verdicts. Kernel evidence lives
> in [phase15-re-opt.md](phase15-re-opt.md) (WS-E / H6.3). The gate tables
> here are the verdict evidence of an opt-in path, read from the CSVs in
> `bench/results/` (listed in `bench/README.md`); shipped defaults stay in
> [benchmarks.md](benchmarks.md). Platform limits are only in
> [uwp-constraints.md](uwp-constraints.md).

**Status (2026-10-02): D3 shipped.** Qwen2.5-Coder-3B and LFM2.5-1.2B decode
on the d3d12 backend by default (catalogue `gpu_layers`); other GGUF models
stay on the CPU. History: D1 = `D2-matmul-only`; D2a = PASS; D2 = FAIL (peak
RAM, Coder-3B H9). **D2-r2 = PASS** on both gate models (q8 activations,
one-copy tied embedding, smaller compute reserve, zero-size ops kept on the
GPU; CI `1.6.0.1156`): Coder-3B decodes 1.60× and LFM2.5-1.2B 1.64×. Next is
D3, the per-model default (done; see below). `gguf_gpu_layers.txt` remains
the operator override.

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
| peak RAM ≤ CPU            | **FAIL** 2044 → 2583 MB      | **FAIL** 783 → 1048 MB         | 311 → 514 MB              |
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
  2110 MB, still above the CPU's 2044, and LFM2.5-1.2B decode 1.37×, below the
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
- Peak RAM: Coder-3B 2583 → 2203 MB (CPU 2044), LFM2.5-1.2B 1048 → 946 MB
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

- peak RAM with GPU layers ≤ **3584 MB**, the product peak gate every
  catalogue model meets;
- **Δ ≤ 200 MB** over the same package's CPU run;
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

| Criterion                        | Coder-3B (Q4_K_M)              | LFM2.5-1.2B (QAD Q4_0)         |
| -------------------------------- | ------------------------------ | ------------------------------ |
| decode ≥ 1.4×                    | **PASS** 14.14 → 22.69 (1.60×) | **PASS** 40.0 → 65.5 (1.64×)   |
| prefill P=1000 ≥ 0.9×            | **PASS** 45.7 → 58.2 (1.27×)   | **PASS** 108.9 → 198.4 (1.82×) |
| peak ≤ 3584 MB, Δ ≤ 200 MB (512) | **PASS** 2044 → 2203 (+159)    | **PASS** 783 → 946 (+163)      |
| same, P ≈ 1000                   | **PASS** 2078 → 2203 (+125)    | **PASS** 806 → 962 (+156)      |
| GPU memory ≤ budget (§7)         | PASS 1851                      | PASS 678                       |
| H9 ≥ CPU                         | **PASS** 6/8 = 6/8             | **PASS** 6/8 = 6/8             |
| `validate-console.sh all`        | PASS (knob = 99 and off)       | PASS (knob = 99 and off)       |

UI process, H9 requests end to end: 7.27 → 11.33 tok/s on Coder-3B (1.56×)
and 14.87 → 23.19 on LFM2.5-1.2B (1.56×). LFM2.5-350M (informative): decode
0.99×, peak +98 MB; it is never a candidate.

**Verdict: D2-r2 PASS on Coder-3B and LFM2.5-1.2B.** Both are D3 candidates
for a GPU default.

**History of the run.**

- The first D2-r2 run (CI `1.6.0.1148`, `2026-10-02-d2r2-gguf-gpu.csv`)
  failed Coder-3B at P ≈ 1000 with +220 MB.
- Cause: on a prompt longer than one ubatch, the ubatches without output rows
  give the last layer a zero-column FFN. The backend refused it, and the
  scheduler copied those weights, `token_embd` included, out of
  `D3D12_Weights` to the CPU.
- Fix: zero-size ops are accepted as no-ops. Coder-3B now measures +125 MB,
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
  own `n_ctx`. Coder-3B ships at 4096, while D2-r2 measured 2048.
- The package carrying the manifest change must pass `validate-console.sh all`
  in the default configuration (no knob).
- The same package must pass H9 via the LAN API, and the log must show the
  layers on D3D12 for the chosen models only.

**D3 result (2026-10-02, CI `1.6.0.1159`): shipped.**

- **Coder-3B at its catalogue `n_ctx` 4096**
  (`bench/results/2026-10-02-d3-coder-ctx4096.csv`): decode 1.60×, prefill
  1.28–1.35×, peak +150 / +126 MB. PASS.
- **Default configuration, no knob:**
  - `validate-console.sh all` ALL PASS;
  - H9 via the LAN API (`2026-10-02-d3-default-h9.jsonl`): Coder-3B 6/8,
    LFM2.5-1.2B 6/8, LFM2.5-350M 4/8, the same as the CPU.
- **Placement**, on a clean log with one request per model: LFM2.5-350M
  loaded on the CPU, LFM2.5-1.2B and Coder-3B on D3D12 (660 / 1834 MiB in
  `D3D12_Weights`).
- **Override:** with `gguf_gpu_layers.txt` set to `0`, LFM2.5-1.2B loaded on
  the CPU.

## Decision log

| Date       | Decision                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    |
| ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-10-01 | Design: ggml backend `d3d12`, matmul-only first, opt-in. D1 gates and the D2 product gate predeclared before any run.                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                       |
| 2026-10-01 | **D1 = `D2-matmul-only`** (CI `1.6.0.1117`): round trip 49.7 µs, simulated Coder-3B token 21.16 ms with sync, in-XAML PASS; weights in DEFAULT heaps (CPU-visible heaps 0.40×). D2 gates unchanged; measure D2 in the UI process too.                                                                                                                                                                                                                                                                                                                                                                                                       |
| 2026-10-01 | D2a implementation: two buffer types (`D3D12_Host` default, `D3D12_Weights` extra) so weights skip the host buft via `supports_op`; Q4_0 / Q4_K / Q6_K kernels; D2a selftest gate predeclared.                                                                                                                                                                                                                                                                                                                                                                                                                                              |
| 2026-10-02 | **D2a = PASS** (run 3, CI `1.6.0.1125`): 12/12 correct, every decode shape ≥ 102 GB/s with the width picked by K (64 threads below K = 4096, 128 from it). Runs 1–2 failed on speed and stay recorded.                                                                                                                                                                                                                                                                                                                                                                                                                                      |
| 2026-10-02 | **D2 = FAIL** (CI `1.6.0.1138`): decode 1.59× / 1.58×, prefill 1.32× / 1.74×, `validate-console.sh all` PASS on Coder-3B / LFM2.5-1.2B; peak RAM +539 / +265 MB over the CPU and Coder-3B H9 5/8 vs 6/8. The backend stays opt-in (`gguf_gpu_layers.txt`), default off. Tied lm_head on the CPU measured and dropped (1.2B decode 1.37×). Follow-ups #309 #310 #312 #313; 350M (0.95×) is never a candidate (#311).                                                                                                                                                                                                                         |
| 2026-10-02 | **RAM criterion for D2-r2/D3 changed by decision, before the run:** peak ≤ 3584 MB, Δ ≤ 200 MB over CPU, GPU ≤ budget. "Peak ≤ CPU" cannot pass any GPU path: the D3D12 runtime alone costs ~44 MB, and the GPU compute buffer is committed, not touched lazily. Gate A PASS (CI `1.6.0.1144`). Fix B so far: Coder-3B 2583 → 2203 MB, 1.2B 1048 → 946 MB.                                                                                                                                                                                                                                                                                  |
| 2026-10-02 | **D2-r2** (CI `1.6.0.1148`): LFM2.5-1.2B **PASS** (decode 1.62×, prefill 1.81×, peak +163/+156 MB, H9 6/8 = 6/8, validate PASS) → D3 candidate. Coder-3B **FAIL** on RAM only (+220 MB at P ≈ 1000; decode 1.55×, H9 6/8 = 6/8) → CPU default; the lever left is lazy commit of the GPU compute buffer (#309).                                                                                                                                                                                                                                                                                                                              |
| 2026-10-02 | **Review fix before merge** (CI `1.6.0.1155`). `n_outputs_max = 1` made llama.cpp abort on any batch asking for more rows: prompt-lookup verify batches and embedding sessions with GPU layers. The cap is now 1 + draft with prompt lookup and llama.cpp's default for embeddings. On console, prompt lookup with GPU layers drafted 44 and accepted 31, and `/api/embed` with the knob passed 17 live requests. Prefill and decode keep 1, so the D2-r2 numbers stand (Coder-3B g99 rerun: 22.4 tok/s, 2203 MB). The selftest now times a median of 5 runs, after a single sample swung 78–109 GB/s; it passes 14/14, GET_ROWS bit-exact. |
| 2026-10-02 | **D2-r2 = PASS** (CI `1.6.0.1156`): Coder-3B decode 1.60×, prefill 1.27×, peak +159/+125 MB; LFM2.5-1.2B 1.64× / 1.82×, +163/+156 MB. H9 6/8 = 6/8 on both, validate ALL PASS with the knob and without. Coder-3B's +220 MB on `1.6.0.1148` was zero-column ops refused by `supports_op`, which made the scheduler copy weights to the CPU; fixed. Next: D3.                                                                                                                                                                                                                                                                                |
| 2026-10-02 | **D3 shipped** (CI `1.6.0.1159`): catalogue `gpu_layers` + `resolve_gguf_gpu_layers` (the operator file overrides; `0` forces the CPU). Qwen2.5-Coder-3B (re-measured at `n_ctx` 4096: decode 1.60×, peak +150/+126 MB) and LFM2.5-1.2B run on d3d12 by default. Default-configuration validate ALL PASS, H9 unchanged, placement checked per model. Follow-ups #310 #313.                                                                                                                                                                                                                                                                  |
