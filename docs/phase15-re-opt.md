# Phase 15 — reverse engineering & inference optimization

> **Living SSOT for the RE + optimization campaign** that aims to make the
> 3B–4B class usable on Xbox Series S. Performance numbers still live in
> [benchmarks.md](benchmarks.md) / `bench/results/`; hypotheses inventory in
> [phase7-hypotheses.md](phase7-hypotheses.md); checklist in
> [`../ROADMAP.md`](../ROADMAP.md) Phase 15. This file owns **campaign method,
> workstream status, and decision log** so those pages do not grow a second
> narrative table.

**Currency:** 2026-08-08. Attack order: **W2 closed for product default**
(console M3 FAIL ≥1.4× gate → stays opt-in OFF); **W3 gpubw M6 PASS** —
Series S STREAM **119.07 GB/s** (checksum_ok, 1 GiB) ≥ 100 GB/s kill → **H6 eng
motivated**.

## Goal

Raise perceived decode (and/or TTFT) for coding/quality tiers without violating
AppContainer limits and without reopening falsified paths. The binding
constraint is **bytes read per token / effective bandwidth**, not framework
quality ([ROADMAP.md](../ROADMAP.md) Phase 15).

## Baseline freeze (do not re-measure without cause)

Headline console rows from [benchmarks.md](benchmarks.md) (Series S, t6):

| Role | Model | Decode | Prefill | Peak MB | Source |
| --- | --- | ---: | ---: | ---: | --- |
| Default chat | LFM2.5-350M Q4_K_M | **94.9** | 438.1 | 320 | `phase13b-threadsbatch-after` |
| Balanced | LFM2.5-1.2B Q4_K_M | **37.9** | 76.2 | 811 | `phase7-lfm` |
| Quality | LFM2-2.6B Q4_K_M | **18.4** | 32.0 | 1623 | `phase7-lfm` |
| Coding 3B (Phase 15 target) | Qwen2.5-Coder-3B Q4_K_M | **14.0** | 46.2 | 2116 | `phase14-console` |
| Peer dense 3B | Llama-3.2-3B Q3_K_S | **14.2** | 19.5 | 1824 | `phase7-scale` |

Physics already measured (cite, do not re-derive):

| Fact | Home |
| --- | --- |
| CPU decode ~12–13 GB/s effective; membw ceiling | [benchmarks.md](benchmarks.md) membw section, `membw.flag` |
| Bus 224 GB/s theoretical | ROADMAP Phase 15 |
| DML fp16 ~34 GB/s; int4 non-fused GEMM | [uwp-constraints.md](uwp-constraints.md) §5/§12 |
| GGUF UWP = heap (no mmap) | [uwp-constraints.md](uwp-constraints.md) §1 |
| t6 cap (t7/t8 livelock) | §5f |
| One resident model (`SessionHub`) | [architecture.md](architecture.md) |
| H2 MoE FAIL; draft-model speculative FAIL on chat | [phase7-hypotheses.md](phase7-hypotheses.md) |
| Prompt-lookup pregate 1.53× code / 1.00× chat (host physics) | `bench/results/phase15-spec-pregate.csv` |
| Prompt-lookup **console M3** 1.04× code / 0.99× chat | `bench/results/phase15-spec-w2-console.csv` |

## Findings (W2 closed — read this before reopening)

1. **Host pregate ≠ console tok/s.** Pregate predicted ~1.53× on code from
   acceptance × prefill:decode ratio on a throttled host. On Series S the same
   path measures **1.04×** (`qwen25-coder-3b`, t6, n_predict=96, median of
   recorded runs). Speculation **fires** (code ~32 accepts / ~62 drafts) but
   multi-token verify still streams weights; BW-bound decode does not turn
   partial accepts into ≥1.4×.
2. **Lead-first correctness.** Draft tokens must not share the lead’s first
   classic decode batch: old [lead+draft] logits diverge from greedy even with
   zero accepts. Fix: classic lead commit, then draft-only verify batch
   (`decode_loop.h`). Host greedy MATCH after that.
3. **Hybrid / SWA KV cannot tail-rewind.** `seq_rm` refuse → abort generation +
   clear error; probe disables prompt-lookup for that model, does not corrupt
   shipping `lfm25-350m`.
4. **Product default OFF.** Opt-in only: CLI `--prompt-lookup`,
   `SessionParams::prompt_lookup`, headless `bench_prompt_lookup.txt`.
5. **Console package path for measurement:** CI MSVC `xllama-appx` + openappx
   (not Linux uwp-crossbuild alone — activation `0x8027025b`; see
   [crossbuild-console.md](crossbuild-console.md)).

## Do not reopen

GPU decode@360M · DML int4 decode config-only · mmap load win · extdata→1B fp16
GPU · MoE H2 · draft-model speculative as default · host tok/s as product claims ·
**prompt-lookup as product default without a new ≥1.4× console CSV** · treating
crossbuild `/MT` or store-`/MD`+sanitized-libcpmt as a launchable shipping path
until APP-CRT import parity with CI is proven on device.

## Method

1. Hypothesis card: claim, measure, PASS, FAIL, kill — written **before** eng.
2. One variable per CSV; `run_index` ≥ 3 for shipping claims; median + min–max.
3. Host timings are **not** product tok/s (throttled). Host is for TDD and
   acceptance rates only.
4. GPU/EP claims need checksum or logit parity (silent CPU fallback is a real
   failure mode).
5. Same-PR SSOT update; console `validate-console.sh all` before ship.
6. FAIL → “Do not reopen” within a day + issue close with CSV.

### RE tooling inventory

| Tool | Path / entry | Use |
| --- | --- | --- |
| CPU STREAM | `membw.flag` / `xllama-cli --membw` / `include/xllama/membw.h` | Decode BW ceiling |
| RAM ceiling | `ramceil.flag` | Heap upper bound |
| DML profile | `scripts/profile-dml-run.sh`, `analyze_ort_profile.py` | EP / fused node |
| GPU util sample | `scripts/xbox-gpu-sample.sh` | WDP systemperf |
| Spec pregate | `scripts/bench-spec-pregate.sh`, `analyze-spec-pregate.py` | H3 host prediction |
| Spec W2 host A/B | `scripts/bench-spec-w2.sh` | TDD / acceptance rates (not product tok/s) |
| Spec W2 console A/B | `scripts/bench-spec-w2-console.sh` | M3 gate on Series S |
| GPU STREAM (W3) | `gpubw.flag` / `scripts/bench-gpubw.sh` / `include/xllama/gpubw.h` | Own CS read + checksum; kill 100 GB/s |
| Q4_K GEMV (H6.1) | `gpugemv.flag` / `scripts/bench-gpugemv.sh` / `include/xllama/gpugemv.h` | Dequant-in-register; soft G2 40 GB/s packed |
| Logit parity | `scripts/validate-logit-parity.sh` | DML correctness |
| GPU mem | `gpu_mem_info()` in `platform.cpp` | VRAM budget |

### Opaque surfaces (RE targets)

| Surface | Status |
| --- | --- |
| DirectML `DmlFusedNode` / low-bit GEMM | Understood: no fused int4; not fixable in-app |
| DML RMSNorm kernel | Worked around (`-v2` asset); watch driver updates |
| Driver metacommands | Opt-out experiment FAIL; not the #91 cause |
| ggml Q4_K CPU kernels / repack | Repack enabled (#155); decode still BW-bound |
| #130 max_length valley mechanism | Mitigated (warm-up); profile still open |
| RDNA2 outside DirectML | **W3 M6 PASS** — own CS STREAM **119.07 GB/s** on Series S |

## Workstreams

| ID | Name | Issue | Status |
| --- | --- | --- | --- |
| WS0 | Baseline freeze + this doc | — | **done** (this file) |
| WS-A | W2 prompt-lookup speculative | #210 | **closed for default** — host PASS; console M3 **1.04× FAIL** gate; opt-in remains |
| WS-B | W3 gpubw STREAM + Q4 GEMV spike | #211 | **closed PASS** — STREAM **119.07 GB/s** Series S (`1.5.2.853`); Q4 GEMV moves to #228 |
| WS-C | #130 DML valley mechanism profile | #130 | opportunistic on console |
| WS-D | H5 BitNet desk survey | — | parallel desk, no eng yet |
| WS-E | H6/H7 GGUF GPU path | #228 | **parked** — H6.1 G1 PASS / G2 FAIL (2.15 GB/s); Decision 2026-08-08 |

### WS-A detail (W2)

| Slice | Content | Gate |
| --- | --- | --- |
| W2.1 | `prompt_lookup_draft` pure function | **done** — host doctest |
| W2.2 | multi-token verify (`llama_batch_init`, same sampler) | **done** — `decode_loop.h` |
| W2.3 | rollback + runtime degrade on `seq_rm` refuse | **done** — per-generation disable |
| W2.4 | `SessionParams` / CLI `--prompt-lookup`, default OFF | **done** |
| W2.5 | console A/B `qwen25-coder-3b` | **done 2026-08-07** — CI package `1.5.2.920`, Series S t6, n_predict=96, runs=3. Code **14.15 → 14.74 tok/s (1.04×) FAIL** ≥1.4×; chat 14.70 → 14.56 (0.99×) PASS ≥0.98×. Peak ~2.0 GB. Spec active: code ~32/62 draft accepts; chat low draft. CSV: `bench/results/phase15-spec-w2-console.csv` |

Regression must-pass: `longchat`, `kvsnap`, `gguf`, shipping default `lfm25-350m`
unchanged (hybrid cache cannot tail-rewind — probe disables, does not corrupt).

### WS-B kill criterion (predeclared) — **applied 2026-08-08**

Own compute-shader STREAM read on ~1 GB VRAM buffer, checksum-verified:

- **&lt; 100 GB/s** → H6 “Do not reopen”, close #211
- **≥ 100 GB/s** → open H6 eng plan

**Result:** **119.07 GB/s** → PASS → #211 closed as research gate; eng continues in
**#228**. Never use the Agility D3D12 factory; headless `gpubw.flag` only.

### WS-E / H6.1 — Q4_K GEMV density (measure only)

Own CS that reads ggml-compatible `block_q4_K` (144 B), dequants in register,
and accumulates `y = Wx` (no fp16 materialize). Headless `gpugemv.flag` /
CLI `--gpugemv` / `scripts/bench-gpugemv.sh`.

| Gate | Criterion |
| --- | --- |
| G1 correctness | `d3d12_ran` + residual vs CPU ref ≤ 1e-2 (or checksum) |
| G2 density (soft) | G1 + **packed_gbs ≥ 40** (packed Q4_K bytes / s) |

Default tile: N=K=8192 (~36 MiB packed). CSV:
`bench/results/phase15-gpugemv.csv`.

**Console 2026-08-08 (CI `1.5.2.860`):** `packed_gbs=2.15`,
`max_abs_err≈4.6e-5`, `checksum_ok=1`, `d3d12_ran=1` → **G1 PASS / G2 FAIL**.
Naive one-thread-per-row kernel is **compute-bound**, not BW-bound; does not
disprove denser CS designs, but **this spike does not clear soft density ≥40**.
**No product tok/s claim.**

## Milestones

| M | Deliverable | Exit |
| --- | --- | --- |
| M0 | This doc + ROADMAP/README links | done |
| M1 | W2.1–W2.3 host + tests | ctest PASS |
| M2 | W2.4 opt-in + host acceptance CSV | acceptance vs pregate |
| M3 | Console W2 A/B + full gates | **measured** — code 1.04× **FAIL** gate; chat OK; peak OK |
| M4 | Product default decision (after M3 numbers) | **OFF** (opt-in only); CHANGELOG |
| M5 | gpubw STREAM spike (code + flag + DXIL) | **done** (eng); multi-dim Dispatch for 1 GiB; host helpers unit-tested |
| M6 | console measure vs 100 GB/s | **PASS** — Series S **119.07 GB/s**, checksum_ok, 1024 MB, CI `1.5.2.853`; CSV `bench/results/phase15-gpubw.csv` |
| M7 | #130 closed | §5e verdict |
| M8 | H5 survey note | go/no-go |
| M9 | H6.1 Q4_K GEMV measure (code + flag + DXIL) | **measured** — G1 PASS / G2 FAIL |
| M9+ | H6 full decode eng | **parked** (Decision 2026-08-08); reopen only with new density PASS or revised gate |

## Decision log

| Date | Decision |
| --- | --- |
| 2026-07-29 | H3 pregate: draft-model rejected; prompt-lookup proceeds (k=2) |
| 2026-07-30 | H2 MoE FAIL on console; W1 closed |
| 2026-07-30 | Shared `decode_loop.h` groundwork for W2 |
| 2026-08-07 | Campaign plan: W2 first; speculative default **after** M3; console this week |
| 2026-08-07 | W2.1: `include/xllama/speculative.h` `prompt_lookup_draft` (n=2, k=2) |
| 2026-08-07 | W2.2–W2.4: verify batch + seq_rm degrade + `--prompt-lookup` / `SessionParams` default OFF; host suite green |
| 2026-08-07 | W2 fixes: always trim verify tail; `rewind_failed` aborts + clears KV; headless `bench_prompt_lookup.txt`; host `bench-spec-w2.sh` + console `bench-spec-w2-console.sh` |
| 2026-08-07 | Console path: **uwp-crossbuild** `build-project.sh --property XllamaBackend=unified` → **openappx** pack/sign/deploy (not Windows VM) |
| 2026-08-07 | **Correctness fix:** commit lead with classic decode first; draft batch only after lead. Old [lead+draft] batch diverged from greedy even with 0 accepts. Host greedy MATCH after fix. |
| 2026-08-07 | **uwp-crossbuild gotchas for xllama:** (1) VCLibs PackageDependency in AppxManifest; (2) `ops.h` CACHE_LINE_SIZE under clang-cl; (3) store-CRT + `libcpmt` MT is incorrect for `/MD` (gotcha 21). **Resolution for product launch:** use **CI MSVC** (`build-uwp` / `xllama-appx`). Crossbuild `/MT` and experimental store-`/MD`+sanitized-libcpmt **link** but fail Xbox activation `0x8027025b` (registry/KERNEL32 imports vs CI APP-CRT). hello-uwp still launches. Details: [crossbuild-console.md](crossbuild-console.md). |
| 2026-08-07 | Host tests: full suite PASS after W2. Control deploy: CI package `1.5.2.910` launches and loads GGUF on Series S. |
| 2026-08-07 | **M3 console A/B (W2.5):** `qwen25-coder-3b` CI W2 `1.5.2.920`. Code 1.04× FAIL (≥1.4×); chat 0.99×. Speculation works (code ~50% draft accept) but BW-bound console does not convert accepts into ≥1.4× tok/s. **Product default remains OFF.** Next campaign focus: W3 gpubw (#211). |
| 2026-08-08 | **Docs consolidated:** Findings section above; H3 card in phase7 updated; model-matrix gap closed; crossbuild launch path remains CI-only. PR #226 holds the W2 eng + CSVs. |
| 2026-08-08 | **W3 gpubw spike (#211) eng:** `include/xllama/gpubw.h`, AOT DXIL `shaders/gpubw_stream.hlsl` → `shaders/generated/gpubw_stream_dxil.h`, UWP `gpubw.flag` / CLI `--gpubw`, `scripts/bench-gpubw.sh`. System `D3D12CreateDevice` only (no Agility). Multi-dim Dispatch for 1 GiB (≤65535/dim). |
| 2026-08-08 | **M6 console gpubw PASS:** Series S CI package `1.5.2.853`, 1 GiB buffer, 3 iters, **read=119.07 GB/s**, `checksum_ok=1`, `d3d12_ran=1` → kill gate **PASS** (≥100). CSV: `bench/results/phase15-gpubw.csv`. **H6 eng is motivated** (own CS beats DirectML BW lens). |
| 2026-08-08 | **#211 closed** as the research/measure gate (PASS). Eng handoff: **#228**. PR #227 carries the probe + multi-dim Dispatch + docs. |
| 2026-08-08 | **H6.1 eng start (#228):** `gpugemv` Q4_K GEMV density probe (AOT DXIL, system D3D12, pure CPU ref). Soft G2 ≥40 GB/s packed; G1 residual. Not a SessionHub backend. |
| 2026-08-08 | **H6.1 console:** Series S `1.5.2.860`, N=K=8192, **packed_gbs=2.15**, max_abs_err≈4.6e-5 → **G1 PASS / G2 FAIL**. Naive CS compute-bound. CSV `bench/results/phase15-gpugemv.csv`. |
| 2026-08-08 | **H6 eng parked (#228).** Binding constraint remains bytes/token; G2 FAIL shows no free ride on STREAM 119 GB/s. No full GGUF GPU backend without new density PASS or explicit gate rewrite. Focus → product hygiene. |

## Related issues

- #210 W2 prompt-lookup (eng shipped opt-in; default OFF after M3)
- #211 W3 gpubw gate — **closed PASS** (119.07 GB/s); PR #227
- #228 H6 eng follow-up — **parked** after H6.1 G2 FAIL
- #130 DML max_length valley mechanism
- #216 kvsnap intermittent (watch on console gates)
