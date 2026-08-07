# Phase 15 — reverse engineering & inference optimization

> **Living SSOT for the RE + optimization campaign** that aims to make the
> 3B–4B class usable on Xbox Series S. Performance numbers still live in
> [benchmarks.md](benchmarks.md) / `bench/results/`; hypotheses inventory in
> [phase7-hypotheses.md](phase7-hypotheses.md); checklist in
> [`../ROADMAP.md`](../ROADMAP.md) Phase 15. This file owns **campaign method,
> workstream status, and decision log** so those pages do not grow a second
> narrative table.

**Currency:** 2026-08-07. Attack order locked: **W2 first**, then W3 gpubw;
speculative product default decided **after** console M3.

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
| Prompt-lookup pregate 1.53× code / 1.00× chat | `bench/results/phase15-spec-pregate.csv` |

## Do not reopen

GPU decode@360M · DML int4 decode config-only · mmap load win · extdata→1B fp16
GPU · MoE H2 · draft-model speculative as default · host tok/s as product claims.

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
| Spec pregate | `scripts/bench-spec-pregate.sh`, `analyze-spec-pregate.py` | H3 prediction |
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
| RDNA2 outside DirectML | **Open — W3 gpubw** |

## Workstreams

| ID | Name | Issue | Status |
| --- | --- | --- | --- |
| WS0 | Baseline freeze + this doc | — | **done** (this file) |
| WS-A | W2 prompt-lookup speculative | #210 | **host correctness PASS** (greedy MATCH); host suite green; **console A/B needs CI MSVC package** (crossbuild install OK, launch `0x8027025b`) |
| WS-B | W3 gpubw STREAM + Q4 GEMV spike | #211 | queued after W2 ≥ M2 |
| WS-C | #130 DML valley mechanism profile | #130 | opportunistic on console |
| WS-D | H5 BitNet desk survey | — | parallel desk, no eng yet |
| WS-E | H6/H7 GGUF GPU path | — | **only if** W3 ≥ 100 GB/s |

### WS-A detail (W2)

| Slice | Content | Gate |
| --- | --- | --- |
| W2.1 | `prompt_lookup_draft` pure function | **done** — host doctest |
| W2.2 | multi-token verify (`llama_batch_init`, same sampler) | **done** — `decode_loop.h` |
| W2.3 | rollback + runtime degrade on `seq_rm` refuse | **done** — per-generation disable |
| W2.4 | `SessionParams` / CLI `--prompt-lookup`, default OFF | **done** |
| W2.5 | console A/B `qwen25-coder-3b` | pending — `scripts/bench-spec-w2-console.sh` + headless `bench_prompt_lookup.txt`; **deploy CI `xllama-appx` with W2**, not crossbuild-only (see [crossbuild-console.md](crossbuild-console.md)) |

Regression must-pass: `longchat`, `kvsnap`, `gguf`, shipping default `lfm25-350m`
unchanged (hybrid cache cannot tail-rewind — probe disables, does not corrupt).

### WS-B kill criterion (predeclared)

Own compute-shader STREAM read on ~1 GB VRAM buffer, checksum-verified:

- **&lt; 100 GB/s** → H6 “Do not reopen”, close #211
- **≥ 100 GB/s** → open H6 eng plan

Never use the Agility D3D12 factory; headless `gpubw.flag` only.

## Milestones

| M | Deliverable | Exit |
| --- | --- | --- |
| M0 | This doc + ROADMAP/README links | done |
| M1 | W2.1–W2.3 host + tests | ctest PASS |
| M2 | W2.4 opt-in + host acceptance CSV | acceptance vs pregate |
| M3 | Console W2 A/B + full gates | ≥1.4× code; 9/9 console |
| M4 | Product default decision (after M3 numbers) | CHANGELOG |
| M5–M6 | gpubw spike + measure | kill/pass H6 |
| M7 | #130 closed | §5e verdict |
| M8 | H5 survey note | go/no-go |
| M9+ | H6/H7 only if M6 PASS | new phase |

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

## Related issues

- #210 W2 prompt-lookup
- #211 W3 gpubw gate
- #130 DML max_length valley mechanism
- #216 kvsnap intermittent (watch before M3)
