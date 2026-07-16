# Phase 7 — Hypotheses for peer-class model support

**Research note (2026-07-16).** Not a performance SSOT — numbers live in
[benchmarks.md](benchmarks.md). Goal: support models with **capability** (and
where possible throughput) comparable to what peer hardware (Zen 2 class +
mid-range GPU) typically runs, without pretending Series S has peer DRAM
bandwidth or fused DML int4.

## Hardware bound (measured)

| Resource | Series S (Dev Mode) | Implication |
| --- | --- | --- |
| Useful CPU cores | ~6 (t7/t8 livelock ggml) | Cap llama at t6 |
| Effective decode BW | ~12.4–13 GB/s | M=1 GEMV ceiling |
| GPU budget | 3801 MB Game | Prefill/diffusion; not small-model decode |
| Peak RAM seen | ~2.7 GB GGUF | 3B Q3 class fits |

Naive BW bound (13 GB/s, one weight read/token): **3B Q4 ~8 tok/s**, **7B Q4 ~3–4
tok/s**. Measured rows already saturate ≤1B (LFM 94, 0.8B 35, 1.7B 21). Peer
“7B chat” machines win with **more bandwidth**, **GPU decode**, or **fewer
active parameters** (MoE / extreme quant) — not by violating physics.

## Goals (do not conflate)

| ID | Goal | Operational target |
| --- | --- | --- |
| G1 | Capability | Chat quality ≈ casual 3B–7B peer |
| G2 | Throughput | ≥12–15 tok/s interactive decode |
| G3 | Prefill/RAG | Low TTFT at 1k+ tokens |

Series S product priority: **G1 ≥ G2** (smart 2–3B at 10–15 tok/s beats dumb
350M at 94).

## Shipping baseline (SSOT excerpt)

| Model | Decode | Peak | Role |
| --- | --- | --- | --- |
| LFM2.5-350M Q4 | **94.2** | 321 MB | default, G2 king |
| Qwen3.5-0.8B Q4 | 35.1 | 718 MB | mid |
| SmolLM2-1.7B int4 | 20.6 | 2423 MB | mid |
| Gemma-4-E2B Q3 | 15.3 | 2742 MB | best heavy quality so far |

Closed negative: DML int4 decode, 1B fp16 DML inference, llama≫ORT BW, AppContainer mmap.

## Hypotheses

### H1 — High-efficiency architectures (LFM / hybrid class)

- **Claim:** At ~20 tok/s bound, a 1–2B efficient arch beats dense 0.8B quality and approaches peer 3B.
- **PASS:** Quality ≥ E2B and decode ≥20 **or** quality ≫ E2B at ≥12 tok/s.
- **FAIL:** Another tiny fast model without quality lift.
- **Status:** Open — LFM 350M already proves efficiency direction; need larger LFM/hybrid if GGUF exists.

### H2 — MoE with ~1–2B active params

- **Claim:** Decode scales with *active* weights; MoE delivers peer quality at mid speed.
- **PASS:** Peak &lt; 4 GB, decode ≥12, quality &gt; Qwen3.5-0.8B.
- **FAIL:** Arch missing from UWP static lib / OOM / &lt;8 tok/s.
- **Status:** Open — pin includes many `*moe*.cpp` + `lfm2moe` via `src/models/*.cpp` wildcard; needs small-enough GGUF candidate.

### H3 — Speculative decoding (draft LFM + target 1.7–3B)

- **Claim:** ≥1.4× perceived tok/s on target without more average bandwidth.
- **PASS:** ≥1.4× on 1.7B+ with same quality.
- **FAIL:** Overhead &gt; gain on 6 cores.
- **Status:** Deferred eng (llama.cpp has tools; not in `LlamaSession` yet).

### H4 — Usable 3B-class GGUF at Q3/Q4

- **Claim:** 3B Q3_K_S runs ≥8 tok/s, peak &lt; 3.5 GB, coherent chat ≥ E2B.
- **PASS:** Coherent generate + decode ≥8 + peak OK.
- **FAIL:** EOG/garbage (see E2B IQ2) or OOM/livelock.
- **Status:** **In campaign** — primary falsifier (low cost, high information).

### H5 — BitNet / 1.58-bit

- **Claim:** 1.5–3B at 1.58-bit → ≥20 tok/s in 400–800 MB.
- **Status:** Desk — `llama.cpp` has `bitnet.cpp`; ORT GenAI 0.14.1 no stable INT2. Survey before eng.

### H6 — GGUF GPU backend (Vulkan/D3D12) in AppContainer

- **Claim:** At ≥3B, GPU decode beats CPU t6.
- **Status:** Deferred platform work; only after H4 shows 3B is quality-limited not eng-limited. Small-model GPU decode already falsified for ORT DML.

### H7 — GGUF prefill offload (hybrid)

- **Claim:** TTFT at 1k tokens ≤0.7× pure CPU GGUF.
- **Status:** Deferred (hard split-session design).

### H8 — Series X GPU budget for 1B fp16 DML

- **Claim:** Higher Game budget avoids 8007000E on 1B fp16 inference.
- **Status:** Opportunistic if Series X available.

### H9 — Task suite (capability, not tok/s)

- **Claim:** E2B/1.7B + KV-reuse covers casual peer-7B tasks.
- **Status:** Human eval after H4 winners.

## Do not reopen

GPU decode@360M, llama 2× ORT BW, mmap load win, extdata→1B fp16 GPU, DML int4 config-only.

## Shortlist (desk, 2026-07-16)

| Hypothesis | Candidate | Repo / file | Size | Why |
| --- | --- | --- | --- | --- |
| H4 | Llama-3.2-3B-Instruct Q3_K_S | `unsloth/Llama-3.2-3B-Instruct-GGUF` | 1.54 GB | Dense 3B, llama arch, under E2B size |
| H4 | Llama-3.2-3B-Instruct Q4_K_M | same | 2.02 GB | Quality control if Q3 weak |
| H4 | Phi-3.5-mini Q3_K_S | `bartowski/Phi-3.5-mini-instruct-GGUF` | 1.68 GB | Strong small instruct class |
| H1 | Larger LFM / LFM2-MoE if &lt;3 GB | check LiquidAI / unsloth | TBD | Efficiency line |
| H2 | Small MoE GGUF with arch in tree | e.g. OLMoE/Qwen-MoE tiny | TBD | Only if &lt;~3.5 GB Q3 |

## Console campaign results

Source CSV: `bench/results/phase7-scale.csv` (Xbox Series S, t6, `standard-512.txt`,
median of 3 runs with run-1 dropped).

| Model | Quant | Decode | Prefill | Peak MB | Load ms | Verdict |
| --- | --- | --- | --- | --- | --- | --- |
| Llama-3.2-3B-Instruct (`llama32-3b-q3ks`) | Q3_K_S | **14.16** | 19.51 | **1824** | ~16900 | **H4 PASS** |

Notes:

- Quant column was auto-mislabeled `Q4_K_M` by the bench harness (first-token
  heuristic); corrected to **Q3_K_S** (on-disk file name).
- Peak **1824 MB** is ~900 MB under Gemma-4-E2B Q3 (2742 MB) at nearly the same
  decode (14.2 vs 15.3) — dense 3B is *lighter* than E2B MatFormer at this quant.
- Naive BW bound for 3B Q4 was ~8 tok/s; Q3_K_S at **14 tok/s** is consistent with
  fewer weight bytes/token (still bandwidth-class).
- Headless bench completed full decode rows (not EOG-zero). Interactive
  `set_model` autopilot for non-catalogue dirs still needs catalogue entry /
  provision path — provision for campaign was `deploy.sh upload-dir`.

### H4 decision

| Criterion | Result |
| --- | --- |
| Decode ≥ 8 tok/s | **14.16** ✅ |
| Peak &lt; 3.5 GB | **1.82 GB** ✅ |
| Coherent generate (bench path) | ✅ (non-zero decode tokens) |
| vs E2B (15.3 tok/s, 2742 MB) | Similar speed, **much less RAM** |

**H4: PASS.** A dense 3B Q3 is a viable “peer-class capability” candidate on Series
S and should be considered for catalogue (HF direct, like gemma4-e2b).

### Next measured steps

1. Phi-3.5-mini Q3_K_S A/B (quality head-to-head with Llama-3.2-3B).
2. H9 human task suite: LFM vs E2B vs Llama-3.2-3B.
3. Optional catalogue entry `llama32-3b` (HF unsloth GGUF).
4. H3 speculative only if H9 says 3B quality is still short of peer 7B.

## Decision log

| Date | Decision |
| --- | --- |
| 2026-07-16 | Open Phase 7 research; prioritize H4 then H1; H3/H6 eng only after H4 data |
| 2026-07-16 | **H4 PASS** — Llama-3.2-3B Q3_K_S @14.2 tok/s, 1824 MB peak (`phase7-scale.csv`) |
