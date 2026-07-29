# Phase 7 — Hypotheses for peer-class model support

**Research note (2026-07-16).** Not a performance SSOT — numbers live in
[benchmarks.md](benchmarks.md). Goal: support models with **capability** (and
where possible throughput) comparable to what peer hardware (Zen 2 class +
mid-range GPU) typically runs, without pretending Series S has peer DRAM
bandwidth or fused DML int4.

## Hardware bound (measured)

| Resource            | Series S (Dev Mode)      | Implication                               |
| ------------------- | ------------------------ | ----------------------------------------- |
| Useful CPU cores    | ~6 (t7/t8 livelock ggml) | Cap llama at t6                           |
| Effective decode BW | ~12.4–13 GB/s            | M=1 GEMV ceiling                          |
| GPU budget          | 3801 MB Game             | Prefill/diffusion; not small-model decode |
| Peak RAM seen       | ~2.7 GB GGUF             | 3B Q3 class fits                          |

Naive BW bound (13 GB/s, one weight read/token): **3B Q4 ~8 tok/s**, **7B Q4 ~3–4
tok/s**. Measured rows already saturate ≤1B (LFM 94, 0.8B 35, 1.7B 21). Peer
“7B chat” machines win with **more bandwidth**, **GPU decode**, or **fewer
active parameters** (MoE / extreme quant) — not by violating physics.

## Goals (do not conflate)

| ID  | Goal        | Operational target               |
| --- | ----------- | -------------------------------- |
| G1  | Capability  | Chat quality ≈ casual 3B–7B peer |
| G2  | Throughput  | ≥12–15 tok/s interactive decode  |
| G3  | Prefill/RAG | Low TTFT at 1k+ tokens           |

Series S product priority: **G1 ≥ G2** (smart 2–3B at 10–15 tok/s beats dumb
350M at 94).

## Shipping baseline roles

The catalogue roles at the close of this campaign are LFM2.5-350M as the fast
default, LFM2.5-1.2B as balanced, and LFM2-2.6B as quality. Llama-3.2-3B remains
the dense peer-class comparator. Current throughput, RAM and quality scores are
generated in [benchmarks.md](benchmarks.md); this research log does not mirror
that table.

Closed negative: DML int4 decode, 1B fp16 DML inference, llama≫ORT BW, AppContainer mmap.

## Hypotheses

### H1 — High-efficiency architectures (LFM / hybrid class)

- **Claim:** At ~20 tok/s bound, a 1–2B efficient arch beats dense 0.8B quality and approaches peer 3B.
- **PASS:** Quality ≥ E2B and decode ≥20 **or** quality ≫ E2B at ≥12 tok/s.
- **FAIL:** Another tiny fast model without quality lift.
- **Status:** **PASS (2026-07-17)** — LFM2.5-1.2B reaches 37.9 tok/s / 811 MB and
  matches E2B at H9 6/8; LFM2-2.6B reaches 18.4 tok/s / 1623 MB and beats E2B
  at H9 7/8. Both are catalogue options; default stays 350M.

### H2 — MoE with ~1–2B active params

- **Claim:** Decode scales with _active_ weights; MoE delivers peer quality at mid-speed.
- **PASS:** Peak &lt; 4 GB, decode ≥12, quality &gt; Qwen3.5-0.8B.
- **FAIL:** Arch missing from UWP static lib / OOM / &lt;8 tok/s.
- **Status:** Open — candidate admitted on measure, awaiting the console run.
  The ceiling that blocked it is measured (below) and the host load fits; what is
  still missing is on-device decode tok/s and peak, i.e. the PASS/FAIL itself.
  Desk survey 2026-07-29 against pin `b10093-1-g6d5a910c5`, whose
  `src/models/*.cpp` wildcard already compiles `lfm2moe.cpp`, `granite-moe.cpp`,
  `qwen3moe.cpp`, `olmoe.cpp` and ~20 more.

  **Candidate: LFM2.5-8B-A1B** (`LiquidAI/LFM2.5-8B-A1B-GGUF`, `unsloth/…` for the
  low quants) — same family as the shipping default, ~1.5B of 8.3B active (≈1/5.5).
  Peak estimated at weights × 1.12, the measured load overhead of the catalogue
  GGUFs (`qwen25-coder-3b` 1840→2116 MB, `lfm2-2.6b` 1491→1623 MB):

  | Quant             | Weights     | Est. peak   | vs H2 gate (4 GB) |
  | ----------------- | ----------- | ----------- | ----------------- |
  | Q4_K_M (official) | 5156 MB     | ~5.8 GB     | over              |
  | UD-IQ4_XS         | 4265 MB     | ~4.8 GB     | over              |
  | UD-Q3_K_M         | 3940 MB     | ~4.4 GB     | over              |
  | **UD-IQ3_S**      | **3571 MB** | **~4.0 GB** | **on the line**   |
  | UD-Q2_K_XL        | 2926 MB     | ~3.3 GB     | under             |
  | UD-IQ2_M          | 2755 MB     | ~3.1 GB     | under             |

  So the hypothesis turned on whether IQ3_S fits, which no number in the repo
  could answer — the only RAM figure was one incidental `avail_phys` 5.0 GB log
  line, and the gates are acceptance policy.

  **Ceiling measured on console 2026-07-29** (MSIX 1.5.1.762, `ramceil.flag` via
  `scripts/bench-ramceil.sh`, raw `bench/results/phase15-ramceil.csv`):
  **4864 MB of heap committed, 4893 MB peak working set**, 38 × 128 MB steps with
  every page faulted in. Process overhead held at exactly 29 MB across all 38
  steps, so committed tracks resident 1:1. `avail_phys` fell linearly from
  ~5113 MB to 240 MB — which both confirms the historical 5.0 GB figure and shows
  it is nearly all spendable.

  Two caveats bound how far this number may be carried:
  1. **It is a lower bound.** The probe stopped on its own 256 MB `avail_phys`
     floor, not on a failed allocation — it was never told how much more it could
     have taken. The ceiling is ≥ 4893 MB, not = 4893 MB.
  2. **It is a headless number.** Measured with no model, no XAML and no
     compositor. An in-app load also pays the compositor's D3D12 device, the ggml
     compute buffers and the KV cache, so the usable in-app ceiling is lower by an
     amount this probe does not measure.

  **Host load, measured** (same GGUF, `xllama-cli` on Linux): the pin loads
  `lfm2moe` and answers coherently at `UD-IQ3_S` with **peak RSS 3502 MB** —
  weights **+2.8%**, not the +12% extrapolated above from the dense catalogue
  models. The estimate was the wrong prior: a dense model re-reads every weight
  each token, while this one touches 4 experts of 32, so its transient buffers
  are sized for the active slice. Treat the table's "est. peak" column as the
  conservative bound it turned out to be, not as the expected value.

  **Consequence:** `UD-IQ3_S` (3502 MB measured on host, ~4.0 GB estimated)
  clears both the H2 4 GB gate and
  the measured ceiling with ~900 MB of headroom, so H2 proceeds at a quant whose
  quality is worth measuring. Q2 is no longer the only option, which matters
  because a Q2 result would have tested the quantization rather than the
  architecture — the E2B IQ2_M garbage precedent (H4 FAIL mode) is the warning,
  with the caveat that it was neither a UD quant nor an MoE, where the low bits
  land on experts rather than on attention. `UD-IQ4_XS` (~4.8 GB) fits the
  measured ceiling but breaks the 4 GB gate: taking it would be a product
  decision to raise the gate, not a measurement.

  Rejected on the same pass: **granite-3.1-3b-a800m** Q4_K_M (2017 MB, ~2.3 GB
  peak) fits comfortably and would be fast, but 800M active is ~1B-class quality —
  it answers "cheap decode", not H2's "peer quality at mid-speed".

### H3 — Speculative decoding (draft LFM + target 1.7–3B)

- **Claim:** ≥1.4× perceived tok/s on target without more average bandwidth.
- **PASS:** ≥1.4× on 1.7B+ with same quality.
- **FAIL:** Overhead &gt; gain on 6 cores.
- **Status:** Deferred eng (llama.cpp has tools; not in `LlamaSession` yet).
  Dependency forks are available and explicitly in scope if H3 or a measured
  kernel/repack bottleneck requires changes below xllama's API layer.

### H4 — Usable 3B-class GGUF at Q3/Q4

- **Claim:** 3B Q3_K_S runs ≥8 tok/s, peak &lt; 3.5 GB, coherent chat ≥ E2B.
- **PASS:** Coherent generate + decode ≥8 + peak OK.
- **FAIL:** EOG/garbage (see E2B IQ2) or OOM/livelock.
- **Status:** **PASS** (2026-07-16) — Llama-3.2-3B preferred; Phi-3.5-mini also gates OK.

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
- **Status:** Opportunistic if Series X available. **#91 allowlist applies**
  (`dml_text_model_ok`): even a passing H8 asset must decompose RMSNorm and
  prove logit parity on **that** DML asset before joining the allowlist — do
  **not** run bare `validate-logit-parity.sh` (default
  `MODEL=smollm2-360m-cpu-int4` is a CPU path). Use an explicit pair, e.g.
  `MODEL=<native-DML-1B-fp16-catalogue-name> ./scripts/validate-logit-parity.sh <matching-golden.bin>`
  (or `MODEL=smollm2-360m-dml-fp16-v2` with its matching golden for the smaller
  DML probe). The DML RMSNorm kernel fault (`dml-rmsnorm-fix-runbook.md`) may
  or may not affect Series X.

### H9 — Task suite (capability, not tok/s)

- **Claim:** E2B/1.7B + KV-reuse covers casual peer-7B tasks.
- **Status:** Human eval after H4 winners.

## Do not reopen

GPU decode@360M, llama 2× ORT BW, mmap load win, extdata→1B fp16 GPU, DML int4 config-only.

## Shortlist (desk, 2026-07-16)

| Hypothesis | Candidate                        | Repo / file                            | Size    | Why                                                   |
| ---------- | -------------------------------- | -------------------------------------- | ------- | ----------------------------------------------------- |
| H4         | Llama-3.2-3B-Instruct Q3_K_S     | `unsloth/Llama-3.2-3B-Instruct-GGUF`   | 1.54 GB | **Measured preferred** — catalogue `llama32-3b`       |
| H4         | Llama-3.2-3B-Instruct Q4_K_M     | same                                   | 2.02 GB | Quality control if Q3 weak (not needed after Q3 PASS) |
| H4         | Phi-3.5-mini Q3_K_S              | `bartowski/Phi-3.5-mini-instruct-GGUF` | 1.68 GB | **Measured** — loses A/B; no catalogue                |
| H1         | LFM2.5-1.2B-Instruct Q4_K_M      | `LiquidAI/LFM2.5-1.2B-Instruct-GGUF`   | 697 MB  | **PASS** — balanced catalogue tier                    |
| H1         | LFM2-2.6B Q4_K_M                 | `LiquidAI/LFM2-2.6B-GGUF`              | 1.46 GB | **PASS** — quality catalogue tier                     |
| H2         | Small MoE GGUF with arch in tree | e.g. OLMoE/Qwen-MoE tiny               | TBD     | Only if &lt;~3.5 GB Q3                                |

## Console campaign results

Source CSV: `bench/results/phase7-scale.csv` (Xbox Series S, t6, `standard-512.txt`,
median of 3 runs with run-1 dropped).

| Model                                     | Quant  | Decode    | Prefill | Peak MB  | Load ms | Verdict                 |
| ----------------------------------------- | ------ | --------- | ------- | -------- | ------- | ----------------------- |
| Llama-3.2-3B-Instruct (`llama32-3b-q3ks`) | Q3_K_S | **14.16** | 19.51   | **1824** | ~16900  | **H4 PASS · preferred** |
| Phi-3.5-mini-instruct (`phi35-mini-q3ks`) | Q3_K_S | **11.31** | 15.29   | **2453** | ~24200  | **H4 PASS · loses A/B** |

Notes:

- Quant column was auto-mislabeled `Q4_K_M` by the bench harness (first-token
  heuristic); corrected to **Q3_K_S** (on-disk file name) for both rows.
- Peak **1824 MB** (Llama) is ~900 MB under Gemma-4-E2B Q3 (2742 MB) at nearly
  the same decode (14.2 vs 15.3) — dense 3B is _lighter_ than E2B MatFormer at
  this quant. Phi-3.5-mini (~3.8B) peaks **2453 MB** — still under the 3.5 GB
  gate, but ~630 MB heavier than Llama at the same quant class.
- Naive BW bound for 3B Q4 was ~8 tok/s; Q3_K_S at **14 tok/s** (Llama) /
  **11 tok/s** (Phi) is consistent with weight-bytes/token (still bandwidth-class).
- Headless bench completed full decode rows (not EOG-zero) for both. Provision
  for campaign: `deploy.sh upload-dir` (Phi not catalogue; Llama is).
- **Template:** campaign headless used ChatML (default at the time). **Phi-3
  template now lands** as `ChatFormatKind::Phi3` (`model_is_phi` /
  `chat_format_for`) so interactive / H9 quality A/B is fair if re-run; speed and
  peak RAM from this campaign remain valid either way.

### H4 decision

| Criterion                      | Llama-3.2-3B                     | Phi-3.5-mini                |
| ------------------------------ | -------------------------------- | --------------------------- |
| Decode ≥ 8 tok/s               | **14.16** ✅                     | **11.31** ✅                |
| Peak &lt; 3.5 GB               | **1.82 GB** ✅                   | **2.45 GB** ✅              |
| Coherent generate (bench path) | ✅                               | ✅                          |
| vs E2B (15.3 tok/s, 2742 MB)   | Similar speed, **much less RAM** | Slower, still under E2B RAM |

**H4: PASS.** Dense ~3B Q3 is viable on Series S. **Llama-3.2-3B is the preferred
peer-class dense candidate** (faster + lighter). Phi-3.5-mini also clears H4
gates but loses the speed/RAM A/B; no catalogue entry unless H9 quality later
overturns that.

### H1 LFM campaign

Sources: `bench/results/phase7-lfm.csv`, `phase7-lfm-long.csv` and
`phase7-h9.jsonl` (Series S, MSIX 1.2.0.536, 2026-07-17). Performance rows are
three-run campaigns with run 1 discarded; t4/t5/t6 were swept and t6 won.

| Model                       | Decode | Prefill | Peak MB | Long decode |  H9 | Verdict                |
| --------------------------- | -----: | ------: | ------: | ----------: | --: | ---------------------- |
| LFM2.5-1.2B-Instruct Q4_K_M |  37.88 |   76.16 |     811 |       35.38 | 6/8 | **H1 PASS · balanced** |
| LFM2-2.6B Q4_K_M            |  18.36 |   32.04 |    1623 |       17.72 | 7/8 | **H1 PASS · quality**  |

H9 uses eight deterministic API tasks at temperature 0 / seed 42. Baselines:
Gemma-4-E2B 6/8, Llama-3.2-3B 5/8, LFM2.5-350M 4/8. Every model failed the same
multi-step arithmetic task. The 2.6B uniquely combines the best aggregate score
with correct abstention, while the 1.2B clears the ≥20 tok/s branch by matching
E2B quality. Both therefore satisfy the predeclared H1 gate without changing
the first-launch default.

Persistent-session validation also passes: turn-2 KV reuse is **19.36×** faster
than cold re-prefill on 1.2B and **20.02×** on 2.6B
(`bench/results/phase7-lfm-kv.csv`).

### Next measured steps

1. ~~Optional catalogue entry `llama32-3b`~~ — **done** (manifest + Llama-3 template).
2. ~~Phi-3.5-mini Q3_K_S speed/RAM A/B~~ — **done** (Llama preferred; see above).
3. ~~H9 deterministic task suite: LFM vs E2B vs Llama-3.2-3B~~ — **done**.
4. H3 speculative: use the available llama.cpp fork only after a target/draft A/B
   predicts ≥1.4× and the quality need justifies dependency work.

## Decision log

| Date       | Decision                                                                                                      |
| ---------- | ------------------------------------------------------------------------------------------------------------- |
| 2026-07-16 | Open Phase 7 research; prioritize H4 then H1; H3/H6 eng only after H4 data                                    |
| 2026-07-16 | **H4 PASS** — Llama-3.2-3B Q3_K_S @14.2 tok/s, 1824 MB peak (`phase7-scale.csv`)                              |
| 2026-07-16 | Catalogue **`llama32-3b`** (HF Q3_K_S) + `ChatFormatKind::Llama3`; default stays LFM                          |
| 2026-07-16 | **Phi-3.5-mini Q3_K_S A/B** — 11.31 tok/s, 2453 MB; H4 PASS but loses to Llama on speed+RAM; **no catalogue** |
| 2026-07-17 | **H1 PASS** — LFM2.5-1.2B 37.88 tok/s, 811 MB, H9 6/8; catalogue balanced tier                                |
| 2026-07-17 | **H1 PASS** — LFM2-2.6B 18.36 tok/s, 1623 MB, H9 7/8; catalogue quality tier; 350M remains default            |
