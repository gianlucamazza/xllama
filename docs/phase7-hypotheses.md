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
- **Status:** **FAIL (2026-07-30)** — measured on console, and the claim does not
  survive. `lfm25-8b-a1b` at `UD-IQ3_S` decodes at **14.50 tok/s** against the
  dense `qwen25-coder-3b`'s 14.0, for **+1437 MiB** of peak; and because it reasons
  on every turn it spends 3-25× more tokens than its reply contains, making it
  ~4× slower in perceived latency. The bandwidth premise was confirmed (631 MB
  read/token vs 645 predicted) — the cost simply moves elsewhere. Full result,
  including why H9 was not applicable, under "H2 measured on console" below.
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

  **Host load, measured** (same GGUF, `xllama-cli` on Linux, `-t 6 -n 64`): the
  pin loads `lfm2moe` and answers coherently at `UD-IQ3_S`, peak RSS **3495 MB**
  with a 264 MiB compute buffer.

  **That host peak does not predict the console peak, and must not be quoted as
  if it did.** The host path leaves llama.cpp's `use_mmap` at its default, so RSS
  counts only the weight pages actually faulted in; the console has no mmap
  (§1) and reads the whole file into the heap. The two numbers measure different
  things, and for a MoE they diverge in the dangerous direction — low on host.
  The console peak estimate therefore stays **weights × ~1.12 ≈ 4.0 GB**; the
  table above is unrevised.

  What the host run _does_ establish is that sparse activation is real and
  observable: 3495 MB of resident set against **3571 MB of weights plus a 264 MiB
  compute buffer** means a sizeable share of weight pages was never touched. The
  dense comparator on the same host and the same prompt goes the other way —
  `qwen25-coder-3b` Q4_K_M reaches 3225 MB against 1840 MB of weights, every page
  resident. Caveat on the fraction: experts accumulate with tokens generated, so
  "untouched" at `-n 64` is not a fixed property of the model. The mechanism H2
  bets on is visible; its magnitude is not measured here.

  **Consequence:** `UD-IQ3_S` (~4.0 GB estimated console peak)
  clears both the H2 4 GB gate and
  the measured ceiling with ~900 MB of headroom, so H2 proceeds at a quant whose
  quality is worth measuring. Q2 is no longer the only option, which matters
  because a Q2 result would have tested the quantization rather than the
  architecture — the E2B IQ2_M garbage precedent (H4 FAIL mode) is the warning,
  with the caveat that it was neither a UD quant nor an MoE, where the low bits
  land on experts rather than on attention. `UD-IQ4_XS` (~4.8 GB) fits the
  measured ceiling but breaks the 4 GB gate: taking it would be a product
  decision to raise the gate, not a measurement.

  ### H2 measured on console 2026-07-30 — the claim is falsified

  MSIX 1.5.2.798, `standard-512`, 3 recorded runs after warmup
  (`bench/results/phase15-moe-console.csv`):

  |         | `lfm25-8b-a1b` UD-IQ3_S | `qwen25-coder-3b` Q4_K_M |
  | ------- | ----------------------: | -----------------------: |
  | decode  |         **14.50** tok/s |               14.0 tok/s |
  | prefill |                   18.97 |                     46.2 |
  | peak    |            **3553 MiB** |                 2116 MiB |
  | load    |                  19.5 s |                        — |

  **No speed advantage over a dense 3B, at +1437 MiB.** Decode spread across the
  three runs was 14.50 / 14.61 / 14.50, so this is not noise.

  **The bandwidth premise was right; the hypothesis was still wrong.** Decomposing
  the measurement gives ~631 MB read per token against the **645 MB predicted**
  from 4-of-32 expert activation — a 2% error. Inactive experts genuinely are not
  read. What H2 did not anticipate is that the bottleneck _moves_: what remains is
  not bandwidth, and it is ~5× what a dense model of the same active parameter
  count would pay. Two candidate causes, **not separated by this measurement**:
  `IQ3_S` is an i-quant, whose dequantization is far more expensive than `Q4_K_M`,
  and gathering 4 experts of 32 is a scattered access a dense model never pays.
  Separating them needs a dense IQ3 comparator the catalogue does not have.

  A limit of that decomposition, stated rather than hidden: `T(n) = W + n·C`
  assumes `W` is constant, which **breaks on a MoE** — in prefill each token of the
  batch may activate different experts, so distinct weights read grow with batch
  size. Prefill at 18.97 against the dense 1.5B's 96.6 is the symptom. The decode
  figure is measured and stands; the 16.3/52.7 split reads as _"the cost is not
  where we expected"_, not as a compute measurement.

  **Perceived latency is worse still, and this is what settles it.** The model
  reasons on every turn. Measured over the LAN endpoint at temperature 0:
  "capital of Italy, one word" → answer `Roma`, **102 completion tokens** (4
  visible); "explain in two sentences" → a correct 72-word answer, **401
  completion tokens** (~95 visible). That is 3-25× more tokens than the reply
  contains. At 14.5 tok/s the two-sentence answer takes **27.7 s**, against ~6.8 s
  for the same visible output from the dense 3B. The MoE is not merely no faster —
  it is roughly **4× slower in perceived latency**.

  **H9 was not run, and could not have been.** Its tasks cap generation at 16-80
  tokens (`bench/eval/phase7-h9.json`), which a model that spends ~100 tokens
  reasoning before answering cannot clear: every task would score 0 while measuring
  the budget, not the model. Quality was therefore probed at an adequate budget
  instead, and the model answers correctly and coherently — the failure is not one
  of capability.

  Not a product defect, worth recording: the LAN endpoint reports this case
  correctly. An empty `content` arrives with `finish_reason: "length"` and
  `completion_tokens: 64`, which is exactly the OpenAI contract's way of saying
  "the budget was consumed" — a client can tell it apart from "the model had
  nothing to say". `validate-api.sh chat` reads only the content and so reports
  FAIL on any thinking model; that is a gate limitation, not an app one.

  **Verdict: H2 FAIL on its claim.** "Decode scales with _active_ weights" holds
  for bandwidth and does not survive contact with the rest of the cost. The
  catalogue entry stays in `model-matrix.md` §A3 with this result attached; the
  model does not enter a product tier. The 3.5 GB product-gate question that a
  speed PASS would have opened is moot — at equal decode, worse perceived latency
  and +1437 MiB there is no case to weigh.

  **Do not reopen at a lower quant.** Q2 would trade the one thing that worked
  (quality) against a cost that is not bandwidth-bound, i.e. it would test the
  quantization rather than the architecture. A future MoE is worth measuring only
  if it is **not** an i-quant and its expert gather is cheaper — the two candidate
  causes above, which remain unseparated.

  Rejected on the same pass: **granite-3.1-3b-a800m** Q4_K_M (2017 MB, ~2.3 GB
  peak) fits comfortably and would be fast, but 800M active is ~1B-class quality —
  it answers "cheap decode", not H2's "peer quality at mid-speed".

### H3 — Speculative decoding (draft LFM + target 1.7–3B)

- **Claim:** ≥1.4× perceived tok/s on target without more average bandwidth.
- **PASS:** ≥1.4× on 1.7B+ with same quality.
- **FAIL:** Overhead &gt; gain on 6 cores.
- **Status:** **Implemented (opt-in) + product-default FAIL.** Pre-gate
  (2026-07-29) rejected draft-model (1.43× code / **0.81× chat**) and admitted
  draft-free prompt lookup k=2 (host physics 1.53× / 1.00×). Phase 15 W2 (#210)
  landed in `LlamaSession` / `decode_loop.h` / CLI / headless
  `bench_prompt_lookup.txt` (default **OFF**). Console M3 (2026-08-07,
  Series S, `qwen25-coder-3b`): **1.04× code FAIL** vs ≥1.4× ship gate; chat
  0.99×; peak ~2.0 GB — CSV `bench/results/phase15-spec-w2-console.csv`.
  Campaign SSOT: [phase15-re-opt.md](phase15-re-opt.md). **Do not turn on as
  product default** without a new console CSV that clears ≥1.4×. Cost model and
  vocab table below remain the pre-gate evidence (host), not console tok/s.

  The pin gates speculation on `common_speculative_are_compatible`
  (`common/speculative.cpp:64`) and **throws** when it fails, so vocab identity
  is a hard precondition, not a quality knob. Replicating that function exactly
  against catalogue GGUFs, vocab-only (raw: `bench/results/phase15-spec-vocab.csv`):

  | target                | draft               | verdict | why                                     |
  | --------------------- | ------------------- | ------- | --------------------------------------- |
  | `qwen25-coder-3b`     | `qwen25-coder-0.5b` | **OK**  | 151936 tokens, 0 differing texts        |
  | `qwen25-coder-1.5b`   | `qwen25-coder-0.5b` | **OK**  | identical vocab                         |
  | `lfm25-1.2b-thinking` | `LFM2.5-350M`       | **OK**  | 65536 tokens, 0 differing texts         |
  | `qwen3-1.7b`          | `qwen25-coder-0.5b` | no      | same size, **4 token texts differ**     |
  | `LFM2.5-8B-A1B`       | `LFM2.5-350M`       | no      | 128000 vs 65536 tokens; differs at id 5 |

  Two of those negatives are worth keeping:
  1. **Same vocab size is not the same vocab.** Qwen3-1.7B and Qwen2.5-Coder both
     report 151936 tokens and pass every size check, then diverge at id 151665
     (`<tool_response>` vs `<|PAD_TOKEN|>`). A pairing rule based on vocab size —
     or on family name — would have shipped a pair that throws at session start.
  2. **The MoE candidate has no draft in the catalogue.** LFM2.5-8B-A1B carries a
     **128000**-token vocab against LFM2.5-350M's 65536, differing from id 5 up.
     Despite the shared name it is not the shipping model's tokenizer lineage, so
     **H2 and H3 do not compose on it**: speeding up the MoE would need a draft
     trained on its own tokenizer, which the catalogue does not have and this
     project does not pretrain.

  So W2 proceeds on `qwen25-coder-3b` ← `qwen25-coder-0.5b`, the pair the plan
  named, and the runtime guard that refuses an incompatible pair is the first
  code item rather than an afterthought.

  **Pre-gate measured 2026-07-29** (`scripts/bench-spec-pregate.sh`, raw
  `bench/results/phase15-spec-pregate.csv`, batch curve
  `…-pregate-batched.txt`). The gate this project predeclared is a **prediction**
  before any engineering, so the two hardware-independent quantities were measured
  on host and combined with the console-measured cost model.

  **The cost model.** Decompose a forward pass as `T(n) = W + n·C` — `W` the weight
  read, paid once, `C` the marginal compute per token in the batch. From the
  console figures for `qwen25-coder-3b` (prefill 46.2, decode 14.0):
  **W = 49.8 ms, C = 21.6 ms**, so compute is **30%** of a decode step. The same
  fit across `coder-0.5b` and `coder-1.5b` returns an effective weight-read
  bandwidth of 43 / 35 / 39 GB/s — consistent, which is what makes the
  decomposition credible. The prefill:decode ratio on the 3B is only **3.3:1**,
  where a GPU shows ~50:1; that ratio is the whole story, because it means
  verifying k+1 tokens in one pass is **not** nearly free.

  Ceilings that follow, at α = 1 and k → ∞: **1.90×** with a draft model
  (`C_draft + C_target`), **3.30×** without one (`C_target`). The gate asks 1.4×,
  which sits inside a narrow band rather than comfortably below the maximum.

  `llama-batched-bench` confirms the shape rather than a favourable exception:
  per-step cost went 1261 → 2163 → 2888 → 5687 ms for batches of 1/2/4/8, i.e.
  **roughly linear with no bandwidth-bound plateau at small n**. Had the middle
  region been free, every ceiling here would be too low. It is not. (Absolute
  values there are throttled and the per-sequence KV grows with the batch, so
  that table is structural evidence only — the W/C ratio comes from the console.)

  **Measured acceptance, and the number that actually decides.**

  Target `qwen25-coder-3b` (baseline 71.4 ms/token):

  | variant     |   k | regime | drafted |     α | drafting rounds | predicted |
  | ----------- | --: | ------ | ------: | ----: | --------------: | --------: |
  | draft model |   2 | code   |      88 | 97.7% |            100% | **1.43×** |
  | draft model |   2 | chat   |     140 | 42.9% |             82% | **0.81×** |
  | draft model |   4 | code   |     108 | 96.3% |             91% | **1.48×** |
  | draft model |   4 | chat   |     188 | 44.7% |             51% | **0.67×** |
  | n-gram      |   2 | code   |      94 | 84.0% |             85% | **1.53×** |
  | n-gram      |   2 | chat   |      10 | 40.0% |          **4%** | **1.00×** |
  | n-gram      |   4 | code   |     136 | 69.1% |             49% | **1.16×** |
  | n-gram      |   4 | chat   |      20 | 10.0% |              4% | **0.96×** |

  Target `lfm25-1.2b-thinking` ← `lfm25-350m` (baseline 27.2 ms/token):

  | variant     |   k | regime | drafted |     α | drafting rounds | predicted |
  | ----------- | --: | ------ | ------: | ----: | --------------: | --------: |
  | draft model |   2 | code   |      88 | 96.6% |             99% | **1.24×** |
  | draft model |   2 | chat   |     102 | 76.5% |             86% | **1.01×** |

  The deciding column is not α, it is **how often the variant drafts at all**. On
  open-ended chat prompt-lookup finds no match and simply declines — 10 drafted
  tokens against the draft model's 140 for the same prompt — so it degrades to
  plain decode. The draft model always drafts, so a bad regime makes it pay for
  80 rejected tokens at 16.0 ms each, and it ends up **19% slower than no
  speculation at all**.

  A note against my own first model: computing this per-round, as if drafting
  always happened, gave the n-gram 0.97× on chat and erased the exact property
  that makes it safe. The frequency-aware form above is the honest one.

  **Verdict: the draft-model variant is rejected, the draft-free one proceeds.**
  A feature that is 1.43× on code and 0.81× on chat is not shippable as a default
  and the regime is not knowable in advance; 1.53× / 1.00× is. Dropping the draft
  model also drops 533 MB of resident weights, the second threadpool that would
  have risked the t7/t8 livelock, and the vocab-compatibility constraint entirely
  — which is what makes the pair table above interesting rather than binding.
  Raising k does not rescue the draft model, it deepens the hole: k=4 buys 1.48× on
  the regime that was already passing and takes chat from 0.81× to **0.67×**, a
  third slower than no speculation, because a longer wrong draft is a longer wrong
  draft. The table is generated by `scripts/analyze-spec-pregate.py` from the raw
  CSV, so every predicted figure here can be re-derived rather than trusted.

  **Two results that fell out of the same run.**

  **k = 2 is the setting, not a starting point.** Prompt lookup drops from 1.53×
  at k=2 to **1.16× at k=4**, because per-token acceptance decays with draft
  length (84.0% → 69.1%) while every extra token in the verify batch costs a full
  `C`. With compute at 30% of a decode step there is no long-draft regime to grow
  into — the k that wins is the smallest one that wins at all. At k=4 the chat
  regime even goes marginally negative (0.96×), so the safety property is a
  property of k=2, not of prompt lookup in general.

  **Speculation is only interesting on the largest target.** The same draft-model
  experiment on the shipping LFM pair (`lfm25-1.2b-thinking` ← `lfm25-350m`) gives
  **1.24× on code and 1.01× on chat** — a clean fail against the gate, at α 96.6%,
  which is as favourable as acceptance gets. The reason is a ratio, not a defect:
  the draft costs 10.5 ms against a 27.2 ms baseline (**39%**), where the Coder
  pair costs 16.0 against 71.4 (**22%**). A smaller target makes its own draft
  proportionally more expensive, so the technique is worth least exactly where the
  model is already fast, and worth most where it is slow. That is convenient for
  the product and unhelpful for a general recommendation.

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
- **Status:** **W3 gate PASS (2026-08-08).** Own CS STREAM on Series S:
  **119.07 GB/s** read over 1 GiB VRAM, checksum-verified
  (`bench/results/phase15-gpubw.csv`, package `1.5.2.853`). Kill was
  &lt; 100 GB/s → Do not reopen; measured ≥ 100 → **H6 eng is open** in
  **#228** (own compute path, not DirectML). Small-model GPU decode via ORT
  DML remains falsified; this result only shows raw VRAM bandwidth our D3D12
  code can reach.

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

GPU decode@360M, llama 2× ORT BW, mmap load win, extdata→1B fp16 GPU, DML int4
config-only, MoE H2 as a decode win, draft-model speculative as a default
(open-chat regression). Campaign tracking: [phase15-re-opt.md](phase15-re-opt.md).

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
