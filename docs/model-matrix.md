# Model matrix — xllama inventory

**SSOT for “what have we tested / shipped / rejected” across text models.**
Performance _numbers_ for comparison rows remain generated in
[benchmarks.md](./benchmarks.md) from `bench/results/` +
`bench/benchmark-summary.json`. This page owns **status, roles, catalogue
ids, templates, licenses, and campaign notes** — not runtime contracts.

| Concern                                                        | Home                                                         |
| -------------------------------------------------------------- | ------------------------------------------------------------ |
| Structure (`n_ctx`, `role`, ChatFormat, out-of-scope surfaces) | [architecture.md](./architecture.md)                         |
| How to add / select models                                     | [model-selection.md](./model-selection.md)                   |
| Catalogue data                                                 | [`../uwp/models/manifest.json`](../uwp/models/manifest.json) |
| Tok/s tables                                                   | [benchmarks.md](./benchmarks.md) only                        |

Last updated: **2026-10-02**. Newest entries: D3 — `qwen25-coder-3b` and `lfm25-1.2b-instruct` run on the d3d12 GPU backend by default (catalogue `gpu_layers`, [gguf-gpu-decode.md](./gguf-gpu-decode.md)), with Flash Attention kept on that path since the FA fix (decode 1.90× / 1.88× the CPU); the CPU columns here stay the CPU measurements, and the default-configuration rows are in [benchmarks.md](./benchmarks.md). Earlier: H9 re-run with the corrected
`grounded_qa` scorer (#243) and `lfm25-230m` back on QAD Q4_0; Series S catalogue gates
(`2026-09-30-catalogue-gates`): §A1 `phi4-mini` T3 PASS (#268), LFM2.5 QAD
Q4_0 PASS for `lfm25-350m` / `lfm25-1.2b-instruct` (#270); §F MiniCPM5-1B (#267) and Gemma-3-1B (#269)
rejected — measured; §I arch-watch reopen gates
(#273 MoE, #274 BitNet, #275 SSM/RWKV + mm), then §F MiniCPM5 renderer shipped / T3 not booked (2026-08-10), WS-E
closed (no consumer); then §A1 `lfm25-230m` shipped as the floor tier and
`gemma3-270m`'s H9 measured (2026-08-10), §A3 H2 MoE FAIL (2026-07-30), §D
per-arch `can_shift` (2026-07-29), §A2 phase14 (2026-07-27).

## How to read the columns

| Column              | Meaning                                                                     |
| ------------------- | --------------------------------------------------------------------------- |
| **Catalogue**       | Id in `uwp/models/manifest.json` (`—` = not shipped)                        |
| **Status**          | `shipping` / `catalogue` / `campaign-only` / `host-smoke` / `rejected`      |
| **Console metrics** | Xbox Series S Dev Mode (see [benchmarks.md](./benchmarks.md))               |
| **H9**              | Deterministic 8-task suite (`2026-09-30-h9-rescore.jsonl`), temperature 0   |
| **Template**        | `chat_format_for` selection                                                 |
| **Role**            | Catalogue `role` (`coding` → denser token estimate + coding system default) |
| **n_ctx**           | Session context (0/omit → `kDefaultNCtx` 2048)                              |

Host figures are **not comparable** to console tok/s (different CPU, build type,
and thermal). They only prove _load + generate + template_.

---

## A. Shipping / catalogue text models (console or host)

### A1. Console-validated performance (Series S)

Headline metrics match the generated table in [benchmarks.md](./benchmarks.md).
All rows below are **CPU-bound decode** unless backend says DirectML.

| Model                 |                  Catalogue |  Params | Quant    | Backend   |   Prefill |    Decode | Peak MB | H9      | Template          | Role | n_ctx | Evidence                                                               |
| --------------------- | -------------------------: | ------: | -------- | --------- | --------: | --------: | ------: | ------- | ----------------- | ---- | ----: | ---------------------------------------------------------------------- |
| LFM2.5-230M           |               `lfm25-230m` |    230M | Q4_0 QAD | llama.cpp | **633.6** | **132.4** |     236 | 1/8     | ChatML            | —    |  2048 | `2026-09-30-catalogue-gates` · floor                                   |
| LFM2.5-350M           |               `lfm25-350m` |    350M | Q4_0 QAD | llama.cpp | **374.4** | **101.6** |     311 | 4/8     | ChatML            | —    |  2048 | `2026-09-30-catalogue-gates` · **default chat**                        |
| Gemma-3-270M          |              `gemma3-270m` |    270M | Q4_K_M   | llama.cpp |     395.0 |      76.8 |     368 | 3/8     | Gemma             | —    |  2048 | `phase6-gemma` · H9 `2026-09-30-h9-rescore.jsonl`                      |
| SmolLM2-360M          |    `smollm2-360m-cpu-int4` |    360M | int4     | ORT CPU   |     262.4 |      74.8 |     708 | —       | ChatML            | —    |  2048 | `t6-shipped-confirm`                                                   |
| SmolLM2-360M          |              (same family) |    360M | Q4_K_M   | llama.cpp |     141.5 |      62.9 |     402 | —       | ChatML            | —    |  2048 | `phase35-llamacpp-scaling`                                             |
| SmolLM2-360M DML v2   | `smollm2-360m-dml-fp16-v2` |    360M | fp16     | ORT DML   |     236.7 |      44.4 |    1268 | —       | ChatML            | —    |  2048 | `phase2-dml` · #91 parity OK                                           |
| LFM2.5-1.2B Instruct  |      `lfm25-1.2b-instruct` |    1.2B | Q4_0 QAD | llama.cpp |     109.7 |  **40.0** |     783 | **6/8** | ChatML            | —    |  2048 | `2026-09-30-catalogue-gates` · H1 PASS balanced · **GPU default (D3)** |
| Qwen3.5-0.8B          |              `qwen35-0.8b` |    0.8B | Q4_K_M   | llama.cpp |      98.1 |      35.1 |     718 | —       | ChatML + no-think | —    |  2048 | `phase5-gguf`                                                          |
| SmolLM2-1.7B          |    `smollm2-1.7b-cpu-int4` |    1.7B | int4     | ORT CPU   |      54.9 |      20.6 |    2423 | —       | ChatML            | —    |  2048 | `phase35-1b-cpu`                                                       |
| LFM2-2.6B             |                `lfm2-2.6b` |    2.6B | Q4_K_M   | llama.cpp |      32.0 |  **18.4** |    1623 | **7/8** | ChatML            | —    |  2048 | `phase7-lfm` · H1 PASS quality                                         |
| Gemma-4-E2B           |               `gemma4-e2b` | ~2B eff | Q3_K_S   | llama.cpp |      26.1 |      15.3 |    2742 | 6/8     | Gemma             | —    |  2048 | `phase6-gemma`                                                         |
| Llama-3.2-3B Instruct |               `llama32-3b` |      3B | Q3_K_S   | llama.cpp |      19.5 |  **14.2** |    1824 | 5/8     | Llama-3           | —    |  2048 | `phase7-scale` · H4 preferred                                          |
| Phi-3.5-mini          |                          — |    3.8B | Q3_K_S   | llama.cpp |      15.3 |      11.3 |    2453 | —       | Phi-3             | —    |  2048 | `phase7-scale` · H4 PASS, loses A/B                                    |
| Phi-4-mini            |                `phi4-mini` |    3.8B | Q4_K_M   | llama.cpp |      33.1 |      11.2 |    2765 | **6/8** | Phi-3             | —    |  2048 | `2026-09-30-catalogue-gates` · **T3 PASS** peer                        |
| Gemma-4-E2B IQ2       |            (upgraded away) | ~2B eff | IQ2_M    | llama.cpp |      13.5 |       9.9 |    2534 | —       | Gemma             | —    |  2048 | historical; EOG on long prompts                                        |
| SmolLM2-360M DML int4 |                          — |    360M | int4     | ORT DML   |     0–153 |   **8.8** |     999 | —       | ChatML            | —    |  2048 | **rejected** wrong logits / slow                                       |

Notes:

- Hybrid LFM: KV tail-rewind unsupported (#170a); front-drop context shift OK (#169).
- **QAD Q4_0 ripin** (#270, console regression 2026-09-30,
  `2026-09-30-catalogue-gates`): `lfm25-230m`, `lfm25-350m` and
  `lfm25-1.2b-instruct` pin Liquid [QAD Q4_0](https://www.liquid.ai/blog/qad)
  with the same H9, higher decode and lower peak. The 230M first failed (1/8
  against a recorded 2/8). That 2/8 counted a wrong `grounded_qa` answer
  ("ciascuno occupa 730 MB"). With the corrected scorer (#243) both artefacts
  score 1/8, so QAD passes. `lfm2-2.6b` stays LFM2 Q4_K_M, since no QAD was
  published for that generation.
- **QAD trade-off, same build** (1.6.0.1072, `2026-09-30-qad-prefill-ab`,
  interleaved runs): Q4_0 QAD costs about 15% prefill on all three ids
  (230M 757 → 635, 350M 439 → 374, 1.2B 131 → 110 tok/s) and gains 8–13%
  decode (121.7 → 131.4, 90.0 → 101.6, 36.9 → 40.1) at 2–4% less peak. An
  earlier note that the 1.2B prefill "rises" compared builds from different
  eras (the 76.2 predates #168) and was wrong.
- **H9 source** (#243): since 2026-09-30 the published H9 column comes from one
  full re-run with the corrected scorer (`2026-09-30-h9-rescore.jsonl`).
  `phase7-h9.jsonl` stays as raw history. Only `lfm25-230m`'s Q4_K_M score
  changed (2/8 → 1/8).
- Qwen3.5 (`qwen35`): `can_shift` false (imrope) — overflow fail-fast + trim, no
  RoPE shift. Qwen3 (`qwen3`) is a different arch and **does** shift — measured, see §D.
- DML text routing allowlist: only `smollm2-360m-dml-fp16-v2` (`dml_text_model_ok`).
- Thinking models get **no KV snapshot** (#170b): the saved history is stripped
  while the resident KV holds the full chain of thought, so the #170a prefix diff
  always diverges and the snapshot would buy nothing. In-conversation delta reuse
  is unaffected.
- Prefill is chunked at `n_batch` for every arch (`fit_prompt` bounds the prompt,
  `LlamaSession::generate` splits it): a prompt past the logical batch is no longer
  an abort, and one past `n_ctx` is a clean `prompt too long`.

### A2. Phase 14 — console-validated (Series S, 2026-07-27)

MSIX **1.5.1.737** (unified), t6, `standard-512` prompt, 2 recorded runs after
warmup. Source: `bench/results/phase14-console.csv` (also in generated
[benchmarks.md](./benchmarks.md)).

| Model                | Catalogue             | Quant  | n_ctx |   Prefill |   Decode | Decode spread |  Peak MB | Status                                                                  |
| -------------------- | --------------------- | ------ | ----: | --------: | -------: | ------------- | -------: | ----------------------------------------------------------------------- |
| Qwen2.5-Coder-0.5B   | `qwen25-coder-0.5b`   | Q4_K_M |  4096 | **148.2** | **62.4** | 56.8–68.0     |      533 | **console PASS** · coding fast                                          |
| LFM2.5-1.2B-Thinking | `lfm25-1.2b-thinking` | Q4_K_M |  2048 | **130.4** | **36.7** | 36.7–36.8     |      811 | **console PASS** · CoT strip · catalogue **n_predict 1024** (#223)      |
| Qwen2.5-Coder-1.5B   | `qwen25-coder-1.5b`   | Q4_K_M |  4096 |  **96.6** | **26.1** | 25.7–26.5     |     1179 | **console PASS** · coding balanced                                      |
| Qwen3-1.7B           | `qwen3-1.7b`          | Q4_K_M |  2048 |  **89.5** | **21.8** | 21.7–21.9     |     1398 | **console PASS** · chat upgrade                                         |
| Qwen2.5-Coder-3B     | `qwen25-coder-3b`     | Q4_K_M |  4096 |  **46.2** | **14.0** | 13.9–14.1     | **2116** | **console PASS** · coding quality (under 3.5 GB) · **GPU default (D3)** |

Host Release cross-check (`phase14-host-validation.csv`): quality PASS for the
coding tier + Qwen3-1.7B (Q3_K_M Coder-3B FAIL; **Q4 only** ships). Console peaks
are the product figures (Coder-3B **2116 MB**).

Thinking: catalogue `lfm25-1.2b-thinking` with `model_is_thinking` +
`strip_thinking_blocks` in `postprocess_output` (display/persist answer only).
Selecting the model applies catalogue **`n_predict: 1024`** (UI default was 512).
Short easy prompts complete under that budget (`thinkdone` gate); multi-step
arithmetic can still exhaust CoT without closing `</think>` (reasoning-only
stand-in) — not a formatting bug (#223).
⚠️ **The `console PASS` above is a load-and-decode result, not H9 quality.**
This model is **absent from the H9 suite** (which evaluates the _instruct_
sibling). **#223 closed:** short prompts complete under catalogue **n_predict
1024** (`thinkdone` gate, Series S); multi-step arithmetic can still exhaust
CoT without closing `</think>` (measured train prompt @ 768 → reasoning-only
stand-in). Evidence: `bench/results/phase15-thinking-complete.csv`.

The tok/s above were recorded on MSIX **1.5.1.737**, which predates the phase14
code. The code path itself — catalogue `n_ctx`/`role` at session open, the
Qwen2.5/Qwen3 template split, think stripping — was validated separately on
**1.5.1.759** with the console gate suite (`gguf`, `routing`, `longchat`, `kvsnap`,
`genroom`, `coderpaste`, `thinkcut`; see [architecture.md](architecture.md) for what
each asserts). Two capability figures come from that run rather than from a bench:
a coding session prefilled **3437 tokens in chunks** past the 2048 logical batch
without aborting, and a 24 KB paste was refused with `prompt too long` while the
app stayed up.

**Re-validated on the shipped package** (MSIX **1.5.2.789**, the artifact attached
to the v1.5.2.0 release, built from the tagged commit): all **eight** gates PASS,
the seven above plus `settings`. The capability figures are reproduced —
3437 tokens chunked, the 24 KB paste refused — and `genroom` measured 1535 tokens
exact after trimming with **513 left for a requested 512**, the fix for the silent
~250-token truncation. `routing` is the one worth naming: it had been green over a
dead feature because it pinned `n_predict=128`, and now runs at the shipping
default (long turn → GPU at 1769 tok, short → CPU at 23).

**Re-validated on v1.5.3.0** (MSIX **1.5.3.873**, release asset for tag `v1.5.3.0`,
Series S 2026-08-08): all **nine** console gates PASS
(`routing`, `settings`, `gguf`, `longchat`, `kvsnap`, `coderpaste`, `thinkcut`,
`genroom`, `taesd`). The `gguf` gate now also asserts a **non-empty conversation
title** on disk (empty-title fix in this cut). Capability figures from 1.5.2
still hold on this package.

**Post-1.5.3 hygiene (shipped in **v1.5.4.0**, MSIX **1.5.4.887**):** **ten**
console gates — the nine above plus **`thinkdone`** (#223). `thinkcut` +
`thinkdone` both PASS; `kvsnap` stress **6× `all` PASS** after the #216 save
race fix (PR #232). Intermediate verify on 1.5.3.881 before the 1.5.4 cut.

**Out of budget / deferred:** Qwen2.5-Coder-7B+, Qwen3-Coder MoE 30B+, Devstral 24B,
DeepSeek-Coder-V2-Lite, StarCoder2 / DS-Coder 1.3B. LFM2.5-8B-A1B was on this list
at ~5 GB (its official Q4_K_M); it moved to A3 below once the heap ceiling was
measured and a 3.57 GB quant turned out to be admissible.

### A3. Catalogued, measured, not shipping

An entry lands here when it is provisionable but is **not** part of a product tier —
the catalogue is what the app can be pointed at, not a claim that it should be. A
row keeps its measurement so the negative result is not re-derived by the next
person who wonders.

| Model         | Catalogue      | Quant    | Weights |    Est. peak | Status                                                                                                                      |
| ------------- | -------------- | -------- | ------: | -----------: | --------------------------------------------------------------------------------------------------------------------------- |
| LFM2.5-8B-A1B | `lfm25-8b-a1b` | UD-IQ3_S | 3571 MB | **3553 MiB** | **H2 FAIL** (2026-07-30) · 32 experts / 4 active · 14.50 tok/s = the dense 3B at +1437 MiB, and ~4× worse perceived latency |

Why it is admissible at all: the console heap ceiling was measured at 4864 MB
committed (`phase15-ramceil`), and that is what decided IQ3_S over Q2 — a Q2 result
would have indicted the quantization rather than the architecture. The arch already
compiles into the UWP static lib via the `src/models/*.cpp` wildcard. Two risks are
on record before the run: the ~4.0 GB estimate **breaks the 3.5 GB product gate**
even though it clears H2's 4 GB one, and the model reasons on every turn without
saying so in its name, so it is wired as a thinking model.

### A4. Catalogued, console T3 pending

These ids are in [`../uwp/models/manifest.json`](../uwp/models/manifest.json)
(in-app download, Device Portal `provision-models.sh`, Ollama `POST /api/pull`)
but have **no Series S bench**. Host smoke is not a console claim. Measurement
harness: [console-validation-runbook.md](./console-validation-runbook.md)
(Campaign T3 — new GGUF catalogue candidate).

None. The 2026-09-30 gates (`2026-09-30-catalogue-gates`) closed every row:
`phi4-mini` moved to §A1, MiniCPM5-1B and Gemma-3-1B to §F.

---

## B. Diffusion

| Model         | Catalogue       | Role  | Console        | Notes                            |
| ------------- | --------------- | ----- | -------------- | -------------------------------- |
| SD-Turbo fp16 | `sd-turbo-fp16` | image | PASS (Phase 5) | 3-stage DirectML; optional TAESD |

Numbers: [benchmarks.md](./benchmarks.md) / `phase5-diffuse.csv`.

---

## C. Training / personalize

| Artefact                    | Catalogue                            | Status                          |
| --------------------------- | ------------------------------------ | ------------------------------- |
| Personalized adapter/merged | `personalized` (LocalState override) | Phase 11 UI + Lane B gates PASS |
| Runtime LoRA                | any `kind:gguf` + `lora` field       | Supported (llama.cpp only)      |

---

## D. Capability matrix by architecture class

`can_shift` is a **runtime** property (`llama_memory_can_shift` && `n_swa == 0`) and
the app logs it on every load (`[xllama] session: can_shift=…`), so this column is
measured per GGUF arch instead of inferred from the family name. Measured
2026-07-29 (per-arch, hence platform-independent): `lfm2` **1**, `qwen3` **1**,
`qwen35` **0**. The previous version of this table assumed Qwen3 behaved like
Qwen3.5 and said **no** for both — wrong, and it under-sold `qwen3-1.7b`, which
shifts.

| Arch class                                 | Examples in tree                                       | KV shift         | KV tail rewind | ORT     | GGUF           | Notes                  |
| ------------------------------------------ | ------------------------------------------------------ | ---------------- | -------------- | ------- | -------------- | ---------------------- |
| Dense standard (`llama`, `qwen2`, `qwen3`) | Llama-3.2, Qwen2.5-Coder, SmolLM2 GGUF, **Qwen3-1.7B** | yes\*            | yes            | if ONNX | yes            | \*if not SWA           |
| Hybrid attn+recurrent (`lfm2`)             | LFM2 / LFM2.5 / Thinking                               | yes (front-drop) | **no**         | no      | yes            | #170a degrade          |
| imrope / mrope (`qwen35`)                  | Qwen3.5                                                | **no**           | N/A            | no      | yes            | #169 fail-fast         |
| SWA                                        | some modern                                            | **no**           | careful        | —       | if arch in pin | `n_swa==0` gate        |
| MoE small active                           | `lfm25-8b-a1b` (§A3, not shipping)                     | TBD              | TBD            | no      | **H2 FAIL**    | need ≤~3.5 GB GGUF; §I |
| BitNet 1.58                                | —                                                      | —                | —              | no      | H5 desk        | not shipping; §I       |

Reopen gates for MoE, BitNet, and SSM/RWKV + small multimodal live in [§I](#i-arch-watch),
not in this capability table. Those watches do not reopen the rejected H2 / H5
models.

---

## E. Product roles (picker intent)

| Role                             | Catalogue ids                             | Product note                                                                                     |
| -------------------------------- | ----------------------------------------- | ------------------------------------------------------------------------------------------------ |
| Floor (smallest shipping)        | `lfm25-230m`                              | Fastest and lightest chat (132.4 tok/s, 236 MB, QAD Q4_0); H9 1/8, two tasks below `gemma3-270m` |
| Default chat (unified)           | `lfm25-350m`                              | First launch                                                                                     |
| Balanced / quality chat          | `lfm25-1.2b-instruct`, `lfm2-2.6b`        | H1 tiers                                                                                         |
| Peer dense chat                  | `llama32-3b`, `gemma4-e2b`                | Advanced / heavy                                                                                 |
| Coding fast / balanced / quality | `qwen25-coder-0.5b`, `…-1.5b`, `…-3b`     | `role:coding`, `n_ctx` 4096                                                                      |
| Chat upgrade (Qwen3)             | `qwen3-1.7b`                              | no-think; context shift OK (measured)                                                            |
| Reasoning                        | `lfm25-1.2b-thinking`                     | CoT stripped for display                                                                         |
| ORT routing pair                 | `smollm2-360m-cpu-int4` + `…-dml-fp16-v2` | Auto GPU only on long first turn                                                                 |
| Image                            | `sd-turbo-fp16`                           | Image dialog                                                                                     |
| Personalized                     | `personalized`                            | After on-device train                                                                            |

---

## F. Survey rejected / deferred (desk 2026-07-27)

| Candidate                  | Decision           | Reason                                                                       |
| -------------------------- | ------------------ | ---------------------------------------------------------------------------- |
| Qwen3-Coder-30B-A3B GGUF   | reject for console | weight size                                                                  |
| Devstral Small             | reject             | weight size                                                                  |
| DeepSeek-Coder-V2-Lite     | reject             | weight size                                                                  |
| Qwen2.5-Coder-7B           | defer              | interactive decode too low                                                   |
| StarCoder2-3B              | defer              | weaker than Qwen2.5-Coder-3B                                                 |
| Phi-4-mini                 | shipped — T3 PASS  | `phi4-mini` (#268): H9 6/8 vs `llama32-3b` 5/8, 11.2 tok/s, 2765 MB, see §A1 |
| Qwen3-1.7B/4B general      | defer              | chat upgrade, not coding                                                     |
| Granite 3.1 2B / 4.0 micro | defer              | generalist; lower priority                                                   |

**Phase 16 (desk + measured, 2026-08-10)** — campaign SSOT:
[phase16-model-scouting.md](phase16-model-scouting.md).

| Candidate                        | Decision                 | Reason                                                                                                                                                                                            |
| -------------------------------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Qwen3.5-2B (`unsloth` Q4_K_M)    | reject — measured        | H16.1a FAIL on both halves: `peak_ws_mb` 1421 > 1398 and decode 19.25 < 19.6 (`phase16-gguf`)                                                                                                     |
| Maincoder-1B (`Maincode` Q4_K_M) | reject — measured        | H16.1b: memory passes (843 ≤ 900) but decode 33.49 < the card's 33.9 (1.283× vs a 1.3× bar) (`phase16-gguf`)                                                                                      |
| MiniCPM5-1B (`openbmb`)          | reject — measured        | T3 2026-09-30 (#267): decode 39.7 ≥ 34.1 and peak 804 ≤ 811 pass, but H9 **3/8** < 6/8 (wrong answers, no `<think>` leak). Removed from the catalogue (`2026-09-30-catalogue-gates`)              |
| Gemma-3-1B (ggml-org)            | reject — measured        | T3 2026-09-30 (#269): dominated by `lfm25-1.2b-instruct`: H9 5/8, decode 35.1, peak 929 MB (all worse than 6/8 / 40.0 / 783); early EOG on the 512-token bench prompt. Removed from the catalogue |
| SmolLM3-3B GGUF (ggml-org)       | reject — cost not bought | WS-A's ≤4 slots allocated; loses the 4th head-to-head to MiniCPM5-1B; self-set PASS bar of H9 8/8 (catalogue best is 7/8)                                                                         |
| EmbeddingGemma-300M (ggml-org)   | reject — licence         | Origin repo `gated: manual` (anon HEAD 401), so a console download is impossible; the ungated mirror ships no licence/notice                                                                      |
| Supra2-100M-Instruct             | reject — duplicate bet   | Same floor role as `lfm25-230m`; ~30 MB of projected gain needs an in-house quant, a re-host and a name-matched stop token                                                                        |
| granite-embedding-english-r2     | reject — duplicate bet   | Same WS-E slot as nomic-embed-text-v1.5 at +15 MB and new-renderer cost; its own FAIL branch concedes to nomic                                                                                    |
| multilingual-e5-small            | reject — duplicate bet   | Same WS-E slot; the multilingual axis has no named consumer, and its GGUF tokenizer is SPM over an XLM-R Unigram vocabulary                                                                       |

**Shape gate (measured 2026-09-30)** — SSOT:
[shape-gate-plan.md](shape-gate-plan.md). Do not reopen without a new fact
(new backend, new quant, new operator).

| Candidate                                    | Decision          | Reason                                                                                                                                                                  |
| -------------------------------------------- | ----------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Custom `lfm2` shape at ~1.1B (depth 8–32)    | reject — measured | G1 FAIL: best iso-parameter depth decodes 1.024× the random LFM2.5-1.2B clone (bar 1.30×). Decode at ~1B is bytes-per-token bound; depth moves it ±4% (`shape-gate`)    |
| Custom `lfm2` vocabulary (32768 / 16384 ids) | reject — measured | G2 FAIL: decode 1.078× / 1.129×, but the same text needs 1.103× / 1.242× the tokens → 0.978× / 0.909× text throughput. Gain equals the bytes saved, no hidden head cost |

## G. Embedding API catalogue (host and Series S validated)

Release CPU smoke on 2026-09-23 used `xllama-cli --embed`, a retrieval-style
input, and `/proc/<pid>/status` `VmHWM` for peak RSS. All three candidates
loaded and returned finite, unit-normalized native-width vectors. Only the two
under the 3584 MiB host gate are catalogued. These host figures are not Xbox
measurements. Series S protocol and maximum-input memory validation passed on
2026-09-28; the source/package-bound outcomes live in
[`2026-09-28-console-api-validation.json`](../bench/results/2026-09-28-console-api-validation.json).
BGE-M3 and Nomic each pass 17 contract requests on the main MSVC package.
The maximum BGE input is bounded by the API logical batch, not its larger
catalogue context; see [API limits](api-endpoint.md#embeddings-apiembed-apiembeddings-v1embeddings).
Dedicated embedding throughput and retrieval-quality benchmarks remain open.
Process high-water marks do not establish a separate Nomic peak-memory bound.

| Model                   | Catalogue            | Arch           | Quant  | Pooling | n_ctx | Width | License    | Host peak RSS | Status                                                     |
| ----------------------- | -------------------- | -------------- | ------ | ------- | ----: | ----: | ---------- | ------------: | ---------------------------------------------------------- |
| BGE-M3                  | `embed-bge-m3`       | bert           | Q8_0   | CLS     |  8192 |  1024 | MIT        |       746 MiB | **host + Series S contract PASS**                          |
| Nomic Embed Text v2 MoE | `embed-nomic-v2-moe` | nomic-bert-moe | Q8_0   | mean    |   512 |   768 | Apache-2.0 |       632 MiB | **host + Series S contract PASS** · task prefixes required |
| Qwen3-Embedding-4B      | —                    | qwen3          | Q4_K_M | last    |  2048 |  2560 | Apache-2.0 |      4411 MiB | **not catalogued** · exceeds 3584 MiB host gate            |

Model SHA-256 pins: BGE-M3 `950f4a8e5e19477a6d3c26d2f162233c20002c601f75e4b002e3239997821167`, Nomic `36c5817bc25f379e62021f49efde05b10ed3b0c93ab8059c43173a7a5de73565`. Qwen candidate pin (not catalogued): `2b0cf8f17b4c723c27303015383c27ec4bf2d8314bb677d05e920dd70bb0f16b`.

---

## H. Gaps still open

1. ~~Console campaign phase14~~ — **done**.
2. ~~Thinking product path~~ — **done** (`model_is_thinking` + strip for display).
3. ~~**Ship MSIX** with phase14 code~~ — **done**; shipped in v1.5.2.0 and
   carried through the current cut (see [`../CHANGELOG.md`](../CHANGELOG.md)).
4. ~~**H3 speculative decoding**~~ — **done as opt-in** (Phase 15 W2 #210:
   draft-free prompt-lookup k=2). Console M3 **1.04× FAIL** product-default gate
   on `qwen25-coder-3b`; stays OFF. See [phase15-re-opt.md](phase15-re-opt.md).
5. **FIM / completions API** — out of scope (second prompt surface).
6. ~~Next models~~ — **done 2026-08-10.** The Phase 16 campaign
   ([phase16-model-scouting.md](phase16-model-scouting.md)) closed with one model
   shipped (`lfm25-230m`, §A1) and everything else rejected, deferred or blocked;
   the verdicts are in §F. Of the 2026-07-27 seeds, LFM2.5-230M shipped and the
   rest were not displaced. WS-E (embeddings) now has a named LAN API consumer;
   BGE-M3 and Nomic v2 MoE pass host screening and Series S contract validation
   (see §G); dedicated throughput/retrieval-quality measurements remain open. WS-F (ASR): microphone capture PASS on Series S 2026-09-30 (#241). No ASR model is catalogued yet.
7. ~~**W3 gpubw** (#211)~~ — **closed PASS** Series S **119.07 GB/s** STREAM;
   H6 eng **#228** (`docs/phase15-re-opt.md`).
8. ~~**MiniCPM5-1B Series S T3**~~ — **closed** 2026-09-30: reject — measured
   (H9 3/8), see §F (#267).
9. **Arch-watch** — desk-only reopen gates in [§I](#i-arch-watch) (#273 MoE,
   #274 BitNet, #275 SSM/RWKV + mm). Not a catalogue seed.

---

## I. Arch-watch

Standing desk watches for architectures that were measured or surveyed and
**must not re-enter a product tier** unless every listed gate holds. These are
not catalogue seeds and do **not** reopen the rejected H2 / H5 models. A
candidate that appears still follows the ladder in
[architecture.md](./architecture.md) (host smoke, then console bench, then
manifest). No new catalogue id without a Series S PASS.

Tracking: [#273](https://github.com/gianlucamazza/xllama/issues/273) MoE,
[#274](https://github.com/gianlucamazza/xllama/issues/274) BitNet 1.58,
[#275](https://github.com/gianlucamazza/xllama/issues/275) SSM/RWKV + small
multimodal. Verdicts: [phase7-hypotheses.md](./phase7-hypotheses.md) H2 / H5.

### I1. MoE (post H2 FAIL)

Closed: H2 “decode scales with active weights” **FAIL** on Series S
(2026-07-30). `lfm25-8b-a1b` matched the dense 3B on decode, paid +~1.4 GB, and
was ~4× worse in perceived latency. Result stays in §A3. Do not re-argue that
run. Gate: #273.

**Do not reopen unless all hold:**

- Official (or pin-compatible) GGUF on the current llama.cpp pin — **no fork**.
- Peak working set **≤ ~3.5 GB** at product `n_ctx` (Series S Dev Mode envelope).
- Decode **strictly above** the shipping dense peer at the same quality bar
  (Llama-3.2-3B / Phi-4-mini class), not “≈ active-param dense”.
- Prefill / perceived TTFT not several× worse than the dense peer for typical
  chat turns.
- Speculative draft, if any, must be a real catalogue draft with matching vocab
  — H2+H3 do not compose on orphan MoEs.

Out of scope: lowering quant on 8B-A1B to “make H2 pass” (would indict quant,
not the arch claim). granite-3.1-3b-a800m already rejected on the same pass.

**Scan 2026-09-30 (pin `7fe450e`): no candidate.** Maple-Preview 20B-A1B
(arch `maple` is on the pin) ships nothing smaller than a 4.98 GB TQ1_0 GGUF,
above the envelope. AliceAI-T5-35B-A0.6B and Nemotron-3.5-Lightning-30B-A3B
fail on total size. New supporting fact: at ~1B, dense decode on Series S is
bytes-per-token bound (shape gate, see §F), which is the bar any MoE must beat
on bytes actually read, not on active parameters.

### I2. BitNet 1.58 (post H5 NO-GO)

Closed: H5 **NO-GO** on absence of artefact, not on measured merit (2026-08-10,
Phase 15 M8). The pin already carries `bitnet`. No downloadable sub-4B weights
trained at ≤2 bits. Gate: #274.

**Do not reopen unless all hold:**

- Public weights for a **sub-4B** model **trained** at ≤2 bits (not PTQ of
  BF16, and not post-hoc IQ2).
- GGUF loads on the **current product pin** — not a bitnet.cpp / custom fork.
- Envelope target from H5: roughly 400–800 MB class aiming ≥20 tok/s on
  Series S (re-measure; those numbers are hypotheses).
- Host smoke + console T-gate before any catalogue id.

Out of scope: 8B–20B ternary MoE, robotics VLA, diffusion LMs with custom
runtimes, recipe-only papers. Post-hoc IQ2 on a normal model is a different
(discouraged) bet — IQ2_M garbage precedent.

**Scan 2026-09-30 (pin `7fe450e`): no candidate.** No new sub-4B checkpoint
trained at ≤2 bits. Ternary-Bonsai-1.7B/4B and Uluka-Comet-1.5B are
derived from BF16 Qwen models (out by the first rule). The only new
bitnet-b1.58-2B-4T files are community requants, and one of them names its own
llama.cpp fork. The pin carries `GGML_TYPE_Q2_0` (block 64, since before `b29c606`); a
checkpoint in that type would still have to pass the first rule.

### I3. SSM/RWKV + small multimodal (desk)

Track non-Transformer / hybrid and small multimodal options that might fit
Series S without repeating closed kills. **Do not reopen unless** the arch is
on the UWP llama.cpp pin, a GGUF exists, and the Series S envelope holds.
Diffusion **text** and custom runtimes stay out (WS-D closed; image SD-Turbo
remains product). Gate: #275.

| Class                      | Interest            | Do not reopen unless                                                                 |
| -------------------------- | ------------------- | ------------------------------------------------------------------------------------ |
| SSM / Mamba-class          | Long-ctx efficiency | Arch in UWP llama.cpp pin + GGUF ≤ envelope                                          |
| RWKV                       | RNN-ish decode      | Same — pin + GGUF + console                                                          |
| Gemma-3-4B + mmproj        | Vision              | Peak + mmproj often breaks Series S; text-only 1B rejected — measured, see §F (#269) |
| Diffusion LM (byte / mask) | Novelty             | Custom runtime ≠ product pin — reject unless llama.cpp path                          |

**Scan 2026-09-30 (pin `7fe450e`), with host smoke on the new pin:**

| Candidate                                            | Verdict                   | Reason                                                                                                                                                                                                                                                                         |
| ---------------------------------------------------- | ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| DFM Mimir v1.5 (`hrm_text`, 1.8B, Q4_K_M 1.17 GB)    | reject — host smoke FAIL  | First loadable on `7fe450e`. Stock pin generates incoherent text, with or without a template (upstream is causal-only; the model is a PrefixLM, and its GGUF card says stock compatibility "is not implied"). KV 1.5 GB at `n_ctx` 2048 puts it at ~3.2 GB, at the envelope.   |
| RWKV7-G1j 1.5B / 2.9B (`rwkv7`, Q4_K_M 1.02/1.92 GB) | defer — no product reason | Loads and generates coherent text on the pin. It is a reasoning (`<think>`) model, so it pays the thinking tax (§A3), and at ~1 GB its decode is bytes-bound like the dense peers. Its only edge is O(1) state for long chats; reopen when a long-context surface asks for it. |
| Mamba-3, Mage-VL                                     | reject                    | Arch not on the pin                                                                                                                                                                                                                                                            |
| LFM2.5-VL-3B                                         | reject                    | 2.6B text tower plus mmproj; above the small-multimodal class                                                                                                                                                                                                                  |

No catalogue ids without a console PASS against the shipping LFM2.5 peers
(`lfm25-1.2b-instruct` at 1B; MiniCPM5-1B was rejected, §F). Nearby
closed kills (do not re-argue): diffusion **image** SD-Turbo remains product;
diffusion **text** WS-D closed; ORT DML int4 text reject; speculative default
FAIL (opt-in only).

---

## See also

- Generated tok/s table → [benchmarks.md](./benchmarks.md)
- Catalogue data → [`../uwp/models/manifest.json`](../uwp/models/manifest.json)
- Selection / add-your-own → [model-selection.md](./model-selection.md)
- Phase 7 research → [phase7-hypotheses.md](./phase7-hypotheses.md)
- Runtime structure → [architecture.md](./architecture.md)
- Arch-watch issues → #273 · #274 · #275
