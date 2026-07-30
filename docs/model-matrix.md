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

Last updated: **2026-07-27** (phase14 **console** validation on Series S).

## How to read the columns

| Column              | Meaning                                                                     |
| ------------------- | --------------------------------------------------------------------------- |
| **Catalogue**       | Id in `uwp/models/manifest.json` (`—` = not shipped)                        |
| **Status**          | `shipping` / `catalogue` / `campaign-only` / `host-smoke` / `rejected`      |
| **Console metrics** | Xbox Series S Dev Mode (see [benchmarks.md](./benchmarks.md))               |
| **H9**              | Deterministic 8-task suite (`phase7-h9.jsonl`), temperature 0               |
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

| Model                 |                  Catalogue |  Params | Quant  | Backend   |   Prefill |   Decode | Peak MB | H9      | Template          | Role | n_ctx | Evidence                                         |
| --------------------- | -------------------------: | ------: | ------ | --------- | --------: | -------: | ------: | ------- | ----------------- | ---- | ----: | ------------------------------------------------ |
| LFM2.5-350M           |               `lfm25-350m` |    350M | Q4_K_M | llama.cpp | **438.1** | **94.9** |     320 | 4/8     | ChatML            | —    |  2048 | `phase13b-threadsbatch-after` · **default chat** |
| Gemma-3-270M          |              `gemma3-270m` |    270M | Q4_K_M | llama.cpp |     395.0 |     76.8 |     368 | —       | Gemma             | —    |  2048 | `phase6-gemma`                                   |
| SmolLM2-360M          |    `smollm2-360m-cpu-int4` |    360M | int4   | ORT CPU   |     262.4 |     74.8 |     708 | —       | ChatML            | —    |  2048 | `t6-shipped-confirm`                             |
| SmolLM2-360M          |              (same family) |    360M | Q4_K_M | llama.cpp |     141.5 |     62.9 |     402 | —       | ChatML            | —    |  2048 | `phase35-llamacpp-scaling`                       |
| SmolLM2-360M DML v2   | `smollm2-360m-dml-fp16-v2` |    360M | fp16   | ORT DML   |     236.7 |     44.4 |    1268 | —       | ChatML            | —    |  2048 | `phase2-dml` · #91 parity OK                     |
| LFM2.5-1.2B Instruct  |      `lfm25-1.2b-instruct` |    1.2B | Q4_K_M | llama.cpp |      76.2 | **37.9** |     811 | **6/8** | ChatML            | —    |  2048 | `phase7-lfm` · H1 PASS balanced                  |
| Qwen3.5-0.8B          |              `qwen35-0.8b` |    0.8B | Q4_K_M | llama.cpp |      98.1 |     35.1 |     718 | —       | ChatML + no-think | —    |  2048 | `phase5-gguf`                                    |
| SmolLM2-1.7B          |    `smollm2-1.7b-cpu-int4` |    1.7B | int4   | ORT CPU   |      54.9 |     20.6 |    2423 | —       | ChatML            | —    |  2048 | `phase35-1b-cpu`                                 |
| LFM2-2.6B             |                `lfm2-2.6b` |    2.6B | Q4_K_M | llama.cpp |      32.0 | **18.4** |    1623 | **7/8** | ChatML            | —    |  2048 | `phase7-lfm` · H1 PASS quality                   |
| Gemma-4-E2B           |               `gemma4-e2b` | ~2B eff | Q3_K_S | llama.cpp |      26.1 |     15.3 |    2742 | 6/8     | Gemma             | —    |  2048 | `phase6-gemma`                                   |
| Llama-3.2-3B Instruct |               `llama32-3b` |      3B | Q3_K_S | llama.cpp |      19.5 | **14.2** |    1824 | 5/8     | Llama-3           | —    |  2048 | `phase7-scale` · H4 preferred                    |
| Phi-3.5-mini          |                          — |    3.8B | Q3_K_S | llama.cpp |      15.3 |     11.3 |    2453 | —       | Phi-3             | —    |  2048 | `phase7-scale` · H4 PASS, loses A/B              |
| Gemma-4-E2B IQ2       |            (upgraded away) | ~2B eff | IQ2_M  | llama.cpp |      13.5 |      9.9 |    2534 | —       | Gemma             | —    |  2048 | historical; EOG on long prompts                  |
| SmolLM2-360M DML int4 |                          — |    360M | int4   | ORT DML   |     0–153 |  **8.8** |     999 | —       | ChatML            | —    |  2048 | **rejected** wrong logits / slow                 |

Notes:

- Hybrid LFM: KV tail-rewind unsupported (#170a); front-drop context shift OK (#169).
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

| Model                | Catalogue             | Quant  | n_ctx |   Prefill |   Decode | Decode spread |  Peak MB | Status                                                  |
| -------------------- | --------------------- | ------ | ----: | --------: | -------: | ------------- | -------: | ------------------------------------------------------- |
| Qwen2.5-Coder-0.5B   | `qwen25-coder-0.5b`   | Q4_K_M |  4096 | **148.2** | **62.4** | 56.8–68.0     |      533 | **console PASS** · coding fast                          |
| LFM2.5-1.2B-Thinking | `lfm25-1.2b-thinking` | Q4_K_M |  2048 | **130.4** | **36.7** | 36.7–36.8     |      811 | **console PASS** · `strip_thinking_content` for display |
| Qwen2.5-Coder-1.5B   | `qwen25-coder-1.5b`   | Q4_K_M |  4096 |  **96.6** | **26.1** | 25.7–26.5     |     1179 | **console PASS** · coding balanced                      |
| Qwen3-1.7B           | `qwen3-1.7b`          | Q4_K_M |  2048 |  **89.5** | **21.8** | 21.7–21.9     |     1398 | **console PASS** · chat upgrade                         |
| Qwen2.5-Coder-3B     | `qwen25-coder-3b`     | Q4_K_M |  4096 |  **46.2** | **14.0** | 13.9–14.1     | **2116** | **console PASS** · coding quality (under 3.5 GB)        |

Host Release cross-check (`phase14-host-validation.csv`): quality PASS for the
coding tier + Qwen3-1.7B (Q3_K_M Coder-3B FAIL; **Q4 only** ships). Console peaks
are the product figures (Coder-3B **2116 MB**).

Thinking: catalogue `lfm25-1.2b-thinking` with `model_is_thinking` +
`strip_thinking_blocks` in `postprocess_output` (display/persist answer only).
⚠️ **The `console PASS` above is a load-and-decode result, not a quality one.**
This model is **absent from the H9 suite** (which evaluates the _instruct_
sibling), and nothing yet shows a reasoning turn completing: on console it spent
**768 tokens** on a two-step time addition without closing `<think>`, so the UI
showed the truncated-reasoning stand-in. Open as
[#223](https://github.com/gianlucamazza/xllama/issues/223); the tier stays in the
catalogue but its usable budget is unmeasured.

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

| Arch class                                 | Examples in tree                                       | KV shift         | KV tail rewind | ORT     | GGUF           | Notes              |
| ------------------------------------------ | ------------------------------------------------------ | ---------------- | -------------- | ------- | -------------- | ------------------ |
| Dense standard (`llama`, `qwen2`, `qwen3`) | Llama-3.2, Qwen2.5-Coder, SmolLM2 GGUF, **Qwen3-1.7B** | yes\*            | yes            | if ONNX | yes            | \*if not SWA       |
| Hybrid attn+recurrent (`lfm2`)             | LFM2 / LFM2.5 / Thinking                               | yes (front-drop) | **no**         | no      | yes            | #170a degrade      |
| imrope / mrope (`qwen35`)                  | Qwen3.5                                                | **no**           | N/A            | no      | yes            | #169 fail-fast     |
| SWA                                        | some modern                                            | **no**           | careful        | —       | if arch in pin | `n_swa==0` gate    |
| MoE small active                           | (none shipping)                                        | TBD              | TBD            | no      | H2 open        | need ≤~3.5 GB GGUF |
| BitNet 1.58                                | —                                                      | —                | —              | no      | H5 desk        | not shipping       |

---

## E. Product roles (picker intent)

| Role                             | Catalogue ids                             | Product note                          |
| -------------------------------- | ----------------------------------------- | ------------------------------------- |
| Default chat (unified)           | `lfm25-350m`                              | First launch                          |
| Balanced / quality chat          | `lfm25-1.2b-instruct`, `lfm2-2.6b`        | H1 tiers                              |
| Peer dense chat                  | `llama32-3b`, `gemma4-e2b`                | Advanced / heavy                      |
| Coding fast / balanced / quality | `qwen25-coder-0.5b`, `…-1.5b`, `…-3b`     | `role:coding`, `n_ctx` 4096           |
| Chat upgrade (Qwen3)             | `qwen3-1.7b`                              | no-think; context shift OK (measured) |
| Reasoning                        | `lfm25-1.2b-thinking`                     | CoT stripped for display              |
| ORT routing pair                 | `smollm2-360m-cpu-int4` + `…-dml-fp16-v2` | Auto GPU only on long first turn      |
| Image                            | `sd-turbo-fp16`                           | Image dialog                          |
| Personalized                     | `personalized`                            | After on-device train                 |

---

## F. Survey rejected / deferred (desk 2026-07-27)

| Candidate                  | Decision           | Reason                                |
| -------------------------- | ------------------ | ------------------------------------- |
| Qwen3-Coder-30B-A3B GGUF   | reject for console | weight size                           |
| Devstral Small             | reject             | weight size                           |
| DeepSeek-Coder-V2-Lite     | reject             | weight size                           |
| Qwen2.5-Coder-7B           | defer              | interactive decode too low            |
| StarCoder2-3B              | defer              | weaker than Qwen2.5-Coder-3B          |
| Phi-4-mini                 | defer (chat peer)  | not coding; re-open if H9 peer needed |
| Qwen3-1.7B/4B general      | defer              | chat upgrade, not coding              |
| Granite 3.1 2B / 4.0 micro | defer              | generalist; lower priority            |

---

## G. Gaps still open

1. ~~Console campaign phase14~~ — **done**.
2. ~~Thinking product path~~ — **done** (`model_is_thinking` + strip for display).
3. **Ship MSIX** with phase14 code (catalogue download, n_ctx/role, think strip).
4. **H3 speculative decoding** (Coder-0.5B draft + 1.5B/3B target).
5. **FIM / completions API** — out of scope (second prompt surface).
6. Optional next models: Qwen3-4B-2507, Phi-4-mini, Gemma-3-1B, LFM2.5-230M.

---

## See also

- Generated tok/s table → [benchmarks.md](./benchmarks.md)
- Catalogue data → [`../uwp/models/manifest.json`](../uwp/models/manifest.json)
- Selection / add-your-own → [model-selection.md](./model-selection.md)
- Phase 7 research → [phase7-hypotheses.md](./phase7-hypotheses.md)
- Runtime structure → [architecture.md](./architecture.md)
