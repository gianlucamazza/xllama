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
- Qwen3.5: `can_shift` false (imrope) — overflow fail-fast + trim, no RoPE shift.
- DML text routing allowlist: only `smollm2-360m-dml-fp16-v2` (`dml_text_model_ok`).

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

Caveats on console MSIX 1.5.1.737 (pre-phase14 code): tok/s valid; new MSIX
needed for catalogue download, n_ctx/role, clean templates, and think stripping.

**Out of budget / deferred:** Qwen2.5-Coder-7B+, Qwen3-Coder MoE 30B+, Devstral 24B,
DeepSeek-Coder-V2-Lite, LFM2.5-8B-A1B (~5 GB), StarCoder2 / DS-Coder 1.3B.

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

| Arch class            | Examples in tree                       | KV shift         | KV tail rewind | ORT     | GGUF           | Notes                                                                 |
| --------------------- | -------------------------------------- | ---------------- | -------------- | ------- | -------------- | --------------------------------------------------------------------- |
| Dense standard        | Llama-3.2, Qwen2.5-Coder, SmolLM2 GGUF | yes\*            | yes            | if ONNX | yes            | \*if not SWA                                                          |
| Hybrid attn+recurrent | LFM2 / LFM2.5 / Thinking               | yes (front-drop) | **no**         | no      | yes            | #170a degrade                                                         |
| imrope / mrope        | Qwen3.5 (measured), Qwen3 (assumed)    | **no**           | N/A            | no      | yes            | #169 fail-fast; `can_shift` is runtime — only `qwen35-0.8b` was gated |
| SWA                   | some modern                            | **no**           | careful        | —       | if arch in pin | `n_swa==0` gate                                                       |
| MoE small active      | (none shipping)                        | TBD              | TBD            | no      | H2 open        | need ≤~3.5 GB GGUF                                                    |
| BitNet 1.58           | —                                      | —                | —              | no      | H5 desk        | not shipping                                                          |

---

## E. Product roles (picker intent)

| Role                             | Catalogue ids                             | Product note                     |
| -------------------------------- | ----------------------------------------- | -------------------------------- |
| Default chat (unified)           | `lfm25-350m`                              | First launch                     |
| Balanced / quality chat          | `lfm25-1.2b-instruct`, `lfm2-2.6b`        | H1 tiers                         |
| Peer dense chat                  | `llama32-3b`, `gemma4-e2b`                | Advanced / heavy                 |
| Coding fast / balanced / quality | `qwen25-coder-0.5b`, `…-1.5b`, `…-3b`     | `role:coding`, `n_ctx` 4096      |
| Chat upgrade (Qwen3)             | `qwen3-1.7b`                              | no-think; shift unmeasured       |
| Reasoning                        | `lfm25-1.2b-thinking`                     | CoT stripped for display         |
| ORT routing pair                 | `smollm2-360m-cpu-int4` + `…-dml-fp16-v2` | Auto GPU only on long first turn |
| Image                            | `sd-turbo-fp16`                           | Image dialog                     |
| Personalized                     | `personalized`                            | After on-device train            |

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
