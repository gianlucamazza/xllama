# MoE streaming: models to validate

Started 2026-11-16. This desk list records candidate checkpoints and filters
considered for optional SSD expert streaming (#479–#481). The
[MoE engineering program](../superpowers/plans/2026-11-24-moe-engineering-program.md)
owns current sequencing and acceptance. Stage status stays with the
[readiness report](moe-upstream-readiness.md) and the corresponding tracker.
Catalogue admission remains governed by [#273] and
[model-matrix.md §I1](../model-matrix.md#i1-moe-post-h2-fail).

Sizes below come from public file listings and third-party pages (sources at
the end). They are desk figures: read the exact byte count from the chosen
repository before a download, and bind it by SHA-256 in the campaign.

## Architecture and execution evidence, 2026-11-27

The rows describe the specific checkpoints below. A GGUF quantization label describes a file recipe. Expert matrix types, offsets, strides and quantization block sizes still have to be read from each actual tensor. In the pinned loaders, separate gate/up expert weights have GGML dimensions `[embedding, expert_ff, experts]` and down has `[expert_ff, embedding, experts]`. Gemma's optional fused gate/up uses `[embedding, 2 × expert_ff, experts]`. The proposed provider streams only routed matrices. Shared and dense FFNs, router parameters, biases and scales stay resident with the model. Recurrent state belongs to the session. [Pinned model loaders and graphs](https://github.com/ggml-org/llama.cpp/tree/d81235049384534c167caea52b85a694f6103d14/src/models) · [xLlama routed weight names](../../src/bridge/moe_streaming.cpp).

| GGUF architecture | Candidate quant                                | Routed weights                      | Shared FFN                        | Router behavior at the pin                                                                   | Recurrent state               |
| ----------------- | ---------------------------------------------- | ----------------------------------- | --------------------------------- | -------------------------------------------------------------------------------------------- | ----------------------------- |
| `lfm2moe`         | 8B / 24B Q4_K_M; separate resident 8B UD-IQ3_S | Separate                            | None                              | GGUF gating function, selection bias, normalized selected weights and scale                  | Short convolution + attention |
| `bailingmoe2`     | Ling-mini-2.0 Q4_K_M                           | Separate                            | Yes                               | GGUF gating function, optional selection bias, GGUF normalization and scale                  | None                          |
| `bailingmoe3`     | Ling-3.0-tiny Q4_K_M                           | Separate                            | Yes                               | GGUF gating function and selection bias                                                      | Gated delta net + attention   |
| `granitehybrid`   | Granite-4.0-H-Tiny Q3_K_S                      | Separate                            | When shared width is positive     | Softmax, normalized selected weights and scale                                               | Mamba2 + attention            |
| `ernie4_5-moe`    | ERNIE-4.5-21B-A3B Q4_K_M                       | Separate                            | When shared width is positive     | Softmax, optional selection bias, normalized selected weights and scale                      | None                          |
| `smallthinker`    | SmallThinker-21B Q4_0                          | Separate                            | None                              | Router projects the pre-attention input, then passes its logits to the MoE FFN               | None                          |
| `gpt-oss`         | gpt-oss-20b MXFP4                              | Separate                            | None                              | Router bias, softmax over the selected logits, per-expert projection biases                  | None                          |
| `gemma4`          | 26B-A4B Q4_0 / Q4_K_M                          | Separate or fused gate/up plus down | Dense FFN alongside routed branch | Scaled router input, softmax and per-expert projection scaling                               | None                          |
| `afmoe`           | Trinity Mini Q4_K_M candidate                  | Separate                            | When shared count is positive     | GGUF gating function (sigmoid by default), selection bias, normalization and scale from GGUF | None                          |
| `maple`           | Maple-Preview 20B-A1B TQ1_0 / TQ2_0            | Separate                            | None                              | Softmax, normalized selected weights                                                         | None                          |

The [fixture campaign, Task 4](../superpowers/plans/2026-11-16-moe-streaming-all-models.md#task-4-small-moe-fixtures-per-architecture) records a resident host load **and decode** of small random-weight GGUFs for eight families. In the table, **PASS fixture** refers only to that historical run. **PASS checkpoint** identifies an actual model run. **NOT_RUN** means no qualifying evidence for the named path. **BLOCKED** identifies a missing dependency. **DEFERRED** records an intentional scope or sequencing decision.

| Architecture / checkpoint | Host CPU load and execution                   | Full resident vs streamed numerical parity           | Native SSD provider / Windows-UWP acceptance      | Xbox Series S evidence                                                                                                                    |
| ------------------------- | --------------------------------------------- | ---------------------------------------------------- | ------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------- |
| `lfm2moe` 8B Q4_K_M       | PASS fixture and resident reference           | BLOCKED, streamed rows for both cache sizes          | BLOCKED #479 → #480 → #481                        | Fork CPU reached native EOS. The fixed 24B trace capacity produced `partial` / `stop_reason: error`. Resident compute allocation refused. |
| `lfm2moe` 24B Q4_K_M      | PASS fixture; full resident reference BLOCKED | BLOCKED, resident reference and streamed comparisons | BLOCKED #479 → #480 → #481                        | Fork CPU two-cache runs reached EOS.                                                                                                      |
| `lfm2moe` 8B UD-IQ3_S     | PASS checkpoint, resident                     | NOT_RUN for this quant                               | DEFERRED, separate quant from the Q4_K_M campaign | Resident CPU checkpoint measured; experimental catalogue entry.                                                                           |
| `granitehybrid`           | PASS checkpoint, resident                     | NOT_RUN, resident program                            | DEFERRED, resident candidate                      | Resident CPU T3 and paired H9 PASS on their recorded builds.                                                                              |
| `bailingmoe2`             | PASS fixture                                  | NOT_RUN                                              | DEFERRED M4                                       | NOT_RUN                                                                                                                                   |
| `bailingmoe3`             | Pinned loader; fixture and checkpoint NOT_RUN | NOT_RUN                                              | DEFERRED #526, after `bailingmoe2` Task 5         | NOT_RUN                                                                                                                                   |
| `ernie4_5-moe`            | PASS fixture                                  | NOT_RUN                                              | DEFERRED M4                                       | NOT_RUN                                                                                                                                   |
| `smallthinker`            | PASS fixture                                  | NOT_RUN                                              | DEFERRED M4                                       | NOT_RUN                                                                                                                                   |
| `gpt-oss`                 | PASS fixture                                  | NOT_RUN                                              | DEFERRED M4                                       | NOT_RUN                                                                                                                                   |
| `gemma4` 26B-A4B          | PASS fixture                                  | NOT_RUN                                              | DEFERRED M4                                       | NOT_RUN for this MoE checkpoint                                                                                                           |
| `afmoe`                   | PASS fixture                                  | NOT_RUN                                              | DEFERRED M4 / #524                                | NOT_RUN                                                                                                                                   |
| `maple`                   | PASS fixture                                  | NOT_RUN                                              | DEFERRED M4                                       | NOT_RUN                                                                                                                                   |

For every row, selected-expert D3D12 execution on the upstream candidate is **DEFERRED** under M1/#421 and the #522 pin decision; an earlier fork's GPU experiments have their own provenance. Qualifying Series X hardware runs are **NOT_RUN** for these paths. Stable streamed support remains **BLOCKED** on native CPU parity and device acceptance. SmallThinker Q4_0, GPT-OSS MXFP4 and Maple ternary have no selected-expert GPU gate for those types. Granite's product decision belongs to #525. The catalogued Gemma 4 E2B has a different checkpoint and graph from the routed 26B-A4B here. [Program and approvals](../superpowers/plans/2026-11-24-moe-engineering-program.md) · [Stage receipts](moe-upstream-readiness.md) · [Resident model matrix](../model-matrix.md#i1-moe-post-h2-fail) · [Ling-3.0-tiny dependency](https://github.com/gianlucamazza/xllama/issues/526).

## Status, 2026-11-16

| Model                                                                                                                 | Streamed on the console                                                                                                         | Resident on the console                                                                 | Host fixture                                  |
| --------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- | --------------------------------------------- |
| LFM2.5-24B-A2B Q4_K_M                                                                                                 | 0.72 / 1.00 tok/s (2 / 512 slots), EOS ([evidence](evidence/2026-11-16-moe-24b-console/README.md))                              | does not fit                                                                            | `lfm2moe` loads and decodes                   |
| LFM2.5-8B-A1B Q4_K_M                                                                                                  | 1.18 / 2.47 tok/s, native EOS; record flagged by the 24B-sized trace ([evidence](evidence/2026-11-16-moe-8b-console/README.md)) | loads, compute buffer refused ([rows](../../bench/results/2026-11-16-moe-8b/README.md)) | same                                          |
| LFM2.5-8B-A1B UD-IQ3_S (H2)                                                                                           | not run                                                                                                                         | 14.42 tok/s decode, 3560 MiB                                                            | same                                          |
| Ling-mini-2.0, ERNIE-4.5-21B-A3B, SmallThinker-21B, gpt-oss-20b, Gemma 4 26B-A4B, Trinity Mini, Maple-Preview 20B-A1B | needs the provider beyond `lfm2moe` (historical architecture proposal, Task 5)                                                  | —                                                                                       | each architecture's fixture loads and decodes |

Maple-Preview 20B-A1B (`maple`) joins the list: its official TQ1_0/TQ2_0
files (4.98 / 5.9 GB, 1B active) exceed the resident #273 file-size envelope.
A future CPU streaming trial would test whether they fit the runtime budget. Fixtures:
`scripts/make-moe-fixture.py`.

## Filters

A model enters the list only if all hold:

1. **Architecture on the llama.cpp pin** (`d812350`, `src/llama-arch.cpp`).
   Whether the fork's provider defers the routed tensors of that architecture
   is a separate check, done at PR 2 (#480); only `lfm2moe` has been run so
   far.
2. **Fits the LocalState volume.** `Q:` is 30 GiB shared by every sideloaded
   app, and xllama alone held 18.4 GB on 2026-11-15
   ([uwp-constraints.md §9](../uwp-constraints.md#9-disk-budget-developer-mode-partition)).
   Working cap: about 15 GB per checkpoint, after removing other models.
3. **Few active bytes per token.** On a storage-bound path,
   `tok/s ≈ read GB/s ÷ missed active GB`
   ([ssd-inference-assessment.md §2](../ssd-inference-assessment.md#2-the-physics)),
   so active parameters, not total, decide the ceiling. Kept: about 4B active
   or fewer.
4. **A GGUF in a type the stage under test executes.** The fork's GPU bridge
   covers Q4_K and Q6_K only; Q4_0, MXFP4 and TQ1_0 checkpoints can serve the
   CPU stage alone.

## Order of validation

Desk recommendation as of 2026-11-16.

### 0. Initial resident scout: `lfm25-8b-a1b`

The original resident scout used UD-IQ3_S, already measured on Series S (H2). The frozen CPU parity campaign uses **Q4_K_M** for both resident and streamed runs, including both cache sizes. Keep the UD-IQ3_S device measurements in their original comparison. [8B declaration](../../bench/moe-streaming/parity-preparation-8b.json) · [Resident measurements](../model-matrix.md#i1-moe-post-h2-fail).

### 1. Target: LFM2.5-24B-A2B

`lfm2moe`, Q4_K_M 14,526,258,464 bytes (pinned by the fork), about 2B active,
always-thinking. The fork's first Series S CPU runs used it
([community-forks.md](../community-forks.md), vetax-by).
At the disk cap: needs other models removed before the download.

### 2. Second architecture, best bytes per token

| Model             | Arch           | Total / active | GGUF                       | Notes                                                                                           |
| ----------------- | -------------- | -------------- | -------------------------- | ----------------------------------------------------------------------------------------------- |
| Ling-mini-2.0     | `bailingmoe2`  | 16.3B / 1.4B   | Q4_K_M ≈ 10.0 GB, official | 256 fine-grained experts: small units for the cache; fewest active bytes listed                 |
| ERNIE-4.5-21B-A3B | `ernie4_5-moe` | 21B / 3B       | Q4_K_M ≈ 13.5 GB           | Instruct and Thinking variants                                                                  |
| SmallThinker-21B  | `smallthinker` | 21B / 3B       | Q4_0 only, third-party     | Designed for storage offload (router before attention, so experts can prefetch); CPU stage only |
| gpt-oss-20b       | `gpt-oss`      | 21B / 3.6B     | MXFP4 ≈ 12.1 GB            | Native MXFP4: CPU stage only                                                                    |

Ling-mini-2.0 goes first: a different routing architecture with the smallest
active set, and an official GGUF in a GPU-bridge type.

### 3. Borderline on disk

| Model           | Arch     | Total / active | GGUF                                            | Blocker                                                                 |
| --------------- | -------- | -------------- | ----------------------------------------------- | ----------------------------------------------------------------------- |
| Gemma 4 26B-A4B | `gemma4` | 26B / 4B       | Q4_0 (QAT, official) 14.44 GB; Q4_K_M ≈ 16.8 GB | Q4_0 is CPU-only; Q4_K_M exceeds the cap; most active bytes in the list |
| Trinity Mini    | `afmoe`  | 26B / 3B       | Q4_K_M ≈ 15 GB (estimate)                       | Size unverified; licence stated differently on two pages                |

### Not on the list

- **Over the disk cap at Q4:** Qwen3.5-35B-A3B and Qwen3.6-35B-A3B
  (Q4_K_M ≈ 20.8 GB), Qwen3-30B-A3B (≈ 17.4 GB), Qwen3-Coder-30B-A3B
  (≈ 18.6 GB), Nemotron 3 Nano 30B-A3B (24.66 GB), Nemotron-3.5-Lightning
  30B-A3B and AliceAI-T5-35B-A0.6B (#273 scan, 2026-09-30).
- **Resident candidates:** Granite-4.0-H-Tiny was evaluated at Q3_K_S; its official Q4_K_M exceeds the resident gate in file bytes alone. OLMoE-1B-7B-0125 and Trinity Nano are also tracked through #273 and #524. The table above records the checkpoint-specific evidence.
- **Too many active parameters:** anything above about 4B active (for
  example Granite-4.0-H-Small 32B-A9B, Phi-3.5-MoE 42B-A6.6B).

### Header qualification for Qwen, DeepSeek and Ling (2026-11-28)

The pinned source recognizes all three GGUF architectures. These five artifacts still need the native CPU provider in #480. The GGUF file length and LFS digest are publisher metadata. A verified HTTP byte range covering each complete header supplied the tensor names, sizes, third-dimension strides and quantization types. The tensor payload itself was not downloaded. Sizes in the following table are bytes. The **resident tensor floor** sums all tensor payloads except the routed gate, up and down matrices. It excludes KV or recurrent state, expert cache, repacking, workspace, allocator overhead and the rest of the application.

| Checkpoint and exact conversion revision                                                                                                                      | Total / active parameters, weight licence |     File bytes | Resident tensor floor | Routed bytes on one all-miss decode position | CPU routed types observed                      |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------- | -------------: | --------------------: | -------------------------------------------: | ---------------------------------------------- |
| [Qwen3-30B-A3B UD-IQ2_XXS](https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF/tree/d5b1d57bd0b504ac62ae6c725904e96ef228dc74)                                   | 30.5B / 3.3B, Apache-2.0                  | 10,362,262,080 |           996,177,920 |                                  585,007,104 | IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS |
| [Qwen3-Coder-30B-A3B-Instruct UD-IQ2_XXS](https://huggingface.co/unsloth/Qwen3-Coder-30B-A3B-Instruct-GGUF/tree/b17cb02dd882d5b6ab62fc777ad2995f19668350)     | 30.5B / 3.3B, Apache-2.0                  | 10,333,691,040 |           997,488,640 |                                  583,139,328 | Same six types, different layer counts         |
| [DeepSeek-V2-Lite-Chat Q4_K_M](https://huggingface.co/mradermacher/DeepSeek-V2-Lite-Chat-GGUF/tree/9ee9f1f8f39f673e1022e176d586aef4228f630d)                  | 16B / 2.4B, DeepSeek Model License        | 10,364,416,768 |           833,058,816 |                                  893,190,144 | Q4_K, Q5_0, Q8_0                               |
| [DeepSeek-Coder-V2-Lite-Instruct Q4_K_M](https://huggingface.co/bartowski/DeepSeek-Coder-V2-Lite-Instruct-GGUF/tree/8f248fa2072348f77a8bc37754e470de1f61866e) | 16B / 2.4B, DeepSeek Model License        | 10,364,416,768 |           833,058,816 |                                  893,190,144 | Q4_K, Q5_0, Q8_0                               |
| [Ling-3.0-tiny Q4_K_M](https://huggingface.co/inclusionAI/Ling-3.0-tiny-GGUF/tree/01b850e29ed5b44e2b810160ff980a9bb353296f)                                   | 7.9B / 1.3B, MIT                          |  4,823,894,944 |           624,909,952 |                                  262,029,312 | Q4_K, Q6_K                                     |

For each routed three-dimensional tensor take its byte count divided by its expert count, multiply by the selected experts per layer, then sum. This is a logical, cold, uncoalesced payload bound for a single decode position. Cache hits, prefill batching and physical storage behavior require a route trace. All five checkpoints have **NOT_RUN** full resident/streamed CPU logits, Windows/UWP acceptance, GPU parity, Series S/X throughput and peak working set for these conversions.

- **Qwen.** Each checkpoint selects 8 of 128 experts across 48 layers. Plain FP16 K+V for 4 KV heads of width 128 is 98,304 B/position (192/384 MiB at 2,048/4,096 positions before allocator overhead). Template hashes differ. Qualify tool and coding cases on the Coder checkpoint, chat on the general checkpoint and each low-bit recipe against its own higher-precision control. [Chat card](https://huggingface.co/Qwen/Qwen3-30B-A3B/tree/ad44e777bcd18fa416d9da3bd8f70d33ebb85d39) · [Coder card](https://huggingface.co/Qwen/Qwen3-Coder-30B-A3B-Instruct/tree/b2cff646eb4bb1d68355c01b18ae02e7cf42d120) · [#720](https://github.com/gianlucamazza/xllama/issues/720).
- **DeepSeek.** Both artifacts contain the older `attn_kv_b` layout, 26 routed layers with a dense first block and two shared experts. Plain FP16 K+V payload is 276,480 B/position (540/1,080 MiB for the same contexts). A separate conversion and measurement must establish any compressed MLA layout. Their embedded template bytes match, while the checkpoints have different full-file digests. [Chat weights and licence](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite-Chat/tree/85864749cd611b4353ce1decdb286193298f64c7) · [Coder weights and licence](https://huggingface.co/deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct/tree/e434a23f91ba5b4923cf6c9d9a238eb4a08e3a11) · [#721](https://github.com/gianlucamazza/xllama/issues/721).
- **Ling.** The hybrid graph leaves shared experts, router bias, recurrent state and attention resident. Measure the actual hybrid cache and workspace at two contexts. `bailingmoe2` parity comes before `bailingmoe3`; preserve the checkpoint's thinking template in the comparison. [Publisher card](https://huggingface.co/inclusionAI/Ling-3.0-tiny/tree/9a98e35fe1c9ee255f78dd64771c7ae15a799481) · [#526](https://github.com/gianlucamazza/xllama/issues/526).
- **Multi-shard GGUF.** Capture every named shard, size, digest and split metadata as one generation. Test shard-one resolution, completeness, previous-generation retention, free space for replacement and recovery. Multi-file storage alone provides no expert routing, remote execution or expected decode rate. [#722](https://github.com/gianlucamazza/xllama/issues/722).

## What each validation needs

Per model, in the stage order of the MoE engineering program: checkpoint SHA-256 and template
bound before the run; host fixture and, where the host can hold it,
full-checkpoint resident-versus-streamed parity (#480); a console CPU run with
two cache sizes, native stop and normal teardown (#481 CPU); GPU stage only
for Q4_K/Q6_K files (#481 GPU). Speed is compared against the dense peer of
[#273] only after correctness passes.

## Sources

- LFM2.5-24B-A2B: [Liquid AI model card](https://huggingface.co/LiquidAI/LFM2.5-24B-a2b), [Unsloth](https://huggingface.co/unsloth/LFM2.5-24B-A2B), [size guide](https://www.developersdigest.tech/local-models/lfm2-5-24b-a2b)
- Ling-mini-2.0: [inclusionAI GGUF](https://huggingface.co/inclusionAI/Ling-mini-2.0-GGUF), [mradermacher GGUF](https://huggingface.co/mradermacher/Ling-mini-2.0-GGUF)
- ERNIE-4.5-21B-A3B: [bartowski GGUF](https://huggingface.co/bartowski/baidu_ERNIE-4.5-21B-A3B-Thinking-GGUF)
- SmallThinker: [arXiv 2507.20984](https://arxiv.org/abs/2507.20984v2), [third-party GGUF](https://hface.pages.dev/Mungert/SmallThinker-21BA3B-Instruct-GGUF)
- gpt-oss-20b: [Unsloth GGUF files](https://huggingface.co/unsloth/gpt-oss-20b-GGUF/tree/main), [ModelFit](https://modelfit.io/quant-compare/gpt-oss-20b/)
- Gemma 4 26B-A4B: [Gemma docs](https://ai.google.dev/gemma/docs/core), [Q4_0 GGUF](https://huggingface.co/VeigaPunk/gemma4-hvm-official-q4), [Jetson AI Lab](https://www.jetson-ai-lab.com/models/gemma4-26b-a4b)
- Trinity: [Trinity Mini GGUF](https://huggingface.co/arcee-ai/Trinity-Mini-GGUF), [Arcee Trinity](https://www.arcee.ai/trinity)
- Qwen3.5/3.6-35B-A3B: [Unsloth GGUF](https://huggingface.co/unsloth/Qwen3.5-35B-A3B-GGUF), [ModelFit](https://modelfit.io/quant-compare/qwen3.5-35b-a3b/); Qwen3-30B-A3B: [ModelFit](https://modelfit.io/quant-compare/qwen3-30b-a3b/)
- Nemotron 3 Nano: [bartowski GGUF](https://huggingface.co/bartowski/nvidia_Nemotron-3-Nano-30B-A3B-GGUF)
- Granite-4.0-H-Tiny: [bartowski GGUF](https://huggingface.co/bartowski/ibm-granite_granite-4.0-h-tiny-GGUF); OLMoE: [allenai GGUF](https://huggingface.co/allenai/OLMoE-1B-7B-0125-GGUF/tree/main)

[#273]: https://github.com/gianlucamazza/xllama/issues/273
