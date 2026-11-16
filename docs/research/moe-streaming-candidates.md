# MoE streaming: models to validate

Desk list, 2026-11-16. Which MoE checkpoints the optional SSD expert-streaming
path (#479–#481, [plan](../superpowers/plans/2026-11-16-ssd-moe-upstream.md))
should be validated on, and in what order. It selects inputs for those
stages; it admits nothing to the catalogue and changes no verdict in [#273]
or [model-matrix.md §I1](../model-matrix.md#i1-moe-post-h2-fail). Status of
each stage stays with the trackers.

Sizes below come from public file listings and third-party pages (sources at
the end). They are desk figures: read the exact byte count from the chosen
repository before a download, and bind it by SHA-256 in the campaign.

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

### 0. Parity anchor: `lfm25-8b-a1b` (already on the console)

LFM2.5-8B-A1B, arch `lfm2moe` (the family of the fork's target; confirm the
GGUF `general.architecture` of both files before the run), `UD-IQ3_S`,
3571 MB, already provisioned and measured resident on the Series S (H2,
[model-matrix.md §A3](../model-matrix.md)). It fits the RAM of a modest host
and of the console, so it can carry what the 24B cannot here:

- **full-checkpoint resident-versus-streamed parity** on the host (the
  readiness report's second prerequisite: the 14.5 GB target does not fit a
  9.7 GiB host);
- **resident-versus-streamed on the console**, same checkpoint, same
  prompt, with the cache sized from 2 slots to "everything fits", so the
  streaming cost is measured against a known resident decode rate.

A Q4_K_M copy (about 5 GB) would also exercise the GPU bridge's types; the
`UD-IQ3_S` file does not.

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
- **No usable GGUF:** Maple-Preview 20B-A1B (`maple` is on the pin; only a
  4.98 GB TQ1_0 was published at the last scan). One active billion makes it
  a strong candidate once a Q4_K file exists.
- **Fits in RAM, so no streaming needed:** Granite-4.0-H-Tiny 7B-A1B
  (Q4_K_M ≈ 3.9 GB), OLMoE-1B-7B-0125 (4.21 GB), Trinity Nano 6B-A1B. These
  belong to the resident #273 gate, not to this path.
- **Too many active parameters:** anything above about 4B active (for
  example Granite-4.0-H-Small 32B-A9B, Phi-3.5-MoE 42B-A6.6B).

## What each validation needs

Per model, in the stage order of the plan: checkpoint SHA-256 and template
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
