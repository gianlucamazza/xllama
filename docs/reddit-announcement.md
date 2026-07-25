# Reddit announcement draft — xllama

This file contains publication-ready copy for `r/LocalLLaMA`. The benchmark
figures are intentionally linked to the repository's generated performance
summary rather than maintained as a second source of truth.

## Suggested title

I got a 2020 Xbox Series S running local LLMs at up to 93 tok/s — and learned why the GPU often loses

## Alternative titles

- xllama: local LLM chat, Stable Diffusion and on-device personalization on an Xbox Series S
- What does a 2020 Xbox console actually do well for local AI? I measured it
- Local LLMs on Xbox Series S: 350M to 3B models, KV-cache reuse and a 7-second image pipeline

## Main post

### TL;DR

I built [xllama](https://github.com/gianlucamazza/xllama), an open-source UWP
application that runs local LLM chat and Stable Diffusion on an Xbox Series S/X
in Dev Mode.

The interesting part is not that a console can run a small model. It is the
workload split: on this hardware the Zen 2 CPU is better for autoregressive
token generation, while the RDNA 2 GPU is useful for batched prompt processing
and diffusion. The app routes around those constraints instead of assuming that
the GPU must always be faster.

Best single fix of the project: one missing preprocessor define
(`GGML_USE_CPU_REPACK` was compiled but never enabled) was silently costing
**62% of GGUF prompt-processing throughput** — 241 → 394 tok/s from a one-line
build change.

There is no cloud inference involved. The app has a gamepad-oriented chat UI,
first-launch model downloads, an optional OpenAI-compatible LAN endpoint, and a
Linux CLI for development and reproducible checks.

Demo: [local chat and image generation on Xbox Series S](https://github.com/gianlucamazza/xllama/releases/download/v1.2.0.0/xllama-demo-v1.2.0.mp4)

Source: [github.com/gianlucamazza/xllama](https://github.com/gianlucamazza/xllama)

### What is running on the console?

- GGUF models through a llama.cpp CPU path.
- ONNX Runtime GenAI models through CPU or DirectML, selected per model and workload.
- KV-cache reuse for multi-turn conversations.
- A from-scratch C++ Stable Diffusion pipeline using DirectML.
- Preference capture and two personalization lanes: host PEFT LoRA and an
  experimental in-process partial fine-tune.

The shipped unified build starts with LFM2.5-350M and downloads models on
demand; the MSIX itself does not contain model weights.

### Measured results

These are measurements from an Xbox Series S in Dev Mode. The current generated
table, raw CSV evidence and selection policy are in the
[benchmark summary](https://github.com/gianlucamazza/xllama/blob/main/docs/benchmarks.md).
The rows below are single-run measurements unless the linked summary says
otherwise, so they should be read as engineering results rather than a formal
benchmark ranking.

| Model        | Quantization / backend |         Decode | Peak RAM |
| ------------ | ---------------------- | -------------: | -------: |
| LFM2.5-350M  | Q4_K_M, llama.cpp CPU  | **93.0 tok/s** |   320 MB |
| Gemma-3-270M | Q4_K_M, llama.cpp CPU  | **76.8 tok/s** |   368 MB |
| SmolLM2-360M | int4, ORT CPU          | **74.8 tok/s** |   708 MB |
| LFM2.5-1.2B  | Q4_K_M, llama.cpp CPU  | **37.9 tok/s** |   811 MB |
| LFM2-2.6B    | Q4_K_M, llama.cpp CPU  | **18.4 tok/s** |  1623 MB |
| Llama-3.2-3B | Q3_K_S, llama.cpp CPU  | **14.2 tok/s** |  1824 MB |

For context, the parity-validated DirectML fp16 SmolLM2 asset measures 236.7
prompt tok/s cold, 44.4 decode tok/s and 1268 MB peak RAM — and DirectML turns
out to lazily compile its kernels on first use, so the app now warms the GPU
session at load time and the first real request runs at the warm regime
(~870 prompt tok/s measured). Its value is first-turn
prompt processing; CPU int4 remains faster for decode, and the CPU path wins
subsequent turns because DirectML cannot reuse the conversation KV cache.

KV reuse changes the feel of larger models considerably:

- SmolLM2-360M: 4.87× faster turn-2 prefill.
- Gemma-3-270M: 4.07× faster turn-2 prefill.
- LFM2.5-1.2B: 19.36× faster turn-2 prefill.
- LFM2-2.6B: 20.02× faster turn-2 prefill.

The GPU's clearer win is image generation: SD-Turbo produces a coherent
512×512 image in about 6.9 seconds on the console, with the pipeline creating
and releasing the text encoder, UNet and VAE sessions sequentially to stay
within the GPU memory budget.

### A small quality check

The repository also contains an eight-task deterministic H9 suite covering
Italian, arithmetic, JSON extraction, grounded QA, summarization, translation,
multi-turn memory and abstention. It is a compact promotion gate, not a general
LLM evaluation.

The current results are:

- LFM2.5-1.2B: 6/8.
- LFM2-2.6B: 7/8.
- Gemma-4-E2B: 6/8.
- Llama-3.2-3B: 5/8.
- LFM2.5-350M: 4/8.

### What surprised me

The obvious assumption was that the Xbox GPU would be the main route for LLM
inference. It was not.

Autoregressive decode is effectively a batch-1, memory-bound workload. At this
model scale, CPU int4 kernels beat DirectML GPU execution. DirectML int4 is
particularly poor because the current path dequantizes weights to fp16 before
the GEMM instead of using a fused low-bit GPU kernel.

The build system can quietly cost you more than any kernel: the ggml
repacked-weight GEMM path (`repack.cpp`) was being **compiled but never
enabled** on the Xbox build — one missing define. Turning it on raised GGUF
prompt processing by 62% (241.9 → 393.2 tok/s on the same model and prompt)
with decode and RAM unchanged. Worth `grep`-ing your own builds for.

DirectML lazily compiles kernels on first use — the first generate in a
process ran at ~60% of steady-state prefill and ~80% of steady-state decode.
The fix is boring and effective: run a throwaway real-length generate at model
load, before the user's first prompt.

The GPU becomes useful when the workload is batched: prompt prefill and
diffusion. Even there, correctness comes first. An earlier DirectML text graph
produced incorrect logits on the Series S. The current `-v2` asset decomposes
the problematic RMSNorm graph, passes logit-parity validation, and is the only
GPU text asset allowed by the routing policy.

The detailed engineering account is in the
[technical report](https://github.com/gianlucamazza/xllama/blob/main/docs/technical-report.md)
and the [DirectML RMSNorm postmortem](https://github.com/gianlucamazza/xllama/blob/main/docs/dml-rmsnorm-fix-runbook.md).

### On-device personalization

The project includes an experimental Lane B partial fine-tuning path that runs
inside the app. The host marker gate and the Xbox Series S marker gate both
pass; the console run reached the marker end-to-end with 1195 MB peak working
set and 446 seconds wall time.

That is evidence that the execution lane works, not a claim that arbitrary
fine-tuning is fast or production-ready on a console. The current design keeps
host PEFT LoRA, console partial fine-tuning and future training work explicitly
separate. See the [training architecture](https://github.com/gianlucamazza/xllama/blob/main/docs/training-architecture.md).

### How to try it

You need an Xbox Series S or X in Dev Mode and a Linux or Windows host. The
Dev Mode activation is a one-time Microsoft fee; the project is not affiliated
with Microsoft.

1. Download the latest app package from the
   [GitHub releases page](https://github.com/gianlucamazza/xllama/releases).
2. Deploy it through Xbox Device Portal, or build it on Windows with the
   documented UWP script.
3. On first launch, let the app download the default chat model.

The [installation guide](https://github.com/gianlucamazza/xllama/blob/main/docs/install-release.md)
and [user guide](https://github.com/gianlucamazza/xllama/blob/main/docs/using-the-app.md)
cover the current flow. The repository also includes Linux CMake presets and
host tests for the shared bridge code.

### Limitations and open questions

- This targets Xbox Dev Mode, not a retail-game sandbox or a normal consumer
  sideloading workflow.
- The benchmark table mixes models, quantizations and runtimes by design; it is
  not a claim that one model is universally better than another.
- Most published rows are single measurements. Repeated-run reporting is now
  supported by the harness, and new campaigns will include spread.
- The LAN API is experimental, unauthenticated, foreground-only and disabled
  by default.
- The on-device training lane is experimental and currently uses a constrained
  partial-fine-tune design.

I would especially like feedback from people working on low-memory inference,
DirectML, console hardware, KV-cache management or practical local model
evaluation. What model family or workload would you test next on this hardware?

## Optional technical comment

The benchmark methodology and raw evidence are maintained in one place:
[docs/benchmarks.md](https://github.com/gianlucamazza/xllama/blob/main/docs/benchmarks.md)
and [`bench/results/`](https://github.com/gianlucamazza/xllama/tree/main/bench/results).

The headline numbers are decode tok/s and peak working set, measured on the
target console. Prefill tok/s is reported separately because it represents a
different workload. The harness records prompt length, generated-token count,
`max_length`, thread count and backend metadata so that a rate can be turned
back into an approximate timing instead of being treated as a standalone score.

The most useful result is probably not the 93 tok/s number. It is the routing
rule: CPU for decode and conversation continuation, GPU for batch-heavy work,
and correctness gates before enabling a model in GPU routing.

## Short version

I built [xllama](https://github.com/gianlucamazza/xllama), an open-source UWP
app for local LLM chat and Stable Diffusion on an Xbox Series S/X in Dev Mode.

On the Series S, the current measurements include:

- LFM2.5-350M Q4_K_M: 93.0 tok/s decode (394.8 prompt tok/s), 320 MB RAM.
- LFM2.5-1.2B Q4_K_M: 37.9 tok/s, 811 MB RAM.
- LFM2-2.6B Q4_K_M: 18.4 tok/s, 1623 MB RAM.
- KV-cache reuse: up to about 20× faster turn-2 prefill.
- SD-Turbo 512×512: about 6.9 seconds.

The main lesson is that the CPU wins autoregressive decode at this scale, while
the GPU is useful for prompt batching and diffusion. The project includes a
gamepad UI, GGUF and ORT backends, model downloads, an optional LAN API, Linux
tests, and an experimental on-device personalization lane.

The full results and limitations are in the
[benchmark summary](https://github.com/gianlucamazza/xllama/blob/main/docs/benchmarks.md).
Feedback on models, runtimes and workloads to test next is welcome.
