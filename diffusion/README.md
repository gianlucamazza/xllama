# diffusion/ — image generation on Xbox (SD-Turbo via ONNX Runtime DirectML)

The text-decode work established that the Series S GPU loses at LLM decode because
that workload is its _worst_ case (M=1, dispatch-bound, non-fused int4 on DML —
see `docs/uwp-constraints.md §12`). Image generation (diffusion) is the _opposite_:
compute-bound fp16 batch on 64×64 latents — the case where DML already wins
(prefill). This directory is the model-side toolchain for running a distilled
few-step diffusion model (SD-Turbo, 1 step) on the console GPU.

## Status

- ✅ **Model validated on the owned layer (2026-07-08).** SD-Turbo exported to
  ONNX and run through **ONNX Runtime (CPU EP)** generates a coherent 512×512
  image in ~13 s / 1 step. This is the exact artifact + runtime that runs on the
  Xbox via the DirectML EP — only the EP (and thus the speed) differs.
- ✅ **Plain ORT DirectML foundation proven** by the image spike
  (`uwp/image-spike.cpp`, PR #3): a compute-bound conv model compiles, links, and
  runs through the DirectML EP in the UWP AppContainer.
- ✅ **C++ pipeline built + host-validated (2026-07-08).** `uwp/diffuse.cpp` runs
  three ORT DirectML sessions (text_encoder → 1× UNet → VAE decode) behind the
  `diffuse.flag` headless mode. Its correctness-critical logic — the CLIP
  byte-BPE tokenizer (`include/xllama/diffusion/clip_tokenizer.h`), the
  EulerDiscreteScheduler (`euler_scheduler.h`), fp16 conversion (`half.h`), and
  the PNG writer (`png_writer.h`) — is asserted against golden vectors captured
  from this Python reference (`gen_golden_vectors.py`) in `tests/test_diffusion.cpp`
  (638 assertions, all green). The ORT DirectML orchestration is CI-compile-
  validated; runtime validation is on console per
  `docs/console-validation-runbook.md §7`.
- ⏳ **Next (console)**: a deployable **fp16** SD-Turbo export (GPU/Olive — see
  Memory/deployability below), then run `diffuse.flag` on the Xbox.

## Reproduce the export

The toolchain is version-sensitive: the export only works with the **legacy
torch ONNX tracer** (torch 2.4.x), because optimum 1.23 does not support torch's
newer dynamo/onnxscript exporter (it hits external-data-naming and
LayerNormalization opset-downgrade bugs). Use **Python 3.10** (3.14 lacks wheels
for this era). Do not bump the pins piecemeal.

```bash
python3.10 -m venv venv
./venv/bin/pip install -r diffusion/requirements.txt \
    --extra-index-url https://download.pytorch.org/whl/cpu
./diffusion/export_sd_turbo.sh                 # -> sd-turbo-onnx/ + sd-turbo-onnx-fp16/
./venv/bin/python diffusion/generate_onnx.py   # validate: writes sd_turbo_onnx.png
```

## Memory / deployability

SD-Turbo (SD1.5-class) fits the console GPU budget in fp16 (~2.4 GB total:
UNet ~1.65 GB, text_encoder ~0.65 GB, vae_decoder ~0.1 GB), and — unlike a 1.7B
LLM in fp16 (a single 3.4 GB weight blob that exceeds the 2 GB ONNX protobuf
limit) — each SD component is individually **< 2 GB**, so all save self-contained
and avoid the AppContainer `weakly_canonical` crash (§8). **These sizes are
confirmed.**

**Getting a runnable fp16 model.** `convert_fp16.py` (CPU) is reliable for the
size analysis above but leaves a mixed-type node in the UNet timestep embedding
(`/time_proj/Mul`) that ORT rejects at load — onnxconverter_common does not fully
type the SD UNet on CPU. **Fix (verified 2026-07-08)**: pass the `/time_proj/*`
node names as `node_block_list` to `convert_float_to_float16` — the timestep
embedding stays an fp32 island with automatic Cast boundaries, and the rest of
the UNet converts cleanly. This is the deployable-artifact path used for the
console (own export, locally validatable end-to-end).

⚠️ **Trap (verified 2026-07-08)**: the ORT-team pre-exports
([`onnxruntime/sd-turbo`](https://hf.co/onnxruntime/sd-turbo),
`tlwu/sd-turbo-onnxruntime`) look ideal on paper (fp16, each component
self-contained < 2 GB) but their UNet/VAE use `com.microsoft.NhwcConv`, which has
**no CPU kernel** (`NOT_IMPLEMENTED` at session init) — they are CUDA-optimized
graphs. They cannot be validated on our owned layer and may not map to DML
either; do not deploy them unvalidated. Other sources for a clean fp16 graph:

1. **Pre-exported fp16 DirectML models** (recommended, no GPU needed) — e.g.
   [`nmkd/stable-diffusion-1.5-onnx-fp16`](https://huggingface.co/nmkd/stable-diffusion-1.5-onnx-fp16),
   [`sharpbai/stable-diffusion-v1-5-onnx-directml-fp16`](https://huggingface.co/sharpbai/stable-diffusion-v1-5-onnx-directml-fp16),
   or the [Amblyopius/Stable-Diffusion-ONNX-FP16](https://github.com/Amblyopius/Stable-Diffusion-ONNX-FP16)
   toolkit. **Verified 2026-07-08**: the nmkd SD1.5 fp16 components all load in
   ORT (UNet input `sample: tensor(float16)`, 1.7 GB → < 2 GB), unlike the CPU
   `convert_fp16.py` output. These ship with **external data** (`unet/weights.pb`),
   so run `scripts/merge_onnx_external_data.py` on each component to make it
   self-contained before deploying (AppContainer `weakly_canonical`, §8).
2. **GPU export**: `optimum-cli export onnx --model stabilityai/sd-turbo --fp16
--device cuda …`, or Microsoft **Olive**'s SD DirectML optimization.

Speed note: SD1.5 needs ~20–25 steps (slow); the console target is a **few-step**
model — SD-Turbo/SDXL-Turbo (1 step) or an LCM variant (2–4 steps) in fp16. The
fp32 SD-Turbo path here is validated for quality; pair it with a clean fp16 UNet
from source (1) or (2) for the console.

## Deploy + run on console (planned)

Upload `sd-turbo-onnx-fp16/{text_encoder,unet,vae_decoder}` to LocalState, then a
future `diffuse.flag` headless mode (mirroring `image.flag`) runs the C++
pipeline and writes the PNG for Device Portal fetch. The pipeline is: CLIP-tokenize
the prompt → text_encoder → init latent → (1×) UNet denoise with the scheduler →
scale + vae_decoder → PNG. Model licensing note: SD-Turbo is under Stability's
research/community license — for a public demo, swap to an openly-licensed
few-step model (e.g. an LCM/SDXL-Turbo-Apache variant).
