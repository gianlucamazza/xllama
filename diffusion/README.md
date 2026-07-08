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
- ⏳ **Next**: fp16 components for the 3801 MB GPU budget (below), then the C++
  pipeline (3 ORT DirectML sessions + scheduler + CLIP tokenizer) in the app.

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

⚠️ **fp16 conversion caveat**: `convert_fp16.py` (CPU) is reliable for the size
analysis above but leaves a mixed-type node in the UNet timestep embedding
(`/time_proj/Mul`) that ORT rejects at load — onnxconverter_common does not fully
type the SD UNet on CPU. For a **runnable** fp16 model, export on a GPU box with
`optimum-cli export onnx --model stabilityai/sd-turbo --fp16 --device cuda …`, or
use Microsoft **Olive**'s SD DirectML optimization (the officially recommended
path). The fp32 ONNX pipeline is fully validated (above); fp16 is a
memory-optimization step that needs the GPU exporter for a correct graph.

## Deploy + run on console (planned)

Upload `sd-turbo-onnx-fp16/{text_encoder,unet,vae_decoder}` to LocalState, then a
future `diffuse.flag` headless mode (mirroring `image.flag`) runs the C++
pipeline and writes the PNG for Device Portal fetch. The pipeline is: CLIP-tokenize
the prompt → text_encoder → init latent → (1×) UNet denoise with the scheduler →
scale + vae_decoder → PNG. Model licensing note: SD-Turbo is under Stability's
research/community license — for a public demo, swap to an openly-licensed
few-step model (e.g. an LCM/SDXL-Turbo-Apache variant).
