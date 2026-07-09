#!/usr/bin/env python3
# Copyright (c) 2024 Venere Labs
# SPDX-License-Identifier: MIT
#
# Export the TAESD tiny decoder (madebyollin/taesd, MIT) as a drop-in
# vae_decoder/model.onnx for the console diffusion pipeline.
#
# Contract preserved (uwp/diffuse.cpp expects the SD VAE decoder interface):
#   input  latent_sample [1,4,64,64] = scheduler latent / 0.18215
#   output sample        [1,3,512,512] in [-1,1]
# TAESD wants RAW scheduler latents and emits [0,1], so the wrapper multiplies
# the input back by 0.18215 and maps the output to [-1,1]. Validate with:
#   validate_pipeline.py <sd_model_dir> out.png "<prompt>" 1 42 <taesd_out_dir>
#
#   venv310/bin/python diffusion/export_taesd.py [out_dir]
import sys

import torch
from diffusers import AutoencoderTiny

OUT = sys.argv[1] if len(sys.argv) > 1 else "taesd-decoder"
VAE_SCALE = 0.18215


class TaesdAsSdVae(torch.nn.Module):
    def __init__(self, taesd):
        super().__init__()
        self.decoder = taesd.decoder

    def forward(self, latent_sample):
        # latent_sample arrives pre-divided by VAE_SCALE (SD VAE contract);
        # TAESD wants the raw latent back. The diffusers AutoencoderTiny
        # decoder already emits in the SD [-1,1] convention (falsified the
        # [0,1] assumption: remapping crushed the image dark) — pass through.
        return self.decoder(latent_sample * VAE_SCALE).clamp(-1.0, 1.0)


def main():
    taesd = AutoencoderTiny.from_pretrained(
        "madebyollin/taesd", torch_dtype=torch.float32
    )
    model = TaesdAsSdVae(taesd).eval()
    x = torch.randn(1, 4, 64, 64)

    import os

    os.makedirs(f"{OUT}/vae_decoder", exist_ok=True)
    path = f"{OUT}/vae_decoder/model.onnx"
    torch.onnx.export(  # legacy tracer — same recipe as the SD export (README)
        model,
        (x,),
        path,
        input_names=["latent_sample"],
        output_names=["sample"],
        opset_version=17,
        dynamic_axes={
            "latent_sample": {0: "batch", 2: "height", 3: "width"},
            "sample": {0: "batch"},
        },
    )
    # Load-test at EXTENDED (the console cap) and report size.
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    s = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    out = s.run(["sample"], {"latent_sample": x.numpy()})[0]
    print(
        f"[taesd] exported {path} ({os.path.getsize(path) / 1e6:.1f} MB), "
        f"load OK, out {out.shape} range [{out.min():.2f}, {out.max():.2f}]"
    )


if __name__ == "__main__":
    main()
