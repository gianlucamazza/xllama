#!/usr/bin/env python3
# Copyright (c) 2024 Venere Labs
# SPDX-License-Identifier: MIT
#
# Validate the exported ONNX SD-Turbo pipeline by generating one image through
# ONNX Runtime (the same runtime that runs on the Xbox via the DirectML EP; here
# on the CPU EP). This proves the ONNX artifacts are correct on the layer we own
# before any C++/DirectML/console work.
#
#   ./venv/bin/python diffusion/generate_onnx.py [onnx_dir] [out.png] ["prompt"]
import sys, time
import numpy as np
from optimum.onnxruntime import ORTStableDiffusionPipeline

onnx_dir = sys.argv[1] if len(sys.argv) > 1 else "sd-turbo-onnx"
out_png = sys.argv[2] if len(sys.argv) > 2 else "sd_turbo_onnx.png"
prompt = (
    sys.argv[3]
    if len(sys.argv) > 3
    else "a red sports car on a mountain road at sunset, cinematic, highly detailed"
)

print(f"[onnx] loading {onnx_dir} via ORT (CPU EP) ...", flush=True)
pipe = ORTStableDiffusionPipeline.from_pretrained(onnx_dir)
print(f"[onnx] generating: {prompt}", flush=True)
t0 = time.time()
# SD-Turbo is a 1-step distilled model with classifier-free guidance disabled.
img = pipe(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]
img.save(out_png)
a = np.asarray(img).astype(float)
print(
    f"[onnx] done {time.time() - t0:.1f}s -> {out_png} {img.size} "
    f"mean={a.mean():.1f} std={a.std():.1f} (std>10 => real image)"
)
