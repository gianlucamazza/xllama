#!/usr/bin/env python3
# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
#
# Validate an exported ONNX SD-Turbo pipeline by generating one image through
# ONNX Runtime (the same runtime that runs on the Xbox via the DirectML EP; here
# on the CPU EP). This proves the ONNX artifacts are correct on the layer we own
# before any C++/DirectML/console work.
#
#   ./venv/bin/python diffusion/generate_onnx.py [onnx_dir] [out.png] ["prompt"]
#
# ADR 0003 retired optimum's ORTStableDiffusionPipeline: this is now a thin
# front-end over validate_pipeline.py, which replays uwp/diffuse.cpp's exact
# 1-step recipe (tokenize -> text_encoder -> Euler -> UNet -> VAE decode) and
# works on both the fp32 export and the fp16 console artifacts.
import os
import runpy
import sys

onnx_dir = sys.argv[1] if len(sys.argv) > 1 else "sd-turbo-onnx"
out_png = sys.argv[2] if len(sys.argv) > 2 else "sd_turbo_onnx.png"
prompt = (
    sys.argv[3]
    if len(sys.argv) > 3
    else "a red sports car on a mountain road at sunset, cinematic, highly detailed"
)

here = os.path.dirname(os.path.abspath(__file__))
sys.argv = [os.path.join(here, "validate_pipeline.py"), onnx_dir, out_png, prompt, "1", "42"]
runpy.run_path(sys.argv[0], run_name="__main__")
