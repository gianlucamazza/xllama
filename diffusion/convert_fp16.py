#!/usr/bin/env python3
# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
#
# Convert the fp32 SD-Turbo ONNX components to fp16 for the Xbox Series S GPU
# budget (3801 MB), producing the deployable console artifacts: each component
# self-contained < 2 GB (ONNX protobuf single-file limit + the merged-model
# requirement that avoids the AppContainer `weakly_canonical` crash,
# docs/uwp-constraints.md §8).
#
# Converter choice (verified 2026-07-08): onnxruntime.transformers'
# OnnxModel.convert_float_to_float16 — the ORT team's corrected fp16 pass.
# onnxconverter_common leaves mixed-type nodes in ALL three SD components
# (regardless of keep_io_types / shape-infer / node_block_list) and its output is
# rejected by ORT at load. See diffusion/README.md "Getting a runnable fp16 model".
#
# Load caveat baked into the pipeline: sessions must cap graph optimization at
# ORT_ENABLE_EXTENDED (ORT_ENABLE_ALL crashes on these graphs) — both
# uwp/diffuse.cpp and validate_pipeline.py do. This script load-tests each
# converted component at EXTENDED before declaring success.
#
# Usage:  python diffusion/convert_fp16.py <onnx_dir_in> <onnx_dir_out>
import os
import shutil
import sys

import onnx
import onnxruntime as ort
from onnxruntime.transformers.onnx_model import OnnxModel

IN = sys.argv[1] if len(sys.argv) > 1 else "sd-turbo-onnx"
OUT = sys.argv[2] if len(sys.argv) > 2 else "sd-turbo-onnx-fp16"
# Components needed for txt2img (vae_encoder is only for img2img).
COMPONENTS = ["text_encoder", "unet", "vae_decoder"]
PROTOBUF_LIMIT = 2 * 1024**3

os.makedirs(OUT, exist_ok=True)
# Copy the non-weight pipeline files (scheduler, tokenizer, configs) verbatim.
for name in os.listdir(IN):
    src = os.path.join(IN, name)
    if os.path.isdir(src) and name not in COMPONENTS + ["vae_encoder"]:
        shutil.copytree(src, os.path.join(OUT, name), dirs_exist_ok=True)
    elif os.path.isfile(src):
        shutil.copy2(src, os.path.join(OUT, name))

failed = 0
for comp in COMPONENTS:
    src = os.path.join(IN, comp, "model.onnx")
    if not os.path.exists(src):
        print(f"[skip] {comp}: no model.onnx")
        continue
    print(f"[fp16] converting {comp} ...", flush=True)
    m = OnnxModel(onnx.load(src))  # loads external data if present
    m.convert_float_to_float16(keep_io_types=False)
    dst_dir = os.path.join(OUT, comp)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, "model.onnx")
    m.save_model_to_file(dst, use_external_data_format=False)  # self-contained
    disk = os.path.getsize(dst)
    if disk >= PROTOBUF_LIMIT:
        print(f"[fp16] {comp}: {disk // (1024 * 1024)} MB EXCEEDS 2GB — not deployable")
        failed += 1
        continue
    # Load-test at the same optimization level the console pipeline uses.
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    try:
        s = ort.InferenceSession(
            dst, sess_options=so, providers=["CPUExecutionProvider"]
        )
        ins = [(i.name, i.type) for i in s.get_inputs()]
        del s
        print(
            f"[fp16] {comp}: {disk // (1024 * 1024)} MB, LOAD OK (EXTENDED), in: {ins}"
        )
    except Exception as e:
        print(f"[fp16] {comp}: LOAD FAIL: {str(e)[:200]}")
        failed += 1

print(f"[fp16] done -> {OUT}" + (f"  ({failed} FAILED)" if failed else ""))
sys.exit(1 if failed else 0)
