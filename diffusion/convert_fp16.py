#!/usr/bin/env python3
# Copyright (c) 2024 Venere Labs
# SPDX-License-Identifier: MIT
#
# Convert the fp32 SD-Turbo ONNX components to fp16 and report their on-disk size
# against the Xbox Series S GPU budget (3801 MB). Each component is saved
# self-contained when < 2 GB (the ONNX protobuf single-file limit + the
# merged-model requirement that avoids the AppContainer `weakly_canonical` crash,
# docs/uwp-constraints.md §8).
#
# ⚠️ KNOWN LIMITATION (verified 2026-07-08): this CPU-side conversion is reliable
# for SIZE analysis only. It leaves a mixed-type node in the SD UNet timestep
# embedding (`/time_proj/Mul`: float vs float16) that makes ORT reject the model
# at load. onnxconverter_common's fp16 pass does not fully type the SD UNet on
# CPU. For a RUNNABLE fp16 model, export directly with a GPU:
#     optimum-cli export onnx --model stabilityai/sd-turbo --fp16 --device cuda ...
# or use Microsoft Olive's SD DirectML optimization. Sizes confirmed here:
# UNet fp16 ~1.65 GB (< 2 GB, self-contained), text_encoder ~0.65 GB, vae ~0.1 GB
# → ~2.4 GB total, fits the 3801 MB budget with all components AppContainer-safe.
#
# Usage:  python diffusion/convert_fp16.py <onnx_dir_in> <onnx_dir_out>
import sys, os, shutil
import onnx
from onnxconverter_common import float16

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

for comp in COMPONENTS:
    src = os.path.join(IN, comp, "model.onnx")
    if not os.path.exists(src):
        print(f"[skip] {comp}: no model.onnx")
        continue
    print(f"[fp16] converting {comp} ...", flush=True)
    m = onnx.load(src)  # loads external data if present
    # Convert the WHOLE graph (I/O included) to fp16. keep_io_types=True leaves
    # fp32 islands that break SD UNets with a mixed-type error at the timestep
    # embedding (`/time_proj/Mul`: float vs float16); an all-fp16 graph is
    # consistent and the ORT pipeline feeds fp16 accordingly.
    m16 = float16.convert_float_to_float16(
        m, keep_io_types=False, disable_shape_infer=True
    )
    dst_dir = os.path.join(OUT, comp)
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, "model.onnx")
    size = m16.ByteSize()
    if size < PROTOBUF_LIMIT:
        onnx.save(m16, dst)  # self-contained (AppContainer-safe)
        ext = "self-contained"
    else:
        onnx.save(
            m16,
            dst,
            save_as_external_data=True,
            location="model.onnx_data",
            all_tensors_to_one_file=True,
        )
        ext = "EXTERNAL DATA (>2GB — AppContainer risk!)"
    disk = os.path.getsize(dst) + (
        os.path.getsize(dst + "_data") if os.path.exists(dst + "_data") else 0
    )
    print(f"[fp16] {comp}: {disk // (1024 * 1024)} MB  ({ext})")
print(f"[fp16] done -> {OUT}")
