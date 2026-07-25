#!/usr/bin/env bash
# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
#
# Export stabilityai/sd-turbo to ONNX (fp32) with optimum, then convert the
# txt2img components to fp16 for the Xbox Series S GPU budget. Requires the
# pinned toolchain in requirements.txt on Python 3.10.
#
#   python3.10 -m venv venv && ./venv/bin/pip install -r diffusion/requirements.txt \
#       --extra-index-url https://download.pytorch.org/whl/cpu
#   ./diffusion/export_sd_turbo.sh
#
# Output: sd-turbo-onnx/ (fp32) and sd-turbo-onnx-fp16/ (console-ready, each
# component self-contained < 2 GB).
set -euo pipefail
VENV="${VENV:-./venv}"
OUT_FP32="${OUT_FP32:-sd-turbo-onnx}"
OUT_FP16="${OUT_FP16:-sd-turbo-onnx-fp16}"
HERE="$(cd "$(dirname "$0")" && pwd)"

export HF_HUB_DISABLE_TELEMETRY=1

# fp32 export on CPU (optimum's --fp16 needs a CUDA GPU). Legacy tracer via
# torch 2.4.1; opset defaults to the model's native (>=17, no downgrade).
"$VENV/bin/optimum-cli" export onnx \
	--model stabilityai/sd-turbo \
	--task text-to-image \
	"$OUT_FP32"

# Convert the txt2img components (text_encoder, unet, vae_decoder) to fp16.
"$VENV/bin/python" "$HERE/convert_fp16.py" "$OUT_FP32" "$OUT_FP16"

echo "Done. Console-ready fp16 components in $OUT_FP16/"
