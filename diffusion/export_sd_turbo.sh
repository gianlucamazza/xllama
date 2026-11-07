#!/usr/bin/env bash
# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
#
# Export stabilityai/sd-turbo to ONNX (fp32) with the first-party dynamo
# exporter (export_sd_turbo.py, ADR 0003), then convert the txt2img components
# to fp16 for the Xbox Series S GPU budget. Requires the pinned toolchain in
# requirements.txt on Python 3.12 (3.11+).
#
#   python3.12 -m venv venv && ./venv/bin/pip install -r diffusion/requirements.txt \
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

# fp32 export on CPU: text_encoder, unet, vae_decoder with the console I/O
# contract (checked after each component), each verified on ORT CPU against
# eager torch. Pass REVISION=<hub commit> to pin the checkpoint.
"$VENV/bin/python" "$HERE/export_sd_turbo.py" \
	--model stabilityai/sd-turbo \
	${REVISION:+--revision "$REVISION"} \
	--out "$OUT_FP32" \
	--verify

# Convert the txt2img components (text_encoder, unet, vae_decoder) to fp16.
"$VENV/bin/python" "$HERE/convert_fp16.py" "$OUT_FP32" "$OUT_FP16"

echo "Done. Console-ready fp16 components in $OUT_FP16/"
