#!/usr/bin/env bash
# package-catalogue-ort-model.sh — stage flat models-v1 assets from a merged ORT GenAI dir
#
# Usage:
#   ./scripts/package-catalogue-ort-model.sh <catalogue-name> <model-dir> [out-dir]
#
# Example:
#   ./scripts/merge_onnx_external_data.py ~/.cache/xllama-hf-fix/smollm2-360m-dml-fp16
#   ./scripts/package-catalogue-ort-model.sh smollm2-360m-dml-fp16 \
#       ~/.cache/xllama-hf-fix/smollm2-360m-dml-fp16 build/catalogue-assets
#   gh release upload models-v1 build/catalogue-assets/smollm2-360m-dml-fp16_*.json \
#       build/catalogue-assets/smollm2-360m-dml-fp16_model.onnx --clobber
set -euo pipefail

NAME="${1:?catalogue name (e.g. smollm2-360m-dml-fp16)}"
SRC="${2:?model directory with genai_config.json + merged model.onnx}"
OUT="${3:-build/catalogue-assets}"

if [[ ! -f "$SRC/genai_config.json" ]]; then
	echo "error: $SRC/genai_config.json missing" >&2
	exit 1
fi
if [[ ! -f "$SRC/model.onnx" ]]; then
	echo "error: $SRC/model.onnx missing (merge external data first)" >&2
	exit 1
fi

mkdir -p "$OUT"
for f in genai_config.json tokenizer.json tokenizer_config.json model.onnx; do
	if [[ ! -f "$SRC/$f" ]]; then
		echo "error: $SRC/$f missing" >&2
		exit 1
	fi
	cp -f "$SRC/$f" "$OUT/${NAME}_${f}"
	echo "  ${NAME}_${f} ($(du -h "$OUT/${NAME}_${f}" | cut -f1))"
done
echo "Upload: gh release upload models-v1 $OUT/${NAME}_* --clobber"