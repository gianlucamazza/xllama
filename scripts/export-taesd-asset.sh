#!/usr/bin/env bash
# Export the TAESD VAE ONNX for the models-v1 GitHub Release and print upload steps.
# Usage: ./scripts/export-taesd-asset.sh [out_dir]
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-$ROOT/build/taesd-vae-asset}"
VENV="${XLLAMA_DIFFUSION_VENV:-$HOME/.cache/xllama-diffusion/venv310}"
PY="$VENV/bin/python"
ASSET_NAME="sd-turbo-fp16_taesd_vae_decoder_model.onnx"

if [[ ! -x "$PY" ]]; then
	echo "error: diffusion venv not found at $PY" >&2
	echo "  Create with diffusion/README.md, or set XLLAMA_DIFFUSION_VENV." >&2
	exit 1
fi

mkdir -p "$OUT"
"$PY" "$ROOT/diffusion/export_taesd.py" "$OUT"

ONNX="$OUT/vae_decoder/model.onnx"
if [[ ! -f "$ONNX" ]]; then
	echo "error: export failed — $ONNX missing" >&2
	exit 1
fi

# Release assets are flat; copy to the canonical name for gh upload.
cp -f "$ONNX" "$OUT/$ASSET_NAME"
BYTES=$(stat -c%s "$OUT/$ASSET_NAME" 2>/dev/null || stat -f%z "$OUT/$ASSET_NAME")
MB=$(awk "BEGIN {printf \"%.1f\", $BYTES/1e6}")
echo ""
echo "Exported: $OUT/$ASSET_NAME (${MB} MB)"
echo ""
echo "Upload to models-v1 (requires gh auth):"
echo "  gh release upload models-v1 \"$OUT/$ASSET_NAME\" --clobber"
echo ""
echo "Then enable TAESD in the app (Image dialog toggle or settings.json diffuse_taesd_vae)."