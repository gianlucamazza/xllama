#!/usr/bin/env bash
# quantize.sh — wrapper around llama-quantize for GGUF preparation
#
# Usage:
#   ./scripts/quantize.sh input.gguf output.gguf [QUANT_TYPE]
#
# QUANT_TYPE defaults to Q4_K_M (recommended for Xbox Series S 8 GB budget).
# See llama.cpp docs for all supported types.

set -euo pipefail

INPUT="${1:-}"
OUTPUT="${2:-}"
QUANT_TYPE="${3:-Q4_K_M}"

if [[ -z "$INPUT" || -z "$OUTPUT" ]]; then
	echo "Usage: $0 <input.gguf> <output.gguf> [QUANT_TYPE]" >&2
	echo "       QUANT_TYPE: Q4_K_M (default), Q4_K_S, IQ4_XS, Q8_0, ..." >&2
	exit 1
fi

QUANTIZE_BIN="./llama.cpp/build/bin/llama-quantize"
if [[ ! -x "$QUANTIZE_BIN" ]]; then
	echo "Error: $QUANTIZE_BIN not found. Build llama.cpp first:" >&2
	echo "  cmake -B llama.cpp/build llama.cpp && cmake --build llama.cpp/build -j" >&2
	exit 1
fi

if [[ ! -f "$INPUT" ]]; then
	echo "Error: input file not found: $INPUT" >&2
	exit 1
fi

echo "Quantizing $INPUT → $OUTPUT (${QUANT_TYPE}) ..."
"$QUANTIZE_BIN" "$INPUT" "$OUTPUT" "$QUANT_TYPE"

SIZE=$(du -sh "$OUTPUT" | cut -f1)
echo "Done. Output: $OUTPUT (${SIZE})"
