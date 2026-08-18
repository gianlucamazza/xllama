#!/usr/bin/env bash
# quantize.sh — wrapper around llama-quantize for GGUF preparation
#
# Usage:
#   ./scripts/quantize.sh input.gguf output.gguf [QUANT_TYPE]
#
# QUANT_TYPE defaults to Q4_K_M (recommended for Xbox Series S 8 GB budget).
# See llama.cpp docs for all supported types.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT="${1:-}"
OUTPUT="${2:-}"
QUANT_TYPE="${3:-Q4_K_M}"

if [[ -z "$INPUT" || -z "$OUTPUT" ]]; then
	echo "Usage: $0 <input.gguf> <output.gguf> [QUANT_TYPE]" >&2
	echo "       QUANT_TYPE: Q4_K_M (default), Q4_K_S, IQ4_XS, Q8_0, ..." >&2
	exit 1
fi

# llama-quantize is built by our own presets, which vendor llama.cpp as a
# subdirectory — there is no separate llama.cpp/build/ tree and configuring one
# is not how this repo builds. LLAMA_QUANTIZE overrides for a hand-built binary.
QUANTIZE_CANDIDATES=(
	"${LLAMA_QUANTIZE:-}"
	"${REPO_ROOT}/build/linux-release/bin/llama-quantize"
	"${REPO_ROOT}/build/linux-test/bin/llama-quantize"
)
QUANTIZE_BIN=""
for cand in "${QUANTIZE_CANDIDATES[@]}"; do
	if [[ -n "$cand" && -x "$cand" ]]; then
		QUANTIZE_BIN="$cand"
		break
	fi
done

if [[ -z "$QUANTIZE_BIN" ]]; then
	echo "Error: llama-quantize not found. Tried:" >&2
	for cand in "${QUANTIZE_CANDIDATES[@]}"; do
		[[ -n "$cand" ]] && echo "  $cand" >&2
	done
	echo "Build it with a preset (it comes with the llama.cpp subdirectory):" >&2
	echo "  cmake --preset linux-release" >&2
	echo "  cmake --build build/linux-release -j\$(nproc) --target llama-quantize" >&2
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
