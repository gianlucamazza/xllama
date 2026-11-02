#!/usr/bin/env bash
# quantize.sh — wrapper around llama-quantize for GGUF preparation
#
# Usage:
#   ./scripts/quantize.sh [--imatrix FILE] input.gguf output.gguf [QUANT_TYPE]
#
# QUANT_TYPE defaults to Q4_K_M (recommended for Xbox Series S 8 GB budget).
# --imatrix passes a calibration file (from llama-imatrix) to llama-quantize.
# See llama.cpp docs for all supported types.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMATRIX=""
POSITIONAL=()
while [[ $# -gt 0 ]]; do
	case "$1" in
	--imatrix)
		if [[ $# -lt 2 ]]; then
			echo "Error: --imatrix needs a file" >&2
			exit 1
		fi
		IMATRIX="$2"
		shift 2
		;;
	*)
		POSITIONAL+=("$1")
		shift
		;;
	esac
done

INPUT="${POSITIONAL[0]:-}"
OUTPUT="${POSITIONAL[1]:-}"
QUANT_TYPE="${POSITIONAL[2]:-Q4_K_M}"

if [[ -z "$INPUT" || -z "$OUTPUT" ]]; then
	echo "Usage: $0 [--imatrix FILE] <input.gguf> <output.gguf> [QUANT_TYPE]" >&2
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

QUANTIZE_ARGS=()
if [[ -n "$IMATRIX" ]]; then
	if [[ ! -f "$IMATRIX" ]]; then
		echo "Error: imatrix file not found: $IMATRIX" >&2
		exit 1
	fi
	QUANTIZE_ARGS+=(--imatrix "$IMATRIX")
fi

echo "Quantizing $INPUT → $OUTPUT (${QUANT_TYPE}${IMATRIX:+, imatrix $IMATRIX}) ..."
"$QUANTIZE_BIN" "${QUANTIZE_ARGS[@]}" "$INPUT" "$OUTPUT" "$QUANT_TYPE"

SIZE=$(du -sh "$OUTPUT" | cut -f1)
echo "Done. Output: $OUTPUT (${SIZE})"
