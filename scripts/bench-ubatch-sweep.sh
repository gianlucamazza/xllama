#!/usr/bin/env bash
# Sweep llama.cpp prefill micro-batch (n_ubatch) and print prompt tok/s per value.
# The prefill physical chunk is the only TTFT-relevant batching knob on the CPU
# path (n_batch just caps the logical batch). Use this to find the ubatch that
# maximises prompt throughput for a given model on a given CPU.
#
# Usage: ./scripts/bench-ubatch-sweep.sh <model.gguf> [prompt-repeat] [ubatch-list]
#   prompt-repeat : how many times to repeat the filler sentence (default 70,
#                   ~700 prompt tokens — long enough for prefill to dominate).
#   ubatch-list   : space-separated values (default "128 256 512 1024").
#
# Host reference (i7-1165G7, Qwen3.5-0.8B-Q4_K_M, 701-token prompt, 2026-07-14):
# repeated sweeps DISAGREE on the ubatch=128 value (82.5 vs 117.4 tok/s across two
# runs) — on a loaded dev laptop the measurement is noise-dominated and shows no
# reproducible ubatch trend. Run this on a quiet machine (or the Xbox in Game
# mode) with several passes before trusting any single number.
set -euo pipefail

MODEL="${1:?usage: bench-ubatch-sweep.sh <model.gguf> [repeat] [ubatch-list]}"
REPEAT="${2:-70}"
UBATCHES="${3:-128 256 512 1024}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLI="${SCRIPT_DIR}/../build/linux-release/bin/xllama-cli"
[ -x "$CLI" ] || {
	echo "build first: cmake --build build/linux-release --target xllama-cli" >&2
	exit 1
}

# Deterministic filler prompt (no python dependency).
SENTENCE="The quick brown fox jumps over the lazy dog. "
PROMPT=""
for _ in $(seq 1 "$REPEAT"); do PROMPT+="$SENTENCE"; done

printf '%-10s %-14s %-14s\n' "ubatch" "prompt_tok/s" "decode_tok/s"
for ub in $UBATCHES; do
	line="$("$CLI" -m "$MODEL" -p "$PROMPT" -n 16 --ubatch "$ub" --seed 1 2>&1 |
		grep -oE 'prompt=[0-9.]+ tok/s decode=[0-9.]+' || true)"
	p="$(echo "$line" | grep -oE 'prompt=[0-9.]+' | cut -d= -f2)"
	d="$(echo "$line" | grep -oE 'decode=[0-9.]+' | cut -d= -f2)"
	printf '%-10s %-14s %-14s\n' "$ub" "${p:-?}" "${d:-?}"
done
