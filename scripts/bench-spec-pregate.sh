#!/usr/bin/env bash
# bench-spec-pregate.sh — decide whether H3 is worth implementing, on host, before
# a line of speculative code exists.
#
# Usage:
#   ./scripts/bench-spec-pregate.sh [--out FILE] [--n-predict N] [--build DIR]
#
# Why a pre-gate at all: docs/phase7-hypotheses.md predeclares that H3 proceeds
# "only after a target/draft A/B predicts >=1.4x". Decomposing the published
# console numbers as T(n) = W + n*C (W = weight read, C = per-token compute) puts
# C at 30% of a decode step on qwen25-coder-3b — the prefill:decode ratio there is
# only 3.3:1, where a GPU would show 50:1. That caps speculative decoding at 1.9x
# with a draft model even at 100% acceptance, and the 0.5B draft costs 16 ms/token
# (22% of a k=2 round). Whether any variant clears 1.4x therefore turns on the
# acceptance rate, which is a property of the model pair and the prompt — NOT of
# the hardware. So it can be measured here and combined with the console-measured
# W and C to predict the on-device gain without a deploy.
#
# What runs:
#   1. llama-batched-bench at batch 1/2/4/8 — measures W and C directly instead of
#      extrapolating them from steady-state prefill, and shows whether a small
#      batch reaches the GEMM efficiency a long prefill does. If it does not, the
#      case for speculation gets worse, not better.
#      This step can also overturn the estimate in the FAVOURABLE direction, and
#      that is why it runs first: W and C above were fitted from two endpoints
#      (batch 1 and a ~300-token prefill) on the assumption T(n) is linear. If a
#      batch of 2-8 is still bandwidth-bound — the extra tokens' compute hiding
#      under the one weight read — the true marginal cost is well below 21.6 ms
#      and every ceiling quoted here is too low. Nothing in the repo measures
#      that middle region; the sweep is the only thing that settles it.
#   2. llama-speculative-simple on the vocab-compatible pairs — reports n_drafted,
#      n_accept and accept% directly (examples/speculative-simple.cpp:342-346).
#   3. llama-lookup on the same prompts — the draft-free n-gram variant, whose
#      draft cost is zero and whose ceiling is therefore 3.3x rather than 1.9x.
#
# Two prompt regimes, reported SEPARATELY: prompt-lookup lives on repetition, so
# averaging a code-editing prompt with an open-ended chat prompt hides exactly the
# fact that decides whether the feature is worth shipping.
#
# The upstream tools are already configured (LLAMA_BUILD_EXAMPLES/TOOLS/COMMON ON
# in CMakeLists.txt) but the build directory accumulates artifacts from several
# pins — rebuild the targets before trusting them, or they fail with a libstdc++
# symbol lookup error.
set -euo pipefail

BUILD_DIR="build/linux-release"
OUT="bench/results/phase15-spec-pregate.csv"
N_PREDICT=128
THREADS=6

while [[ $# -gt 0 ]]; do
	case "$1" in
	--out)
		OUT="$2"
		shift 2
		;;
	--n-predict)
		N_PREDICT="$2"
		shift 2
		;;
	--build)
		BUILD_DIR="$2"
		shift 2
		;;
	--threads)
		THREADS="$2"
		shift 2
		;;
	-h | --help)
		sed -n '2,33p' "$0"
		exit 0
		;;
	*)
		echo "unknown argument: $1" >&2
		exit 2
		;;
	esac
done

BIN="$BUILD_DIR/bin"
for t in llama-batched-bench llama-speculative-simple llama-lookup; do
	if [[ ! -x "$BIN/$t" ]]; then
		echo "missing $BIN/$t — build it first:" >&2
		echo "  cmake --build $BUILD_DIR -j6 --target llama-batched-bench llama-speculative-simple llama-lookup" >&2
		exit 1
	fi
done

CODER_3B="build/coding-models/qwen25-coder-3b-Q4_K_M.gguf"
CODER_05B="build/coding-models/qwen25-coder-0.5b-Q4_K_M.gguf"
LFM_12B="build/coding-models/lfm25-1.2b-thinking-Q4_K_M.gguf"
LFM_350M="$HOME/.cache/xllama-gguf/LFM2.5-350M-Q4_K_M.gguf"

for m in "$CODER_3B" "$CODER_05B" "$LFM_12B" "$LFM_350M"; do
	[[ -f "$m" ]] || {
		echo "missing model: $m" >&2
		exit 1
	}
done

P_CODE="bench/prompts/spec-code-edit.txt"
P_CHAT="bench/prompts/spec-chat-open.txt"

mkdir -p "$(dirname "$OUT")"
if [[ ! -f "$OUT" ]]; then
	echo "kind,target,draft,regime,n_draft_max,n_drafted,n_accept,accept_pct,decode_tok_s,host,date" >"$OUT"
fi

DATE="$(date -Iseconds)"
HOST="linux-host"
LOGDIR="$(mktemp -d)"
trap 'rm -rf "$LOGDIR"' EXIT

# --- 1. batch cost curve: pins W and C for the target ------------------------
# Kept next to the CSV, not in the temp dir: this table IS the W/C measurement,
# and its shape does not fit the per-pair CSV schema.
BATCHED_OUT="${OUT%.csv}-batched.txt"
echo "== batched-bench: qwen25-coder-3b (batch 1/2/4/8) =="
"$BIN/llama-batched-bench" -m "$CODER_3B" -c 4096 -b 2048 -ub 512 \
	-npp 128 -ntg 8 -npl 1,2,4,8 -t "$THREADS" 2>&1 | tee "$BATCHED_OUT" | tail -20

# --- 2/3. acceptance rate, per variant and per regime ------------------------
run_spec() {
	local tgt="$1" tgt_name="$2" dft="$3" dft_name="$4" prompt="$5" regime="$6" kmax="$7"
	echo "== speculative: $tgt_name <- $dft_name [$regime] k=$kmax =="
	local log="$LOGDIR/spec-$tgt_name-$regime.txt"
	# --draft-max / --draft-min were REMOVED in this pin (arg.cpp:4112-4126 turns
	# them into a hard error); the current names are --spec-draft-n-max/-n-min.
	"$BIN/llama-speculative-simple" -m "$tgt" -md "$dft" -f "$prompt" \
		-c 4096 -n "$N_PREDICT" -t "$THREADS" \
		--spec-type draft-simple --spec-draft-n-max "$kmax" --spec-draft-n-min 1 \
		2>&1 | tee "$log" | tail -12 || true
	local drafted accept pct tps
	drafted=$(grep -oP 'n_drafted\s*=\s*\K[0-9]+' "$log" | tail -1)
	accept=$(grep -oP 'n_accept\s*=\s*\K[0-9]+' "$log" | tail -1)
	pct=$(grep -oP 'accept\s*=\s*\K[0-9.]+' "$log" | tail -1)
	tps=$(grep -oP 'decoded.*speed:\s*\K[0-9.]+' "$log" | tail -1)
	echo "draft_model,$tgt_name,$dft_name,$regime,$kmax,${drafted:-},${accept:-},${pct:-},${tps:-},$HOST,$DATE" >>"$OUT"
}

run_lookup() {
	local tgt="$1" tgt_name="$2" prompt="$3" regime="$4" kmax="$5"
	echo "== lookup (n-gram, no draft model): $tgt_name [$regime] k=$kmax =="
	local log="$LOGDIR/lookup-$tgt_name-$regime.txt"
	# llama-lookup is self-contained (it does not load a draft model at all) and
	# reports t_draft as well, which is how the "draft cost is zero" claim gets
	# checked rather than assumed.
	"$BIN/llama-lookup" -m "$tgt" -f "$prompt" -c 4096 -n "$N_PREDICT" \
		-t "$THREADS" --spec-draft-n-max "$kmax" 2>&1 | tee "$log" | tail -14 || true
	local drafted accept pct tps
	drafted=$(grep -oP 'n_drafted\s*=\s*\K[0-9]+' "$log" | tail -1)
	accept=$(grep -oP 'n_accept\s*=\s*\K[0-9]+' "$log" | tail -1)
	pct=$(grep -oP 'accept\s*=\s*\K[0-9.]+' "$log" | tail -1)
	tps=$(grep -oP 'decoded.*speed:\s*\K[0-9.]+' "$log" | tail -1)
	echo "ngram,$tgt_name,-,$regime,$kmax,${drafted:-},${accept:-},${pct:-},${tps:-},$HOST,$DATE" >>"$OUT"
}

for k in 2 4; do
	run_spec "$CODER_3B" qwen25-coder-3b "$CODER_05B" qwen25-coder-0.5b "$P_CODE" code "$k"
	run_spec "$CODER_3B" qwen25-coder-3b "$CODER_05B" qwen25-coder-0.5b "$P_CHAT" chat "$k"
	run_lookup "$CODER_3B" qwen25-coder-3b "$P_CODE" code "$k"
	run_lookup "$CODER_3B" qwen25-coder-3b "$P_CHAT" chat "$k"
done

# LFM is the shipping default family, so it gets a data point too.
run_spec "$LFM_12B" lfm25-1.2b-thinking "$LFM_350M" lfm25-350m "$P_CODE" code 2
run_spec "$LFM_12B" lfm25-1.2b-thinking "$LFM_350M" lfm25-350m "$P_CHAT" chat 2

echo
echo "batched-bench table: $BATCHED_OUT"
echo "wrote $OUT"
