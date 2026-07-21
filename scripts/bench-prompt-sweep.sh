#!/usr/bin/env bash
# bench-prompt-sweep.sh — sweep prompt length across backends on the console.
#
# Usage:
#   source ~/.config/xllama/xbox-env
#   ./scripts/bench-prompt-sweep.sh [--models "a b"] [--tokens "150 300 ..."]
#                                   [--runs N] [--out FILE]
#                                   [--ctx N] [--n-predict N]
#
# Why this exists: the 600-token routing threshold (include/xllama/routing_policy.h)
# is not a measured value. It is the midpoint between two sample points — 285 and
# ~1050 prompt tokens — so the real CPU/GPU prefill crossover is unlocalised across
# a 765-token interval. Worse, that matrix was taken on the pre-#91 asset
# `smollm2-360m-dml-fp16`, which dml_text_model_ok() now excludes; the shipping
# `-v2` asset has no long-prompt row anywhere in bench/results/. This sweep
# measures the crossover on the asset routing actually uses.
#
# Each point is a full bench-xbox-ort.sh invocation (upload prompt/model, restart
# the app headless via bench.flag, poll, median of --runs minus the warmup). The
# actual prefill token count lands in the n_prompt_tok column, so the targets
# below only need to spread the samples — they do not need to be exact.
#
# Runtime: ~40-60 s per run. Defaults (6 lengths x 2 backends x 4 runs) ≈ 40 min.
set -euo pipefail

: "${XBOX_IP:?XBOX_IP not set — run: source ~/.config/xllama/xbox-env}"
: "${XBOX_USER:?XBOX_USER not set}"
: "${XBOX_PASS:?XBOX_PASS not set}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODELS="smollm2-360m-cpu-int4 smollm2-360m-dml-fp16-v2"
TOKENS="150 300 550 800 1100 1600"
RUNS=4
CTX=0      # 0 = engine default; #130 varies it to test the band hypothesis
NPREDICT=0 # 0 = engine default
OUT="${REPO_ROOT}/bench/results/phase12-dml-crossover.csv"

while [[ $# -gt 0 ]]; do
	case "$1" in
	--models)
		MODELS="$2"
		shift 2
		;;
	--tokens)
		TOKENS="$2"
		shift 2
		;;
	--runs)
		RUNS="$2"
		shift 2
		;;
	--ctx)
		CTX="$2"
		shift 2
		;;
	--n-predict)
		NPREDICT="$2"
		shift 2
		;;
	--out)
		OUT="$2"
		shift 2
		;;
	*)
		echo "Unknown arg: $1" >&2
		exit 1
		;;
	esac
done

TMPDIR_LOCAL=$(mktemp -d)
trap 'rm -rf "$TMPDIR_LOCAL"' EXIT

# Build one prompt file per target length from the long-1k prose. The ChatML
# wrapper is stripped: main_loop re-applies the model's chat template, so leaving
# the markup in would double-template the prompt (the mistake baked into
# bench/prompts/long-1k.txt itself).
echo "--- Generating prompts ---"
# shellcheck disable=SC2086  # word splitting of $TOKENS is intended
python3 - "$REPO_ROOT" "$TMPDIR_LOCAL" $TOKENS <<'PY'
import re, sys
repo, tmp = sys.argv[1], sys.argv[2]
targets = [int(t) for t in sys.argv[3:]]
body = open(f"{repo}/bench/prompts/long-1k.txt").read()
m = re.search(r'<\|im_start\|>user\n(.*?)(?:<\|im_end\|>|$)', body, re.S)
pool = (m.group(1) if m else body).strip().split()
# ~1.3 tokens per English word; repeat the pool when a target outruns the source.
for t in targets:
    n_words = max(1, int(t / 1.3))
    words = (pool * (n_words // len(pool) + 1))[:n_words]
    path = f"{tmp}/prompt-{t}.txt"
    open(path, "w").write(" ".join(words))
    print(f"  target {t:>5} tok -> {n_words} words, {len(' '.join(words))} bytes")
PY

echo ""
echo "--- Sweep: $(wc -w <<<"$MODELS") backends x $(wc -w <<<"$TOKENS") lengths x ${RUNS} runs ---"
echo "Output: $OUT"
echo ""

FAILED=0
for model in $MODELS; do
	for t in $TOKENS; do
		echo "=========================================================="
		echo "  $model @ target ${t} tok"
		echo "=========================================================="
		extra=()
		((CTX > 0)) && extra+=(--ctx "$CTX")
		((NPREDICT > 0)) && extra+=(--n-predict "$NPREDICT")
		if ! "${SCRIPT_DIR}/bench-xbox-ort.sh" "$model" \
			--prompt "${TMPDIR_LOCAL}/prompt-${t}.txt" \
			--runs "$RUNS" --out "$OUT" "${extra[@]+"${extra[@]}"}"; then
			echo "  WARN: point failed ($model @ ${t} tok) — continuing" >&2
			FAILED=$((FAILED + 1))
		fi
		echo ""
	done
done

echo "=========================================================="
if ((FAILED > 0)); then
	echo "Sweep finished with ${FAILED} failed point(s) — the CSV is incomplete."
else
	echo "Sweep complete."
fi
if [[ -f "$OUT" ]]; then
	echo "Rows in $OUT:"
	column -s, -t "$OUT"
else
	echo "No output file — every point failed." >&2
fi
exit $((FAILED > 0 ? 1 : 0))
