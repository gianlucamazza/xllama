#!/usr/bin/env bash
# bench-spec-w2.sh — Phase 15 W2 host A/B for draft-free prompt-lookup (#210).
#
# Usage:
#   ./scripts/bench-spec-w2.sh --model PATH.gguf [--out FILE] [--n-predict N] \
#       [--threads T] [--cli BIN] [--runs N]
#
# What it measures (host):
#   * Correctness: same greedy text with and without --prompt-lookup on both
#     pre-gate prompt regimes (code + chat). A mismatch is a FAIL, not a speed
#     number.
#   * Acceptance: n_drafted / n_spec_accepted from SPEC_STATS (hardware-
#     independent; compare to phase15-spec-pregate).
#   * Timing on host is logged but NOT a product claim (throttled laptop).
#
# Console timing (the real PASS ≥1.4× gate) is a separate step — deploy the
# same binary and re-run with the console CLI/bench once the MSIX is ready.
# See docs/phase15-re-opt.md §W2.5.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

MODEL=""
OUT="bench/results/phase15-spec-w2-host.csv"
N_PREDICT=128
THREADS=6
RUNS=1
CLI=""
BUILD_DIR="build/linux-release"

while [[ $# -gt 0 ]]; do
	case "$1" in
	--model)
		MODEL="$2"
		shift 2
		;;
	--out)
		OUT="$2"
		shift 2
		;;
	--n-predict)
		N_PREDICT="$2"
		shift 2
		;;
	--threads)
		THREADS="$2"
		shift 2
		;;
	--runs)
		RUNS="$2"
		shift 2
		;;
	--cli)
		CLI="$2"
		shift 2
		;;
	--build)
		BUILD_DIR="$2"
		shift 2
		;;
	-h | --help)
		sed -n '2,22p' "$0"
		exit 0
		;;
	*)
		echo "unknown argument: $1" >&2
		exit 2
		;;
	esac
done

if [[ -z "$MODEL" ]]; then
	# Prefer the shipping coding 3B pin if present; else 0.5B for a fast smoke.
	for cand in \
		build/coding-models/qwen25-coder-3b-Q4_K_M.gguf \
		build/coding-models/qwen25-coder-0.5b-Q4_K_M.gguf; do
		if [[ -f "$cand" ]]; then
			MODEL="$cand"
			break
		fi
	done
fi
if [[ -z "$MODEL" || ! -f "$MODEL" ]]; then
	echo "need --model PATH.gguf (no default GGUF found under build/coding-models/)" >&2
	exit 2
fi

if [[ -z "$CLI" ]]; then
	if [[ -x "$BUILD_DIR/bin/xllama-cli" ]]; then
		CLI="$BUILD_DIR/bin/xllama-cli"
	elif [[ -x "build/linux-test/bin/xllama-cli" ]]; then
		CLI="build/linux-test/bin/xllama-cli"
	else
		echo "xllama-cli not found; build with cmake --preset linux-release && cmake --build build/linux-release -j\$(nproc)" >&2
		exit 2
	fi
fi

P_CODE="bench/prompts/spec-code-edit.txt"
P_CHAT="bench/prompts/spec-chat-open.txt"
for p in "$P_CODE" "$P_CHAT"; do
	[[ -f "$p" ]] || {
		echo "missing prompt fixture: $p" >&2
		exit 2
	}
done

mkdir -p "$(dirname "$OUT")"
HOST="$(hostname -s 2>/dev/null || echo host)"
DATE="$(date -u +%Y-%m-%d)"
MODEL_BASE="$(basename "$MODEL")"

echo "kind,model,regime,prompt_lookup,run_index,n_eval,t_eval_ms,decode_tok_s,n_drafted,n_spec_accepted,peak_ws_mb,text_sha256,success,host,date" >"$OUT"

run_one() {
	local regime="$1" prompt_file="$2" lookup="$3" run_index="$4"
	local args=(-m "$MODEL" -p "$(cat "$prompt_file")" -n "$N_PREDICT" -t "$THREADS" --greedy)
	if [[ "$lookup" == "1" ]]; then
		args+=(--prompt-lookup)
	fi
	local tmp out_txt err_txt
	tmp="$(mktemp)"
	out_txt="$(mktemp)"
	err_txt="$(mktemp)"
	# stdout = streamed tokens; stderr = log + SPEC_STATS
	set +e
	"$CLI" "${args[@]}" >"$out_txt" 2>"$err_txt"
	local rc=$?
	set -e
	local stats
	stats="$(grep -E '^SPEC_STATS ' "$err_txt" | tail -1 || true)"
	local success=0 n_eval=0 t_eval=0 decode=0 drafted=0 accepted=0 peak=0
	if [[ -n "$stats" ]]; then
		# SPEC_STATS success=1 n_eval=... t_eval_ms=... decode_tok_s=... n_drafted=... n_spec_accepted=... peak_ws_mb=...
		success="$(echo "$stats" | grep -oP 'success=\K[0-9]+' || echo 0)"
		n_eval="$(echo "$stats" | grep -oP 'n_eval=\K[0-9]+' || echo 0)"
		t_eval="$(echo "$stats" | grep -oP 't_eval_ms=\K[0-9.]+' || echo 0)"
		decode="$(echo "$stats" | grep -oP 'decode_tok_s=\K[0-9.]+' || echo 0)"
		drafted="$(echo "$stats" | grep -oP 'n_drafted=\K[0-9]+' || echo 0)"
		accepted="$(echo "$stats" | grep -oP 'n_spec_accepted=\K[0-9]+' || echo 0)"
		peak="$(echo "$stats" | grep -oP 'peak_ws_mb=\K[0-9]+' || echo 0)"
	fi
	if [[ $rc -ne 0 ]]; then
		success=0
	fi
	local sha
	sha="$(sha256sum "$out_txt" | awk '{print $1}')"
	echo "w2,$MODEL_BASE,$regime,$lookup,$run_index,$n_eval,$t_eval,$decode,$drafted,$accepted,$peak,$sha,$success,$HOST,$DATE" >>"$OUT"
	# Stash text for equality check (last run of each config wins as reference).
	cp "$out_txt" "$tmp"
	echo "$tmp|$sha|$success"
	rm -f "$err_txt"
}

echo "== W2 host A/B model=$MODEL_BASE n_predict=$N_PREDICT threads=$THREADS runs=$RUNS =="

fail=0
for regime_pair in "code|$P_CODE" "chat|$P_CHAT"; do
	regime="${regime_pair%%|*}"
	prompt_file="${regime_pair##*|}"
	echo "-- regime=$regime"

	ref_sha=""
	for lookup in 0 1; do
		last_sha=""
		last_ok=0
		for ((r = 1; r <= RUNS; r++)); do
			echo "   prompt_lookup=$lookup run=$r ..."
			result="$(run_one "$regime" "$prompt_file" "$lookup" "$r")"
			tmp="${result%%|*}"
			rest="${result#*|}"
			sha="${rest%%|*}"
			ok="${rest##*|}"
			last_sha="$sha"
			last_ok="$ok"
			rm -f "$tmp"
			if [[ "$ok" != "1" ]]; then
				echo "   FAIL: run did not succeed (prompt_lookup=$lookup regime=$regime)" >&2
				fail=1
			fi
		done
		if [[ "$lookup" == "0" ]]; then
			ref_sha="$last_sha"
		else
			if [[ -n "$ref_sha" && "$last_sha" != "$ref_sha" ]]; then
				echo "   FAIL: greedy text mismatch with vs without prompt-lookup (regime=$regime)" >&2
				echo "         baseline=$ref_sha  lookup=$last_sha" >&2
				fail=1
			else
				echo "   OK: greedy text identical (regime=$regime)"
			fi
		fi
		if [[ "$last_ok" != "1" ]]; then
			fail=1
		fi
	done
done

echo
echo "wrote $OUT"
column -t -s, "$OUT" 2>/dev/null || cat "$OUT"

if [[ $fail -ne 0 ]]; then
	echo "W2 host A/B: FAIL" >&2
	exit 1
fi
echo "W2 host A/B: PASS (correctness). Console timing still required for ≥1.4× gate."
exit 0
