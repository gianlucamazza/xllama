#!/usr/bin/env bash
# bench-spec-w2-console.sh — Phase 15 W2.5 on-console A/B (wrapper).
#
# Runs bench-xbox-ort.sh twice per regime (baseline / --prompt-lookup) and
# writes a compact summary CSV plus the median ratio.
#
# Usage:
#   source ~/.config/xllama/xbox-env
#   ./scripts/bench-spec-w2-console.sh [model] [--runs N] [--n-predict N] [--out FILE]
#
# Default model: qwen25-coder-1.5b if 3b is not on the device (gate model is
# qwen25-coder-3b — upload it first for the declared ≥1.4× PASS).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${XBOX_ENV:-$HOME/.config/xllama/xbox-env}"

MODEL="${1:-}"
shift || true
N_RUNS=4
N_PREDICT=128
THREADS=6
OUT="${REPO_ROOT}/bench/results/phase15-spec-w2-console.csv"
RAW_OUT="${REPO_ROOT}/bench/results/phase15-spec-w2-console-raw.csv"

while [[ $# -gt 0 ]]; do
	case "$1" in
	--runs)
		N_RUNS="$2"
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
	--out)
		OUT="$2"
		shift 2
		;;
	-h | --help)
		sed -n '2,16p' "$0"
		exit 0
		;;
	*)
		# bare model name if first arg was a flag
		if [[ -z "$MODEL" && "$1" != --* ]]; then
			MODEL="$1"
			shift
		else
			echo "unknown: $1" >&2
			exit 2
		fi
		;;
	esac
done

: "${XBOX_IP:?source ~/.config/xllama/xbox-env}"
PFN=$("${SCRIPT_DIR}/deploy.sh" pfn 2>/dev/null || true)
[[ -n "$PFN" ]] || {
	echo "xllama not installed on console" >&2
	exit 1
}

# Auto-pick model if unset: prefer 3b, else 1.5b, else 0.5b.
if [[ -z "$MODEL" ]]; then
	listing=$(curl -sk -u "${XBOX_USER}:${XBOX_PASS}" \
		"https://${XBOX_IP}:11443/api/filesystem/apps/files?knownfolderid=LocalAppData&packagefullname=${PFN}&path=%5CLocalState%5Cmodels" \
		2>/dev/null || true)
	for cand in qwen25-coder-3b qwen25-coder-1.5b qwen25-coder-0.5b; do
		if [[ "$listing" == *"\"Name\": \"${cand}\""* || "$listing" == *"\"Name\":\"${cand}\""* ]]; then
			MODEL="$cand"
			break
		fi
	done
fi
[[ -n "$MODEL" ]] || {
	echo "no coding model on device; upload qwen25-coder-3b first" >&2
	exit 1
}

echo "=== W2 console A/B model=${MODEL} n_predict=${N_PREDICT} runs=${N_RUNS} ==="
: >"$RAW_OUT"
echo "kind,model,regime,prompt_lookup,decode_tok_s_median,n_runs,host,date" >"$OUT"
DATE="$(date -u +%Y-%m-%d)"

run_regime() {
	local regime="$1" prompt="$2" lookup="$3"
	local tag out_csv
	out_csv="${REPO_ROOT}/bench/results/phase15-spec-w2-${regime}-pl${lookup}.csv"
	# Must not leave an empty file: bench-xbox-ort refuses to append when the
	# first line is not the canonical CSV header (empty ≠ header).
	rm -f "$out_csv"
	local args=(
		"$MODEL"
		--threads "$THREADS"
		--n-predict "$N_PREDICT"
		--runs "$N_RUNS"
		--prompt "$prompt"
		--out "$out_csv"
	)
	if [[ "$lookup" == "1" ]]; then
		args+=(--prompt-lookup)
	fi
	echo "--- regime=${regime} prompt_lookup=${lookup} ---"
	"${SCRIPT_DIR}/bench-xbox-ort.sh" "${args[@]}"
	# Append raw rows
	if [[ -s "$out_csv" ]]; then
		tail -n +2 "$out_csv" >>"$RAW_OUT" || true
	fi
	# Median decode from recorded runs (skip header; use field decode_tok_s)
	local med
	med=$(python3 - <<PY
import csv
rows=[]
with open("$out_csv", newline="") as f:
    r=csv.DictReader(f)
    for row in r:
        try:
            rows.append(float(row["decode_tok_s"]))
        except Exception:
            pass
rows=sorted(rows)
if not rows:
    print("0")
else:
    n=len(rows)
    print(rows[n//2] if n%2 else 0.5*(rows[n//2-1]+rows[n//2]))
PY
	)
	local host="xbox-series-s-t${THREADS}"
	[[ "$lookup" == "1" ]] && host="${host}-plookup"
	echo "w2,${MODEL},${regime},${lookup},${med},$(grep -cve '^\s*$' "$out_csv" || echo 0),${host},${DATE}" >>"$OUT"
	echo "  median decode_tok_s=${med}"
}

for regime_pair in "code|${REPO_ROOT}/bench/prompts/spec-code-edit.txt" "chat|${REPO_ROOT}/bench/prompts/spec-chat-open.txt"; do
	regime="${regime_pair%%|*}"
	prompt="${regime_pair##*|}"
	run_regime "$regime" "$prompt" 0
	run_regime "$regime" "$prompt" 1
done

echo
echo "=== Summary (${OUT}) ==="
python3 - <<PY
import csv
from collections import defaultdict
path="$OUT"
by=defaultdict(dict)
with open(path, newline="") as f:
    for row in csv.DictReader(f):
        by[row["regime"]][row["prompt_lookup"]]=float(row["decode_tok_s_median"] or 0)
for regime in ("code","chat"):
    off=by.get(regime,{}).get("0",0.0)
    on=by.get(regime,{}).get("1",0.0)
    ratio=(on/off) if off>0 else 0.0
    if regime=="code":
        gate="PASS" if ratio>=1.4 else "FAIL"
    else:
        gate="PASS" if ratio>=0.98 else "FAIL"
    print(f"{regime}: baseline={off:.2f} plookup={on:.2f} ratio={ratio:.2f}x  [{gate}]")
    if regime=="code" and "$MODEL"!="qwen25-coder-3b":
        print(f"  note: gate model is qwen25-coder-3b; this run used $MODEL")
PY
echo "raw rows: $RAW_OUT"
