#!/usr/bin/env bash
# run-xab.sh — Xbox AI Benchmark orchestration over the existing console harnesses.
#
# Usage:
#   source ~/.config/xllama/xbox-env
#   ./scripts/run-xab.sh --models lfm25-350m,lfm25-1.2b-instruct --include text,kv,h9 --out bench/results/xab-2026-08-26
#   ./scripts/run-xab.sh --dry-run

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROFILE="series-s-devmode"
MODELS="lfm25-230m,lfm25-350m,lfm25-1.2b-instruct"
INCLUDE="text"
OUT="${REPO_ROOT}/bench/results/xab-$(date -u +%F)"
RUNS=4
PROMPT="${REPO_ROOT}/bench/prompts/standard-512.txt"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
	case "$1" in
	--profile) PROFILE="${2:?--profile requires a value}"; shift 2 ;;
	--models) MODELS="${2:?--models requires a comma-separated value}"; shift 2 ;;
	--include) INCLUDE="${2:?--include requires a comma-separated value}"; shift 2 ;;
	--out) OUT="$2"; shift 2 ;;
	--runs) RUNS="$2"; shift 2 ;;
	--prompt) PROMPT="$2"; shift 2 ;;
	--dry-run) DRY_RUN=1; shift ;;
	-h|--help) sed -n '2,9p' "$0"; exit 0 ;;
	*) echo "unknown argument: $1" >&2; exit 2 ;;
	esac
done

[[ "$PROFILE" == "series-s-devmode" ]] || {
	echo "unsupported profile: $PROFILE" >&2
	exit 2
}
[[ -f "$PROMPT" ]] || { echo "prompt not found: $PROMPT" >&2; exit 1; }
[[ "$RUNS" =~ ^[1-9][0-9]*$ ]] || { echo "--runs must be positive" >&2; exit 2; }
IFS=',' read -r -a WORKLOAD_LIST <<<"$INCLUDE"
for workload in "${WORKLOAD_LIST[@]}"; do
	case "$workload" in text|kv|diffusion|h9) ;; *) echo "unsupported workload: $workload" >&2; exit 2 ;; esac
done

mkdir -p "$OUT"
IFS=',' read -r -a MODEL_LIST <<<"$MODELS"
for model in "${MODEL_LIST[@]}"; do
	[[ -n "$model" && "$model" =~ ^[a-zA-Z0-9._-]+$ ]] || { echo "invalid model id: $model" >&2; exit 2; }
done

run_text() {
	local model="$1"
	local result="${OUT}/${model}.csv"
	local cmd=("${SCRIPT_DIR}/bench-xbox-ort.sh" "$model" --runs "$RUNS" --threads 6
		--prompt "$PROMPT" --out "$result")
	printf 'XAB %s\n' "${cmd[*]}"
	if ((DRY_RUN == 0)); then
		"${cmd[@]}"
		python3 "${SCRIPT_DIR}/validate-benchmark.py" "$result"
	fi
}

run_kv() {
	local model="$1"
	local result="${OUT}/${model}-kv.csv"
	local cmd=("${SCRIPT_DIR}/bench-xbox-kv.sh" "$model" --prompt "$PROMPT" --runs "$RUNS" --out "$result")
	printf 'XAB %s\n' "${cmd[*]}"
	if ((DRY_RUN == 0)); then "${cmd[@]}"; fi
}

if [[ ",$INCLUDE," == *,text,* || ",$INCLUDE," == *,kv,* ]]; then
	if ((DRY_RUN == 0)); then
		: "${XBOX_IP:?source ~/.config/xllama/xbox-env}"
		: "${XBOX_USER:?source ~/.config/xllama/xbox-env}"
		: "${XBOX_PASS:?source ~/.config/xllama/xbox-env}"
	fi
fi
for model in "${MODEL_LIST[@]}"; do
	[[ ",$INCLUDE," == *,text,* ]] && run_text "$model"
	[[ ",$INCLUDE," == *,kv,* ]] && run_kv "$model"
done

if [[ ",$INCLUDE," == *,h9,* ]]; then
	cmd=("${SCRIPT_DIR}/eval-xbox-models.sh" --models "$MODELS" --out "${OUT}/h9.jsonl")
	printf 'XAB %s\n' "${cmd[*]}"
	if ((DRY_RUN == 0)); then "${cmd[@]}"; fi
fi
if [[ ",$INCLUDE," == *,diffusion,* ]]; then
	log="${OUT}/diffusion.log"
	cmd=("${SCRIPT_DIR}/validate-console.sh" taesd)
	printf 'XAB %s > %s\n' "${cmd[*]}" "$log"
	if ((DRY_RUN == 0)); then "${cmd[@]}" >"$log" 2>&1; fi
fi

python3 - "$OUT" "$PROFILE" "$PROMPT" "$RUNS" "$INCLUDE" "${MODEL_LIST[@]}" <<'PY'
import json
import sys
from pathlib import Path

out, profile, prompt, runs, include, *models = sys.argv[1:]
payload = {
    "schema_version": 1,
    "profile": profile,
    "workloads": include.split(","),
    "models": models,
    "prompt": str(Path(prompt).as_posix()),
    "runs_requested": int(runs),
    "raw_results": "one CSV per model",
    "sidecars": "required before publishing a headline claim",
}
Path(out, "manifest.json").write_text(
    json.dumps(payload, indent=2) + "\n", encoding="utf-8"
)
PY
echo "Wrote XAB manifest: ${OUT}/manifest.json"
