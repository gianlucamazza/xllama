#!/usr/bin/env bash
# bench-xbox.sh — end-to-end benchmark orchestrator for Xbox Series S
#
# Usage:
#   ./scripts/bench-xbox.sh <model.gguf> [config.json] [n_runs]
#
# Required env vars:
#   XBOX_IP, XBOX_USER, XBOX_PASS
#
# Example:
#   export XBOX_IP=192.168.1.42 XBOX_USER=devuser XBOX_PASS=secret
#   ./scripts/bench-xbox.sh ~/models/qwen3-1.7b-Q4_K_M.gguf bench/config/phase1-qwen3-1.7b.json

set -euo pipefail

MODEL_PATH="${1:-}"
CONFIG="${2:-}"
N_RUNS="${3:-3}"

if [[ -z "$MODEL_PATH" ]]; then
	echo "Usage: $0 <model.gguf> [config.json] [n_runs]" >&2
	exit 1
fi
if [[ ! -f "$MODEL_PATH" ]]; then
	echo "Error: model file not found: $MODEL_PATH" >&2
	exit 1
fi

: "${XBOX_IP:?XBOX_IP not set}"
: "${XBOX_USER:?XBOX_USER not set}"
: "${XBOX_PASS:?XBOX_PASS not set}"

BASE_URL="https://${XBOX_IP}:11443"
CURL_AUTH=(--basic -u "${XBOX_USER}:${XBOX_PASS}" -k -sS)
APP_ID="VenereLabs.xllama"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Extract a JSON string field using jq (preferred) or python3 (fallback).
json_get_string() {
	local file="$1"
	local key="$2"
	if command -v jq &>/dev/null; then
		jq -r "${key} // empty" "$file"
	else
		python3 -c "import json,sys; d=json.load(open('$file')); print(d.get('$key',''))" 2>/dev/null || true
	fi
}

# Compute median of a numeric column from a CSV by header name.
# Usage: csv_median <file> <header_name>
csv_median() {
	local file="$1"
	local col_name="$2"
	awk -v col="$col_name" '
    NR == 1 {
        for (i = 1; i <= NF; i++) {
            gsub(/^"|"$/, "", $i)
            if ($i == col) { c = i; break }
        }
        next
    }
    c { vals[++n] = $c }
    END {
        if (n == 0) exit 1
        # sort via system call (gawk extension)
        cmd = "sort -n"
        for (i = 1; i <= n; i++) printf "%.6f\n", vals[i] | cmd
        close(cmd)
    }
    ' FS="," "$file" | awk '{
        a[NR] = $1
    } END {
        n = NR
        if (n == 0) exit 1
        if (n % 2 == 1) { print a[int(n/2)+1]; exit }
        print (a[n/2] + a[n/2+1]) / 2
    }'
}

# ---------------------------------------------------------------------------
# Xbox Device Portal helpers
# ---------------------------------------------------------------------------

dp() {
	curl "${CURL_AUTH[@]}" "$@"
}

dpw() {
	curl "${CURL_AUTH[@]}" -H "X-CSRF-Token:${CSRF_TOKEN}" "$@"
}

get_pfn() {
	dp "${BASE_URL}/api/app/packagemanager/packages" |
		python3 -c "
import sys, json
data = json.load(sys.stdin)
packages = data.get('InstalledPackages', [])
for p in packages:
    if '${APP_ID}' in p.get('PackageRelativeId', ''):
        print(p.get('PackageFullName', ''))
        break
" 2>/dev/null
}

upload_file() {
	local local_path="$1"
	local pfn="$2"
	local remote_dir="$3"
	local path_param="\\LocalState\\${remote_dir}"
	[[ -z "$remote_dir" ]] && path_param="\\LocalState"
	echo "  Uploading $(basename "$local_path") → LocalFolder/${remote_dir}/ ..."
	dpw -X POST \
		-F "file=@${local_path};type=application/octet-stream" \
		"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${pfn}&path=${path_param}" \
		>/dev/null
}

delete_remote_file() {
	local pfn="$1"
	local filename="$2"
	dpw -X DELETE \
		"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${pfn}&path=\\LocalState&filename=${filename}" \
		>/dev/null 2>&1 || true
}

download_remote_file() {
	local pfn="$1"
	local filename="$2"
	local dest="$3"
	dp -o "$dest" \
		"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${pfn}&path=\\LocalState&filename=${filename}"
}

restart_app() {
	local pfn="$1"
	dpw -X DELETE "${BASE_URL}/api/taskmanager/app?package=${pfn}" >/dev/null 2>&1 || true
	sleep 2
	local pfamily
	# shellcheck disable=SC2001  # regex uses [^_] class, not expressible with bash ${//}
	pfamily=$(echo "$pfn" | sed 's/_[0-9][0-9.]*_[^_]*__/_/')
	local aumid
	aumid=$(printf '%s!xllama' "$pfamily" | base64 -w0)
	dpw -X POST -d "" "${BASE_URL}/api/taskmanager/app?appid=${aumid}" >/dev/null 2>&1 || true
}

wait_for_done_marker() {
	local pfn="$1"
	local timeout_s="${2:-300}"
	local elapsed=0
	echo "  Waiting for bench-result.csv.done (timeout: ${timeout_s}s) ..."
	while ((elapsed < timeout_s)); do
		local status
		status=$(dp "${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${pfn}&path=\\LocalState&filename=bench-result.csv.done" 2>&1) || true
		if [[ "$status" != *"404"* && -n "$status" ]]; then
			echo "  Done marker found after ${elapsed}s."
			return 0
		fi
		sleep 10
		((elapsed += 10))
	done
	echo "  Timeout waiting for done marker!" >&2
	return 1
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

echo "=== xllama bench-xbox ==="
echo "  Model:  $MODEL_PATH"
echo "  Runs:   $N_RUNS (dropping run 1 as warmup)"
echo "  Xbox:   $XBOX_IP"

CSRF_TOKEN=$(dp "${BASE_URL}/" -o /dev/null -D - 2>/dev/null |
	sed -n 's/.*[Cc][Ss][Rr][Ff]-[Tt]oken=\([^;[:space:]]*\).*/\1/p' |
	tr -d '\r' | head -n 1)

if [[ -z "$CSRF_TOKEN" ]]; then
	echo "Warning: failed to extract CSRF token. POST requests may fail." >&2
fi

PFN=$(get_pfn)
if [[ -z "$PFN" ]]; then
	echo "Error: xllama not found in installed packages. Deploy it first:" >&2
	echo "  ./scripts/deploy.sh xllama_*.appx" >&2
	exit 1
fi
echo "  PFN: $PFN"

MODEL_FILENAME=$(basename "$MODEL_PATH")
TMPDIR_LOCAL=$(mktemp -d)
trap 'rm -rf "$TMPDIR_LOCAL"' EXIT

echo ""
echo "--- Uploading model ---"
upload_file "$MODEL_PATH" "$PFN" ""

echo "$MODEL_FILENAME" >"${TMPDIR_LOCAL}/model.txt"
upload_file "${TMPDIR_LOCAL}/model.txt" "$PFN" ""

echo "bench" >"${TMPDIR_LOCAL}/bench.flag"
upload_file "${TMPDIR_LOCAL}/bench.flag" "$PFN" ""

echo ""
echo "--- Uploading prompt ---"
PROMPT_FILE="${REPO_ROOT}/bench/prompts/standard-512.txt"
if [[ -n "$CONFIG" && -f "$CONFIG" ]]; then
	pf=$(json_get_string "$CONFIG" "prompt_file")
	[[ -n "$pf" ]] && PROMPT_FILE="${REPO_ROOT}/${pf}"
fi
cp "$PROMPT_FILE" "${TMPDIR_LOCAL}/prompt.txt"
upload_file "${TMPDIR_LOCAL}/prompt.txt" "$PFN" ""

# Run N_RUNS iterations
declare -a CSV_ROWS=()
for ((run = 1; run <= N_RUNS; run++)); do
	echo ""
	echo "--- Run $run / $N_RUNS ---"

	delete_remote_file "$PFN" "bench-result.csv"
	delete_remote_file "$PFN" "bench-result.csv.done"
	sleep 1

	echo "  Starting xllama ..."
	restart_app "$PFN"

	if ! wait_for_done_marker "$PFN" 300; then
		echo "  Run $run timed out, skipping." >&2
		continue
	fi

	local_csv="${TMPDIR_LOCAL}/run${run}.csv"
	download_remote_file "$PFN" "bench-result.csv" "$local_csv"
	data_row=$(tail -n +2 "$local_csv" 2>/dev/null | head -1)
	if [[ -n "$data_row" ]]; then
		CSV_ROWS+=("$data_row")
		echo "  Row: $data_row"
	fi
done

# Collect all valid rows (excluding warmup run 1) into a temp CSV for median.
echo ""
echo "--- Computing median ---"
RESULT_CSV="${REPO_ROOT}/bench/results/phase1-cpu.csv"

MEDIAN_TMP=$(mktemp)
trap 'rm -f "$MEDIAN_TMP"' EXIT

if ((${#CSV_ROWS[@]} < 2)); then
	echo "Warning: fewer than 2 successful runs; using all available." >&2
	if ((${#CSV_ROWS[@]} > 0)); then
		echo "${CSV_ROWS[0]}" >>"$MEDIAN_TMP"
	fi
else
	# Write header + valid rows (skip warmup) to temp CSV
	printf '%s\n' "model,quant,backend,n_ctx,n_threads,prompt_tok_s,decode_tok_s,peak_ws_mb,load_ms,host,date" >>"$MEDIAN_TMP"
	for row in "${CSV_ROWS[@]:1}"; do
		printf '%s\n' "$row" >>"$MEDIAN_TMP"
	done
fi

if [[ -s "$MEDIAN_TMP" ]]; then
	MEDIAN_VAL=$(csv_median "$MEDIAN_TMP" "decode_tok_s")
	MEDIAN_ROW=$(awk -v v="$MEDIAN_VAL" 'NR>1 && $7 == v {print; exit}' FS="," "$MEDIAN_TMP")
	if [[ -z "$MEDIAN_ROW" ]]; then
		# fallback: first data row if exact match fails (floating point)
		MEDIAN_ROW=$(tail -n +2 "$MEDIAN_TMP" | head -1)
	fi
	echo "$MEDIAN_ROW" >>"$RESULT_CSV"
	echo "Appended to $RESULT_CSV:"
	echo "  $MEDIAN_ROW"
else
	echo "Error: no valid run data collected." >&2
	exit 1
fi

echo ""
echo "Done. Results in bench/results/phase1-cpu.csv"
