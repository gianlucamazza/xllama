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
CURL_AUTH="--digest -u ${XBOX_USER}:${XBOX_PASS} -k -sS"
APP_ID="VenereLabs.xllama"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

dp() {
	# Calls Device Portal REST endpoint; returns body
	curl $CURL_AUTH "$@"
}

get_pfn() {
	# Get Package Full Name (PFN) of xllama from installed packages list
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
	local remote_dir="$3" # e.g. "models" or "" for LocalFolder root
	local filename
	filename=$(basename "$local_path")
	local path_param="\\${remote_dir}"
	[[ -z "$remote_dir" ]] && path_param="\\"
	echo "  Uploading $(basename "$local_path") → LocalFolder/${remote_dir}/ ..."
	dp -X POST \
		-F "file=@${local_path};type=application/octet-stream" \
		"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${pfn}&path=${path_param}" \
		>/dev/null
}

delete_remote_file() {
	local pfn="$1"
	local filename="$2"
	dp -X DELETE \
		"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${pfn}&path=\\${filename}" \
		>/dev/null 2>&1 || true
}

download_remote_file() {
	local pfn="$1"
	local filename="$2"
	local dest="$3"
	dp -o "$dest" \
		"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${pfn}&path=\\${filename}"
}

restart_app() {
	local pfn="$1"
	# Kill running instance
	dp -X DELETE "${BASE_URL}/api/taskmanager/app?package=${pfn}" >/dev/null 2>&1 || true
	sleep 2
	# Launch
	dp -X POST "${BASE_URL}/api/taskmanager/app?appid=${pfn}" >/dev/null 2>&1 || true
}

wait_for_done_marker() {
	local pfn="$1"
	local timeout_s="${2:-300}"
	local elapsed=0
	echo "  Waiting for bench-result.csv.done (timeout: ${timeout_s}s) ..."
	while ((elapsed < timeout_s)); do
		local status
		status=$(dp "${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${pfn}&path=\\bench-result.csv.done" 2>&1) || true
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

# Get PFN
PFN=$(get_pfn)
if [[ -z "$PFN" ]]; then
	echo "Error: xllama not found in installed packages. Deploy it first:" >&2
	echo "  ./scripts/deploy.sh xllama_*.appx" >&2
	exit 1
fi
echo "  PFN: $PFN"

MODEL_FILENAME=$(basename "$MODEL_PATH")

# Upload model if not already present
echo ""
echo "--- Uploading model ---"
upload_file "$MODEL_PATH" "$PFN" "models"

# Write model.txt so the app knows which model to load
TMPDIR_LOCAL=$(mktemp -d)
echo "$MODEL_FILENAME" >"${TMPDIR_LOCAL}/model.txt"
upload_file "${TMPDIR_LOCAL}/model.txt" "$PFN" ""

# Upload prompt
echo ""
echo "--- Uploading prompt ---"
PROMPT_FILE="${REPO_ROOT}/bench/prompts/standard-512.txt"
if [[ -n "$CONFIG" && -f "$CONFIG" ]]; then
	pf=$(python3 -c "import json; d=json.load(open('$CONFIG')); print(d.get('prompt_file',''))" 2>/dev/null || true)
	[[ -n "$pf" ]] && PROMPT_FILE="${REPO_ROOT}/${pf}"
fi
cp "$PROMPT_FILE" "${TMPDIR_LOCAL}/prompt.txt"
upload_file "${TMPDIR_LOCAL}/prompt.txt" "$PFN" ""

# Run N_RUNS iterations
declare -a CSV_ROWS=()
for ((run = 1; run <= N_RUNS; run++)); do
	echo ""
	echo "--- Run $run / $N_RUNS ---"

	# Clean up previous result marker
	delete_remote_file "$PFN" "bench-result.csv"
	delete_remote_file "$PFN" "bench-result.csv.done"
	sleep 1

	# Restart app
	echo "  Starting xllama ..."
	restart_app "$PFN"

	# Wait for completion
	if ! wait_for_done_marker "$PFN" 300; then
		echo "  Run $run timed out, skipping." >&2
		continue
	fi

	# Download result CSV
	local_csv="${TMPDIR_LOCAL}/run${run}.csv"
	download_remote_file "$PFN" "bench-result.csv" "$local_csv"
	# Skip header line, take data row
	data_row=$(tail -n +2 "$local_csv" 2>/dev/null | head -1)
	if [[ -n "$data_row" ]]; then
		CSV_ROWS+=("$data_row")
		echo "  Row: $data_row"
	fi
done

# Compute median of runs 2..N (drop warmup run 1)
echo ""
echo "--- Computing median ---"
RESULT_CSV="${REPO_ROOT}/bench/results/phase1-cpu.csv"

if ((${#CSV_ROWS[@]} < 2)); then
	echo "Warning: fewer than 2 successful runs; using all available." >&2
	MEDIAN_ROW="${CSV_ROWS[0]:-}"
else
	# Use runs 2..N (index 1..)
	VALID_ROWS=("${CSV_ROWS[@]:1}")
	# Extract decode_tok_s (field 7) and find median via sort+middle
	MEDIAN_ROW=$(printf '%s\n' "${VALID_ROWS[@]}" | sort -t, -k7 -n | awk -v n="${#VALID_ROWS[@]}" 'NR==int(n/2)+1')
fi

if [[ -n "$MEDIAN_ROW" ]]; then
	echo "$MEDIAN_ROW" >>"$RESULT_CSV"
	echo "Appended to $RESULT_CSV:"
	echo "  $MEDIAN_ROW"
else
	echo "Error: no valid run data collected." >&2
	exit 1
fi

rm -rf "$TMPDIR_LOCAL"
echo ""
echo "Done. Results in bench/results/phase1-cpu.csv"
