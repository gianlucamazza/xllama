#!/usr/bin/env bash
# bench-xbox-ort.sh — ORT GenAI benchmark orchestrator for Xbox Series S
#
# Usage:
#   source ~/.config/xllama/xbox-env
#   ./scripts/bench-xbox-ort.sh <model-dir-name> [--threads N] [--runs N] [--prompt file]
#
# Arguments:
#   model-dir-name   Model directory name in LocalState/models/ (e.g. smollm2-360m-cpu-int4)
#   --threads N      Upload genai_config-threads-N.json and tag CSV row with t<N>
#   --runs N         Number of bench runs (default: 3, warmup run 1 is dropped)
#   --prompt file    Path to prompt file (default: bench/prompts/standard-512.txt)
#
# Required env: XBOX_IP, XBOX_USER, XBOX_PASS
#
# Output: median row appended to bench/results/phase1-cpu.csv
#
# Notes:
#   - The model must already be in LocalState\models\<name>\ on the console
#     (copied from MSIX on first launch, or uploaded via deploy.sh upload-dir).
#   - bench.flag is consumed by the app and deleted; this script re-uploads it
#     for each run.
#   - Thread variants require bench-xbox-ort.sh from a v0.3.1+ MSIX that reads
#     bench_threads.txt. With v0.3.0 the n_threads CSV column shows detect_threads().

set -euo pipefail

MODEL_NAME="${1:-smollm2-360m-cpu-int4}"
N_THREADS=0
N_RUNS=3
PROMPT_FILE=""

shift || true
while [[ $# -gt 0 ]]; do
	case "$1" in
	--threads)
		N_THREADS="${2:?--threads requires a value}"
		shift 2
		;;
	--runs)
		N_RUNS="${2:?--runs requires a value}"
		shift 2
		;;
	--prompt)
		PROMPT_FILE="${2:?--prompt requires a file}"
		shift 2
		;;
	*)
		echo "Unknown argument: $1" >&2
		exit 1
		;;
	esac
done

: "${XBOX_IP:?XBOX_IP not set — run: source ~/.config/xllama/xbox-env}"
: "${XBOX_USER:?XBOX_USER not set}"
: "${XBOX_PASS:?XBOX_PASS not set}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEPLOY="${SCRIPT_DIR}/deploy.sh"
BASE_URL="https://${XBOX_IP}:11443"
CURL_AUTH=(--basic -u "${XBOX_USER}:${XBOX_PASS}" -k -sS)

echo "=== xllama bench-xbox-ort ==="
echo "  Model:   $MODEL_NAME"
echo "  Threads: ${N_THREADS:-auto}"
echo "  Runs:    $N_RUNS (run 1 warmup, dropped from median)"
echo "  Xbox:    $XBOX_IP"

# ---------------------------------------------------------------------------
# CSRF token (required for POST/DELETE)
# ---------------------------------------------------------------------------
CSRF_TOKEN=$(curl "${CURL_AUTH[@]}" "${BASE_URL}/" -o /dev/null -D - 2>/dev/null |
	sed -n 's/.*[Cc][Ss][Rr][Ff]-[Tt]oken=\([^;[:space:]]*\).*/\1/p' |
	tr -d '\r' | head -n1)
[[ -z "$CSRF_TOKEN" ]] && echo "Warning: no CSRF token — POST/DELETE may fail" >&2

# ---------------------------------------------------------------------------
# Package full name
# ---------------------------------------------------------------------------
PFN=$("${DEPLOY}" pfn 2>/dev/null)
[[ -z "$PFN" ]] && {
	echo "Error: xllama not found — deploy it first" >&2
	exit 1
}
echo "  PFN: $PFN"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
TMPDIR_LOCAL=$(mktemp -d)
trap 'rm -rf "$TMPDIR_LOCAL"' EXIT

_curl_post_file() {
	local local_path="$1" path_param="$2" filename="${3:-}"
	local form_entry
	if [[ -n "$filename" ]]; then
		form_entry="${local_path};filename=${filename};type=application/octet-stream"
	else
		form_entry="${local_path};type=application/octet-stream"
	fi
	curl "${CURL_AUTH[@]}" -H "X-CSRF-Token:${CSRF_TOKEN}" -X POST \
		-F "file=@${form_entry}" \
		"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=${path_param}" \
		>/dev/null
}

# Upload file to LocalState root (or subdir via remote_dir like "models\\<name>")
upload_to_localstate() {
	local local_path="$1" remote_dir="${2:-}"
	local path_param="%5CLocalState"
	[[ -n "$remote_dir" ]] && path_param="%5CLocalState%5C${remote_dir//\\/%5C}"
	echo "  Uploading $(basename "$local_path") → LocalState\\${remote_dir} ..."
	_curl_post_file "$local_path" "$path_param"
}

# Upload file to LocalState with an explicit remote filename
upload_as() {
	local local_path="$1" remote_dir="${2:-}" remote_name="$3"
	local path_param="%5CLocalState"
	[[ -n "$remote_dir" ]] && path_param="%5CLocalState%5C${remote_dir//\\/%5C}"
	echo "  Uploading $(basename "$local_path") → LocalState\\${remote_dir}\\${remote_name} ..."
	_curl_post_file "$local_path" "$path_param" "$remote_name"
}

# Download file from LocalState root
download_from_localstate() {
	local remote_name="$1" dest="$2"
	curl "${CURL_AUTH[@]}" -o "$dest" \
		"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=%5CLocalState&filename=${remote_name}" \
		2>/dev/null || true
}

# Delete file from LocalState root
delete_from_localstate() {
	local remote_name="$1"
	curl "${CURL_AUTH[@]}" -H "X-CSRF-Token:${CSRF_TOKEN}" -X DELETE \
		"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=%5CLocalState&filename=${remote_name}" \
		>/dev/null 2>&1 || true
}

# Restart app via Device Portal
restart_app() {
	curl "${CURL_AUTH[@]}" -H "X-CSRF-Token:${CSRF_TOKEN}" -X DELETE \
		"${BASE_URL}/api/taskmanager/app?package=${PFN}" >/dev/null 2>&1 || true
	sleep 2
	local pfamily
	# shellcheck disable=SC2001
	pfamily=$(echo "$PFN" | sed 's/_[0-9][0-9.]*_[^_]*__/_/')
	local aumid
	aumid=$(printf '%s!xllama' "$pfamily" | base64 -w0)
	curl "${CURL_AUTH[@]}" -H "X-CSRF-Token:${CSRF_TOKEN}" -X POST -d "" \
		"${BASE_URL}/api/taskmanager/app?appid=${aumid}" >/dev/null 2>&1 || true
}

# Wait for bench-result.csv.done (polling)
wait_for_done() {
	local timeout_s="${1:-300}" elapsed=0
	echo "  Waiting for bench-result.csv.done (timeout ${timeout_s}s)..."
	while ((elapsed < timeout_s)); do
		local resp
		resp=$(curl "${CURL_AUTH[@]}" \
			"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=%5CLocalState&filename=bench-result.csv.done" \
			2>/dev/null) || true
		if [[ -n "$resp" && "$resp" != *"404"* && "$resp" != *"error"* ]]; then
			echo "  Done after ${elapsed}s."
			return 0
		fi
		sleep 10
		((elapsed += 10))
	done
	echo "  Timeout waiting for bench-result.csv.done" >&2
	return 1
}

# Compute median of a numeric CSV column by header name
csv_median() {
	local file="$1" col_name="$2"
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
# Upload genai_config variant for thread tuning (if --threads N specified)
# ---------------------------------------------------------------------------
if [[ "$N_THREADS" -gt 0 ]] 2>/dev/null; then
	THREADS_CONFIG="${REPO_ROOT}/bench/configs/genai_config-threads-${N_THREADS}.json"
	if [[ -f "$THREADS_CONFIG" ]]; then
		echo ""
		echo "--- Uploading genai_config (intra_op_num_threads=${N_THREADS}) ---"
		upload_as "$THREADS_CONFIG" "models\\${MODEL_NAME}" "genai_config.json"
	else
		echo "Warning: $THREADS_CONFIG not found — using existing genai_config.json on device" >&2
	fi
fi

# ---------------------------------------------------------------------------
# Prepare local bench files
# ---------------------------------------------------------------------------
PROMPT_SRC="${PROMPT_FILE:-${REPO_ROOT}/bench/prompts/standard-512.txt}"
[[ ! -f "$PROMPT_SRC" ]] && {
	echo "Error: prompt file not found: $PROMPT_SRC" >&2
	exit 1
}
cp "$PROMPT_SRC" "${TMPDIR_LOCAL}/prompt.txt"

# model.txt — tells inference-bridge which model dir to use
printf '%s' "$MODEL_NAME" >"${TMPDIR_LOCAL}/model.txt"

# bench_threads.txt — tells inference-bridge what n_threads to write in CSV (v0.3.1+)
printf '%d' "$N_THREADS" >"${TMPDIR_LOCAL}/bench_threads.txt"

# bench.flag — consumed by app on each start; must be re-uploaded per run
printf 'bench' >"${TMPDIR_LOCAL}/bench.flag"

# ---------------------------------------------------------------------------
# Run N_RUNS iterations
# ---------------------------------------------------------------------------
declare -a CSV_ROWS=()

for ((run = 1; run <= N_RUNS; run++)); do
	echo ""
	echo "--- Run $run / $N_RUNS ---"

	delete_from_localstate "bench-result.csv"
	delete_from_localstate "bench-result.csv.done"
	sleep 1

	echo "  Uploading bench artifacts..."
	upload_to_localstate "${TMPDIR_LOCAL}/bench.flag"
	upload_to_localstate "${TMPDIR_LOCAL}/prompt.txt"
	upload_to_localstate "${TMPDIR_LOCAL}/model.txt"
	upload_to_localstate "${TMPDIR_LOCAL}/bench_threads.txt"

	echo "  Starting app..."
	restart_app

	if ! wait_for_done 300; then
		echo "  Run $run timed out — skipping." >&2
		continue
	fi

	local_csv="${TMPDIR_LOCAL}/run${run}.csv"
	download_from_localstate "bench-result.csv" "$local_csv"
	data_row=$(tail -n +2 "$local_csv" 2>/dev/null | head -1)
	if [[ -n "$data_row" ]]; then
		CSV_ROWS+=("$data_row")
		echo "  Row: $data_row"
	else
		echo "  Warning: bench-result.csv empty or missing" >&2
	fi
done

# ---------------------------------------------------------------------------
# Compute median and append to phase1-cpu.csv
# ---------------------------------------------------------------------------
echo ""
echo "--- Computing median ---"
RESULT_CSV="${REPO_ROOT}/bench/results/phase1-cpu.csv"
MEDIAN_TMP=$(mktemp)
trap 'rm -f "$MEDIAN_TMP"' EXIT

if ((${#CSV_ROWS[@]} == 0)); then
	echo "Error: no successful runs collected." >&2
	exit 1
elif ((${#CSV_ROWS[@]} < 2)); then
	echo "Warning: only 1 run collected (warmup not dropped)." >&2
	printf '%s\n' "${CSV_ROWS[0]}" >>"$MEDIAN_TMP"
else
	# Write header + rows skipping warmup (run 1 = index 0)
	printf '%s\n' "model,quant,backend,n_ctx,n_threads,prompt_tok_s,decode_tok_s,peak_ws_mb,load_ms,host,date" >>"$MEDIAN_TMP"
	for row in "${CSV_ROWS[@]:1}"; do
		printf '%s\n' "$row" >>"$MEDIAN_TMP"
	done
fi

if [[ -s "$MEDIAN_TMP" ]]; then
	MEDIAN_VAL=$(csv_median "$MEDIAN_TMP" "decode_tok_s")
	MEDIAN_ROW=$(awk -v v="$MEDIAN_VAL" 'NR>1 && $7==v {print; exit}' FS="," "$MEDIAN_TMP")
	if [[ -z "$MEDIAN_ROW" ]]; then
		MEDIAN_ROW=$(tail -n +2 "$MEDIAN_TMP" | head -1)
	fi
	printf '%s\n' "$MEDIAN_ROW" >>"$RESULT_CSV"
	echo "Appended to $RESULT_CSV:"
	echo "  $MEDIAN_ROW"
else
	echo "Error: no data for median computation." >&2
	exit 1
fi

echo ""
echo "Done. bench/results/phase1-cpu.csv updated."
