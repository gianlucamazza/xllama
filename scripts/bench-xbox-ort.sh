#!/usr/bin/env bash
# bench-xbox-ort.sh — ORT GenAI benchmark orchestrator for Xbox Series S
#
# Usage:
#   source ~/.config/xllama/xbox-env
#   ./scripts/bench-xbox-ort.sh <model-dir-name> [--threads N] [--runs N] [--prompt file]
#                                [--out FILE] [--gpu-sample] [--ctx N] [--n-predict N]
#                                [--max-length N] [--keep-config] [--prompt-lookup]
#
# Arguments:
#   model-dir-name   Model directory name in LocalState/models/ (e.g. smollm2-360m-cpu-int4)
#   --threads N      Upload genai_config-threads-N.json and tag CSV row with t<N>.
#                    The device config is backed up and RESTORED on exit (a swap
#                    left in place would silently affect every later run).
#   --keep-config    Leave the --threads config on the device (skip the restore)
#   --runs N         Number of bench runs (default: 4, warmup run 1 is dropped;
#                    runs 2..N are each appended individually with their run_index
#                    so the summary can report a spread instead of a pre-averaged
#                    point — W1.1). Default 4 yields 3 recorded measurement runs.
#   --prompt file    Path to prompt file (default: bench/prompts/standard-512.txt)
#   --out FILE       Results CSV to append the per-run rows to
#                    (default: bench/results/phase1-cpu.csv; DML runs → phase2-dml.csv)
#   --gpu-sample     Sample Device Portal GPU telemetry (xbox-gpu-sample.sh)
#                    across the runs and print a per-engine summary at the end
#   --ctx N          Override n_ctx via bench_ctx.txt (0 = engine default 2048)
#   --n-predict N    Override n_predict via bench_npredict.txt (0 = default 512)
#   --max-length N   Override max_length via bench_maxlen.txt. 0 = derive as
#                    min(n_ctx, prompt+n_predict); -1 = saturate to n_ctx (what
#                    the shipping app does since #135); >0 = explicit value.
#                    On DirectML this is THE variable governing prefill (#130).
#   --ubatch N       Override llama.cpp n_ubatch (physical prefill chunk) via
#                    bench_ubatch.txt (#172). 0 = llama default (512). GGUF
#                    models only; the ORT path ignores it. The device tags the
#                    CSV host column with -uN so the row carries the variable.
#   --kv-q8          q8_0 KV cache + flash attention via bench_kvq8.txt (#171).
#                    GGUF models only. Host column tagged -kvq8 (same rationale
#                    as -uN); the guard fails if the MSIX ignores the knob.
#   --prompt-lookup  Phase 15 W2 (#210): draft-free n-gram speculative decoding
#                    via bench_prompt_lookup.txt=1. Host column tagged -plookup.
#                    Off (file deleted) when the flag is absent so a prior on
#                    run cannot leak into the next.
#
# Required env: XBOX_IP, XBOX_USER, XBOX_PASS
#
# Output: one row per recorded run appended to the --out CSV (run_index column);
#         the summary generator computes the median and min-max spread from them.
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
N_CTX=0     # 0 = engine default (2048)
N_PREDICT=0 # 0 = engine default (512)
MAX_LEN=0   # 0 = derive min(n_ctx, prompt+n_predict); -1 = saturate to n_ctx; >0 = explicit
UBATCH=0    # 0 = llama default (512); #172 sweep knob, GGUF only
KVQ8=0      # 1 = q8_0 KV + flash attention; #171 A/B knob, GGUF only
PROMPT_LOOKUP=0 # 1 = W2 prompt-lookup; #210 A/B knob, GGUF only
N_RUNS=4    # warmup run 1 dropped; runs 2..N recorded individually (W1.1) → 3 by default
PROMPT_FILE=""
OUT_CSV=""
GPU_SAMPLE=false
KEEP_CONFIG=false # --keep-config: leave a --threads genai_config.json on the device

shift || true
while [[ $# -gt 0 ]]; do
	case "$1" in
	--threads)
		N_THREADS="${2:?--threads requires a value}"
		shift 2
		;;
	--ctx)
		N_CTX="${2:?--ctx requires a value}"
		shift 2
		;;
	--n-predict)
		N_PREDICT="${2:?--n-predict requires a value}"
		shift 2
		;;
	--max-length)
		MAX_LEN="${2:?--max-length requires a value}"
		shift 2
		;;
	--ubatch)
		UBATCH="${2:?--ubatch requires a value}"
		shift 2
		;;
	--kv-q8)
		KVQ8=1
		shift
		;;
	--prompt-lookup)
		PROMPT_LOOKUP=1
		shift
		;;
	--runs)
		N_RUNS="${2:?--runs requires a value}"
		shift 2
		;;
	--prompt)
		PROMPT_FILE="${2:?--prompt requires a file}"
		shift 2
		;;
	--out)
		OUT_CSV="${2:?--out requires a file}"
		shift 2
		;;
	--gpu-sample)
		GPU_SAMPLE=true
		shift
		;;
	--keep-config)
		KEEP_CONFIG=true
		shift
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
echo "  n_ctx:   $([[ $N_CTX -gt 0 ]] && echo "$N_CTX" || echo "default")"
echo "  n_predict: $([[ $N_PREDICT -gt 0 ]] && echo "$N_PREDICT" || echo "default")"
if ((MAX_LEN == 0)); then
	MAX_LEN_LABEL="derived"
elif ((MAX_LEN < 0)); then
	MAX_LEN_LABEL="saturated (n_ctx)"
else
	MAX_LEN_LABEL="$MAX_LEN"
fi
echo "  max_length: $MAX_LEN_LABEL"
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
CONFIG_SWAPPED="" # set when --threads overwrites the device genai_config.json

# One cleanup path for the whole script. --threads overwrites genai_config.json
# on the device (models\<name>\); without a restore, the last thread variant
# stays in force for every later run of the app and every bench that does not
# pass --threads (observed: a t8 sweep left the console on t8). Mirror the
# backup/restore that profile-dml-run.sh already does, and run it on ANY exit,
# so a mid-run failure still restores. --keep-config opts out.
cleanup() {
	if [[ -n "$CONFIG_SWAPPED" && "$KEEP_CONFIG" != "true" ]]; then
		echo "  Restoring original genai_config.json on the device..." >&2
		upload_as "${TMPDIR_LOCAL}/genai_config_orig.json" "models\\${MODEL_NAME}" "genai_config.json" >/dev/null 2>&1 || true
	elif [[ -n "$CONFIG_SWAPPED" ]]; then
		echo "  --keep-config: t${N_THREADS} genai_config.json left on the device" >&2
	fi
	rm -rf "$TMPDIR_LOCAL"
}
trap cleanup EXIT

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
	printf '  Uploading %s → LocalState\\%s\\%s ...\n' "$(basename "$local_path")" "$remote_dir" "$remote_name"
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

# Poll until a LocalState file is confirmed absent (GET returns 404 / error body).
# delete_from_localstate is fire-and-forget; without this confirmation a stale
# marker still present when wait_for_done first polls makes it return 0 instantly
# off the PREVIOUS run's bench-result.csv.done — a silent wrong row, not a timeout.
verify_deleted() {
	local remote_name="$1" tries="${2:-15}" resp
	while ((tries-- > 0)); do
		resp=$(curl "${CURL_AUTH[@]}" \
			"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=%5CLocalState&filename=${remote_name}" \
			2>/dev/null) || true
		if [[ -z "$resp" || "$resp" == *"404"* || "$resp" == *"error"* ]]; then
			return 0
		fi
		delete_from_localstate "$remote_name" # retry the delete, then re-check
		sleep 1
	done
	echo "  Warning: could not confirm ${remote_name} was deleted — result may be stale" >&2
	return 1
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

# ---------------------------------------------------------------------------
# Upload genai_config variant for thread tuning (if --threads N specified)
# ORT GenAI only — skip when the model dir is GGUF (llama.cpp uses
# bench_threads.txt / params.n_threads; a genai_config.json next to a .gguf
# is noise and confused earlier campaigns).
# ---------------------------------------------------------------------------
model_dir_has_gguf() {
	local listing
	listing=$(curl "${CURL_AUTH[@]}" \
		"${BASE_URL}/api/filesystem/apps/files?knownfolderid=LocalAppData&packagefullname=${PFN}&path=%5CLocalState%5Cmodels%5C${MODEL_NAME}" \
		2>/dev/null) || true
	[[ "$listing" == *".gguf"* ]]
}

if [[ "$N_THREADS" -gt 0 ]] 2>/dev/null; then
	if model_dir_has_gguf; then
		echo ""
		echo "--- Skipping genai_config (GGUF model; threads via bench_threads.txt) ---"
	else
		THREADS_CONFIG="${REPO_ROOT}/bench/configs/genai_config-threads-${N_THREADS}.json"
		if [[ -f "$THREADS_CONFIG" ]]; then
			echo ""
			echo "--- Uploading genai_config (intra_op_num_threads=${N_THREADS}) ---"
			# Back up the device's current config FIRST, so the EXIT trap can put
			# it back. Without this the swap is permanent (see cleanup()).
			curl "${CURL_AUTH[@]}" -o "${TMPDIR_LOCAL}/genai_config_orig.json" \
				"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=%5CLocalState%5Cmodels%5C${MODEL_NAME}&filename=genai_config.json" \
				2>/dev/null || true
			if [[ -s "${TMPDIR_LOCAL}/genai_config_orig.json" ]]; then
				CONFIG_SWAPPED=1
			else
				echo "Warning: could not back up the device genai_config.json — it will NOT be restored" >&2
			fi
			upload_as "$THREADS_CONFIG" "models\\${MODEL_NAME}" "genai_config.json"
		else
			echo "Warning: $THREADS_CONFIG not found — using existing genai_config.json on device" >&2
		fi
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
# 0 = leave the engine default; #130 varies these to test the band hypothesis.
printf '%d' "$N_CTX" >"${TMPDIR_LOCAL}/bench_ctx.txt"
printf '%d' "$N_PREDICT" >"${TMPDIR_LOCAL}/bench_npredict.txt"
# Always written, even when 0 — main_loop only ever overwrites these files, so a
# value left by a previous run would otherwise stay in force.
printf '%d' "$MAX_LEN" >"${TMPDIR_LOCAL}/bench_maxlen.txt"
printf '%d' "$UBATCH" >"${TMPDIR_LOCAL}/bench_ubatch.txt"
printf '%d' "$KVQ8" >"${TMPDIR_LOCAL}/bench_kvq8.txt"
printf '%d' "$PROMPT_LOOKUP" >"${TMPDIR_LOCAL}/bench_prompt_lookup.txt"

# bench.flag — consumed by app on each start; must be re-uploaded per run
printf 'bench' >"${TMPDIR_LOCAL}/bench.flag"

# ---------------------------------------------------------------------------
# Run N_RUNS iterations
# ---------------------------------------------------------------------------
declare -a CSV_ROWS=()

# Defined before the run loop, not just before the median: each downloaded row is
# checked against this schema's field count as it arrives (see below).
CSV_HEADER="model,quant,backend,n_ctx,n_threads,prompt_tok_s,decode_tok_s,peak_ws_mb,load_ms,gpu_mem_mb,gpu_budget_mb,n_prompt_tok,n_gen_tok,max_length,host,date,run_index"

SAMPLER_PID=""
if [[ "$GPU_SAMPLE" == "true" ]]; then
	echo "  Starting GPU sampler (systemperf)..."
	"${SCRIPT_DIR}/xbox-gpu-sample.sh" --out "${TMPDIR_LOCAL}/gpu-sample.csv" \
		>"${TMPDIR_LOCAL}/gpu-summary.txt" 2>&1 &
	SAMPLER_PID=$!
fi

for ((run = 1; run <= N_RUNS; run++)); do
	echo ""
	echo "--- Run $run / $N_RUNS ---"

	delete_from_localstate "bench-result.csv"
	delete_from_localstate "bench-result.csv.done"
	# Confirm the marker is actually gone before starting the app, so wait_for_done
	# can't return off a stale bench-result.csv.done from the previous run.
	verify_deleted "bench-result.csv.done"
	# Remove any leftover bench_turns.txt: main_loop treats its presence as the
	# KV-reuse bench trigger, which would hijack this standard bench into kv-bench
	# mode (writes bench-kv-result.csv, never bench-result.csv → this run times out).
	delete_from_localstate "bench_turns.txt"
	sleep 1

	# run_index changes per iteration (unlike the other bench_*.txt knobs), so it
	# is written inside the loop: the device echoes it into the CSV row so repeats
	# are individually recoverable rather than pre-averaged here (W1.1).
	printf '%d' "$run" >"${TMPDIR_LOCAL}/bench_run_index.txt"

	echo "  Uploading bench artifacts..."
	upload_to_localstate "${TMPDIR_LOCAL}/bench.flag"
	upload_to_localstate "${TMPDIR_LOCAL}/prompt.txt"
	upload_to_localstate "${TMPDIR_LOCAL}/model.txt"
	upload_to_localstate "${TMPDIR_LOCAL}/bench_threads.txt"
	upload_to_localstate "${TMPDIR_LOCAL}/bench_ctx.txt"
	upload_to_localstate "${TMPDIR_LOCAL}/bench_npredict.txt"
	upload_to_localstate "${TMPDIR_LOCAL}/bench_maxlen.txt"
	upload_to_localstate "${TMPDIR_LOCAL}/bench_ubatch.txt"
	upload_to_localstate "${TMPDIR_LOCAL}/bench_kvq8.txt"
	if ((PROMPT_LOOKUP != 0)); then
		upload_to_localstate "${TMPDIR_LOCAL}/bench_prompt_lookup.txt"
	else
		# A prior --prompt-lookup run must not leave the knob on.
		delete_from_localstate "bench_prompt_lookup.txt"
		verify_deleted "bench_prompt_lookup.txt" 5 || true
	fi
	upload_to_localstate "${TMPDIR_LOCAL}/bench_run_index.txt"

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
		# The device writes the row; this script owns the header. A build older
		# than the current schema emits fewer fields and they get appended under
		# our header anyway, shifting every column silently (a 14-field row from a
		# pre-n_gen_tok build puts `host` under n_gen_tok and `date` under host).
		# assert_header_matches only guards the local file, so nothing else here
		# would notice. Observed 2026-07-21 with MSIX 1.4.0.615.
		n_fields=$(awk -F, '{print NF}' <<<"$data_row")
		want_fields=$(awk -F, '{print NF}' <<<"$CSV_HEADER")
		if ((n_fields != want_fields)); then
			echo "Error: the console wrote a ${n_fields}-field row; this script expects ${want_fields}." >&2
			echo "  row:    $data_row" >&2
			echo "  header: $CSV_HEADER" >&2
			echo "The installed MSIX is older than the CSV schema — redeploy before benchmarking." >&2
			exit 1
		fi
		# Before anything else: did the device measure the model we ASKED for?
		# model.txt reaches the console over WDP, and a WDP POST can report
		# success while writing nothing (the missing-X-CSRF-Token failure mode
		# documented in uwp-constraints). A build older than 1.5.2 answered a
		# missing model.txt by silently benching smollm2-360m-cpu-int4, so a lost
		# upload produced a genuine row for the wrong model, appended to the
		# results file of the run that was requested. Current builds refuse and
		# write no CSV; this check also covers the older ones, and any future way
		# the two can drift apart.
		got_model=$(awk -F, '{print $1}' <<<"$data_row")
		if [[ "$got_model" != "$MODEL_NAME" ]]; then
			echo "Error: the console benched '${got_model}', not the requested '${MODEL_NAME}'." >&2
			echo "  model.txt likely never reached the device (WDP writes can fail silently)." >&2
			echo "  Nothing was appended — re-run rather than trusting this row." >&2
			exit 1
		fi
		# The CSV schema does NOT change with --max-length, so the arity check
		# above cannot catch an MSIX that ignores bench_maxlen.txt — it would
		# record DERIVED max_lengths under a "saturated" label and the whole
		# experiment would be quietly wrong. Compare what we asked for against
		# what the device reports in the max_length column.
		if ((MAX_LEN != 0)); then
			got=$(awk -F, '{print $14}' <<<"$data_row")
			if ((MAX_LEN < 0)); then
				want=$((N_CTX > 0 ? N_CTX : 2048))
			else
				want=$MAX_LEN
			fi
			if ((got != want)); then
				echo "Error: the console ignored --max-length: row says ${got}, asked ${want}." >&2
				echo "  The installed MSIX predates bench_maxlen.txt — redeploy before measuring." >&2
				exit 1
			fi
		fi
		# Same hazard for --ubatch: the schema has no ubatch column, so an MSIX
		# that ignores bench_ubatch.txt would record default-512 rows under a
		# sweep label. The device tags the host column with -uN; require it.
		if ((UBATCH != 0)); then
			got_host=$(awk -F, '{print $15}' <<<"$data_row")
			if [[ "$got_host" != *"-u${UBATCH}"* ]]; then
				echo "Error: the console ignored --ubatch: host column says '${got_host}', asked -u${UBATCH}." >&2
				echo "  The installed MSIX predates bench_ubatch.txt — redeploy before measuring." >&2
				exit 1
			fi
		fi
		# And for --kv-q8 (#171): require the -kvq8 host tag.
		if ((KVQ8 != 0)); then
			got_host=$(awk -F, '{print $15}' <<<"$data_row")
			if [[ "$got_host" != *"-kvq8"* ]]; then
				echo "Error: the console ignored --kv-q8: host column says '${got_host}'." >&2
				echo "  The installed MSIX predates bench_kvq8.txt — redeploy before measuring." >&2
				exit 1
			fi
		fi
		CSV_ROWS+=("$data_row")
		echo "  Row: $data_row"
	else
		echo "  Warning: bench-result.csv empty or missing" >&2
	fi
done

if [[ -n "$SAMPLER_PID" ]]; then
	kill -TERM "$SAMPLER_PID" 2>/dev/null || true
	wait "$SAMPLER_PID" 2>/dev/null || true
	if [[ -s "${TMPDIR_LOCAL}/gpu-summary.txt" ]]; then
		echo ""
		cat "${TMPDIR_LOCAL}/gpu-summary.txt"
	fi
fi

# ---------------------------------------------------------------------------
# Append each recorded run to the results CSV
# ---------------------------------------------------------------------------
# W1.1: the driver no longer pre-averages. Every recorded run is appended with the
# run_index the device wrote, so the spread is recoverable from the committed CSV;
# scripts/generate-benchmark-summary.py computes the median and min-max from them.
echo ""
echo "--- Appending recorded runs ---"
RESULT_CSV="${OUT_CSV:-${REPO_ROOT}/bench/results/phase1-cpu.csv}"

# Warmup drop matches the previous median's warmup exclusion: run 1 (CSV_ROWS[0])
# is a cold cache and is not recorded, UNLESS it is the only run collected.
if ((${#CSV_ROWS[@]} == 0)); then
	echo "Error: no successful runs collected." >&2
	exit 1
elif ((${#CSV_ROWS[@]} < 2)); then
	echo "Warning: only 1 run collected (warmup not dropped)." >&2
	ROWS_TO_APPEND=("${CSV_ROWS[0]}")
else
	ROWS_TO_APPEND=("${CSV_ROWS[@]:1}") # skip warmup (run 1 = index 0)
fi

# Route DML rows to phase2-dml.csv when --out was not given. All rows in one
# campaign share a backend, so decide from the first row to append.
if [[ -z "$OUT_CSV" && "$(cut -d, -f3 <<<"${ROWS_TO_APPEND[0]}")" == *dml* ]]; then
	RESULT_CSV="${REPO_ROOT}/bench/results/phase2-dml.csv"
fi
[[ ! -f "$RESULT_CSV" ]] && printf '%s\n' "$CSV_HEADER" >"$RESULT_CSV"

# Refuse to append to a file written under an older schema: the row arity would
# not match its header, and csv.DictReader would silently misalign every field
# after the mismatch (host/date landing under the wrong keys). Pick a new --out.
assert_header_matches() {
	local csv="$1" existing
	existing=$(head -1 "$csv")
	if [[ "$existing" != "$CSV_HEADER" ]]; then
		echo "Error: $csv uses a different CSV schema." >&2
		echo "  file:     $existing" >&2
		echo "  expected: $CSV_HEADER" >&2
		echo "Append to a new file with --out, or migrate the old one first." >&2
		exit 1
	fi
}
assert_header_matches "$RESULT_CSV"

for row in "${ROWS_TO_APPEND[@]}"; do
	printf '%s\n' "$row" >>"$RESULT_CSV"
	echo "  appended: $row"
done
echo "Appended ${#ROWS_TO_APPEND[@]} run(s) to $RESULT_CSV"

echo ""
echo "Done. ${RESULT_CSV} updated."
