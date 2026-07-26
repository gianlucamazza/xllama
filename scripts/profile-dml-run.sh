#!/usr/bin/env bash
# profile-dml-run.sh — one profiled DML inference on Xbox, fetch + analyze the
# ORT profiling JSON to prove GPU vs CPU execution (no PIX needed).
#
# Usage:
#   source ~/.config/xllama/xbox-env
#   ./scripts/profile-dml-run.sh [--model NAME] [--prompt FILE] [--gpu-sample]
#                                [--absolute-prefix] [--keep-config] [--no-analyze]
#
# Flow:
#   1. Back up the on-device genai_config.json, upload the profiling variant
#      (bench/configs/genai_config-dml-profile.json: DML EP + enable_profiling
#      + log_severity_level 0).
#   2. Run one bench inference (bench.flag mechanism), wait for the done marker.
#   3. Locate ort_profile_*.json (LocalState root, then models\<name>\ —
#      the profile lands relative to the AppContainer CWD), download it plus
#      the new xllama.log tail and bench-result.csv into
#      bench/results/profiles/<UTC-timestamp>/.
#   4. Restore the original genai_config.json (unless --keep-config).
#   5. Run scripts/analyze_ort_profile.py → greppable "VERDICT:" line.
#
# --absolute-prefix: profile-location ladder step 2 — render
#   genai_config-dml-profile.tpl.json with an absolute LocalState prefix
#   (use when the relative prefix produced no file; the definitive fix is the
#   in-app CWD pin, see docs/uwp-constraints.md).
# --gpu-sample: sample Device Portal GPU telemetry during the run
#   (scripts/xbox-gpu-sample.sh) and print its summary next to the verdict.

set -euo pipefail

MODEL="smollm2-360m-cpu-int4"
PROMPT_FILE=""
GPU_SAMPLE=false
ABSOLUTE_PREFIX=false
KEEP_CONFIG=false
NO_ANALYZE=false

while [[ $# -gt 0 ]]; do
	case "$1" in
	--model)
		MODEL="${2:?--model requires a name}"
		shift 2
		;;
	--prompt)
		PROMPT_FILE="${2:?--prompt requires a file}"
		shift 2
		;;
	--gpu-sample)
		GPU_SAMPLE=true
		shift
		;;
	--absolute-prefix)
		ABSOLUTE_PREFIX=true
		shift
		;;
	--keep-config)
		KEEP_CONFIG=true
		shift
		;;
	--no-analyze)
		NO_ANALYZE=true
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
MODEL_DIR="models\\${MODEL}"

echo "=== xllama profile-dml-run ==="
echo "  Model: $MODEL"
echo "  Xbox:  $XBOX_IP"

# --------------------------------------------------------------------------
# CSRF token (required for POST/DELETE) + package full name
# --------------------------------------------------------------------------
CSRF_TOKEN=$(curl "${CURL_AUTH[@]}" "${BASE_URL}/" -o /dev/null -D - 2>/dev/null |
	sed -n 's/.*[Cc][Ss][Rr][Ff]-[Tt]oken=\([^;[:space:]]*\).*/\1/p' |
	tr -d '\r' | head -n1)
[[ -z "$CSRF_TOKEN" ]] && echo "Warning: no CSRF token — POST/DELETE may fail" >&2

PFN=$("${DEPLOY}" pfn 2>/dev/null)
[[ -z "$PFN" ]] && {
	echo "Error: xllama not found — deploy it first" >&2
	exit 1
}
echo "  PFN: $PFN"

TMPDIR_LOCAL=$(mktemp -d)
trap 'rm -rf "$TMPDIR_LOCAL"' EXIT

# --------------------------------------------------------------------------
# WDP helpers (same patterns as bench-xbox-ort.sh)
# --------------------------------------------------------------------------
_path_param() {
	local remote_dir="${1:-}"
	local p="%5CLocalState"
	[[ -n "$remote_dir" ]] && p="%5CLocalState%5C${remote_dir//\\/%5C}"
	printf '%s' "$p"
}

upload_as() {
	local local_path="$1" remote_dir="$2" remote_name="$3"
	curl "${CURL_AUTH[@]}" -H "X-CSRF-Token:${CSRF_TOKEN}" -X POST \
		-F "file=@${local_path};filename=${remote_name};type=application/octet-stream" \
		"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=$(_path_param "$remote_dir")" \
		>/dev/null
}

download_file() {
	local remote_dir="$1" remote_name="$2" dest="$3"
	curl "${CURL_AUTH[@]}" -o "$dest" \
		"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=$(_path_param "$remote_dir")&filename=${remote_name}" \
		2>/dev/null || true
}

delete_file() {
	local remote_dir="$1" remote_name="$2"
	curl "${CURL_AUTH[@]}" -H "X-CSRF-Token:${CSRF_TOKEN}" -X DELETE \
		"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=$(_path_param "$remote_dir")&filename=${remote_name}" \
		>/dev/null 2>&1 || true
}

# List file names in a LocalState dir (one per line)
list_files() {
	local remote_dir="${1:-}"
	curl "${CURL_AUTH[@]}" \
		"${BASE_URL}/api/filesystem/apps/files?knownfolderid=LocalAppData&packagefullname=${PFN}&path=$(_path_param "$remote_dir")" \
		2>/dev/null |
		python3 -c 'import json,sys
try:
    doc = json.load(sys.stdin)
except Exception:
    sys.exit(0)
for item in doc.get("Items", []):
    print(item.get("Name", ""))' || true
}

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

# --------------------------------------------------------------------------
# 1. Back up on-device genai_config.json, upload profiling variant
# --------------------------------------------------------------------------
echo ""
echo "--- Swapping genai_config.json (profiling variant) ---"
download_file "$MODEL_DIR" "genai_config.json" "${TMPDIR_LOCAL}/genai_config_orig.json"
[[ ! -s "${TMPDIR_LOCAL}/genai_config_orig.json" ]] && {
	echo "Error: could not download current genai_config.json from ${MODEL_DIR}" >&2
	exit 1
}
upload_as "${TMPDIR_LOCAL}/genai_config_orig.json" "$MODEL_DIR" "genai_config.json.bak"

PROFILE_CONFIG="${REPO_ROOT}/bench/configs/genai_config-dml-profile.json"
if [[ "$ABSOLUTE_PREFIX" == "true" ]]; then
	# Ladder step 2: render the template with an absolute LocalState prefix.
	# shellcheck disable=SC2001
	PFAMILY=$(echo "$PFN" | sed 's/_[0-9][0-9.]*_[^_]*__/_/')
	PROFILE_CONFIG="${TMPDIR_LOCAL}/genai_config-dml-profile.abs.json"
	PFAMILY="$PFAMILY" OUT_CONFIG="$PROFILE_CONFIG" \
		TPL_CONFIG="${REPO_ROOT}/bench/configs/genai_config-dml-profile.tpl.json" \
		python3 -c '
import json, os
prefix = (r"C:\Data\Users\DefaultAccount\AppData\Local\Packages"
          + "\\" + os.environ["PFAMILY"] + r"\LocalState\ort_profile")
with open(os.environ["TPL_CONFIG"]) as fp:
    cfg = json.load(fp)
cfg["model"]["decoder"]["session_options"]["enable_profiling"] = prefix
with open(os.environ["OUT_CONFIG"], "w") as fp:
    json.dump(cfg, fp, indent=4)
' || {
		echo "Error: failed to render absolute-prefix config" >&2
		exit 1
	}
	echo "  Using absolute profile prefix (package family: $PFAMILY)"
fi
upload_as "$PROFILE_CONFIG" "$MODEL_DIR" "genai_config.json"

restore_config() {
	if [[ "$KEEP_CONFIG" == "true" ]]; then
		echo "  --keep-config: profiling config left on device (restore: test-dml-config.sh --restore)"
		return 0
	fi
	echo "  Restoring original genai_config.json..."
	upload_as "${TMPDIR_LOCAL}/genai_config_orig.json" "$MODEL_DIR" "genai_config.json"
}

# --------------------------------------------------------------------------
# 2. Clean stale artifacts, snapshot log size, upload bench inputs, run
# --------------------------------------------------------------------------
echo ""
echo "--- Preparing run ---"
while IFS= read -r name; do
	[[ "$name" == ort_profile*.json ]] && {
		echo "  Deleting stale $name"
		delete_file "" "$name"
	}
done < <(list_files "")
while IFS= read -r name; do
	[[ "$name" == ort_profile*.json ]] && {
		echo "  Deleting stale ${MODEL_DIR}\\$name"
		delete_file "$MODEL_DIR" "$name"
	}
done < <(list_files "$MODEL_DIR")
delete_file "" "bench-result.csv"
delete_file "" "bench-result.csv.done"
# main_loop treats a present bench_turns.txt as a switch into the multi-turn KV
# bench, which returns early and writes bench-kv-result.csv instead — the poll
# below would then hang for its full timeout with no indication why. One left by
# scripts/bench-xbox-kv.sh is enough to do it. bench-xbox-ort.sh clears it for
# the same reason.
delete_file "" "bench_turns.txt"
# Same class of hazard, and quieter: bench-xbox-ort.sh --ctx / --n-predict leave
# bench_ctx.txt, bench_npredict.txt and bench_maxlen.txt behind, and main_loop
# honours whatever it finds. A profile run after a sweep would silently inherit
# that sweep's n_ctx / n_predict / max_length — no error, just a profile of the
# wrong configuration. max_length matters most: it is THE variable governing
# DirectML prefill (#130/#135), so a stale one would characterize the wrong point
# invisibly. The bench script only overwrites these, never deletes them, so
# clearing them here is the only thing that makes a profile run self-contained.
delete_file "" "bench_ctx.txt"
delete_file "" "bench_npredict.txt"
delete_file "" "bench_maxlen.txt"
delete_file "" "bench_ubatch.txt"
delete_file "" "bench_kvq8.txt"

# Append-only log: remember current size to slice only the new tail later.
download_file "" "xllama.log" "${TMPDIR_LOCAL}/xllama_before.log"
LOG_OFFSET=$(wc -c <"${TMPDIR_LOCAL}/xllama_before.log" 2>/dev/null || echo 0)

PROMPT_SRC="${PROMPT_FILE:-${REPO_ROOT}/bench/prompts/standard-512.txt}"
[[ ! -f "$PROMPT_SRC" ]] && {
	echo "Error: prompt file not found: $PROMPT_SRC" >&2
	restore_config
	exit 1
}
cp "$PROMPT_SRC" "${TMPDIR_LOCAL}/prompt.txt"
printf '%s' "$MODEL" >"${TMPDIR_LOCAL}/model.txt"
printf '0' >"${TMPDIR_LOCAL}/bench_threads.txt"
printf 'bench' >"${TMPDIR_LOCAL}/bench.flag"

for f in prompt.txt model.txt bench_threads.txt bench.flag; do
	upload_as "${TMPDIR_LOCAL}/${f}" "" "$f"
done

OUTDIR="${REPO_ROOT}/bench/results/profiles/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$OUTDIR"

SAMPLER_PID=""
if [[ "$GPU_SAMPLE" == "true" ]]; then
	echo "  Starting GPU sampler..."
	"${SCRIPT_DIR}/xbox-gpu-sample.sh" --out "${OUTDIR}/gpu-sample.csv" \
		>"${OUTDIR}/gpu-summary.txt" 2>&1 &
	SAMPLER_PID=$!
fi

echo "  Starting app..."
restart_app

RUN_OK=true
wait_for_done 300 || RUN_OK=false

if [[ -n "$SAMPLER_PID" ]]; then
	kill -TERM "$SAMPLER_PID" 2>/dev/null || true
	wait "$SAMPLER_PID" 2>/dev/null || true
fi

# --------------------------------------------------------------------------
# 3. Locate + download profile, log tail, bench CSV
# --------------------------------------------------------------------------
echo ""
echo "--- Collecting artifacts → ${OUTDIR} ---"
PROFILE_LOCAL=""
for dir in "" "$MODEL_DIR"; do
	while IFS= read -r name; do
		[[ "$name" == ort_profile*.json ]] || continue
		echo "  Found profile: ${dir:+${dir}\\}$name"
		download_file "$dir" "$name" "${OUTDIR}/${name}"
		PROFILE_LOCAL="${OUTDIR}/${name}"
	done < <(list_files "$dir")
	[[ -n "$PROFILE_LOCAL" ]] && break
done

download_file "" "xllama.log" "${TMPDIR_LOCAL}/xllama_after.log"
tail -c +"$((LOG_OFFSET + 1))" "${TMPDIR_LOCAL}/xllama_after.log" >"${OUTDIR}/xllama.log" 2>/dev/null || true
download_file "" "bench-result.csv" "${OUTDIR}/bench-result.csv"

restore_config

if [[ "$RUN_OK" != "true" ]]; then
	echo "Error: run timed out — check ${OUTDIR}/xllama.log" >&2
	exit 1
fi

if [[ -z "$PROFILE_LOCAL" ]]; then
	echo ""
	echo "Profile NOT found in LocalState or ${MODEL_DIR}." >&2
	echo "Ladder: rerun with --absolute-prefix; if still missing, deploy an MSIX" >&2
	echo "with the LocalState CWD pin (set_cwd_to_local_folder) and rerun." >&2
	exit 2
fi

# --------------------------------------------------------------------------
# 4. Analyze
# --------------------------------------------------------------------------
if [[ "$NO_ANALYZE" != "true" ]]; then
	echo ""
	echo "--- Analysis ---"
	python3 "${SCRIPT_DIR}/analyze_ort_profile.py" "$PROFILE_LOCAL" --log "${OUTDIR}/xllama.log"
fi

if [[ "$GPU_SAMPLE" == "true" && -s "${OUTDIR}/gpu-summary.txt" ]]; then
	echo ""
	cat "${OUTDIR}/gpu-summary.txt"
	echo "  (3D/compute engine > ~0.3 sustained during decode corroborates GPU execution)"
fi

echo ""
echo "Artifacts: $OUTDIR"
