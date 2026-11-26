#!/usr/bin/env bash
# measure-diffusion-coexistence.sh — does SD-Turbo fit next to a resident chat
# model on the console? (#548)
#
# Image generation runs while the chat model stays resident: the GUI, the LAN
# API and autopilot do not unload the session first, unlike device training.
# This measures it, one cell per model:
#
#   none     headless diffuse.flag process: the pipeline with no chat model
#   <model>  the UI process with the LAN API on: one chat turn makes <model>
#            resident, then POST /v1/images/generations runs next to it
#
# Each cell restarts the app, then records: HTTP status / headless marker,
# wall time, the pipeline's process_peak_ws_mb (diffuse-result.csv, only a
# row written during the cell), the max PrivateWorkingSet sampled from the
# Device Portal, and the xllama.log lines added during the cell that mention
# memory or failure. GPU memory is not observable from outside the app; a GPU
# overflow shows up as a failed generation and its log line.
#
# Usage:
#   source ~/.config/xllama/xbox-env
#   ./scripts/measure-diffusion-coexistence.sh <out-dir> none lfm25-350m ...
#
# Models must already be on the console (scripts/provision-models.sh).
# Writes <out-dir>/cells.csv and <out-dir>/<cell>.log. Leaves api.flag removed.
set -euo pipefail

: "${XBOX_IP:?XBOX_IP not set — run: source ~/.config/xllama/xbox-env}"
: "${XBOX_USER:?XBOX_USER not set}"
: "${XBOX_PASS:?XBOX_PASS not set}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/wdp-curl-auth.sh"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/wdp-file-upload.sh"

[[ $# -ge 2 ]] || {
	echo "usage: $0 <out-dir> none|<model> [...]" >&2
	exit 2
}
OUT="$1"
shift
mkdir -p "$OUT"

wdp_setup_curl_auth
DEPLOY="${SCRIPT_DIR}/deploy.sh"
BASE_URL="https://${XBOX_IP}:11443"
API_URL="http://${XBOX_IP}:11434"
PROMPT="a red sports car on a mountain road at sunset"
SAMPLE_S="${XLLAMA_COEXIST_WS_SAMPLE_S:-2}"

CSRF_TOKEN=$(curl "${CURL_AUTH[@]}" "${BASE_URL}/" -o /dev/null -D - 2>/dev/null |
	sed -n 's/.*[Cc][Ss][Rr][Ff]-[Tt]oken=\([^;[:space:]]*\).*/\1/p' | tr -d '\r' | head -n1)
PFN=$("${DEPLOY}" pfn 2>/dev/null)
[[ -n "$PFN" ]] || {
	echo "Error: xllama not installed" >&2
	exit 1
}
# shellcheck disable=SC2001
PACKAGE_FAMILY=$(echo "$PFN" | sed 's/_[0-9][0-9.]*_[^_]*__/_/')

TMP=$(mktemp -d)
SAMPLER_PID=""
cleanup() {
	[[ -n "$SAMPLER_PID" ]] && kill "$SAMPLER_PID" 2>/dev/null || true
	delete_file api.flag
	delete_file diffuse.flag
	rm -rf "$TMP"
}
trap cleanup EXIT

upload_file() { # <local> [remote-name]
	wdp_upload_file_request \
		"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=%5CLocalState" \
		"$1" "${2:-}"
}
fetch_file() { # <remote> <dest>
	: >"$2"
	curl "${CURL_AUTH[@]}" -o "$2" \
		"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=%5CLocalState&filename=$1" \
		2>/dev/null || true
}
delete_file() {
	curl "${CURL_AUTH[@]}" -H "X-CSRF-Token:${CSRF_TOKEN}" -X DELETE \
		"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=%5CLocalState&filename=$1" \
		>/dev/null 2>&1 || true
}
put_text() { # <remote-name> <content>
	printf '%s' "$2" >"${TMP}/$1"
	upload_file "${TMP}/$1"
}
stop_app() {
	curl "${CURL_AUTH[@]}" -H "X-CSRF-Token:${CSRF_TOKEN}" -X DELETE \
		"${BASE_URL}/api/taskmanager/app?package=${PFN}" >/dev/null 2>&1 || true
	sleep 3
}
start_app() {
	local aumid
	aumid=$(printf '%s!xllama' "$PACKAGE_FAMILY" | base64 | tr -d '\n')
	curl "${CURL_AUTH[@]}" -H "X-CSRF-Token:${CSRF_TOKEN}" -X POST -d "" \
		"${BASE_URL}/api/taskmanager/app?appid=${aumid}" >/dev/null 2>&1 || true
}
log_lines() { "${DEPLOY}" get-log "$PFN" 2>/dev/null | wc -l; }
start_sampler() { # max PrivateWorkingSet (MB) of the app process -> $1
	: >"$1"
	(
		while true; do
			mb=$(curl "${CURL_AUTH[@]}" --max-time "$SAMPLE_S" \
				"${BASE_URL}/api/resourcemanager/processes" 2>/dev/null |
				python3 -I "${REPO_ROOT}/scripts/console_taesd_result.py" private-ws \
					--package-family "$PACKAGE_FAMILY") || true
			if [[ "$mb" =~ ^[0-9]+$ ]]; then
				prev=$(tr -d '[:space:]' <"$1")
				if [[ -z "$prev" ]] || ((mb > prev)); then printf '%s\n' "$mb" >"$1"; fi
			fi
			sleep "$SAMPLE_S"
		done
	) &
	SAMPLER_PID=$!
}
stop_sampler() {
	[[ -n "$SAMPLER_PID" ]] && kill "$SAMPLER_PID" 2>/dev/null || true
	wait "$SAMPLER_PID" 2>/dev/null || true
	SAMPLER_PID=""
}
wait_api() {
	local i
	for ((i = 0; i < 60; i++)); do
		[[ "$(curl -sS -m 4 -o /dev/null -w '%{http_code}' "${API_URL}/" 2>/dev/null)" == 200 ]] && return 0
		sleep 5
	done
	return 1
}
# Peak WS from a diffuse-result.csv row written at or after $2 (UTC ISO).
peak_from_csv() {
	python3 -I - "$1" "$2" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8", errors="replace")))
row = rows[-1] if rows else None
print(row["process_peak_ws_mb"] if row and row.get("date", "") >= sys.argv[2] else "")
PY
}

printf 'cell,outcome,detail,wall_s,process_peak_ws_mb,max_sampled_private_ws_mb\n' >"${OUT}/cells.csv"
for cell in "$@"; do
	echo "=== cell: ${cell}"
	stop_app
	delete_file api.flag
	delete_file diffuse.flag
	delete_file diffuse-result.csv
	delete_file diffuse-out.png.done
	put_text prompt.txt "$PROMPT"
	put_text diffuse-steps.txt 1
	put_text diffuse-seed.txt 42
	put_text diffuse-model.txt sd-turbo-fp16
	log0=$(log_lines)
	run_start=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
	ws_file="${TMP}/ws.${cell}"
	outcome="" detail=""
	t0=$SECONDS
	if [[ "$cell" == none ]]; then
		put_text diffuse.flag go
		start_sampler "$ws_file"
		start_app
		for ((i = 0; i < 120; i++)); do
			fetch_file diffuse-out.png.done "${TMP}/done"
			[[ -s "${TMP}/done" ]] && break
			sleep 5
		done
		stop_sampler
		fetch_file diffuse-progress.txt "${TMP}/stage"
		detail="stage=$(tr -d '\r\n' <"${TMP}/stage")"
		[[ "$detail" == "stage=done" ]] && outcome=ok || outcome=fail
	else
		put_text model.txt "$cell"
		put_text api.flag go
		start_app
		if ! wait_api; then
			outcome=fail detail="api did not come up"
		else
			chat=$(curl -sS -m 600 -o "${TMP}/chat.json" -w '%{http_code}' \
				-H 'Content-Type: application/json' "${API_URL}/v1/chat/completions" \
				-d "{\"model\":\"${cell}\",\"messages\":[{\"role\":\"user\",\"content\":\"Say hi.\"}],\"max_tokens\":8}" || echo 000)
			start_sampler "$ws_file"
			t0=$SECONDS
			img=$(curl -sS -m 900 -o "${TMP}/img.json" -w '%{http_code}' \
				-H 'Content-Type: application/json' "${API_URL}/v1/images/generations" \
				-d "{\"prompt\":\"${PROMPT}\",\"steps\":1,\"seed\":42}" || echo 000)
			stop_sampler
			detail="chat=${chat} image=${img}"
			if [[ "$img" == 200 && "$chat" == 200 ]]; then
				outcome=ok
			else
				outcome=fail
				err=$(python3 -I -c 'import json,sys; print(json.load(open(sys.argv[1])).get("error",{}).get("message","")[:120])' "${TMP}/img.json" 2>/dev/null || true)
				[[ -n "$err" ]] && detail="${detail} error=${err//,/;}"
			fi
		fi
	fi
	wall=$((SECONDS - t0))
	fetch_file diffuse-result.csv "${TMP}/result.csv"
	peak=$(peak_from_csv "${TMP}/result.csv" "$run_start")
	ws=""
	[[ -s "$ws_file" ]] && ws=$(tr -d '[:space:]' <"$ws_file")
	"${DEPLOY}" get-log "$PFN" 2>/dev/null | tail -n "+$((log0 + 1))" |
		grep -E -i 'diffuse|released|gpu-mem|ws after|out of memory|8007000E|E_OUTOFMEMORY|887A0005|DEVICE_REMOVED|session|error' |
		python3 -I "${REPO_ROOT}/scripts/redact_recorded_path.py" --text >"${OUT}/${cell}.log" || true
	printf '%s,%s,%s,%s,%s,%s\n' "$cell" "$outcome" "$detail" "$wall" "$peak" "$ws" >>"${OUT}/cells.csv"
	echo "    ${outcome} ${detail} wall=${wall}s peak_ws=${peak:-?}MB sampled_private_ws=${ws:-?}MB"
done
stop_app
