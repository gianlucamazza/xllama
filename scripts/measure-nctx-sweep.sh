#!/usr/bin/env bash
# measure-nctx-sweep.sh — long contexts on the console once the session memory
# admission bounds the KV (#570).
#
# GGUF n_ctx is capped by the model's trained context, not by a fixed 8192, and
# the admission (memory_admission.h) refuses a load whose KV would not fit. This
# measures what that admits, one cell per <model>:<n_ctx>:<prompt_tokens>:
#
#   1. A temporary LocalState manifest.json override sets the entry's n_ctx.
#      The override is a copy of the bundled entry (uwp/models/manifest.json),
#      so only n_ctx differs.
#   2. The app restarts with the LAN API, and one short turn loads the model.
#      A refused load answers 400 "insufficient memory: …".
#   3. If the model loaded and prompt_tokens > 0: one streamed request with a
#      prompt of about that many tokens (scripts/measure-api-ttfb.py: first
#      content and done, in ms from the client). Then the same prompt
#      non-streamed, which reuses the KV, for the exact usage.prompt_tokens.
#
# Each cell records the max PrivateWorkingSet sampled from the Device Portal and
# the xllama.log lines added during the cell about memory, context or failure.
#
# Usage:
#   source ~/.config/xllama/xbox-env
#   ./scripts/measure-nctx-sweep.sh --hardware <label> <out-dir> \
#       lfm25-350m:16384:14000 llama32-3b:32768:0 ...
#
# Models must already be on the console (scripts/provision-models.sh). The
# LocalState files the sweep overwrites (manifest.json, model.txt, api.flag) are
# saved first and restored at the end, so the LAN API setting and the GUI's model
# survive the run. If api.flag was present the app is started again at the end.
# Writes <out-dir>/cells.csv, <out-dir>/<cell>.log and <out-dir>/<cell>.ttfb.json.
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

HARDWARE=""
if [[ "${1:-}" == "--hardware" ]]; then
	HARDWARE="${2:-}"
	shift 2
fi
[[ -n "$HARDWARE" && $# -ge 2 ]] || {
	echo "usage: $0 --hardware <label> <out-dir> <model>:<n_ctx>:<prompt_tokens> [...]" >&2
	exit 2
}
OUT="$1"
shift
mkdir -p "$OUT"

wdp_setup_curl_auth
DEPLOY="${SCRIPT_DIR}/deploy.sh"
BASE_URL="https://${XBOX_IP}:11443"
API_URL="http://${XBOX_IP}:11434"
SAMPLE_S="${XLLAMA_NCTX_WS_SAMPLE_S:-2}"
REQUEST_TIMEOUT_S="${XLLAMA_NCTX_TIMEOUT_S:-1800}"

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
SAVED_MANIFEST=""
SAVED_MODEL=""
HAD_API_FLAG=0

upload_file() { # <local> [remote-name]
	wdp_upload_file_request \
		"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=%5CLocalState" \
		"$1" "${2:-}"
}
fetch_file() { # <remote> <dest>
	: >"$2"
	curl "${CURL_AUTH[@]}" -f -o "$2" \
		"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=%5CLocalState&filename=$1" \
		2>/dev/null || : >"$2"
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
# The bundled catalogue entry for <model> with n_ctx set, as a one-entry override.
write_override() { # <model> <n_ctx> <dest>
	python3 -I - "${REPO_ROOT}/uwp/models/manifest.json" "$1" "$2" "$3" <<'PY'
import json, sys
src, name, n_ctx, dest = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
entry = next(e for e in json.load(open(src, encoding="utf-8"))["models"] if e["name"] == name)
entry["n_ctx"] = n_ctx
entry["_comment"] = f"measure-nctx-sweep.sh: temporary n_ctx {n_ctx} override (#570)"
json.dump({"_comment": "temporary, removed by measure-nctx-sweep.sh", "models": [entry]},
          open(dest, "w", encoding="utf-8"), indent=2)
PY
}
# Deterministic filler of about <tokens> tokens (numbered sentences), then a
# short instruction. 3.5 chars per token is calibrated on this text with the
# LFM2.5 and Llama-3.2 vocabularies (2026-11-25: 5922 chars were 1700 tokens).
# usage.prompt_tokens records the exact count.
write_prompt() { # <tokens> <dest>
	python3 -I - "$1" "$2" <<'PY'
import sys
tokens, dest = int(sys.argv[1]), sys.argv[2]
words = ["river", "stone", "lantern", "harbor", "meadow", "copper", "signal", "orchard",
         "glacier", "thread", "compass", "ember", "valley", "archive", "falcon", "mirror"]
out, n, i = [], 0, 0
while n < tokens:
    s = f"Note {i}: the {words[i % 16]} near the {words[(i * 7 + 3) % 16]} was logged at {i * 13 % 997}."
    out.append(s)
    n += len(s) / 3.5
    i += 1
out.append("Reply with the single word OK.")
open(dest, "w", encoding="utf-8").write(" ".join(out))
PY
}
chat_json() { # <model> <prompt-file> <dest-json> [max_tokens] -> http code
	python3 -I -c 'import json,sys; print(json.dumps({"model": sys.argv[1], "stream": False,
"max_tokens": int(sys.argv[3]), "messages": [{"role": "user", "content": open(sys.argv[2], encoding="utf-8").read()}]}))' \
		"$1" "$2" "${4:-8}" >"${TMP}/body.json"
	curl -sS -m "$REQUEST_TIMEOUT_S" -o "$3" -w '%{http_code}' -H 'Content-Type: application/json' \
		--data-binary "@${TMP}/body.json" "${API_URL}/v1/chat/completions" || echo 000
}
json_field() { # <file> <python expr on d>
	python3 -I -c 'import json,sys
try:
    d = json.load(open(sys.argv[1], encoding="utf-8"))
    v = eval(sys.argv[2], {"d": d})
except Exception:
    v = ""
print("" if v is None else str(v).replace(",", ";")[:160])' "$1" "$2"
}

cleanup() {
	stop_sampler
	stop_app
	delete_file api.flag
	delete_file manifest.json
	[[ -n "$SAVED_MANIFEST" ]] && upload_file "$SAVED_MANIFEST" manifest.json
	[[ -n "$SAVED_MODEL" ]] && upload_file "$SAVED_MODEL" model.txt
	# The LAN API was on before the sweep: bring the server back up as it was.
	if ((HAD_API_FLAG)); then
		upload_file "${TMP}/api.flag.saved" api.flag
		start_app
	fi
	rm -rf "$TMP"
}
trap cleanup EXIT

stop_app
fetch_file manifest.json "${TMP}/manifest.saved"
if [[ -s "${TMP}/manifest.saved" ]]; then
	SAVED_MANIFEST="${TMP}/manifest.saved"
	echo "saved the existing LocalState manifest.json; it is restored at the end"
fi
fetch_file model.txt "${TMP}/model.saved"
[[ -s "${TMP}/model.saved" ]] && SAVED_MODEL="${TMP}/model.saved"
# App.cpp starts the LAN API when the flag exists; keep the file as it was.
if curl "${CURL_AUTH[@]}" -f -o "${TMP}/api.flag.saved" \
	"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=%5CLocalState&filename=api.flag" \
	2>/dev/null; then
	HAD_API_FLAG=1
	echo "api.flag is present; it is restored at the end"
fi

printf 'cell,model,n_ctx,target_tokens,load_status,prompt_status,prompt_tokens,first_content_ms,done_ms,max_private_ws_mb,detail\n' >"${OUT}/cells.csv"
for cell in "$@"; do
	IFS=: read -r model n_ctx target <<<"$cell"
	name="${model}-${n_ctx}-${target}"
	echo "=== cell: ${name}"
	stop_app
	write_override "$model" "$n_ctx" "${TMP}/override.json"
	upload_file "${TMP}/override.json" manifest.json
	put_text model.txt "$model"
	put_text api.flag go
	log0=$(log_lines)
	ws_file="${TMP}/ws.${name}"
	load="" prompt_status="" prompt_tokens="" first="" done_ms="" detail=""
	start_sampler "$ws_file"
	start_app
	if ! wait_api; then
		load=000 detail="api did not come up"
	else
		printf 'Say hi.' >"${TMP}/hi.txt"
		load=$(chat_json "$model" "${TMP}/hi.txt" "${TMP}/load.json")
		[[ "$load" == 200 ]] || detail=$(json_field "${TMP}/load.json" 'd["error"]["message"]')
		if [[ "$load" == 200 && "$target" -gt 0 ]]; then
			write_prompt "$target" "${TMP}/prompt.txt"
			python3 -I "${SCRIPT_DIR}/measure-api-ttfb.py" --host "$XBOX_IP" --model "$model" \
				--prompt "$(cat "${TMP}/prompt.txt")" --repeats 1 --max-tokens 8 \
				--timeout "$REQUEST_TIMEOUT_S" --hardware "$HARDWARE" \
				--json-out "${OUT}/${name}.ttfb.json" >/dev/null 2>&1 || true
			# The receipt is committed: keep the console's LAN address out of it.
			[[ -f "${OUT}/${name}.ttfb.json" ]] && sed -i "s/${XBOX_IP//./\\.}/console/g" "${OUT}/${name}.ttfb.json"
			first=$(json_field "${OUT}/${name}.ttfb.json" 'd["attempts"][0]["first_content_ms"]')
			done_ms=$(json_field "${OUT}/${name}.ttfb.json" 'd["attempts"][0]["done_ms"]')
			err=$(json_field "${OUT}/${name}.ttfb.json" 'd["attempts"][0]["error"]')
			[[ -n "$err" ]] && detail="stream: ${err}"
			prompt_status=$(chat_json "$model" "${TMP}/prompt.txt" "${TMP}/prompt.json")
			prompt_tokens=$(json_field "${TMP}/prompt.json" 'd["usage"]["prompt_tokens"]')
			[[ "$prompt_status" == 200 ]] ||
				detail="${detail:+${detail}; }$(json_field "${TMP}/prompt.json" 'd["error"]["message"]')"
		fi
	fi
	stop_sampler
	ws=""
	[[ -s "$ws_file" ]] && ws=$(tr -d '[:space:]' <"$ws_file")
	"${DEPLOY}" get-log "$PFN" 2>/dev/null | tail -n "+$((log0 + 1))" |
		grep -E -i 'memory estimate|capped|insufficient|ws after|n_ctx|prompt too long|context shift|stall|out of memory|bad allocation|error|fail' |
		python3 -I "${REPO_ROOT}/scripts/redact_recorded_path.py" --text >"${OUT}/${name}.log" || true
	printf '%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n' "$name" "$model" "$n_ctx" "$target" "$load" \
		"$prompt_status" "$prompt_tokens" "$first" "$done_ms" "$ws" "$detail" >>"${OUT}/cells.csv"
	echo "    load=${load} prompt=${prompt_status:-skip} tokens=${prompt_tokens:-?} first=${first:-?}ms done=${done_ms:-?}ms ws=${ws:-?}MB ${detail}"
done
