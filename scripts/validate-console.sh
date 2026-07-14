#!/usr/bin/env bash
# validate-console.sh — drive the xllama autopilot to validate the live XAML UI
# on Xbox with deterministic PASS/FAIL verdicts, no human at the pad.
#
# Usage:
#   source ~/.config/xllama/xbox-env
#   ./scripts/validate-console.sh <routing|gguf|taesd|all>
#
# Requires: an installed xllama build with the autopilot (>= 1.1.3.0), the
# relevant models already in LocalState (smollm2-360m-dml-fp16 for routing,
# lfm25-350m for gguf, sd-turbo-fp16 for taesd), and XBOX_IP/USER/PASS env.
#
# Contract per subcommand: upload chats/autopilot.json + autopilot.flag,
# restart the app, poll LocalState\autopilot-done.txt, fetch xllama.log, and
# grep the log for the verdict. Exit 0 = PASS, non-zero = FAIL. The autopilot
# ends every script with a `quit` action so the app exits itself.

set -euo pipefail

: "${XBOX_IP:?XBOX_IP not set — run: source ~/.config/xllama/xbox-env}"
: "${XBOX_USER:?XBOX_USER not set}"
: "${XBOX_PASS:?XBOX_PASS not set}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEPLOY="${SCRIPT_DIR}/deploy.sh"
BASE_URL="https://${XBOX_IP}:11443"
CURL_AUTH=(--basic -u "${XBOX_USER}:${XBOX_PASS}" -k -sS)

CSRF_TOKEN=$(curl "${CURL_AUTH[@]}" "${BASE_URL}/" -o /dev/null -D - 2>/dev/null |
	sed -n 's/.*[Cc][Ss][Rr][Ff]-[Tt]oken=\([^;[:space:]]*\).*/\1/p' |
	tr -d '\r' | head -n1)
[[ -z "$CSRF_TOKEN" ]] && echo "Warning: no CSRF token — POST/DELETE may fail" >&2

PFN=$("${DEPLOY}" pfn 2>/dev/null)
[[ -z "$PFN" ]] && {
	echo "Error: xllama not found — deploy it first" >&2
	exit 1
}

TMPDIR_LOCAL=$(mktemp -d)
VAE_CACHE="" # set by validate_taesd; read by its EXIT trap (must be global)
trap 'rm -rf "$TMPDIR_LOCAL"' EXIT

# --- WDP helpers (same idioms as bench-xbox-ort.sh) ------------------------

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

# upload_file <local> [remote_subdir] [remote_name]
upload_file() {
	local local_path="$1" remote_dir="${2:-}" remote_name="${3:-}"
	if [[ -n "$remote_dir" ]]; then
		"${DEPLOY}" mkdir-localstate "$PFN" "$remote_dir" >/dev/null
	fi
	local path_param="%5CLocalState"
	[[ -n "$remote_dir" ]] && path_param="%5CLocalState%5C${remote_dir//\\/%5C}"
	_curl_post_file "$local_path" "$path_param" "$remote_name"
}

# fetch_file <remote_name> <dest> [remote_subdir]
fetch_file() {
	local remote_name="$1" dest="$2" remote_dir="${3:-}"
	local path_param="%5CLocalState"
	[[ -n "$remote_dir" ]] && path_param="%5CLocalState%5C${remote_dir//\\/%5C}"
	curl "${CURL_AUTH[@]}" -o "$dest" \
		"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=${path_param}&filename=${remote_name}" \
		2>/dev/null || true
}

delete_file() {
	local remote_name="$1" remote_dir="${2:-}"
	local path_param="%5CLocalState"
	[[ -n "$remote_dir" ]] && path_param="%5CLocalState%5C${remote_dir//\\/%5C}"
	curl "${CURL_AUTH[@]}" -H "X-CSRF-Token:${CSRF_TOKEN}" -X DELETE \
		"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=${path_param}&filename=${remote_name}" \
		>/dev/null 2>&1 || true
}

restart_app() {
	curl "${CURL_AUTH[@]}" -H "X-CSRF-Token:${CSRF_TOKEN}" -X DELETE \
		"${BASE_URL}/api/taskmanager/app?package=${PFN}" >/dev/null 2>&1 || true
	sleep 2
	local pfamily aumid
	# shellcheck disable=SC2001
	pfamily=$(echo "$PFN" | sed 's/_[0-9][0-9.]*_[^_]*__/_/')
	aumid=$(printf '%s!xllama' "$pfamily" | base64 -w0)
	curl "${CURL_AUTH[@]}" -H "X-CSRF-Token:${CSRF_TOKEN}" -X POST -d "" \
		"${BASE_URL}/api/taskmanager/app?appid=${aumid}" >/dev/null 2>&1 || true
}

# Poll autopilot-done.txt; echoes its content, returns non-zero on timeout.
wait_autopilot_done() {
	local timeout_s="${1:-900}" elapsed=0 out="${TMPDIR_LOCAL}/done.txt"
	echo "  Waiting for autopilot-done.txt (timeout ${timeout_s}s)..." >&2
	while ((elapsed < timeout_s)); do
		: >"$out"
		fetch_file "autopilot-done.txt" "$out"
		# A real marker is "ok" or "error: ..."; a 404 body contains neither.
		if grep -qE '^(ok|error:)' "$out" 2>/dev/null; then
			echo "  Done after ${elapsed}s." >&2
			cat "$out"
			return 0
		fi
		sleep 10
		((elapsed += 10))
	done
	echo "  Timeout waiting for autopilot-done.txt" >&2
	return 1
}

# Upload autopilot.json (from stdin) + a fresh autopilot.flag, clear stale
# markers, restart, and wait. Echoes the done-marker content.
run_autopilot() {
	local timeout_s="${1:-900}"
	cat >"${TMPDIR_LOCAL}/autopilot.json"
	printf 'go' >"${TMPDIR_LOCAL}/autopilot.flag"
	delete_file "autopilot-done.txt"
	delete_file "diffuse-progress.txt"
	# xllama.log is append-only across restarts; clear it so the verdict greps
	# only this run (a stale 887A0036 or routing line would falsify the gate).
	# The app reopens the log lazily on next launch.
	delete_file "xllama.log"
	upload_file "${TMPDIR_LOCAL}/autopilot.json"
	upload_file "${TMPDIR_LOCAL}/autopilot.flag"
	restart_app
	wait_autopilot_done "$timeout_s"
}

fetch_log() {
	fetch_file "xllama.log" "${TMPDIR_LOCAL}/xllama.log"
	echo "${TMPDIR_LOCAL}/xllama.log"
}

# --- §2 routing A/B --------------------------------------------------------

model_provisioned() {
	local model="$1"
	local path_param="%5CLocalState%5Cmodels%5C${model//\\/%5C}"
	local code
	code=$(curl "${CURL_AUTH[@]}" -o /dev/null -w "%{http_code}" \
		"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&filename=genai_config.json&packagefullname=${PFN}&path=${path_param}" 2>/dev/null || echo "000")
	[[ "$code" == "200" ]]
}

validate_routing() {
	echo "=== §2 routing A/B ==="
	if ! model_provisioned "smollm2-360m-dml-fp16"; then
		echo "  FAIL: smollm2-360m-dml-fp16 is not in LocalState\\models\\"
		echo "  Upload the DML fp16 model before routing validation:"
		echo "    PFN=\$(./scripts/deploy.sh pfn)"
		echo "    ./scripts/deploy.sh upload-dir <host-path>/smollm2-360m-dml-fp16 \"\$PFN\" \"models\\\\smollm2-360m-dml-fp16\""
		echo "  (not on models-v1 — USB/Device Portal only; reinstalling the MSIX wipes LocalState)"
		echo "§2 routing: FAIL"
		return 1
	fi
	# Routing=auto is emitted only when settings.json has "routing": 2; seed it
	# (the whole point is that no human could set it via the dialog). Schema
	# mirrors SaveSettings (MainPage.cpp).
	cat >"${TMPDIR_LOCAL}/settings.json" <<'JSON'
{
  "system_prompt": "You are a helpful AI assistant.",
  "model": "smollm2-360m-cpu-int4",
  "kv_reuse": true,
  "routing": 2,
  "gpu_model": "smollm2-360m-dml-fp16",
  "diffuse_taesd_vae": false,
  "sampling": {"temperature": 0.8, "top_p": 0.9, "top_k": 40, "repetition_penalty": 1.1, "n_predict": 256}
}
JSON
	upload_file "${TMPDIR_LOCAL}/settings.json"

	# Build the long decoy conversation (>600 tok) from long-1k.txt, stripping
	# the ChatML wrapper so the stored user content is plain prose. Merge it into
	# the device's existing chat index rather than clobbering it.
	local esca_id="ap-routing-longctx"
	fetch_file "index.json" "${TMPDIR_LOCAL}/existing-index.json" "chats"
	python3 - "$REPO_ROOT" "$TMPDIR_LOCAL" "$esca_id" <<'PY'
import json, re, sys
repo, tmp, cid = sys.argv[1], sys.argv[2], sys.argv[3]
body = open(f"{repo}/bench/prompts/long-1k.txt").read()
m = re.search(r'<\|im_start\|>user\n(.*?)(?:<\|im_end\|>|$)', body, re.S)
user = (m.group(1) if m else body).strip()
ts = 1
conv = {"id": cid, "title": "routing A/B (long context)", "messages": [
    {"role": "user", "content": user, "ts": ts, "partial": False},
    {"role": "assistant", "content": "Understood; ready to continue.", "ts": ts+1, "partial": False},
]}
open(f"{tmp}/{cid}.json", "w").write(json.dumps(conv, ensure_ascii=False))
entry = {"id": cid, "title": conv["title"], "last_modified": ts+1, "n_messages": 2}
idx = []
try:
    idx = json.load(open(f"{tmp}/existing-index.json"))
    if not isinstance(idx, list):
        idx = []
except Exception:
    idx = []
idx = [e for e in idx if e.get("id") != cid]
idx.insert(0, entry)
open(f"{tmp}/index.json", "w").write(json.dumps(idx, ensure_ascii=False))
print(f"  decoy user chars: {len(user)} (~{len(user)//4} tok); index entries: {len(idx)}")
PY
	upload_file "${TMPDIR_LOCAL}/${esca_id}.json" "chats"
	upload_file "${TMPDIR_LOCAL}/index.json" "chats"

	local marker
	marker=$(
		run_autopilot 1300 <<JSON
{"total_timeout_s": 900, "actions": [
  {"op": "load_chat", "id": "${esca_id}"},
  {"op": "send", "text": "continue", "timeout_s": 300},
  {"op": "send", "text": "ok", "timeout_s": 120},
  {"op": "new_chat"},
  {"op": "send", "text": "hello", "timeout_s": 120},
  {"op": "quit"}
]}
JSON
	) || true
	echo "  autopilot: ${marker}"
	local log
	log=$(fetch_log)

	local verdict=0
	if [[ "$marker" != "ok" ]]; then
		echo "  FAIL: autopilot did not finish ok"
		verdict=1
	fi
	# The routing log line uses "auto -> gpu"/"auto -> cpu" (ASCII arrow in the
	# C++ is "→"; match on the tok field to be encoding-robust).
	local gpu_line cpu_line
	gpu_line=$(grep -aE 'routing: auto .* gpu \([0-9]+ tok' "$log" | head -1 || true)
	cpu_line=$(grep -aE 'routing: auto .* cpu \([0-9]+ tok' "$log" | head -1 || true)
	if [[ -z "$gpu_line" ]]; then
		echo "  FAIL: no 'auto -> gpu' routing line for the long-context turn"
		verdict=1
	else
		local ntok
		ntok=$(sed -n 's/.*(\([0-9]\+\) tok.*/\1/p' <<<"$gpu_line")
		if ((ntok <= 600)); then
			echo "  FAIL: gpu-routed turn had only ${ntok} tok (<=600)"
			verdict=1
		else
			echo "  ok: long turn routed to GPU (${ntok} tok)"
		fi
	fi
	if [[ -z "$cpu_line" ]]; then
		echo "  FAIL: no 'auto -> cpu' routing line for the new short chat"
		verdict=1
	else
		echo "  ok: new short chat routed to CPU"
	fi
	if grep -aq '887A0036' "$log"; then
		echo "  FAIL: 887A0036 present — patched GenAI DLL did not take"
		verdict=1
	else
		echo "  ok: no 887A0036 (patched DLL works in XAML)"
	fi
	# Remove the decoy chat so "Understood; ready to continue." does not linger in
	# the History list after a validation run.
	delete_file "${esca_id}.json" "chats"
	fetch_file "index.json" "${TMPDIR_LOCAL}/index-after.json" "chats"
	python3 - "$TMPDIR_LOCAL" "$esca_id" <<'PY'
import json, sys
tmp, cid = sys.argv[1], sys.argv[2]
path = f"{tmp}/index-after.json"
try:
    idx = json.load(open(path))
except Exception:
    idx = []
if isinstance(idx, list):
    idx = [e for e in idx if e.get("id") != cid]
    open(path, "w").write(json.dumps(idx, ensure_ascii=False))
PY
	upload_file "${TMPDIR_LOCAL}/index-after.json" "chats" "index.json"
	echo "  ok: removed decoy chat ${esca_id}"

	[[ $verdict -eq 0 ]] && echo "§2 routing: PASS" || echo "§2 routing: FAIL"
	return $verdict
}

# --- GGUF chat -------------------------------------------------------------

validate_gguf() {
	echo "=== GGUF chat (unified) ==="
	local marker
	marker=$(
		run_autopilot 900 <<'JSON'
{"total_timeout_s": 300, "actions": [
  {"op": "set_model", "name": "lfm25-350m"},
  {"op": "new_chat"},
  {"op": "send", "text": "Say hello in one short sentence.", "timeout_s": 180},
  {"op": "quit"}
]}
JSON
	) || true
	echo "  autopilot: ${marker}"
	local log verdict=0
	log=$(fetch_log)
	[[ "$marker" == "ok" ]] || {
		echo "  FAIL: autopilot did not finish ok"
		verdict=1
	}
	# The interactive llama.cpp session logs a distinct marker on load.
	if grep -aq 'GGUF model loaded via llama.cpp' "$log"; then
		echo "  ok: lfm25-350m loaded via llama.cpp (persistent session)"
	else
		echo "  FAIL: no llama.cpp GGUF session load in the log"
		verdict=1
	fi
	[[ $verdict -eq 0 ]] && echo "GGUF chat: PASS" || echo "GGUF chat: FAIL"
	return $verdict
}

# --- §7c TAESD -------------------------------------------------------------

validate_taesd() {
	echo "=== §7c TAESD image ==="
	local rel="https://github.com/gianlucamazza/xllama/releases/download/models-v1"
	# VAE_CACHE is a GLOBAL (set below) so the EXIT trap can still read it after
	# this function returns — a `local` would be out of scope at trap time and
	# abort the restore under `set -u`.
	VAE_CACHE="${REPO_ROOT}/.cache-validate"
	mkdir -p "$VAE_CACHE"
	# Cache the tiny (TAESD) and full VAE decoders once.
	[[ -f "${VAE_CACHE}/taesd_vae.onnx" ]] ||
		curl -fsSL "${rel}/sd-turbo-fp16_taesd_vae_decoder_model.onnx" -o "${VAE_CACHE}/taesd_vae.onnx"
	[[ -f "${VAE_CACHE}/full_vae.onnx" ]] ||
		curl -fsSL "${rel}/sd-turbo-fp16_vae_decoder_model.onnx" -o "${VAE_CACHE}/full_vae.onnx"

	# Swap TAESD in, restore the full VAE on exit no matter what.
	trap 'echo "  restoring full VAE decoder ..." >&2; upload_file "${VAE_CACHE}/full_vae.onnx" "models\\sd-turbo-fp16\\vae_decoder" "model.onnx"; rm -rf "$TMPDIR_LOCAL"' EXIT
	upload_file "${VAE_CACHE}/taesd_vae.onnx" "models\\sd-turbo-fp16\\vae_decoder" "model.onnx"

	local marker
	marker=$(
		run_autopilot 1300 <<'JSON'
{"total_timeout_s": 600, "actions": [
  {"op": "generate_image", "prompt": "a red sports car on a mountain road at sunset", "steps": 1, "seed": 42, "timeout_s": 600},
  {"op": "quit"}
]}
JSON
	) || true
	echo "  autopilot: ${marker}"
	local verdict=0
	[[ "$marker" == "ok" ]] || {
		echo "  FAIL: autopilot did not finish ok"
		verdict=1
	}
	fetch_file "diffuse-result.csv" "${TMPDIR_LOCAL}/diffuse-result.csv"
	# diffuse-result.csv columns include a vae_ms field; TAESD should be sub-second.
	local vae_ms
	vae_ms=$(
		python3 - "${TMPDIR_LOCAL}/diffuse-result.csv" <<'PY'
import csv, sys
try:
    rows = list(csv.DictReader(open(sys.argv[1])))
    r = rows[-1] if rows else {}
    key = next((k for k in r if k and 'vae' in k.lower() and 'ms' in k.lower()), None)
    print(r.get(key, "") if key else "")
except Exception:
    print("")
PY
	)
	if [[ -n "$vae_ms" ]] && awk "BEGIN{exit !($vae_ms < 1000)}"; then
		echo "  ok: TAESD VAE stage ${vae_ms} ms (<1000)"
	else
		echo "  FAIL: VAE stage ms='${vae_ms}' not sub-second"
		verdict=1
	fi
	[[ $verdict -eq 0 ]] && echo "§7c TAESD: PASS" || echo "§7c TAESD: FAIL"
	return $verdict
}

# --- dispatch --------------------------------------------------------------

CMD="${1:-}"
case "$CMD" in
routing) validate_routing ;;
gguf) validate_gguf ;;
taesd) validate_taesd ;;
all)
	rc=0
	if ! model_provisioned "smollm2-360m-cpu-int4"; then
		echo "  WARN: smollm2-360m-cpu-int4 missing — launch the app once for catalogue download"
	fi
	validate_routing || rc=1
	validate_gguf || rc=1
	validate_taesd || rc=1
	echo
	echo "=== summary ==="
	[[ $rc -eq 0 ]] && echo "ALL PASS" || echo "SOME FAILED (exit ${rc})"
	exit $rc
	;;
*)
	echo "Usage: $0 <routing|gguf|taesd|all>" >&2
	exit 1
	;;
esac
