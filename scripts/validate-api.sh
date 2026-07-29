#!/usr/bin/env bash
# validate-api.sh — deploy api.flag, restart the app, and validate the LAN HTTP
# endpoint (OpenAI-compat) from this host over the network. Deterministic
# PASS/FAIL, no human at the pad. See docs/api-endpoint.md.
#
# Usage:
#   source ~/.config/xllama/xbox-env
#   ./scripts/validate-api.sh <spike|chat|budget|prefs|train|all>
#
#   spike  bind gate only: GET / -> HTTP 200 (proves the StreamSocketListener
#          survives the Series S firewall/PLM). No inference.
#   chat   POST /v1/chat/completions -> non-empty assistant content + a 503-busy
#          check (two concurrent requests). Implies the spike gate first.
#   budget context budget over the wire: a long messages[] is trimmed and answered,
#          an oversized single message is 400 "prompt too long" (not 500).
#   prefs  POST /v1/preferences -> appends a like sample (#118).
#   train  GET /v1/training/status -> JSON with state + usable_samples (#118).
#   all    spike + chat + budget + prefs + train (images need SD-Turbo on device — manual).
#
# Requires: an installed xllama build with the endpoint, a chat model already in
# LocalState (set MODEL, or seed LocalState\model.txt), and XBOX_IP/USER/PASS.
# The app must NOT be running a headless flag (bench/diffuse) — those exit the
# process and never bind the socket.

set -euo pipefail

: "${XBOX_IP:?XBOX_IP not set — run: source ~/.config/xllama/xbox-env}"
: "${XBOX_USER:?XBOX_USER not set}"
: "${XBOX_PASS:?XBOX_PASS not set}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DEPLOY="${SCRIPT_DIR}/deploy.sh"
BASE_URL="https://${XBOX_IP}:11443"
CURL_AUTH=(--basic -u "${XBOX_USER}:${XBOX_PASS}" -k -sS)

PORT="${API_PORT:-11434}"
MODEL="${MODEL:-lfm25-350m}"
API_URL="http://${XBOX_IP}:${PORT}"

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
trap 'rm -rf "$TMPDIR_LOCAL"' EXIT

# --- WDP helpers (same idioms as validate-console.sh) ----------------------

upload_file() {
	local local_path="$1" remote_name="${2:-}"
	local form_entry
	if [[ -n "$remote_name" ]]; then
		form_entry="${local_path};filename=${remote_name};type=application/octet-stream"
	else
		form_entry="${local_path};type=application/octet-stream"
	fi
	curl "${CURL_AUTH[@]}" -H "X-CSRF-Token:${CSRF_TOKEN}" -X POST \
		-F "file=@${form_entry}" \
		"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=%5CLocalState" \
		>/dev/null
}

delete_file() {
	curl "${CURL_AUTH[@]}" -H "X-CSRF-Token:${CSRF_TOKEN}" -X DELETE \
		"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=%5CLocalState&filename=$1" \
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

# Deploy api.flag + model.txt and (re)launch so the endpoint comes up.
start_endpoint() {
	printf 'go' >"${TMPDIR_LOCAL}/api.flag"
	printf '%s' "$MODEL" >"${TMPDIR_LOCAL}/model.txt"
	[[ "$PORT" != "11434" ]] && printf '%s' "$PORT" >"${TMPDIR_LOCAL}/api-port.txt"
	delete_file "bench.flag" # a stale headless flag would exit before binding
	delete_file "diffuse.flag"
	upload_file "${TMPDIR_LOCAL}/api.flag"
	upload_file "${TMPDIR_LOCAL}/model.txt"
	[[ "$PORT" != "11434" ]] && upload_file "${TMPDIR_LOCAL}/api-port.txt"
	restart_app
}

# Poll GET / until it returns 200 or times out.
wait_endpoint() {
	local timeout_s="${1:-120}" elapsed=0 code
	echo "  Waiting for ${API_URL}/ (timeout ${timeout_s}s)..." >&2
	while ((elapsed < timeout_s)); do
		code=$(curl -sS -m 4 -o /dev/null -w "%{http_code}" "${API_URL}/" 2>/dev/null || echo "000")
		if [[ "$code" == "200" ]]; then
			echo "  Endpoint up after ${elapsed}s." >&2
			return 0
		fi
		sleep 5
		((elapsed += 5))
	done
	echo "  Timeout: endpoint never returned 200 (last code ${code})." >&2
	return 1
}

# --- spike gate ------------------------------------------------------------

validate_spike() {
	echo "=== spike: LAN bind + HTTP 200 ==="
	start_endpoint
	if ! wait_endpoint 120; then
		echo "  FAIL: no 200 from ${API_URL}/ — bind blocked or app not serving"
		echo "spike: FAIL"
		return 1
	fi
	local body
	body=$(curl -sS -m 5 "${API_URL}/" || true)
	echo "  GET / -> ${body}"
	if grep -q '"status":"ok"' <<<"$body"; then
		echo "spike: PASS"
		return 0
	fi
	echo "  FAIL: unexpected health body"
	echo "spike: FAIL"
	return 1
}

# --- chat round-trip -------------------------------------------------------

validate_chat() {
	echo "=== chat: /v1/chat/completions ==="
	local req resp verdict=0
	# temperature 0 + fixed seed: deterministic PASS/FAIL. At the default
	# temperature a 350M model occasionally samples a role hallucination
	# ("User\n\n<|end|>"), turning the gate into a coin flip.
	req=$(printf '{"model":"%s","messages":[{"role":"user","content":"Say hello in one short sentence."}],"max_tokens":64,"temperature":0,"seed":42}' "$MODEL")
	resp=$(curl -sS -m 180 -H 'Content-Type: application/json' -d "$req" \
		"${API_URL}/v1/chat/completions" || true)
	local content
	# Parse via stdin: interpolating $resp into a python heredoc broke on any
	# multi-line content (the JSON "\n" became a real newline inside ''' '''),
	# mislabeling valid completions as empty.
	content=$(
		printf '%s' "$resp" | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    print(d.get("choices", [{}])[0].get("message", {}).get("content", ""))
except Exception:
    print("")
'
	)
	if [[ -z "$content" ]]; then
		echo "  FAIL: empty/invalid completion. Raw: ${resp:0:200}"
		verdict=1
	elif [[ "$content" == *"<|"* ]]; then
		# Template-token leak ("User\n\n<|end|>" etc.): the model echoed turn
		# markers instead of answering — a real reply never contains "<|".
		echo "  FAIL: template leak in reply: ${content:0:80}"
		verdict=1
	else
		echo "  ok: assistant replied: ${content:0:80}"
	fi

	# 503-busy: fire two concurrent requests; at least one should be 503 while the
	# single slot is occupied (best-effort — a very fast model may serialize both).
	echo "  checking single-slot 503 (two concurrent requests)..."
	local c1 c2
	curl -sS -m 180 -o /dev/null -w "%{http_code}" -H 'Content-Type: application/json' \
		-d "$req" "${API_URL}/v1/chat/completions" >"${TMPDIR_LOCAL}/c1" 2>/dev/null &
	curl -sS -m 180 -o /dev/null -w "%{http_code}" -H 'Content-Type: application/json' \
		-d "$req" "${API_URL}/v1/chat/completions" >"${TMPDIR_LOCAL}/c2" 2>/dev/null &
	wait
	c1=$(cat "${TMPDIR_LOCAL}/c1" 2>/dev/null || echo "000")
	c2=$(cat "${TMPDIR_LOCAL}/c2" 2>/dev/null || echo "000")
	echo "  concurrent codes: ${c1} ${c2}"
	if [[ "$c1" == "503" || "$c2" == "503" ]]; then
		echo "  ok: single-slot 503 observed"
	else
		echo "  note: no 503 seen (both ${c1}/${c2}) — acceptable if the model served fast"
	fi

	[[ $verdict -eq 0 ]] && echo "chat: PASS" || echo "chat: FAIL"
	return $verdict
}

# --- #118 prefs / training status ------------------------------------------

validate_prefs() {
	echo "=== prefs: POST /v1/preferences ==="
	local req resp code
	req='{"label":"like","messages":[{"role":"user","content":"hi"},{"role":"assistant","content":"hello"}]}'
	code=$(curl -sS -m 30 -o "${TMPDIR_LOCAL}/prefs.json" -w "%{http_code}" \
		-H 'Content-Type: application/json' -d "$req" \
		"${API_URL}/v1/preferences" || echo "000")
	resp=$(cat "${TMPDIR_LOCAL}/prefs.json" 2>/dev/null || true)
	echo "  HTTP ${code}: ${resp:0:120}"
	if [[ "$code" == "200" ]] && grep -qE '"ok"[[:space:]]*:[[:space:]]*true' <<<"$resp"; then
		echo "prefs: PASS"
		return 0
	fi
	echo "prefs: FAIL"
	return 1
}

validate_train() {
	echo "=== train: GET /v1/training/status ==="
	local resp code
	code=$(curl -sS -m 15 -o "${TMPDIR_LOCAL}/train.json" -w "%{http_code}" \
		"${API_URL}/v1/training/status" || echo "000")
	resp=$(cat "${TMPDIR_LOCAL}/train.json" 2>/dev/null || true)
	echo "  HTTP ${code}: ${resp:0:160}"
	if [[ "$code" == "200" ]] && grep -q '"state"' <<<"$resp"; then
		echo "train: PASS"
		return 0
	fi
	echo "train: FAIL"
	return 1
}

# --- context budget over the wire ------------------------------------------

validate_budget() {
	echo "=== budget: a long conversation is trimmed, an oversized message is 400 ==="
	# The LAN endpoint used to have no context budget at all: a long messages[]
	# reached Session::generate and came back 500 — a client error reported as a
	# server one. It now applies the same primitive as the chat UI
	# (xllama::fit_prompt): drop the oldest entries, and 400 only when the final
	# user message alone cannot fit.
	local verdict=0 req resp code

	# A. many old turns + a short question: must answer, not fail.
	req=$(python3 - <<'PYBUDGET'
import json
filler = ("The harbour master logged tide tables, cargo manifests and the crane "
          "maintenance schedule for pier seven, then filed the quarterly figures. ")
msgs = []
for i in range(40):
    msgs.append({"role": "user", "content": f"Item {i:02d}: " + filler * 3})
    msgs.append({"role": "assistant", "content": f"Logged item {i:02d}."})
msgs.append({"role": "user", "content": "Reply with the single word OK."})
print(json.dumps({"model": "__MODEL__", "messages": msgs, "max_tokens": 24}))
PYBUDGET
	)
	req=${req/__MODEL__/$MODEL}
	code=$(curl -sS -m 300 -o "${TMPDIR_LOCAL}/budget-a.json" -w "%{http_code}" \
		-H 'Content-Type: application/json' -d "$req" \
		"${API_URL}/v1/chat/completions" || echo "000")
	resp=$(cat "${TMPDIR_LOCAL}/budget-a.json" 2>/dev/null || true)
	echo "  A HTTP ${code}: ${resp:0:120}"
	if [[ "$code" == "200" ]] && grep -q '"content"' <<<"$resp"; then
		echo "  ok: a history far past n_ctx was trimmed and answered"
	else
		echo "  FAIL: a long conversation must be trimmed, not rejected"
		verdict=1
	fi

	# B. one oversized user message: 400, and the body says why.
	req=$(python3 - <<'PYBUDGET'
import json
print(json.dumps({"model": "__MODEL__",
                  "messages": [{"role": "user", "content": "x " * 20000}],
                  "max_tokens": 24}))
PYBUDGET
	)
	req=${req/__MODEL__/$MODEL}
	code=$(curl -sS -m 300 -o "${TMPDIR_LOCAL}/budget-b.json" -w "%{http_code}" \
		-H 'Content-Type: application/json' -d "$req" \
		"${API_URL}/v1/chat/completions" || echo "000")
	resp=$(cat "${TMPDIR_LOCAL}/budget-b.json" 2>/dev/null || true)
	echo "  B HTTP ${code}: ${resp:0:160}"
	if [[ "$code" == "400" ]] && grep -q 'prompt too long' <<<"$resp"; then
		echo "  ok: refused as a client error, with the numbers"
	else
		echo "  FAIL: an unfittable message must be 400 'prompt too long' (not 500, not 200)"
		verdict=1
	fi

	[[ $verdict -eq 0 ]] && echo "budget: PASS" || echo "budget: FAIL"
	return $verdict
}

# --- dispatch --------------------------------------------------------------

CMD="${1:-}"
case "$CMD" in
spike) validate_spike ;;
chat)
	validate_spike || exit 1
	validate_chat
	;;
budget)
	validate_spike || exit 1
	validate_budget
	;;
prefs)
	validate_spike || exit 1
	validate_prefs
	;;
train)
	validate_spike || exit 1
	validate_train
	;;
all)
	rc=0
	validate_spike || {
		echo "=== summary ==="
		echo "SPIKE FAILED — stopping before chat"
		exit 1
	}
	validate_chat || rc=1
	validate_budget || rc=1
	validate_prefs || rc=1
	validate_train || rc=1
	echo
	echo "=== summary ==="
	[[ $rc -eq 0 ]] && echo "ALL PASS" || echo "SOME FAILED (exit ${rc})"
	exit $rc
	;;
*)
	echo "Usage: $0 <spike|chat|prefs|train|all>" >&2
	exit 1
	;;
esac
