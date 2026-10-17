#!/usr/bin/env bash
# validate-api.sh — deploy api.flag, restart the app, and validate the LAN HTTP
# endpoint (OpenAI-compat) from this host over the network. Deterministic
# PASS/FAIL, no human at the pad. See docs/api-endpoint.md.
#
# Usage:
#   source ~/.config/xllama/xbox-env
#   ./scripts/validate-api.sh <spike|chat|stream|budget|tools|framing|embed|pull|prefs|train|all>
#
#   spike  bind gate only: GET / -> HTTP 200 (proves the StreamSocketListener
#          survives the Series S firewall/PLM). No inference.
#   chat   POST /v1/chat/completions -> non-empty assistant content + a 503-busy
#          check (two concurrent requests). Implies the spike gate first.
#   stream SSE streaming round-trip: text/event-stream, chunked framing, one
#          delta per frame, a final finish_reason chunk and `data: [DONE]`.
#          Implies the spike gate first.
#   budget context budget over the wire: a long messages[] is trimmed and answered,
#          an oversized single message is 400 "prompt too long" (not 500), and
#          invalid generation numbers (max_tokens -1/1.5/INT_MAX, temperature,
#          top_p, seed) are 400 before the model loads (F01).
#   tools  client-side tool proposal (ADR 0004): a named tool_choice comes back
#          as tool_calls with JSON arguments, a tool result is answered,
#          tools+functions is 400, tool_choice none stays plain chat. TOOLS_MODEL
#          (default MODEL) must be a GGUF with a chat template; ORT_MODEL, when
#          set, names an ONNX model that must answer 400.
#   framing exact HTTP framing (ADR 0006): negative/conflicting/oversized
#          Content-Length, chunked, a short body and a stalled body are refused
#          (400/400/413/411/400/408) and a refused POST appends no sample.
#   embed  Ollama native + legacy and OpenAI float/base64 embeddings. Requires
#          EMBED_MODEL to identify an embedding GGUF already on the console.
#   prefs  POST /v1/preferences -> appends a like sample (#118).
#   train  GET /v1/training/status -> JSON with state + usable_samples (#118).
#   all    spike + chat + embed + budget + tools + prefs + train + pull (images are manual).
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

# A trial must never silently measure a different installed package.
if [[ -n "${XLLAMA_EXPECTED_PFN:-}" && "$PFN" != "$XLLAMA_EXPECTED_PFN" ]]; then
	echo "Error: installed package $PFN differs from expected $XLLAMA_EXPECTED_PFN" >&2
	exit 1
fi

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

# --- SSE streaming ---------------------------------------------------------

validate_stream() {
	echo "=== stream: /v1/chat/completions with stream:true (SSE) ==="
	local req headers out code verdict=0 check
	req=$(printf '{"model":"%s","messages":[{"role":"user","content":"Count from one to five."}],"max_tokens":64,"temperature":0,"seed":42,"stream":true}' "${MODEL}")
	headers="${TMPDIR_LOCAL}/stream-headers.txt"
	out="${TMPDIR_LOCAL}/stream-body.txt"
	code=$(curl -sS -m 180 -D "${headers}" -o "${out}" -w "%{http_code}" \
		-H 'Content-Type: application/json' -d "${req}" "${API_URL}/v1/chat/completions" || echo "000")
	if [[ "${code}" != "200" ]]; then
		echo "  FAIL: HTTP ${code} (want 200). Raw: $(head -c 160 "${out}" 2>/dev/null)"
		echo "stream: FAIL"
		return 1
	fi

	# A streamed reply is SSE over chunked transfer: no Content-Length, so a
	# client that only reads a content-length body would hang forever.
	if grep -qi '^content-type: *text/event-stream' "${headers}"; then
		echo "  ok: Content-Type is text/event-stream"
	else
		echo "  FAIL: wrong Content-Type: $(grep -i '^content-type' "${headers}" | head -1)"
		verdict=1
	fi
	if grep -qi '^content-length' "${headers}"; then
		echo "  FAIL: a streamed reply carries a Content-Length"
		verdict=1
	fi

	# Frames are one 'data: <json>' line each, the last one 'data: [DONE]', and
	# the deltas must add up to a non-empty reply with a real finish_reason.
	check=$(
		python3 -c '
import json, sys
lines = open(sys.argv[1], encoding="utf-8", errors="replace").read().splitlines()
frames = [l[len("data: "):] for l in lines if l.startswith("data: ")]
if not frames:
    print("FAIL: no data frames")
    raise SystemExit
if frames[-1].strip() != "[DONE]":
    print("FAIL: stream does not end with data: [DONE]")
    raise SystemExit
text, finish = "", None
for frame in frames[:-1]:
    choice = json.loads(frame).get("choices", [{}])[0]
    text += choice.get("delta", {}).get("content", "") or ""
    if choice.get("finish_reason"):
        finish = choice["finish_reason"]
if not text.strip():
    print("FAIL: no delta content in any frame")
elif finish not in ("stop", "length"):
    print("FAIL: missing or odd finish_reason: %r" % (finish,))
else:
    print("ok: %d frames, %d chars, finish_reason=%s" % (len(frames), len(text), finish))
' "${out}" 2>&1 | tail -n1
	)
	if [[ "${check}" == ok:* ]]; then
		echo "  ${check}"
	else
		echo "  ${check}"
		verdict=1
	fi
	[[ $verdict -eq 0 ]] && echo "stream: PASS" || echo "stream: FAIL"
	return $verdict
}

# --- embedding round-trips -------------------------------------------------

validate_embed() {
	local embed_model="${EMBED_MODEL:-embed-nomic-v2-moe}"
	local evidence_args=()
	if [[ -n "${XLLAMA_API_EVIDENCE_DIR:-}" ]]; then
		evidence_args=(--out "${XLLAMA_API_EVIDENCE_DIR}/${embed_model}.json")
	fi
	python3 "${SCRIPT_DIR}/validate-embedding.py" "$API_URL" "$embed_model" "${evidence_args[@]}"
}

# --- Ollama model pull -----------------------------------------------------

validate_pull() {
	local pull_model="${PULL_MODEL:-lfm25-350m}" pull_kind="${PULL_MODEL_KIND:-chat}"
	local req code verdict=0
	echo "=== pull: /api/pull (${pull_model}) ==="
	req=$(python3 -c 'import json,sys; print(json.dumps({"model":sys.argv[1]}))' "$pull_model")
	code=$(curl -sS -N -m 900 -o "${TMPDIR_LOCAL}/pull-stream.ndjson" -w "%{http_code}" \
		-H 'Content-Type: application/json' -d "$req" "${API_URL}/api/pull" || echo "000")
	if [[ "$code" == "200" ]] && python3 -c '
import json,sys
events=[json.loads(line) for line in open(sys.argv[1]) if line.strip()]
assert events and events[-1] == {"status":"success"}
assert any(e.get("status") == "pulling model" for e in events)
' "${TMPDIR_LOCAL}/pull-stream.ndjson"; then
		echo "  ok: NDJSON pull completed with progress and success records"
	else
		echo "  FAIL: streamed pull HTTP ${code}: $(tail -n 3 "${TMPDIR_LOCAL}/pull-stream.ndjson" 2>/dev/null | tr '\n' ' ')"
		verdict=1
	fi

	req=$(python3 -c 'import json,sys; print(json.dumps({"model":sys.argv[1],"stream":False}))' "$pull_model")
	code=$(curl -sS -m 900 -o "${TMPDIR_LOCAL}/pull-single.json" -w "%{http_code}" \
		-H 'Content-Type: application/json' -d "$req" "${API_URL}/api/pull" || echo "000")
	if [[ "$code" == "200" ]] && python3 -c 'import json,sys; assert json.load(open(sys.argv[1])) == {"status":"success"}' "${TMPDIR_LOCAL}/pull-single.json"; then
		echo "  ok: stream=false returns one success object"
	else
		echo "  FAIL: stream=false HTTP ${code}"
		verdict=1
	fi

	code=$(curl -sS -m 30 -o "${TMPDIR_LOCAL}/pull-unknown.json" -w "%{http_code}" \
		-H 'Content-Type: application/json' -d '{"model":"https://example.invalid/model"}' \
		"${API_URL}/api/pull" || echo "000")
	if [[ "$code" == "404" ]]; then
		echo "  ok: arbitrary remote model sources are rejected"
	else
		echo "  FAIL: arbitrary remote source returned HTTP ${code}"
		verdict=1
	fi

	if [[ "$pull_kind" == "embedding" ]]; then
		req=$(python3 -c 'import json,sys; print(json.dumps({"model":sys.argv[1],"input":"pull smoke query"}))' "$pull_model")
		code=$(curl -sS -m 300 -o "${TMPDIR_LOCAL}/pull-infer.json" -w "%{http_code}" \
			-H 'Content-Type: application/json' -d "$req" "${API_URL}/api/embed" || echo "000")
		if [[ "$code" == "200" ]] && python3 -c 'import json,sys; assert json.load(open(sys.argv[1]))["embeddings"][0]' "${TMPDIR_LOCAL}/pull-infer.json"; then
			echo "  ok: pulled embedding model inferred"
		else
			echo "  FAIL: pulled embedding model inference returned HTTP ${code}"
			verdict=1
		fi
	else
		req=$(python3 -c 'import json,sys; print(json.dumps({"model":sys.argv[1],"messages":[{"role":"user","content":"Say OK."}],"max_tokens":16,"temperature":0}))' "$pull_model")
		code=$(curl -sS -m 300 -o "${TMPDIR_LOCAL}/pull-infer.json" -w "%{http_code}" \
			-H 'Content-Type: application/json' -d "$req" "${API_URL}/v1/chat/completions" || echo "000")
		if [[ "$code" == "200" ]] && python3 -c 'import json,sys; assert json.load(open(sys.argv[1]))["choices"][0]["message"]["content"]' "${TMPDIR_LOCAL}/pull-infer.json"; then
			echo "  ok: pulled chat model inferred"
		else
			echo "  FAIL: pulled chat model inference returned HTTP ${code}"
			verdict=1
		fi
	fi
	[[ $verdict -eq 0 ]] && echo "pull: PASS" || echo "pull: FAIL"
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

# --- HTTP framing over the wire (ADR 0006, audit F02) ---------------------

validate_framing() {
	echo "=== framing: exact Content-Length, refusals, deadlines ==="
	# Raw sockets: curl cannot send a negative length, a short body or a stall.
	# The truncated preferences POST must not append a sample (no side effect).
	XBOX_IP="$XBOX_IP" PORT="$PORT" python3 - <<'PYFRAME'
import json, os, socket, time
host, port = os.environ["XBOX_IP"], int(os.environ["PORT"])

def exchange(raw, close_write=True, timeout=40):
    s = socket.create_connection((host, port), timeout=timeout)
    s.sendall(raw)
    if close_write:
        s.shutdown(socket.SHUT_WR)
    t0 = time.time()
    data = b""
    try:
        while True:
            chunk = s.recv(65536)
            if not chunk:
                break
            data += chunk
    except (socket.timeout, ConnectionResetError):
        # A refused request is answered, then closed; a reset after the
        # status line still carries the answer read so far.
        pass
    s.close()
    status = data.split(b" ", 2)[1].decode() if data.startswith(b"HTTP/") else "none"
    return status, time.time() - t0, data.split(b"\r\n\r\n", 1)[-1][:120].decode(errors="replace")

def samples():
    _, _, body = exchange(b"GET /v1/training/status HTTP/1.1\r\nHost: x\r\n\r\n")
    try:
        return json.loads(body).get("usable_samples")
    except Exception:
        return None

pref = b'{"label":"like","messages":[{"role":"user","content":"framing probe"},{"role":"assistant","content":"ok"}]}'
cases = [
    ("negative Content-Length", b"POST /v1/preferences HTTP/1.1\r\nContent-Length: -1\r\n\r\n" + pref, True, "400"),
    ("conflicting Content-Length", b"POST /v1/preferences HTTP/1.1\r\nContent-Length: 5\r\nContent-Length: 6\r\n\r\nhello", True, "400"),
    ("oversized Content-Length", b"POST /v1/preferences HTTP/1.1\r\nContent-Length: 9000000\r\n\r\n", True, "413"),
    ("chunked", b"POST /v1/preferences HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n", True, "411"),
    ("short body", b"POST /v1/preferences HTTP/1.1\r\nContent-Length: %d\r\n\r\n" % (len(pref) + 100) + pref, True, "400"),
    ("stalled body", b"POST /v1/preferences HTTP/1.1\r\nContent-Length: 10\r\n\r\nabc", False, "408"),
    ("valid request", b"GET / HTTP/1.1\r\nHost: x\r\n\r\n", True, "200"),
]
before = samples()
fail = 0
for name, raw, close_write, want in cases:
    status, secs, body = exchange(raw, close_write)
    ok = status == want
    fail |= not ok
    print(f"  {'ok' if ok else 'FAIL'}: {name} -> {status} in {secs:.1f}s (want {want}) {body}")
after = samples()
if before is None or after != before:
    print(f"  FAIL: usable_samples {before} -> {after}: a refused request had a side effect")
    fail = 1
else:
    print(f"  ok: usable_samples unchanged ({after})")
print("framing: " + ("FAIL" if fail else "PASS"))
raise SystemExit(fail)
PYFRAME
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

	# C. F01: generation numbers are validated before the model loads. The audit
	# sent -1, 1.5 and INT_MAX and got 200 for all three.
	local body want body_want
	for body_want in \
		'"max_tokens":-1|400' '"max_tokens":1.5|400' '"max_tokens":2147483647|400' \
		'"max_completion_tokens":0|400' '"temperature":5|400' '"top_p":-0.5|400' \
		'"seed":1.5|400' '"seed":-1,"max_tokens":8|200'; do
		body=${body_want%|*}
		want=${body_want##*|}
		req=$(printf '{"model":"%s","messages":[{"role":"user","content":"Say OK."}],%s}' "$MODEL" "$body")
		code=$(curl -sS -m 120 -o "${TMPDIR_LOCAL}/budget-c.json" -w "%{http_code}" \
			-H 'Content-Type: application/json' -d "$req" \
			"${API_URL}/v1/chat/completions" || echo "000")
		if [[ "$code" == "$want" ]]; then
			echo "  C ok: {${body}} -> ${code} $(head -c 90 "${TMPDIR_LOCAL}/budget-c.json")"
		else
			echo "  C FAIL: {${body}} -> ${code}, want ${want}: $(head -c 120 "${TMPDIR_LOCAL}/budget-c.json")"
			verdict=1
		fi
	done

	[[ $verdict -eq 0 ]] && echo "budget: PASS" || echo "budget: FAIL"
	return $verdict
}

# --- tool proposals (ADR 0004) -----------------------------------------------

# POST $1 to chat/completions; writes the body to $2 and prints the HTTP code.
post_chat() {
	curl -sS -m 300 -o "$2" -w "%{http_code}" -H 'Content-Type: application/json' \
		-d "$1" "${API_URL}/v1/chat/completions" || echo "000"
}

validate_tools() {
	echo "=== tools: client-side tool proposal (ADR 0004) ==="
	local tools_model="${TOOLS_MODEL:-$MODEL}" verdict=0 req code check
	local schema='[{"type":"function","function":{"name":"get_weather","description":"Current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]'

	# A. named tool_choice: the reply must be a get_weather call with JSON arguments.
	req=$(printf '{"model":"%s","messages":[{"role":"user","content":"What is the weather in Paris?"}],"tools":%s,"tool_choice":{"type":"function","function":{"name":"get_weather"}},"max_tokens":128,"temperature":0,"seed":42}' "$tools_model" "$schema")
	code=$(post_chat "$req" "${TMPDIR_LOCAL}/tools-a.json")
	check=$(python3 -c '
import json, sys
d = json.load(open(sys.argv[1]))
c = d["choices"][0]
calls = c["message"].get("tool_calls") or []
assert calls, "no tool_calls (finish_reason=%s)" % c.get("finish_reason")
f = calls[0]["function"]
assert f["name"] == "get_weather", "name " + f["name"]
json.loads(f["arguments"])
assert calls[0].get("id"), "missing id"
assert c["finish_reason"] == "tool_calls", "finish_reason " + c["finish_reason"]
print("get_weather(" + f["arguments"] + ")")
' "${TMPDIR_LOCAL}/tools-a.json" 2>&1 | tail -n1 || true)
	if [[ "$code" == "200" && "$check" == get_weather* ]]; then
		echo "  A ok: ${check:0:100}"
	else
		echo "  A FAIL: HTTP ${code}: ${check:0:160}"
		verdict=1
	fi

	# B. a tool result after the call: answered, no new call required.
	req=$(printf '{"model":"%s","messages":[{"role":"user","content":"What is the weather in Paris?"},{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":"{\\"city\\":\\"Paris\\"}"}}]},{"role":"tool","tool_call_id":"call_1","content":"18 C, light rain"}],"tools":%s,"max_tokens":96,"temperature":0,"seed":42}' "$tools_model" "$schema")
	code=$(post_chat "$req" "${TMPDIR_LOCAL}/tools-b.json")
	check=$(python3 -c '
import json, sys
m = json.load(open(sys.argv[1]))["choices"][0]["message"]
print("ok" if (m.get("content") or m.get("tool_calls")) else "empty message")
' "${TMPDIR_LOCAL}/tools-b.json" 2>&1 | tail -n1 || true)
	if [[ "$code" == "200" && "$check" == "ok" ]]; then
		echo "  B ok: tool result answered"
	else
		echo "  B FAIL: HTTP ${code}: ${check:0:160}"
		verdict=1
	fi

	# C. tools and functions together: 400 with the reason.
	req=$(printf '{"model":"%s","messages":[{"role":"user","content":"hi"}],"tools":%s,"functions":[{"name":"f","parameters":{"type":"object"}}]}' "$tools_model" "$schema")
	code=$(post_chat "$req" "${TMPDIR_LOCAL}/tools-c.json")
	if [[ "$code" == "400" ]] && grep -q 'cannot be combined' "${TMPDIR_LOCAL}/tools-c.json"; then
		echo "  C ok: tools+functions is 400"
	else
		echo "  C FAIL: HTTP ${code}: $(head -c 160 "${TMPDIR_LOCAL}/tools-c.json")"
		verdict=1
	fi

	# D. tool_choice none: plain chat, text content and no tool_calls.
	req=$(printf '{"model":"%s","messages":[{"role":"user","content":"Say hello in one short sentence."}],"tools":%s,"tool_choice":"none","max_tokens":48,"temperature":0,"seed":42}' "$tools_model" "$schema")
	code=$(post_chat "$req" "${TMPDIR_LOCAL}/tools-d.json")
	check=$(python3 -c '
import json, sys
m = json.load(open(sys.argv[1]))["choices"][0]["message"]
print("ok" if m.get("content") and not m.get("tool_calls") else "unexpected " + json.dumps(m)[:120])
' "${TMPDIR_LOCAL}/tools-d.json" 2>&1 | tail -n1 || true)
	if [[ "$code" == "200" && "$check" == "ok" ]]; then
		echo "  D ok: tool_choice none answered as plain chat"
	else
		echo "  D FAIL: HTTP ${code}: ${check:0:160}"
		verdict=1
	fi

	# E. optional: an ONNX session has no GGUF template, so a proposal is 400.
	if [[ -n "${ORT_MODEL:-}" ]]; then
		req=$(printf '{"model":"%s","messages":[{"role":"user","content":"hi"}],"tools":%s}' "$ORT_MODEL" "$schema")
		code=$(post_chat "$req" "${TMPDIR_LOCAL}/tools-e.json")
		if [[ "$code" == "400" ]] && grep -q 'GGUF chat template' "${TMPDIR_LOCAL}/tools-e.json"; then
			echo "  E ok: ONNX model refused with 400"
		else
			echo "  E FAIL: HTTP ${code}: $(head -c 160 "${TMPDIR_LOCAL}/tools-e.json")"
			verdict=1
		fi
	else
		echo "  E skipped: set ORT_MODEL to check the ONNX 400"
	fi

	[[ $verdict -eq 0 ]] && echo "tools: PASS" || echo "tools: FAIL"
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
tools)
	validate_spike || exit 1
	validate_tools
	;;
framing)
	validate_spike || exit 1
	validate_framing
	;;
stream)
	validate_spike || exit 1
	validate_stream
	;;
prefs)
	validate_spike || exit 1
	validate_prefs
	;;
train)
	validate_spike || exit 1
	validate_train
	;;
pull)
	validate_spike || exit 1
	validate_pull
	;;
embed)
	validate_spike || exit 1
	validate_embed
	;;
# Every check below, in order: spike, chat, stream, embed, budget, tools, framing,
# prefs, train, pull.
all)
	rc=0
	validate_spike || {
		echo "=== summary ==="
		echo "SPIKE FAILED — stopping before chat"
		exit 1
	}
	validate_chat || rc=1
	validate_stream || rc=1
	validate_embed || rc=1
	validate_budget || rc=1
	validate_tools || rc=1
	validate_framing || rc=1
	validate_prefs || rc=1
	validate_train || rc=1
	validate_pull || rc=1
	echo
	echo "=== summary ==="
	[[ $rc -eq 0 ]] && echo "ALL PASS" || echo "SOME FAILED (exit ${rc})"
	exit $rc
	;;
*)
	echo "Usage: $0 <spike|chat|stream|budget|tools|framing|embed|pull|prefs|train|all>" >&2
	exit 1
	;;
esac
