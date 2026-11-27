#!/usr/bin/env bash
# validate-api.sh — deploy api.flag, restart the app, and validate the LAN HTTP
# endpoint (OpenAI-compat) from this host over the network. Deterministic
# PASS/FAIL, no human at the pad. See docs/api-endpoint.md.
#
# Usage:
#   source ~/.config/xllama/xbox-env
#   ./scripts/validate-api.sh <spike|discovery|chat|stream|admission|disconnect|budget|tools|framing|embed|pull|prefs|train|compat|all>
#
#   spike  bind gate only: GET / -> HTTP 200 (proves the StreamSocketListener
#          survives the Series S firewall/PLM). No inference.
#   chat   POST /v1/chat/completions -> non-empty assistant content + a 503-busy
#          check (two concurrent requests). When /health lists
#          reasoning_control (#648): reasoning_effort "none" on a model whose
#          capabilities.reasoning_control is true (REASONING_MODEL, default the
#          first such model installed) returns no reasoning_content, and a
#          switch the policy refuses (a model with reasoning_control false) is
#          400 reasoning_control_unsupported. Implies the spike gate first.
#   stream SSE streaming round-trip: text/event-stream, chunked framing, one
#          delta per frame, a final finish_reason chunk and `data: [DONE]`.
#          Implies the spike gate first.
#   admission compute-permit hold (ADR 0008): a long streamed reply
#          (max_tokens >= 256) holds the permit; the probe waits for the first
#          SSE data chunk, then a second /v1/chat/completions must be HTTP 503
#          busy, and the stream must still finish with data: [DONE]. Address is
#          $XBOX_IP:$API_PORT (no hardcoded host). Implies the spike gate first.
#          For a GUI turn already generating, the same curl (printed by this
#          mode) must be 503.
#   budget context budget over the wire: a long messages[] is trimmed and answered,
#          an oversized single message is 400 "prompt too long" (not 500), and
#          invalid generation numbers (max_tokens -1/1.5/INT_MAX, temperature,
#          top_p, seed) are 400 before the model loads (F01).
#   tools  client-side tool proposal (ADR 0004): a named tool_choice comes back
#          as tool_calls with JSON arguments, a tool result is answered,
#          tools+functions is 400, tool_choice none stays plain chat, and an
#          oversized tools array is 400 "prompt too long" with
#          context_length_exceeded. TOOLS_MODEL (default MODEL) must be a GGUF
#          with a chat template; ORT_MODEL, when set, names an ONNX model that
#          must answer 400.
#   framing exact HTTP framing (ADR 0006): negative/conflicting/oversized
#          Content-Length, chunked, a short body and a stalled body are refused
#          (400/400/413/411/400/408) and a refused POST appends no sample.
#   embed  Ollama native + legacy and OpenAI float/base64 embeddings. Requires
#          EMBED_MODEL to identify an embedding GGUF already on the console.
#   disconnect  a streamed client that leaves during a long prefill releases
#          the session (#581): the keepalive write fails, the prefill aborts,
#          and a probe is served within DISCONNECT_LIMIT_S (default 30) while the
#          prefill alone would take a minute. It runs once for plain chat and
#          once for a streamed tool proposal (#594). DISCONNECT_MODEL (default
#          llama32-3b) must be slow enough for that and a GGUF with a tool
#          template. Not part of `all`.
#   prefs  POST /v1/preferences -> appends a like sample (#118).
#   train  GET /v1/training/status -> JSON with state + usable_samples (#118).
#   compat native Ollama /api/chat and /api/generate (validate-api-compat.py):
#          NDJSON and JSON framing, refused options and fields, think,
#          load-only requests, a tool proposal and its replay, and busy while a
#          native stream holds the permit, options.num_ctx on both protocols,
#          and the keep_alive lease, expiry and unload of a model the API
#          loads (/api/ps). MODEL is a GGUF chat model and
#          TOOLS_MODEL (default MODEL) a tool-capable GGUF; ORT_MODEL, when
#          set, must be refused by the native routes. API_COMPAT_SDKS=1 also
#          runs the installed `openai` and `ollama` Python packages. Not part
#          of `all`.
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
# shellcheck disable=SC1090,SC1091
source "${SCRIPT_DIR}/wdp-curl-auth.sh"
# shellcheck disable=SC1090,SC1091
source "${SCRIPT_DIR}/wdp-file-upload.sh"
wdp_setup_curl_auth
DEPLOY="${SCRIPT_DIR}/deploy.sh"
BASE_URL="https://${XBOX_IP}:11443"

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
trap 'rm -rf "$TMPDIR_LOCAL"; wdp_curl_auth_cleanup' EXIT

# The usage block of every chat reply (#585): the Python checks below that read
# one exec this first. usage_problem returns "" when the block is well formed.
export XLLAMA_USAGE_PY='
def usage_problem(usage, reasoning):
    if not isinstance(usage, dict):
        return "no usage block"
    p, c = usage.get("prompt_tokens"), usage.get("completion_tokens")
    if usage.get("total_tokens") != p + c:
        return "total_tokens != prompt + completion: %r" % usage
    r = (usage.get("completion_tokens_details") or {}).get("reasoning_tokens")
    if not isinstance(r, int) or not 0 <= r <= c:
        return "reasoning_tokens out of [0, completion_tokens]: %r" % usage
    if reasoning and r == 0:
        return "reasoning_content without reasoning_tokens: %r" % usage
    k = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
    if not isinstance(k, int) or not 0 <= k <= p:
        return "cached_tokens out of [0, prompt_tokens]: %r" % usage
    return ""
'

# --- WDP helpers (same idioms as validate-console.sh) ----------------------

upload_file() {
	local local_path="$1" remote_name="${2:-}"
	wdp_upload_file_request \
		"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=%5CLocalState" \
		"$local_path" "$remote_name"
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
	aumid=$(printf '%s!xllama' "$pfamily" | base64 | tr -d '\n')
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

# --- discovery: /health and /v1/models fields (#565, #566) -----------------

validate_discovery() {
	echo "=== discovery: /health version + capabilities, /v1/models fields ==="
	local health models
	health=$(curl -fsS -m 10 "${API_URL}/health" || true)
	models=$(curl -fsS -m 10 "${API_URL}/v1/models" || true)
	echo "  GET /health -> ${health}"
	if python3 -c '
import json, sys
h = json.loads(sys.argv[1])
assert h["status"] == "ok" and h["service"] == "xllama", h
assert isinstance(h["version"], str) and h["version"].count(".") == 3, h["version"]
assert isinstance(h["api"], int) and h["api"] >= 1, h["api"]
assert h["response_formats"] == ["text"], h.get("response_formats")
caps = h["capabilities"]
for key in ("tools", "tools_stream", "embeddings", "images", "pull", "preferences",
            "training_status", "reasoning_content", "reasoning_tokens", "cached_tokens"):
    assert isinstance(caps[key], bool), (key, caps)
m = json.loads(sys.argv[2])
assert m["object"] == "list" and m["data"], m
for entry in m["data"]:
    for key in ("id", "object", "created", "owned_by", "role", "capabilities"):
        assert key in entry, (key, entry)
    for key in ("tools", "reasoning", "embedding"):
        assert isinstance(entry["capabilities"][key], bool), entry
    if caps.get("reasoning_control"):
        assert isinstance(entry["capabilities"]["reasoning_control"], bool), entry
    chat = entry["role"] != "embedding" and entry.get("kind") != "diffusion"
    if caps.get("model_sampling") and chat:
        s = entry["sampling"]
        for key in ("temperature", "top_p", "top_k", "repetition_penalty"):
            assert isinstance(s[key], (int, float)), (key, entry)
            assert s["source"][key] in ("default", "model"), (key, entry)
        assert isinstance(s["top_k"], int), entry
    if entry.get("kind") == "gguf" and entry["role"] != "embedding":
        print("  %s: n_ctx=%s tools=%s reasoning=%s sampling=%s" % (entry["id"],
              entry.get("n_ctx"), entry["capabilities"]["tools"],
              entry["capabilities"]["reasoning"], entry.get("sampling")))
' "$health" "$models"; then
		echo "discovery: PASS"
		return 0
	fi
	echo "discovery: FAIL"
	return 1
}

# --- per-request reasoning control (#648) ----------------------------------

# Part of the chat gate. The models come from /v1/models
# (capabilities.reasoning_control), never from a list kept here.
validate_reasoning_control() {
	local health models picks on_model refused_model refused_effort
	health=$(curl -fsS -m 10 "${API_URL}/health" || true)
	if ! printf '%s' "$health" | python3 -c '
import json, sys
sys.exit(0 if json.load(sys.stdin)["capabilities"].get("reasoning_control") else 1)
' 2>/dev/null; then
		echo "  note: /health does not list reasoning_control: skipped"
		return 0
	fi
	models=$(curl -fsS -m 10 "${API_URL}/v1/models" || true)
	# Prints "<switchable model> <refused model> <effort it refuses>"; "-" when absent.
	picks=$(printf '%s' "$models" | python3 -c '
import json, sys
wanted = sys.argv[1]
chat = [e for e in json.load(sys.stdin)["data"]
        if e["role"] != "embedding" and e.get("kind") != "diffusion"]
on = [e["id"] for e in chat if e["capabilities"].get("reasoning_control")]
if wanted:
    on = [i for i in on if i == wanted] or ["missing:" + wanted]
off = [e for e in chat if not e["capabilities"].get("reasoning_control")]
# A model that reasons cannot be turned off; one that does not cannot be turned on.
off.sort(key=lambda e: not e["capabilities"]["reasoning"])
refused = off[0] if off else None
print(on[0] if on else "-",
      refused["id"] if refused else "-",
      ("none" if refused["capabilities"]["reasoning"] else "high") if refused else "-")
' "${REASONING_MODEL:-}" || echo "- - -")
	read -r on_model refused_model refused_effort <<<"$picks"
	local verdict=0 status
	if [[ "$on_model" == missing:* ]]; then
		echo "  FAIL: REASONING_MODEL ${on_model#missing:} does not report reasoning_control"
		verdict=1
	elif [[ "$on_model" == "-" ]]; then
		echo "  note: no installed model reports reasoning_control: true; switch not checked"
	else
		# Off: the template's no-think path, so no reasoning comes back.
		status=$(post_chat "$(printf '{"model":"%s","messages":[{"role":"user","content":"What is 2+2? Answer with the number."}],"max_tokens":64,"temperature":0,"seed":42,"reasoning_effort":"none"}' "$on_model")" \
			"${TMPDIR_LOCAL}/reasoning-off.json")
		if [[ "$status" == 200 ]] && python3 -c '
import json, sys
m = json.load(open(sys.argv[1]))["choices"][0]["message"]
sys.exit(0 if not m.get("reasoning_content") and m.get("content") else 1)
' "${TMPDIR_LOCAL}/reasoning-off.json"; then
			echo "  ok: ${on_model} with reasoning_effort none answered without reasoning_content"
		else
			echo "  FAIL: ${on_model} reasoning_effort none: HTTP ${status} $(head -c 300 "${TMPDIR_LOCAL}/reasoning-off.json")"
			verdict=1
		fi
		# On: reported, not asserted (a short answer may skip the reasoning).
		status=$(post_chat "$(printf '{"model":"%s","messages":[{"role":"user","content":"What is 2+2? Answer with the number."}],"max_tokens":256,"temperature":0,"seed":42,"reasoning_effort":"high"}' "$on_model")" \
			"${TMPDIR_LOCAL}/reasoning-on.json")
		echo "  note: ${on_model} reasoning_effort high: HTTP ${status}, $(python3 -c '
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print("%d reasoning tokens" % d["usage"]["completion_tokens_details"]["reasoning_tokens"])
except Exception:
    print("no usage")
' "${TMPDIR_LOCAL}/reasoning-on.json")"
	fi
	if [[ "$refused_model" == "-" ]]; then
		echo "  note: every installed chat model reports reasoning_control: true; refusal not checked"
	else
		status=$(post_chat "$(printf '{"model":"%s","messages":[{"role":"user","content":"Hi"}],"reasoning_effort":"%s"}' "$refused_model" "$refused_effort")" \
			"${TMPDIR_LOCAL}/reasoning-refused.json")
		if [[ "$status" == 400 ]] && python3 -c '
import json, sys
err = json.load(open(sys.argv[1]))["error"]
sys.exit(0 if (err.get("code"), err.get("param")) == ("reasoning_control_unsupported", "reasoning_effort") else 1)
' "${TMPDIR_LOCAL}/reasoning-refused.json"; then
			echo "  ok: ${refused_model} reasoning_effort ${refused_effort} is 400 reasoning_control_unsupported"
		else
			echo "  FAIL: ${refused_model} reasoning_effort ${refused_effort}: HTTP ${status} $(head -c 300 "${TMPDIR_LOCAL}/reasoning-refused.json")"
			verdict=1
		fi
	fi
	return $verdict
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
import json, os, sys
exec(os.environ["XLLAMA_USAGE_PY"])
try:
    d = json.load(sys.stdin)
    message = d.get("choices", [{}])[0].get("message", {})
    problem = usage_problem(d.get("usage"), message.get("reasoning_content"))
    print(("USAGE " + problem) if problem else message.get("content", ""))
except Exception:
    print("")
'
	)
	if [[ -z "$content" ]]; then
		echo "  FAIL: empty/invalid completion. Raw: ${resp:0:200}"
		verdict=1
	elif [[ "$content" == "USAGE "* ]]; then
		echo "  FAIL: ${content#USAGE }"
		verdict=1
	elif [[ "$content" == *"<|"* ]]; then
		# Template-token leak ("User\n\n<|end|>" etc.): the model echoed turn
		# markers instead of answering — a real reply never contains "<|".
		echo "  FAIL: template leak in reply: ${content:0:80}"
		verdict=1
	else
		echo "  ok: assistant replied: ${content:0:80}"
	fi

	# #595: the same request again finds its prompt in the resident KV, so most
	# of it is reported as cached (a GGUF model; ORT prefills everything).
	# Explicit text must accept the same prompt as the omitted response_format.
	local cached text_req
	text_req=$(python3 -c 'import json,sys; r=json.loads(sys.argv[1]); r["response_format"]={"type":"text"}; print(json.dumps(r))' "$req")
	cached=$(curl -sS -m 180 -H 'Content-Type: application/json' -d "$text_req" \
		"${API_URL}/v1/chat/completions" | python3 -c '
import json, sys
try:
    u = json.load(sys.stdin)["usage"]
    print("%d %d" % (u["prompt_tokens_details"]["cached_tokens"], u["prompt_tokens"]))
except Exception:
    print("")
' || true)
	local kind
	kind=$(curl -sS -m 10 "${API_URL}/v1/models" | python3 -c '
import json, sys
m = [e for e in json.load(sys.stdin)["data"] if e["id"] == sys.argv[1]]
print(m[0].get("kind", "") if m else "")
' "$MODEL" 2>/dev/null || true)
	if [[ "$kind" != "gguf" ]]; then
		echo "  note: ${MODEL} is not a GGUF model (kind '${kind}'): no prefix reuse to check"
	elif [[ "$cached" =~ ^([0-9]+)\ ([0-9]+)$ ]] && ((BASH_REMATCH[1] * 2 > BASH_REMATCH[2])); then
		echo "  ok: repeated request cached ${BASH_REMATCH[1]} of ${BASH_REMATCH[2]} prompt tokens"
	else
		echo "  FAIL: repeated request did not report most of its prompt as cached: '${cached}'"
		verdict=1
	fi

	# A structured-output request must fail before a missing model is looked up,
	# even when the client asked for SSE. The error is one HTTP JSON response.
	local format_status
	format_status=$(curl -sS -m 15 -o "${TMPDIR_LOCAL}/format-error.json" -w '%{http_code}' \
		-H 'Content-Type: application/json' \
		-d '{"model":"xllama-missing-format-test-model","messages":[{"role":"user","content":"Hi"}],"stream":true,"response_format":{"type":"json_object"}}' \
		"${API_URL}/v1/chat/completions" || true)
	if [[ "$format_status" != 400 ]] || ! python3 -c '
import json, sys
with open(sys.argv[1]) as src:
    err = json.load(src)["error"]
sys.exit(0 if (err.get("code"), err.get("param")) == ("unsupported_response_format", "response_format") else 1)
' "${TMPDIR_LOCAL}/format-error.json"; then
		echo "  FAIL: expected HTTP 400 response_format validation error, got ${format_status}"
		verdict=1
	fi

	validate_reasoning_control || verdict=1

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
import json, os, sys
exec(os.environ["XLLAMA_USAGE_PY"])
lines = open(sys.argv[1], encoding="utf-8", errors="replace").read().splitlines()
frames = [l[len("data: "):] for l in lines if l.startswith("data: ")]
if not frames:
    print("FAIL: no data frames")
    raise SystemExit
if frames[-1].strip() != "[DONE]":
    print("FAIL: stream does not end with data: [DONE]")
    raise SystemExit
text, reasoning, finish, usage = "", "", None, None
for frame in frames[:-1]:
    chunk = json.loads(frame)
    choice = chunk.get("choices", [{}])[0]
    text += choice.get("delta", {}).get("content", "") or ""
    reasoning += choice.get("delta", {}).get("reasoning_content", "") or ""
    if choice.get("finish_reason"):
        finish, usage = choice["finish_reason"], chunk.get("usage")
problem = usage_problem(usage, reasoning)
if not text.strip():
    print("FAIL: no delta content in any frame")
elif problem:
    print("FAIL: final chunk usage: " + problem)
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
	# Exercise each explicit usage mode through the same wire verifier as tool proposals.
	local mode
	for mode in true false; do
		validate_tool_stream "{\"stream_options\":{\"include_usage\":${mode}},${req:1}" text "${mode}" || verdict=1
	done
	# An SDK option this server does not implement is ignored, not refused.
	validate_tool_stream "{\"stream_options\":{\"include_usage\":true,\"include_obfuscation\":false},${req:1}" text true || verdict=1
	local invalid
	for invalid in '"stream":true,"stream_options":[]' \
		'"stream":true,"stream_options":{"include_usage":"yes"}' \
		'"stream":false,"stream_options":{"include_usage":true}'; do
		code=$(post_chat "{${invalid}}" "${TMPDIR_LOCAL}/bad-stream-options.json")
		if [[ "$code" != 400 ]] || ! python3 -c 'import json,sys; assert json.load(open(sys.argv[1]))["error"]["param"] == "stream_options"' "${TMPDIR_LOCAL}/bad-stream-options.json"; then
			echo "  FAIL: invalid stream options accepted or misclassified: ${invalid} (HTTP ${code})"
			verdict=1
		fi
	done
	[[ $verdict -eq 0 ]] && echo "stream: PASS" || echo "stream: FAIL"
	return $verdict
}

# --- client disconnect during prefill (#581, #594) -------------------------

# One leave-and-probe round: the client gives up 3 s into the prefill of
# |long_req|, then a short request must get the session within |limit| s.
disconnect_round() {
	local label="$1" long_req="$2" probe_req="$3" limit="$4" code start elapsed
	local partial="${TMPDIR_LOCAL}/disconnect.sse" curl_status=0
	printf '%s' "$long_req" | curl -sS -N -m 3 -o "$partial" -H 'Content-Type: application/json' \
		--data-binary @- "${API_URL}/v1/chat/completions" 2>/dev/null || curl_status=$?
	if [[ "$curl_status" != 28 ]] || ! grep -q '^data: ' "$partial" ||
		grep -qE '^data: \[DONE\]|"usage":[[:space:]]*\{' "$partial"; then
		echo "  FAIL: ${label} did not stop after the stream opened and before completion (curl ${curl_status})"
		return 1
	fi
	start=$(date +%s)
	while :; do
		code=$(curl -sS -m 120 -o /dev/null -w "%{http_code}" -H 'Content-Type: application/json' \
			-d "$probe_req" "${API_URL}/v1/chat/completions" || echo "000")
		elapsed=$(($(date +%s) - start))
		[[ "$code" != "503" ]] && break
		if ((elapsed > 180)); then
			break
		fi
		sleep 1
	done
	echo "  ${label}: probe after the disconnect: HTTP ${code} in ${elapsed}s (limit ${limit}s)"
	[[ "$code" == "200" ]] && ((elapsed <= limit))
}

validate_disconnect() {
	echo "=== disconnect: a client gone during prefill releases the session ==="
	local model="${DISCONNECT_MODEL:-llama32-3b}" limit="${DISCONNECT_LIMIT_S:-30}"
	local plain_req tool_req probe_req code verdict=0
	# About 1600 tokens: inside a 2048 context, a minute of CPU prefill on a 3B.
	# The nonce keeps a cached prefix from making it fast. The tool variant
	# streams a tool proposal, whose stream opens before generation (#594).
	long_request() {
		python3 -c '
import json, sys, uuid
filler = ("Entry %s: the harbour master logged tide tables, cargo manifests and the crane "
          "maintenance schedule for pier seven. ")
text = "".join(filler % i for i in range(70))
body = {"model": sys.argv[1], "stream": True, "max_tokens": 16,
        "stream_options": {"include_usage": True},
        "messages": [{"role": "user", "content": uuid.uuid4().hex + " " + text +
                      "Summarise the log."}]}
if sys.argv[2] == "tool":
    body["tools"] = [{"type": "function", "function": {
        "name": "file_report", "description": "File a summary of the harbour log",
        "parameters": {"type": "object", "properties": {"summary": {"type": "string"}},
                       "required": ["summary"]}}}]
print(json.dumps(body))
' "$model" "$1"
	}
	plain_req=$(long_request plain)
	tool_req=$(long_request tool)
	probe_req=$(printf '{"model":"%s","messages":[{"role":"user","content":"Say OK."}],"max_tokens":1}' "$model")
	# Load the model first, so the timing below is the prefill, not the load.
	code=$(curl -sS -m 300 -o /dev/null -w "%{http_code}" -H 'Content-Type: application/json' \
		-d "$probe_req" "${API_URL}/v1/chat/completions" || echo "000")
	if [[ "$code" != "200" ]]; then
		echo "  FAIL: warm-up on ${model} was HTTP ${code}"
		echo "disconnect: FAIL"
		return 1
	fi
	disconnect_round "plain stream" "$plain_req" "$probe_req" "$limit" || verdict=1
	disconnect_round "tool proposal stream" "$tool_req" "$probe_req" "$limit" || verdict=1
	[[ $verdict -eq 0 ]] && echo "disconnect: PASS" || echo "disconnect: FAIL"
	return $verdict
}

# --- compute admission (streamed permit hold) ------------------------------

# Drive (b) from a LAN host: the permit stays held until the last SSE event, so
# a second chat request during the stream is 503. (a) is the same URL while a
# GUI turn is generating — this mode prints that curl (XBOX_IP / API_PORT).
validate_admission() {
	echo "=== admission: streamed /v1/chat/completions holds the compute permit ==="
	local stream_req probe_req headers out probe_body stream_code probe_code
	local verdict=0 spid
	stream_req=$(printf '{"model":"%s","messages":[{"role":"user","content":"Count slowly from one to one hundred."}],"max_tokens":256,"temperature":0,"seed":42,"stream":true}' "${MODEL}")
	probe_req=$(printf '{"model":"%s","messages":[{"role":"user","content":"ping"}],"max_tokens":8,"temperature":0,"seed":1}' "${MODEL}")
	headers="${TMPDIR_LOCAL}/admission-headers.txt"
	out="${TMPDIR_LOCAL}/admission-stream.txt"
	probe_body="${TMPDIR_LOCAL}/admission-probe.txt"

	: >"${TMPDIR_LOCAL}/admission-stream-code"
	: >"${out}"
	curl -sS -N -m 180 -D "${headers}" -o "${out}" -w "%{http_code}" \
		-H 'Content-Type: application/json' -d "${stream_req}" \
		"${API_URL}/v1/chat/completions" >"${TMPDIR_LOCAL}/admission-stream-code" 2>/dev/null &
	spid=$!
	# Probe only after the first SSE data chunk is on disk: a fixed sleep can
	# fire before the listener has dispatched, or after a short reply ends.
	deadline=$((SECONDS + 30))
	saw_chunk=0
	while ((SECONDS < deadline)); do
		if grep -q '^data: ' "${out}" 2>/dev/null; then
			saw_chunk=1
			break
		fi
		if ! kill -0 "${spid}" 2>/dev/null; then
			break
		fi
		sleep 0.05
	done
	if [[ "${saw_chunk}" -ne 1 ]]; then
		echo "  FAIL: streamed reply produced no SSE data chunk before the probe window"
		wait "${spid}" || true
		echo "admission: FAIL"
		return 1
	fi
	probe_code=$(curl -sS -m 30 -o "${probe_body}" -w "%{http_code}" \
		-H 'Content-Type: application/json' -d "${probe_req}" \
		"${API_URL}/v1/chat/completions" || echo "000")
	wait "${spid}" || true
	stream_code=$(cat "${TMPDIR_LOCAL}/admission-stream-code" 2>/dev/null || echo "000")

	if [[ "${probe_code}" == "503" ]]; then
		echo "  ok: probe during SSE is 503 busy"
	else
		echo "  FAIL: probe during SSE was HTTP ${probe_code} (want 503). Raw: $(head -c 120 "${probe_body}" 2>/dev/null)"
		verdict=1
	fi
	if [[ "${stream_code}" != "200" ]]; then
		echo "  FAIL: streamed reply HTTP ${stream_code} (want 200)"
		verdict=1
	elif ! grep -q '^data: \[DONE\]' "${out}" 2>/dev/null; then
		echo "  FAIL: streamed reply did not end with data: [DONE]"
		verdict=1
	else
		echo "  ok: streamed reply held the slot through data: [DONE]"
	fi

	echo "  GUI-busy probe (run while a pad turn is generating):"
	printf "    curl -sS -o /tmp/xllama-gui-busy.json -w '%%{http_code}\\n' \\\\\n"
	printf "      -H 'Content-Type: application/json' \\\\\n"
	printf "      -d '%s' \\\\\n" "${probe_req}"
	printf "      '%s/v1/chat/completions'\n" "${API_URL}"
	echo "    expect 503 and {\"error\":{\"message\":\"busy\"}}"
	if [[ "${ADMISSION_GUI:-}" == "1" ]]; then
		echo "  ADMISSION_GUI=1: requiring 503 on a single request (GUI must be generating)"
		probe_code=$(curl -sS -m 30 -o "${probe_body}" -w "%{http_code}" \
			-H 'Content-Type: application/json' -d "${probe_req}" \
			"${API_URL}/v1/chat/completions" || echo "000")
		if [[ "${probe_code}" == "503" ]]; then
			echo "  ok: GUI-overlap probe is 503 busy"
		else
			echo "  FAIL: GUI-overlap probe was HTTP ${probe_code} (want 503)"
			verdict=1
		fi
	fi

	[[ $verdict -eq 0 ]] && echo "admission: PASS" || echo "admission: FAIL"
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
    return status, time.time() - t0, data.split(b"\r\n\r\n", 1)[-1].decode(errors="replace")

def samples():
    status, _, body = exchange(b"GET /v1/training/status HTTP/1.1\r\nHost: x\r\n\r\n")
    try:
        # Parse the complete response, not the bounded diagnostic preview.
        count = json.loads(body).get("usable_samples")
        return count if status == "200" and type(count) is int and count >= 0 else None
    except (ValueError, AttributeError):
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
    print(f"  {'ok' if ok else 'FAIL'}: {name} -> {status} in {secs:.1f}s (want {want}) {body[:120]}")
after = samples()
if before is None or after is None:
    print("  FAIL: training status lacks a valid usable_samples counter; side effects unverified")
    fail = 1
elif after != before:
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
	printf '%s' "$1" | curl -sS -m 300 -o "$2" -w "%{http_code}" -H 'Content-Type: application/json' \
		--data-binary @- "${API_URL}/v1/chat/completions" || echo "000"
}

validate_tool_stream() {
	local request="$1" expected="$2" response
	response=$(printf '%s' "$request" |
		curl -sS -m 300 -o "${TMPDIR_LOCAL}/tool-stream.sse" -w "%{http_code} %{content_type}" \
			-H 'Content-Type: application/json' --data-binary @- "${API_URL}/v1/chat/completions") || return
	if [[ "$response" != "200 text/event-stream"* ]]; then
		echo "  FAIL: expected SSE, got ${response}"
		return 1
	fi
	python3 - "${TMPDIR_LOCAL}/tool-stream.sse" "$expected" "${TMPDIR_LOCAL}/streamed-call.json" "${3:-legacy}" <<'PYTOOLSTREAM'
import json
import os
import sys
from pathlib import Path

exec(os.environ["XLLAMA_USAGE_PY"])

body = Path(sys.argv[1]).read_text()
# SSE comments (": keepalive", #581) are not events.
events = [e for e in body.split("\n\n") if not e.startswith(":")]
assert events[-2:] == ["data: [DONE]", ""], events[-2:]
chunks = []
for event in events[:-2]:
    assert event.startswith("data: ") and "\n" not in event, event
    chunks.append(json.loads(event[6:]))
assert len(chunks) >= 3, chunks
assert all(c["object"] == "chat.completion.chunk" for c in chunks), chunks
assert len({c["id"] for c in chunks}) == 1, chunks
mode = sys.argv[4]
if mode == "true":
    summary = chunks.pop()
    assert summary["choices"] == [], summary
    usage = summary["usage"]
    assert all(c["usage"] is None for c in chunks), chunks
elif mode == "false":
    assert all("usage" not in c for c in chunks), chunks
    usage = None
else:
    usage = chunks[-1]["usage"]
assert all(len(c["choices"]) == 1 for c in chunks), chunks
choices = [c["choices"][0] for c in chunks]
assert all(c["index"] == 0 for c in choices), choices
assert choices[0]["delta"]["role"] == "assistant", choices[0]
assert all(c["finish_reason"] is None for c in choices[:-1]), choices
assert choices[-1]["delta"] == {}, choices[-1]
deltas = [c["delta"] for c in choices]
reasoning = "".join(delta.get("reasoning_content") or "" for delta in deltas)
if mode != "false":
    problem = usage_problem(usage, reasoning)
    assert not problem, problem
calls = [call for delta in deltas for call in delta.get("tool_calls", [])]
if sys.argv[2] == "tool_calls":
    assert choices[-1]["finish_reason"] == "tool_calls", choices[-1]
    assert len(calls) == 1 and calls[0]["index"] == 0 and calls[0]["id"], calls
    assert calls[0]["type"] == "function" and calls[0]["function"]["name"] == "get_weather", calls
    assert json.loads(calls[0]["function"]["arguments"])["city"] == "Paris", calls
    call = calls[0].copy()
    del call["index"]
    Path(sys.argv[3]).write_text(json.dumps(call))
else:
    assert choices[-1]["finish_reason"] in ("stop", "length"), choices[-1]
    assert not calls and "".join(delta.get("content") or "" for delta in deltas).strip(), deltas
PYTOOLSTREAM
}

validate_tools() {
	echo "=== tools: client-side tool proposal (ADR 0004) ==="
	local tools_model="${TOOLS_MODEL:-$MODEL}" verdict=0 req code check stream named_request result_request
	local schema='[{"type":"function","function":{"name":"get_weather","description":"Current weather for a city","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}]'

	# A. named tool_choice: the reply must be a get_weather call with JSON arguments.
	req=$(printf '{"model":"%s","messages":[{"role":"user","content":"What is the weather in Paris?"}],"tools":%s,"tool_choice":{"type":"function","function":{"name":"get_weather"}},"max_tokens":128,"temperature":0,"seed":42}' "$tools_model" "$schema")
	named_request="$req"
	code=$(post_chat "$req" "${TMPDIR_LOCAL}/tools-a.json")
	check=$(python3 -c '
import json, os, sys
exec(os.environ["XLLAMA_USAGE_PY"])
d = json.load(open(sys.argv[1]))
c = d["choices"][0]
problem = usage_problem(d.get("usage"), c["message"].get("reasoning_content"))
assert not problem, problem
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
	result_request="$req"
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
		check=$(python3 -c '
import json, sys
e = json.load(open(sys.argv[1]))["error"]
assert "tools require a GGUF chat template" in e["message"], e
assert e.get("code") == "tools_unsupported_for_model", e
assert e.get("param") == "model", e
assert e.get("type") == "xllama_error", e
print("ok")
' "${TMPDIR_LOCAL}/tools-e.json" 2>&1 | tail -n1 || true)
		if [[ "$code" == "400" && "$check" == "ok" ]]; then
			echo "  E ok: ONNX tool proposal refused with 400 tools_unsupported_for_model"
		else
			echo "  E FAIL: HTTP ${code}: ${check:0:160}"
			verdict=1
		fi

		if curl -fsS -m 10 "${API_URL}/v1/models" | python3 -c '
import json, sys
active = [m["id"] for m in json.load(sys.stdin)["data"] if m.get("active")]
sys.exit(0 if active == [sys.argv[1]] else 1)
' "$tools_model"; then
			echo "  E1 ok: rejected ONNX proposal kept the GGUF resident"
		else
			echo "  E1 FAIL: rejected ONNX proposal changed the resident model"
			verdict=1
		fi

		req=$(printf '{"model":"%s","messages":[{"role":"user","content":"Say hello."}],"tools":%s,"tool_choice":"none","max_tokens":32,"temperature":0}' "$ORT_MODEL" "$schema")
		code=$(post_chat "$req" "${TMPDIR_LOCAL}/tools-e-plain.json")
		check=$(python3 -c '
import json, sys
m = json.load(open(sys.argv[1]))["choices"][0]["message"]
print("ok" if m.get("content") and not m.get("tool_calls") else "unexpected " + str(m)[:120])
' "${TMPDIR_LOCAL}/tools-e-plain.json" 2>&1 | tail -n1 || true)
		if [[ "$code" == "200" && "$check" == "ok" ]]; then
			echo "  E2 ok: ONNX with tool_choice none answered as plain chat"
		else
			echo "  E2 FAIL: HTTP ${code}: ${check:0:160}"
			verdict=1
		fi
	else
		echo "  E skipped: set ORT_MODEL to check the ONNX 400"
	fi

	# F. tool definitions alone overflow n_ctx: 400 with the existing prefix and
	#    code context_length_exceeded. The schema is never dropped (ADR 0004).
	req=$(python3 - <<'PYTOOLS'
import json
# Larger than 8192 tokens even if a catalogue n_ctx is 4096 or 8192 (~4 chars/token).
pad = ("synthetic schema padding for the oversized-tools gate. ") * 80
tools = []
for i in range(39):
    tools.append({
        "type": "function",
        "function": {
            "name": f"synth_{i:02d}",
            "description": pad,
            "parameters": {"type": "object", "properties": {"arg": {"type": "string"}},
                           "required": ["arg"]},
        },
    })
print(json.dumps({"model": "__MODEL__",
                  "messages": [{"role": "user", "content": "Say OK."}],
                  "tools": tools,
                  "max_tokens": 24}))
PYTOOLS
	)
	req=${req/__MODEL__/$tools_model}
	for stream in false true; do
		code=$(post_chat "{\"stream\":${stream},${req:1}" "${TMPDIR_LOCAL}/tools-f.json")
		check=$(python3 -c '
import json, sys
d = json.load(open(sys.argv[1]))
e = d.get("error") or {}
msg = e.get("message") or ""
assert "prompt too long: the request needs" in msg, msg[:160]
assert e.get("code") == "context_length_exceeded", e
assert e.get("param") == "tools", e
print("ok")
' "${TMPDIR_LOCAL}/tools-f.json" 2>&1 | tail -n1 || true)
		if [[ "$code" == "400" && "$check" == "ok" ]]; then
			echo "  F ok: oversized tools are 400 context_length_exceeded param tools (stream=${stream})"
		else
			echo "  F FAIL: HTTP ${code} (stream=${stream}): ${check:0:160}"
			verdict=1
		fi
	done

	# The default layout (usage on the finish chunk) stays covered for tool streams.
	if validate_tool_stream "{\"stream\":true,${named_request:1}" tool_calls; then
		echo "  G0 ok: SSE tool proposal keeps usage on the finish chunk by default"
	else
		echo "  G0 FAIL: default SSE tool-proposal layout"
		verdict=1
	fi
	if validate_tool_stream "{\"stream\":true,\"stream_options\":{\"include_usage\":true},${named_request:1}" tool_calls true; then
		echo "  G ok: SSE returned indexed get_weather and tool_calls finish"
		result_request=$(python3 - "$result_request" "${TMPDIR_LOCAL}/streamed-call.json" <<'PYTOOLRESULT'
import json
import sys
from pathlib import Path

request = json.loads(sys.argv[1])
call = json.loads(Path(sys.argv[2]).read_text())
request["messages"][1]["tool_calls"] = [call]
request["messages"][2]["tool_call_id"] = call["id"]
request["stream"] = True
print(json.dumps(request))
PYTOOLRESULT
		)
		if validate_tool_stream "$result_request" text; then
			echo "  H ok: SSE answered the returned tool call ID and ended with DONE"
		else
			echo "  H FAIL: streamed tool result"
			verdict=1
		fi
	else
		echo "  G FAIL: streamed tool call"
		verdict=1
	fi

	[[ $verdict -eq 0 ]] && echo "tools: PASS" || echo "tools: FAIL"
	return $verdict
}

# --- dispatch --------------------------------------------------------------

CMD="${1:-}"
case "$CMD" in
spike) validate_spike ;;
discovery)
	validate_spike || exit 1
	validate_discovery
	;;
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
admission)
	validate_spike || exit 1
	validate_admission
	;;
disconnect)
	validate_spike || exit 1
	validate_disconnect
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
compat)
	validate_spike || exit 1
	compat_args=(--tool-model "${TOOLS_MODEL:-$MODEL}" --ort-model "${ORT_MODEL:-}")
	if [[ "${API_COMPAT_SDKS:-0}" == 1 ]]; then compat_args+=(--sdks); fi
	python3 "${SCRIPT_DIR}/validate-api-compat.py" "$API_URL" "$MODEL" "${compat_args[@]}"
	;;
# Every check below, in order: spike, discovery, chat, stream, admission,
# embed, budget, tools, framing, prefs, train, pull.
all)
	rc=0
	validate_spike || {
		echo "=== summary ==="
		echo "SPIKE FAILED — stopping before chat"
		exit 1
	}
	validate_discovery || rc=1
	validate_chat || rc=1
	validate_stream || rc=1
	validate_admission || rc=1
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
	echo "Usage: $0 <spike|discovery|chat|stream|admission|disconnect|budget|tools|framing|embed|pull|prefs|train|compat|all>" >&2
	exit 1
	;;
esac
