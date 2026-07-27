#!/usr/bin/env bash
# validate-console.sh — drive the xllama autopilot to validate the live XAML UI
# on Xbox with deterministic PASS/FAIL verdicts, no human at the pad.
#
# Usage:
#   source ~/.config/xllama/xbox-env
#   ./scripts/validate-console.sh <routing|settings|gguf|longchat|kvsnap|taesd|all>
#
# Requires: an installed xllama build with the autopilot (>= 1.1.3.0; the
# settings ops need >= 1.4.0.606), the relevant models already in LocalState.
# The routing gate needs BOTH smollm2-360m-cpu-int4 and smollm2-360m-dml-fp16-v2
# — #91 was root-caused and GPU text routing is live again for the -v2
# parity-validated asset, so the old "dml model not needed" note no longer holds.
# settings needs smollm2-360m-cpu-int4; gguf needs lfm25-350m; taesd needs
# sd-turbo-fp16. Plus XBOX_IP/USER/PASS env. Seed with scripts/provision-models.sh.
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

# Single source of truth for the routing threshold: read it from the header that
# decide_routing() actually uses. Hardcoding it here is what broke this gate when
# the constant was retuned 600 -> 1550 in #129.
ROUTING_THRESHOLD=$(sed -n 's/.*int token_threshold = \([0-9]\+\);.*/\1/p' \
	"${REPO_ROOT}/include/xllama/routing_policy.h" | head -n1)
if ! [[ "$ROUTING_THRESHOLD" =~ ^[0-9]+$ ]]; then
	echo "Error: could not read token_threshold from include/xllama/routing_policy.h" >&2
	exit 1
fi
# The decoy must clear the routing threshold in REAL tokens while staying under
# the context trimmer's budget in ESTIMATED tokens — BuildPrompt runs first and
# drops any turn over budget, which is #133. Both constants come from the same
# header so the decoy tracks a retune of either.
TRIM_BUDGET=$(sed -n 's/.*kMaxPromptTokens = \([0-9]\+\);.*/\1/p' \
	"${REPO_ROOT}/include/xllama/routing_policy.h" | head -n1)
EST_CHARS_PER_TOK=$(sed -n 's/.*kEstimatedCharsPerToken = \([0-9.]\+\);.*/\1/p' \
	"${REPO_ROOT}/include/xllama/routing_policy.h" | head -n1)
if ! [[ "$TRIM_BUDGET" =~ ^[0-9]+$ ]] || [[ -z "$EST_CHARS_PER_TOK" ]]; then
	echo "Error: could not read kMaxPromptTokens / kEstimatedCharsPerToken from" \
		"include/xllama/routing_policy.h" >&2
	exit 1
fi

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

# Remove an injected conversation: the file AND its index entry. Deleting only
# the file leaves a phantom row in the console's History list that fails to
# open — a gate must not leave litter on the device it validated.
remove_chat() {
	local cid="$1"
	delete_file "${cid}.json" "chats"
	fetch_file "index.json" "${TMPDIR_LOCAL}/index-after.json" "chats"
	python3 - "$TMPDIR_LOCAL" "$cid" <<'PY'
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
	# #91 lifted for the rmsfix asset (dml_text_model_ok, routing_policy.h):
	# routing=auto must send the >ROUTING_THRESHOLD-token turn to the GPU model and the short
	# turns to CPU. Both models are preconditions now.
	local m
	for m in smollm2-360m-cpu-int4 smollm2-360m-dml-fp16-v2; do
		if ! model_provisioned "$m"; then
			echo "  FAIL: $m is not in LocalState\\models\\"
			echo "  Seed it first:  ./scripts/provision-models.sh $m"
			echo "§2 routing: FAIL"
			return 1
		fi
	done
	# Routing=auto is emitted only when settings.json has "routing": 2; seed it
	# (the whole point is that no human could set it via the dialog). Schema
	# mirrors SaveSettings (MainPage.cpp).
	cat >"${TMPDIR_LOCAL}/settings.json" <<'JSON'
{
  "system_prompt": "You are a helpful AI assistant.",
  "model": "smollm2-360m-cpu-int4",
  "kv_reuse": true,
  "routing": 2,
  "gpu_model": "smollm2-360m-dml-fp16-v2",
  "diffuse_taesd_vae": false,
  "sampling": {"temperature": 0.8, "top_p": 0.9, "top_k": 40, "repetition_penalty": 1.1, "n_predict": 256}
}
JSON
	upload_file "${TMPDIR_LOCAL}/settings.json"

	# Build the long decoy conversation from long-1k.txt, stripping the ChatML
	# wrapper so the stored user content is plain prose. Merge it into the
	# device's existing chat index rather than clobbering it.
	#
	# The decoy is squeezed between two constants pulling opposite ways:
	#   - it must exceed ROUTING_THRESHOLD in REAL tokens, or routing stays on CPU;
	#   - it must stay under TRIM_BUDGET in ESTIMATED tokens, or BuildPrompt drops
	#     the whole turn before routing ever counts it (#133 — and because a single
	#     oversized turn is dropped entirely, the prompt collapses to ~22 tokens).
	# Size it just under the trimmer's ceiling and let the assertion below check,
	# against the console's own tokenizer, that it landed above the threshold.
	# This gate broke once already (threshold retuned 600 -> 1550 in #129 while the
	# decoy stayed fixed), so every number here derives from the header.
	local esca_id="ap-routing-longctx"
	fetch_file "index.json" "${TMPDIR_LOCAL}/existing-index.json" "chats"
	python3 - "$REPO_ROOT" "$TMPDIR_LOCAL" "$esca_id" "$ROUTING_THRESHOLD" \
		"$TRIM_BUDGET" "$EST_CHARS_PER_TOK" <<'PY'
import json, re, sys
repo, tmp, cid = sys.argv[1], sys.argv[2], sys.argv[3]
threshold, trim_budget, est_cpt = int(sys.argv[4]), int(sys.argv[5]), float(sys.argv[6])
body = open(f"{repo}/bench/prompts/long-1k.txt").read()
m = re.search(r'<\|im_start\|>user\n(.*?)(?:<\|im_end\|>|$)', body, re.S)
user = (m.group(1) if m else body).strip()
# Real chars-per-token measured on console with the SmolLM2 tokenizer: 7100
# chars -> 1329 tokens.
REAL_CPT = 5.34
# The reachable band in REAL tokens is bounded below by the routing threshold and
# above by whatever survives the trimmer. Aim at its MIDPOINT rather than near
# either edge — with the shipping constants the band is only ~135 tokens wide, so
# there is no comfortable side to hug. That narrowness is itself a finding: see
# #133 and the pending threshold re-derivation in #130.
lo = threshold
hi = trim_budget * est_cpt / REAL_CPT
target_chars = int((lo + hi) / 2 * REAL_CPT)
words = user.split()
while len(" ".join(words)) < target_chars:
    words = words * 2
user = " ".join(words)[:target_chars]
est = int(len(user) / est_cpt)
if est >= trim_budget:
    sys.exit(f"  decoy estimate {est} >= trimmer budget {trim_budget} — would be dropped")
if int(len(user) / 5.34) <= threshold:
    sys.exit(f"  decoy cannot clear threshold {threshold} without exceeding the "
             f"trimmer budget {trim_budget} — see #133")
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
print(f"  decoy: {len(user)} chars, trimmer estimate {est} tok (budget {trim_budget}), "
      f"~{int(len(user)/5.34)} real tok (threshold {threshold}); index entries: {len(idx)}")
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
	#
	# #91 lifted for the rmsfix -v2 asset (dml_text_model_ok): the pre-#91
	# assertions are back — the long decoy turn (>ROUTING_THRESHOLD tok) must route auto→gpu
	# and the fresh short turn auto→cpu.
	local gpu_line cpu_line
	gpu_line=$(grep -aE 'routing: auto .* gpu \([0-9]+ tok' "$log" | head -1 || true)
	cpu_line=$(grep -aE 'routing: auto .* cpu \([0-9]+ tok' "$log" | head -1 || true)
	if [[ -z "$gpu_line" ]]; then
		echo "  FAIL: no 'auto -> gpu' routing line — long turn did not route to the DML model"
		# Diagnose the #133 failure mode specifically: if the decoy turn was
		# trimmed, the CPU line reports a token count far below the decoy's size
		# and there is a "context trimmed" line. Without this, the symptom
		# ("no gpu line") points at routing when the cause is the trimmer.
		if grep -aq 'context trimmed' "$log"; then
			echo "  cause: BuildPrompt trimmed the decoy turn before routing saw it (#133)"
			grep -a 'context trimmed' "$log" | tail -1 | sed 's/^/    /'
		elif [[ -n "$cpu_line" ]]; then
			echo "  the long turn was counted as: ${cpu_line}"
		fi
		verdict=1
	else
		local gtok
		gtok=$(sed -n 's/.*(\([0-9]\+\) tok.*/\1/p' <<<"$gpu_line")
		if ((gtok <= ROUTING_THRESHOLD)); then
			echo "  FAIL: GPU-routed turn at ${gtok} tok (threshold is ${ROUTING_THRESHOLD}): ${gpu_line}"
			verdict=1
		else
			echo "  ok: long turn routed to GPU (${gtok} tok)"
		fi
	fi
	if [[ -z "$cpu_line" ]]; then
		echo "  FAIL: no 'auto -> cpu' routing line (short turns never ran on CPU)"
		verdict=1
	else
		local ntok
		ntok=$(sed -n 's/.*(\([0-9]\+\) tok.*/\1/p' <<<"$cpu_line")
		if ((ntok > ROUTING_THRESHOLD)); then
			echo "  FAIL: CPU-routed turn above threshold (${ntok} tok): ${cpu_line}"
			verdict=1
		else
			echo "  ok: short turn routed to CPU (${ntok} tok)"
		fi
	fi
	if grep -aq '887A0036' "$log"; then
		echo "  FAIL: 887A0036 present — patched GenAI DLL did not take"
		verdict=1
	else
		echo "  ok: no 887A0036 (patched DLL works in XAML)"
	fi
	# Remove the decoy chat so "Understood; ready to continue." does not linger in
	# the History list after a validation run.
	remove_chat "${esca_id}"
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

# --- #169 long-chat context shift ------------------------------------------

validate_longchat() {
	echo "=== #169 long-chat context shift ==="
	# Inject a chat whose history exceeds kMaxPromptTokens (in ESTIMATED
	# tokens; each single turn stays under budget or it would be dropped
	# whole, #133), then run turns with KV reuse on the GGUF model. The #169
	# contract: trimmed rounds stay in the reuse regime (prefill = delta, not
	# ~1800 tokens), and when the resident KV overflows n_ctx the session
	# front-drop-evicts and continues — no continuation may fall back to a
	# full re-prefill and no "context full" error may surface.
	local cid="ap-169-longchat"
	fetch_file "index.json" "${TMPDIR_LOCAL}/existing-index.json" "chats"
	python3 - "$TMPDIR_LOCAL" "$cid" "$TRIM_BUDGET" "$EST_CHARS_PER_TOK" <<'PY'
import json, sys
tmp, cid = sys.argv[1], sys.argv[2]
budget, est_cpt = int(sys.argv[3]), float(sys.argv[4])
base = ("Please keep track of item %02d: the quick brown fox jumps over the lazy "
        "dog while the committee reviews the quarterly figures and the harbour "
        "master files a detailed report about tide tables, cargo manifests and "
        "the maintenance schedule of every crane on pier seven. ")
turns, ts = [], 1
for i in range(12):
    turns.append({"role": "user", "content": (base % i) * 3, "ts": ts, "partial": False})
    turns.append({"role": "assistant", "content": f"Noted item {i:02d}; tracking it.",
                  "ts": ts + 1, "partial": False})
    ts += 2
nchars = sum(len(m["content"]) for m in turns)
est = int(nchars / est_cpt)
if est <= budget:
    sys.exit(f"  history estimate {est} tok <= trimmer budget {budget} — no trim, no test")
conv = {"id": cid, "title": "#169 long-chat shift", "messages": turns}
open(f"{tmp}/{cid}.json", "w").write(json.dumps(conv, ensure_ascii=False))
entry = {"id": cid, "title": conv["title"], "last_modified": ts, "n_messages": len(turns)}
try:
    idx = json.load(open(f"{tmp}/existing-index.json"))
    if not isinstance(idx, list):
        idx = []
except Exception:
    idx = []
idx = [e for e in idx if e.get("id") != cid]
idx.insert(0, entry)
open(f"{tmp}/index.json", "w").write(json.dumps(idx, ensure_ascii=False))
print(f"  injected: {len(turns)} messages, est ~{est} tok (budget {budget})")
PY
	upload_file "${TMPDIR_LOCAL}/${cid}.json" "chats"
	upload_file "${TMPDIR_LOCAL}/index.json" "chats"

	# Long user turns guarantee KV growth even when generation stops early at
	# EOG (~85 real tokens of delta per send): the ~1800-token seed plus a few
	# of these must overflow n_ctx 2048 and fire the shift.
	local longq="Please compare the tide tables, the cargo manifests and the crane maintenance schedule across the items you are tracking, and explain in detail which of them would be affected if pier seven were closed for repairs during the spring inspection window, considering both the quarterly figures the committee reviewed and the harbour master's detailed report about scheduling conflicts."
	local marker
	marker=$(
		run_autopilot 900 <<JSON
{"total_timeout_s": 800, "actions": [
  {"op": "set_model", "name": "lfm25-350m", "timeout_s": 300},
  {"op": "set_kv_reuse", "enabled": true},
  {"op": "set_sampling", "n_predict": 256, "temperature": 0.7},
  {"op": "load_chat", "id": "${cid}"},
  {"op": "send", "text": "Summarize item 03 in one sentence.", "timeout_s": 240},
  {"op": "send", "text": "${longq}", "timeout_s": 240},
  {"op": "send", "text": "${longq} Focus on item 05 this time.", "timeout_s": 240},
  {"op": "send", "text": "${longq} Focus on item 09 this time.", "timeout_s": 240},
  {"op": "send", "text": "${longq} Focus on item 11 this time.", "timeout_s": 240},
  {"op": "send", "text": "Which item mentioned pier seven?", "timeout_s": 240},
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
	if grep -aq 'context shift — evicted' "$log"; then
		echo "  ok: context shift fired ($(grep -ac 'context shift — evicted' "$log")x)"
	else
		echo "  FAIL: no context-shift line in the log"
		verdict=1
	fi
	if grep -aq 'retrying with full prefill' "$log"; then
		echo "  FAIL: a continuation fell back to a full prefill"
		verdict=1
	else
		echo "  ok: no continuation fell back to a full prefill"
	fi
	if grep -aq 'context full' "$log"; then
		echo "  FAIL: context-full error in the log"
		verdict=1
	else
		echo "  ok: no context-full error"
	fi
	remove_chat "${cid}"
	[[ $verdict -eq 0 ]] && echo "#169 long-chat: PASS" || echo "#169 long-chat: FAIL"
	return $verdict
}

# --- #170b KV snapshots across a conversation switch -----------------------

validate_kvsnap() {
	echo "=== #170b KV snapshot across a conversation switch ==="
	# Leave a conversation, come back, and the history must NOT be re-read:
	# the snapshot written on the way out is restored on the way back, and the
	# #170a prefix diff turns it into a delta prefill. The measurable claim is
	# the prompt-token count of the returning turn against the cold one.
	local cid="ap-170b-switch"
	fetch_file "index.json" "${TMPDIR_LOCAL}/existing-index.json" "chats"
	python3 - "$TMPDIR_LOCAL" "$cid" "$TRIM_BUDGET" "$EST_CHARS_PER_TOK" <<'PY'
import json, sys
tmp, cid = sys.argv[1], sys.argv[2]
budget, est_cpt = int(sys.argv[3]), float(sys.argv[4])
base = ("Item %02d: the harbour master logged tide tables, cargo manifests and the "
        "crane maintenance schedule for pier seven. ")
turns, ts = [], 1
for i in range(8):
    turns.append({"role": "user", "content": (base % i) * 2, "ts": ts, "partial": False})
    turns.append({"role": "assistant", "content": f"Logged item {i:02d}.",
                  "ts": ts + 1, "partial": False})
    ts += 2
nchars = sum(len(m["content"]) for m in turns)
est = int(nchars / est_cpt)
# Must stay UNDER the trimmer budget: a trimmed round would muddy the signal
# with #169's shift, and this gate is about the snapshot alone.
if est >= budget:
    sys.exit(f"  history estimate {est} tok >= trimmer budget {budget} — would trim")
conv = {"id": cid, "title": "#170b switch", "messages": turns}
open(f"{tmp}/{cid}.json", "w").write(json.dumps(conv, ensure_ascii=False))
entry = {"id": cid, "title": conv["title"], "last_modified": ts, "n_messages": len(turns)}
try:
    idx = json.load(open(f"{tmp}/existing-index.json"))
    if not isinstance(idx, list):
        idx = []
except Exception:
    idx = []
idx = [e for e in idx if e.get("id") != cid]
idx.insert(0, entry)
open(f"{tmp}/index.json", "w").write(json.dumps(idx, ensure_ascii=False))
print(f"  injected: {len(turns)} messages, est ~{est} tok (budget {budget})")
PY
	upload_file "${TMPDIR_LOCAL}/${cid}.json" "chats"
	upload_file "${TMPDIR_LOCAL}/index.json" "chats"

	local marker
	marker=$(
		run_autopilot 900 <<JSON
{"total_timeout_s": 800, "actions": [
  {"op": "set_model", "name": "lfm25-350m", "timeout_s": 300},
  {"op": "set_kv_reuse", "enabled": true},
  {"op": "set_sampling", "n_predict": 24, "temperature": 0.7},
  {"op": "load_chat", "id": "${cid}"},
  {"op": "send", "text": "Summarize item 02 in one sentence.", "timeout_s": 240},
  {"op": "new_chat"},
  {"op": "send", "text": "Say hello.", "timeout_s": 180},
  {"op": "load_chat", "id": "${cid}"},
  {"op": "send", "text": "Now summarize item 05 in one sentence.", "timeout_s": 240},
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
	if grep -aq 'KV state saved' "$log"; then
		echo "  ok: snapshot written on the way out"
	else
		echo "  FAIL: no snapshot was written"
		verdict=1
	fi
	if grep -aq 'KV snapshot restored' "$log"; then
		echo "  ok: snapshot restored on the way back"
	else
		echo "  FAIL: snapshot not restored"
		verdict=1
	fi
	# The point of the feature: the returning turn prefills a delta, not the
	# history it prefilled cold.
	python3 - "$log" <<'PY' || verdict=1
import re, sys
toks = [int(m) for m in re.findall(r"session generate: .*?\((\d+) tok\)",
                                   open(sys.argv[1], errors="ignore").read())]
if len(toks) < 3:
    sys.exit(f"  FAIL: expected 3 prefills in the log, saw {len(toks)}: {toks}")
cold, ret = toks[0], toks[2]
print(f"  cold prefill {cold} tok -> returning prefill {ret} tok")
if ret >= cold / 4:
    sys.exit(f"  FAIL: returning turn re-read the history ({ret} vs {cold} tok)")
print(f"  ok: returning turn pays a delta ({ret} tok, {100.0 * ret / cold:.0f}% of cold)")
PY
	remove_chat "${cid}"
	[[ $verdict -eq 0 ]] && echo "#170b KV snapshot: PASS" || echo "#170b KV snapshot: FAIL"
	return $verdict
}

# --- §7c TAESD -------------------------------------------------------------

validate_taesd() {
	echo "=== §7c TAESD image ==="
	# Full diffusion package required (te + unet + tokenizer). After a clean
	# MSIX install, LocalState is empty — a prior failed taesd run can leave
	# only vae_decoder/ and fail with "text_encoder/model.onnx File doesn't exist".
	if ! model_provisioned "sd-turbo-fp16"; then
		# model_provisioned only checks genai_config; for diffusion probe te.
		local te_code
		te_code=$(curl "${CURL_AUTH[@]}" -o /dev/null -w "%{http_code}" \
			"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=%5CLocalState%5Cmodels%5Csd-turbo-fp16%5Ctext_encoder&filename=model.onnx" 2>/dev/null || echo "000")
		if [[ "$te_code" != "200" ]]; then
			echo "  FAIL: sd-turbo-fp16 incomplete (no text_encoder/model.onnx)"
			echo "  Seed:  ./scripts/provision-models.sh sd-turbo-fp16"
			echo "§7c TAESD: FAIL"
			return 1
		fi
	fi
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

# --- settings ops (routing / sampling / KV-reuse) --------------------------

# The Settings dialog is unreachable in Dev Mode (no text input, #117), so the
# only way to prove the three preference writers work end-to-end is to drive
# them through the autopilot and read back the persisted settings.json.
#
# Deliberately separate from §2 routing: that gate seeds settings.json as a file
# upload and asserts on routing *behaviour*; this one asserts that the in-app
# ops actually mutate and persist state.
validate_settings() {
	echo "=== settings ops (routing / sampling / kv_reuse / taesd / system_prompt) ==="
	if ! model_provisioned "smollm2-360m-cpu-int4"; then
		echo "  FAIL: smollm2-360m-cpu-int4 is not in LocalState\\models\\"
		echo "  Seed it first:  ./scripts/provision-models.sh smollm2-360m-cpu-int4"
		echo "settings ops: FAIL"
		return 1
	fi
	# Seed a baseline that differs from every target value below, so each op has
	# to do real work — a no-op would leave the baseline behind and fail.
	cat >"${TMPDIR_LOCAL}/settings.json" <<'JSON'
{
  "system_prompt": "You are a helpful AI assistant.",
  "model": "smollm2-360m-cpu-int4",
  "kv_reuse": true,
  "routing": 2,
  "gpu_model": "smollm2-360m-dml-fp16-v2",
  "diffuse_taesd_vae": false,
  "sampling": {"temperature": 0.8, "top_p": 0.9, "top_k": 40, "repetition_penalty": 1.1, "n_predict": 256}
}
JSON
	upload_file "${TMPDIR_LOCAL}/settings.json"

	local marker
	marker=$(
		run_autopilot 300 <<'JSON'
{"total_timeout_s": 120, "actions": [
  {"op": "set_routing", "routing": 0},
  {"op": "set_sampling", "temperature": 0.55, "top_p": 0.8, "top_k": 20, "repetition_penalty": 1.25, "n_predict": 128},
  {"op": "set_kv_reuse", "enabled": false},
  {"op": "set_taesd", "enabled": true},
  {"op": "set_system_prompt", "text": "You are a terse assistant."},
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
	local op
	for op in set_routing set_sampling set_kv_reuse set_taesd set_system_prompt; do
		if grep -aq "\[autopilot\] action .* ${op} end" "$log"; then
			echo "  ok: ${op} dispatched"
		else
			echo "  FAIL: no '${op} end' line in the log"
			verdict=1
		fi
	done

	# Read back the persisted state — the ops call SaveSettings(), so every
	# target value must be on disk after the app has quit.
	fetch_file "settings.json" "${TMPDIR_LOCAL}/settings-after.json"
	if ! python3 - "${TMPDIR_LOCAL}/settings-after.json" <<'PY'; then
import json, sys
try:
    s = json.load(open(sys.argv[1]))
except Exception as e:
    print(f"  FAIL: settings.json unreadable after the run: {e}")
    sys.exit(1)
smp = s.get("sampling", {})
# kind: "exact" for bool/int (identity-strict for bools), "near" for floats
# (settings.json is written with %.2f, so compare with a tolerance).
want = [
    ("routing", s.get("routing"), 0, "exact"),
    ("kv_reuse", s.get("kv_reuse"), False, "exact"),
    ("temperature", smp.get("temperature"), 0.55, "near"),
    ("top_p", smp.get("top_p"), 0.8, "near"),
    ("top_k", smp.get("top_k"), 20, "exact"),
    ("repetition_penalty", smp.get("repetition_penalty"), 1.25, "near"),
    ("n_predict", smp.get("n_predict"), 128, "exact"),
    # Seeded false / "You are a helpful AI assistant." above, so a no-op op fails.
    ("diffuse_taesd_vae", s.get("diffuse_taesd_vae"), True, "exact"),
    ("system_prompt", s.get("system_prompt"), "You are a terse assistant.", "exact"),
]
rc = 0
for name, got, exp, kind in want:
    if kind == "exact":
        ok = type(got) is type(exp) and got == exp
    else:
        ok = isinstance(got, (int, float)) and not isinstance(got, bool) \
            and abs(got - exp) < 1e-4
    print(f"  {'ok' if ok else 'FAIL'}: {name} = {got!r} (want {exp!r})")
    if not ok:
        rc = 1
sys.exit(rc)
PY
		verdict=1
	fi

	[[ $verdict -eq 0 ]] && echo "settings ops: PASS" || echo "settings ops: FAIL"
	return $verdict
}

# --- dispatch --------------------------------------------------------------

CMD="${1:-}"
case "$CMD" in
routing) validate_routing ;;
settings) validate_settings ;;
gguf) validate_gguf ;;
longchat) validate_longchat ;;
kvsnap) validate_kvsnap ;;
taesd) validate_taesd ;;
all)
	rc=0
	if ! model_provisioned "smollm2-360m-cpu-int4"; then
		echo "  WARN: smollm2-360m-cpu-int4 missing — launch the app once for catalogue download"
	fi
	validate_routing || rc=1
	validate_settings || rc=1
	validate_gguf || rc=1
	validate_longchat || rc=1
	validate_kvsnap || rc=1
	validate_taesd || rc=1
	echo
	echo "=== summary ==="
	[[ $rc -eq 0 ]] && echo "ALL PASS" || echo "SOME FAILED (exit ${rc})"
	exit $rc
	;;
*)
	echo "Usage: $0 <routing|settings|gguf|longchat|kvsnap|taesd|all>" >&2
	exit 1
	;;
esac
