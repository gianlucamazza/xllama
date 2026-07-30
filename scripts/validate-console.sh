#!/usr/bin/env bash
# validate-console.sh — drive the xllama autopilot to validate the live XAML UI
# on Xbox with deterministic PASS/FAIL verdicts, no human at the pad.
#
# Usage:
#   source ~/.config/xllama/xbox-env
#   ./scripts/validate-console.sh <routing|settings|gguf|longchat|kvsnap|coderpaste|thinkcut|genroom|taesd|all>
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
# kMaxPromptTokens is the estimate ceiling the app actually trims by, and it does
# not depend on n_predict (the reply's room is enforced later and exactly, by
# fit_prompt). trim_ceiling stays as the one place a gate asks for that number, so
# a future retune cannot leave a payload sized against a constant the app no
# longer uses — the #133 failure mode.
DEFAULT_N_CTX=$(sed -n 's/.*kDefaultNCtx = \([0-9]\+\);.*/\1/p' \
	"${REPO_ROOT}/include/xllama/inference_params.h" | head -n1)
RESERVED_GEN=$(sed -n 's/.*kReservedGenerationTokens = \([0-9]\+\);.*/\1/p' \
	"${REPO_ROOT}/include/xllama/routing_policy.h" | head -n1)
if ! [[ "$DEFAULT_N_CTX" =~ ^[0-9]+$ ]] || ! [[ "$RESERVED_GEN" =~ ^[0-9]+$ ]]; then
	echo "Error: could not read kDefaultNCtx / kReservedGenerationTokens" >&2
	exit 1
fi
# trim_ceiling <n_predict> — the estimated-token budget a gate's payload must fit
# under on the shipping context, for the n_predict that gate sets.
# trim_ceiling <n_ctx> — the estimated-token ceiling a gate's payload must fit
# under. n_predict is deliberately NOT an input: the reply's room is enforced
# later and exactly, by fit_prompt.
trim_ceiling() {
	local n_ctx="$1" budget
	if ((n_ctx == DEFAULT_N_CTX)); then
		echo "$TRIM_BUDGET"
		return
	fi
	budget=$((n_ctx - RESERVED_GEN))
	((budget < 256)) && budget=256
	echo "$budget"
}

TMPDIR_LOCAL=$(mktemp -d)
VAE_CACHE="" # set by validate_taesd; read by its EXIT trap (must be global)
# Byte-for-byte copy of xllama.log as it was BEFORE the current autopilot run.
# WDP answers 200 to a DELETE of a file the app holds open and then does not
# unlink it, so "clear the log" is not something a gate can rely on: fetch_log
# slices off whatever this snapshot already contained instead.
LOG_BEFORE="${TMPDIR_LOCAL}/xllama-before.log"
: >"$LOG_BEFORE"
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

stop_app() {
	curl "${CURL_AUTH[@]}" -H "X-CSRF-Token:${CSRF_TOKEN}" -X DELETE \
		"${BASE_URL}/api/taskmanager/app?package=${PFN}" >/dev/null 2>&1 || true
	sleep 2
}

start_app() {
	local pfamily aumid
	# shellcheck disable=SC2001
	pfamily=$(echo "$PFN" | sed 's/_[0-9][0-9.]*_[^_]*__/_/')
	aumid=$(printf '%s!xllama' "$pfamily" | base64 -w0)
	curl "${CURL_AUTH[@]}" -H "X-CSRF-Token:${CSRF_TOKEN}" -X POST -d "" \
		"${BASE_URL}/api/taskmanager/app?appid=${aumid}" >/dev/null 2>&1 || true
}

restart_app() {
	stop_app
	start_app
}

# True when <name> is absent from LocalState (HTTP 404).
file_absent() {
	local remote_name="$1" code
	code=$(curl "${CURL_AUTH[@]}" -o /dev/null -w "%{http_code}" \
		"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=%5CLocalState&filename=${remote_name}" 2>/dev/null || echo "000")
	[[ "$code" == "404" ]]
}

# --- failure screenshots ---------------------------------------------------
#
# Every verdict in this script comes from grepping xllama.log, and once that was
# not enough: the app died at launch with no log, no crash dump and no WER
# report, and only a Device Portal screenshot showed why — a "Sign in to start
# this app (0x8004090a)" dialog (docs/dml-metacommands-runbook.md). The
# capability was already there; the gate simply did not use it.
#
# The poll loop keeps the last two frames; they are written out only when a gate
# fails. Two frames rather than one because of how the three failure classes
# differ:
#   * autopilot "error:" — ApRun writes the marker WITHOUT exiting
#     (MainPage.cpp), so the app is still on screen showing the broken state;
#   * timeout — the app is alive, most likely stuck;
#   * marker "ok" but the log grep rejects — every gate script ends with `quit`
#     and that path does exit, so only the earlier frame still shows the app.
#
# Which gates take frames DURING the run, and why it is a list rather than a
# rate. Exactly one gate asserts a duration — taesd, on a VAE decode under
# 1000 ms — and a screenshot is GPU work on the same SoC. Sampling less often
# everywhere would not remove that collision, only make it rarer while also
# halving the evidence for the eight gates that time nothing. So: every poll for
# the gates with no timing assertion, and none mid-run for taesd.
#
# What taesd gives up is nothing it needed. Its two failures are an image
# generation error, which leaves the app on screen and takes the autopilot
# "error:" path where the end-of-run frame is the right one anyway; and the
# vae_ms assertion, whose evidence is a number already in the log that no
# screenshot improves.
#
# XLLAMA_GATE_SHOTS=0 disables capture entirely.
SHOTS_DIR="${XLLAMA_GATE_SHOTS_DIR:-${TMPDIR:-/tmp}/xllama-gate-shots}"
GATES_NO_MIDRUN_SHOTS=" taesd "
RING_MIDRUN=1

grab_screenshot() {
	local dest="$1"
	curl "${CURL_AUTH[@]}" -o "$dest" --fail "${BASE_URL}/ext/screenshot" >/dev/null 2>&1 || return 1
}

_ring_grab() {
	[[ -f "${TMPDIR_LOCAL}/ring-1.png" ]] &&
		mv -f "${TMPDIR_LOCAL}/ring-1.png" "${TMPDIR_LOCAL}/ring-0.png"
	grab_screenshot "${TMPDIR_LOCAL}/ring-1.png" || rm -f "${TMPDIR_LOCAL}/ring-1.png"
	return 0
}

# Called from the poll loop; keeps ring-0.png (older) and ring-1.png (newer).
ring_tick() {
	[[ "${XLLAMA_GATE_SHOTS:-1}" == "0" ]] && return 0
	((RING_MIDRUN == 1)) || return 0
	_ring_grab
}

ring_reset() { rm -f "${TMPDIR_LOCAL}/ring-0.png" "${TMPDIR_LOCAL}/ring-1.png"; }

# The frame for the moment a run ends. Taken for every gate, including the ones
# that skip mid-run frames — by then there is nothing left to perturb.
ring_tick_now() {
	[[ "${XLLAMA_GATE_SHOTS:-1}" == "0" ]] && return 0
	_ring_grab
}

# Persist whatever the ring holds. Called only on a failing gate.
save_fail_shots() {
	local gate="$1" saved=0 i
	[[ "${XLLAMA_GATE_SHOTS:-1}" == "0" ]] && return 0
	for i in 0 1; do
		[[ -f "${TMPDIR_LOCAL}/ring-${i}.png" ]] || continue
		mkdir -p "$SHOTS_DIR"
		cp "${TMPDIR_LOCAL}/ring-${i}.png" "${SHOTS_DIR}/${gate}-${i}.png"
		saved=$((saved + 1))
	done
	if ((saved > 0)); then
		echo "  Screenshots of the failing run: ${SHOTS_DIR}/${gate}-*.png" \
			"(${gate}-1.png is the later frame)" >&2
	else
		# Not a cadence problem: every run ends with a forced grab, so zero frames
		# means GET /ext/screenshot itself failed — console unreachable, or WDP
		# refusing. Say that, rather than blaming the sampling interval.
		echo "  No screenshot captured for ${gate}: GET /ext/screenshot failed" >&2
	fi
}

# Poll autopilot-done.txt; echoes its content, returns non-zero on timeout.
wait_autopilot_done() {
	local timeout_s="${1:-900}" out="${TMPDIR_LOCAL}/done.txt"
	# Wall-clock deadline rather than a tick count: a poll iteration now also
	# grabs a screenshot, so "10 per iteration" would understate real elapsed
	# time and silently stretch the declared timeout.
	local start=$SECONDS deadline=$((SECONDS + timeout_s))
	echo "  Waiting for autopilot-done.txt (timeout ${timeout_s}s)..." >&2
	while ((SECONDS < deadline)); do
		: >"$out"
		fetch_file "autopilot-done.txt" "$out"
		# A real marker is "ok" or "error: ..."; a 404 body contains neither.
		if grep -qE '^(ok|error:)' "$out" 2>/dev/null; then
			echo "  Done after $((SECONDS - start))s." >&2
			# One last frame: on the "error:" path the app is still on screen
			# showing exactly what broke, and that is the frame worth having.
			ring_tick_now
			cat "$out"
			return 0
		fi
		sleep 10
		ring_tick
	done
	echo "  Timeout waiting for autopilot-done.txt" >&2
	ring_tick_now
	return 1
}

# Upload autopilot.json (from stdin) + a fresh autopilot.flag, clear stale
# markers, restart, and wait. Echoes the done-marker content.
run_autopilot() {
	local timeout_s="${1:-900}"
	cat >"${TMPDIR_LOCAL}/autopilot.json"
	printf 'go' >"${TMPDIR_LOCAL}/autopilot.flag"
	# Frames belong to one run: a gate that fails must not be handed the previous
	# gate's screenshot, which would look like evidence and be a different app state.
	ring_reset
	# Stop the app BEFORE deleting: WDP cannot unlink a file the app still holds
	# open, and delete_file swallows the failure — a previous run's lines then
	# survive into this run's log and falsify the verdict (observed: a "prompt
	# too long" from the prior gate invocation counted against the next one).
	stop_app
	delete_file "autopilot-done.txt"
	delete_file "diffuse-progress.txt"
	# xllama.log is append-only across restarts; clear it so the verdict greps
	# only this run (a stale 887A0036 or routing line would falsify the gate).
	# The app reopens the log lazily on next launch.
	delete_file "xllama.log"
	# Best effort only: when the unlink does not take, snapshot what is there so
	# fetch_log can subtract it. Verdicts must never depend on a DELETE landing.
	# The snapshot is only trusted on a real 200 — fetch_file happily writes a 404
	# body, which would break the prefix match in fetch_log and quietly hand the
	# whole file (previous run included) to the verdict.
	: >"$LOG_BEFORE"
	if ! file_absent "xllama.log"; then
		local snap_code
		snap_code=$(curl "${CURL_AUTH[@]}" -o "$LOG_BEFORE" -w "%{http_code}" \
			"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=%5CLocalState&filename=xllama.log" 2>/dev/null || echo "000")
		[[ "$snap_code" == "200" ]] || : >"$LOG_BEFORE"
	fi
	upload_file "${TMPDIR_LOCAL}/autopilot.json"
	upload_file "${TMPDIR_LOCAL}/autopilot.flag"
	start_app
	wait_autopilot_done "$timeout_s"
}

# The log this run appended, with the pre-run bytes removed. Every verdict greps
# THIS, so a stale line from an earlier run cannot pass or fail a gate — the
# failure mode that made #193's own gate report a rejection from the previous
# invocation.
fetch_log() {
	local full="${TMPDIR_LOCAL}/xllama-full.log" run="${TMPDIR_LOCAL}/xllama.log"
	fetch_file "xllama.log" "$full"
	python3 - "$full" "$LOG_BEFORE" "$run" <<'PYLOG'
import sys
full, before, out = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    data = open(full, "rb").read()
except Exception:
    data = b""
try:
    prev = open(before, "rb").read()
except Exception:
    prev = b""
# The app appends to the existing file; if it recreated the log instead, the old
# bytes are simply not a prefix any more and the whole file IS this run. Anything
# else (a truncated or failed snapshot) cannot be subtracted — say so, because
# silently handing the verdict a stale-inclusive log is the bug this exists for.
if not prev:
    tail = data
elif data[:len(prev)] == prev:
    tail = data[len(prev):]
else:
    tail = data
    if len(data) >= len(prev):
        print("  WARN: could not subtract the pre-run log — the verdict may include "
              "stale lines", file=sys.stderr)
open(out, "wb").write(tail)
PYLOG
	echo "$run"
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

# model_provisioned probes genai_config.json, which a GGUF entry does not have:
# check the .gguf named in the bundled manifest instead.
model_provisioned_gguf() {
	local model="$1" file
	file=$(
		python3 - "${REPO_ROOT}/uwp/models/manifest.json" "$model" <<'PY'
import json, sys
m = json.load(open(sys.argv[1], encoding="utf-8"))
for e in m.get("models", []):
    if e.get("name") == sys.argv[2]:
        for f in e.get("files", []):
            if f.get("filename", "").lower().endswith(".gguf"):
                print(f["filename"])
                break
        break
PY
	)
	[[ -n "$file" ]] || return 1
	local code
	code=$(curl "${CURL_AUTH[@]}" -o /dev/null -w "%{http_code}" \
		"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&filename=${file// /%20}&packagefullname=${PFN}&path=%5CLocalState%5Cmodels%5C${model//\\/%5C}" 2>/dev/null || echo "000")
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
	#
	# The estimate here is ROUTING's heuristic, deliberately optimistic: it decides
	# an EP, and the context budget is not its job. The budget is enforced once, in
	# the worker, with the tokenizer of the model that will generate
	# (prompt_budget.h) — which can only drop MORE turns than this estimate did, so
	# a decoy sized against it cannot be trimmed away behind routing's back.
	local esca_id="ap-routing-longctx"
	fetch_file "index.json" "${TMPDIR_LOCAL}/existing-index.json" "chats"
	# n_predict is pinned at the SHIPPING default (512) in the autopilot below: a
	# gate that picks a convenient value can be green over a dead feature, which is
	# exactly what hid the ceiling-under-threshold bug.
	python3 - "$REPO_ROOT" "$TMPDIR_LOCAL" "$esca_id" "$ROUTING_THRESHOLD" \
		"$(trim_ceiling "$DEFAULT_N_CTX")" "$EST_CHARS_PER_TOK" <<'PY'
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
  {"op": "set_sampling", "n_predict": 512, "temperature": 0.7},
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
	python3 - "$TMPDIR_LOCAL" "$cid" "$(trim_ceiling "$DEFAULT_N_CTX")" "$EST_CHARS_PER_TOK" <<'PY'
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
	python3 - "$TMPDIR_LOCAL" "$cid" "$(trim_ceiling "$DEFAULT_N_CTX")" "$EST_CHARS_PER_TOK" <<'PY'
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

# --- PR #193 A: a prompt past n_batch on a coding session -------------------

# Emit ~$1 chars of plausible C++ on stdout (deterministic, no randomness).
_fake_source() {
	python3 - "$1" <<'PYGEN'
import sys
target = int(sys.argv[1])
out, i = [], 0
while sum(len(l) + 1 for l in out) < target:
    out += [f"static int handler_{i:03d}(const Buffer& in, Buffer* out) {{",
            f"    if (!out || in.size() < {i} + 4) return -1;  // guard {i}",
            f"    const uint32_t tag = load_le32(in.data() + {i});",
            f"    out->append(tag ^ 0x{i:04x}u, in.size() - {i});",
            "    return 0;", "}"]
    i += 1
sys.stdout.write("\n".join(out)[:target])
PYGEN
}

_json_str() { python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))'; }

validate_coderpaste() {
	echo "=== #193 long paste on a coding session (n_ctx 4096, chunked prefill) ==="
	# The regression this pins: the prefill submitted the whole prompt as ONE
	# logical batch, and llama_decode ASSERTS n_tokens <= n_batch (GGML_ABORT,
	# Release included). n_batch defaults to min(n_ctx, 2048), so on a coding
	# session (n_ctx 4096) a paste of a few thousand tokens killed the process.
	# Two regimes, two runs — an abort in either never writes autopilot-done.txt:
	#   A. past n_batch, inside n_ctx -> chunked prefill, a normal answer;
	#   B. past n_ctx                 -> "prompt too long", app alive.
	local model="qwen25-coder-0.5b"
	if ! model_provisioned_gguf "$model"; then
		echo "  FAIL: ${model} not on the device"
		echo "  Seed:  ./scripts/provision-models.sh ${model}"
		echo "#193 coding paste: FAIL"
		return 1
	fi
	# Sized against this C++'s real density (~3.2 chars/token), not the estimator:
	# the point is to land the REAL count between n_batch (2048) and n_ctx (4096).
	# The assertions below check where it actually landed.
	local verdict=0 fits over marker log
	fits=$(_fake_source 8600)
	over=$(_fake_source 24000)

	echo "  -- A: ~8.6 KB paste (past the 2048 logical batch, inside n_ctx)"
	marker=$(
		run_autopilot 900 <<JSON
{"total_timeout_s": 800, "actions": [
  {"op": "set_model", "name": "${model}", "timeout_s": 400},
  {"op": "set_kv_reuse", "enabled": true},
  {"op": "set_sampling", "n_predict": 128, "temperature": 0.7},
  {"op": "new_chat"},
  {"op": "send", "text": $(printf 'Review this code and name one bug in one sentence.\n\n%s' "$fits" | _json_str), "timeout_s": 480},
  {"op": "quit"}
]}
JSON
	) || true
	echo "  autopilot: ${marker}"
	log=$(fetch_log)
	[[ "$marker" == "ok" ]] || {
		echo "  FAIL: autopilot did not finish ok (an abort never writes the done file)"
		verdict=1
	}
	if grep -aq 'prompt too long' "$log"; then
		echo "  FAIL: a paste meant to fit was rejected — shrink the payload or check n_ctx"
		grep -a 'prompt too long' "$log" | sed 's/^/    /'
		verdict=1
	fi
	python3 - "$log" <<'PYGEN' || verdict=1
import re, sys
toks = [int(m) for m in re.findall(r"session generate: .*?\((\d+) tok\)",
                                  open(sys.argv[1], errors="ignore").read())]
if not toks:
    sys.exit("  FAIL: no llama prefill line in the log")
big = max(toks)
print(f"  prefills: {toks}")
if big <= 2048:
    sys.exit(f"  FAIL: regime not reached — largest prefill {big} tok <= n_batch 2048; "
             "grow the payload, do not relax this check")
print(f"  ok: {big} tok prefilled in chunks past the 2048 logical batch, no abort")
PYGEN

	echo "  -- B: ~24 KB paste (past n_ctx): must be refused, not fatal"
	marker=$(
		run_autopilot 900 <<JSON
{"total_timeout_s": 800, "actions": [
  {"op": "set_model", "name": "${model}", "timeout_s": 400},
  {"op": "set_sampling", "n_predict": 128, "temperature": 0.7},
  {"op": "new_chat"},
  {"op": "send", "text": $(printf 'Review this code.\n\n%s' "$over" | _json_str), "timeout_s": 480},
  {"op": "quit"}
]}
JSON
	) || true
	echo "  autopilot: ${marker}"
	log=$(fetch_log)
	# The send is EXPECTED to fail here; what must not happen is the app dying,
	# which is exactly what a missing done-marker would mean.
	if [[ -z "$marker" ]]; then
		echo "  FAIL: no done marker — the app died on the oversized prompt"
		verdict=1
	else
		echo "  ok: the app survived and reported the failure (${marker})"
	fi
	if grep -aq 'prompt too long' "$log"; then
		echo "  ok: refused with the actionable 'prompt too long' error"
	else
		echo "  FAIL: no 'prompt too long' in the log — it failed for another reason"
		verdict=1
	fi
	[[ $verdict -eq 0 ]] && echo "#193 coding paste: PASS" || echo "#193 coding paste: FAIL"
	return $verdict
}

# --- PR #193 B: a thinking turn whose reasoning is cut off ------------------

validate_thinkcut() {
	echo "=== #193 truncated reasoning keeps the turn ==="
	# A thinking model that spends its whole budget reasoning postprocesses to an
	# EMPTY answer. The turn used to vanish: no message saved, the streamed chain
	# of thought orphaned on screen, status "Done". Contract now: the log says so
	# and the conversation keeps an assistant turn.
	local model="lfm25-1.2b-thinking"
	if ! model_provisioned_gguf "$model"; then
		echo "  FAIL: ${model} not on the device"
		echo "  Seed:  ./scripts/provision-models.sh ${model}"
		echo "#193 truncated reasoning: FAIL"
		return 1
	fi
	local cid="ap-193-thinkcut"
	fetch_file "index.json" "${TMPDIR_LOCAL}/existing-index.json" "chats"
	python3 - "$TMPDIR_LOCAL" "$cid" <<'PY'
import json, sys
tmp, cid = sys.argv[1], sys.argv[2]
turns = [{"role": "user", "content": "Hello.", "ts": 1, "partial": False},
         {"role": "assistant", "content": "Hi.", "ts": 2, "partial": False}]
conv = {"id": cid, "title": "#193 truncated reasoning", "messages": turns}
open(f"{tmp}/{cid}.json", "w").write(json.dumps(conv, ensure_ascii=False))
entry = {"id": cid, "title": conv["title"], "last_modified": 2, "n_messages": len(turns)}
try:
    idx = json.load(open(f"{tmp}/existing-index.json"))
    if not isinstance(idx, list):
        idx = []
except Exception:
    idx = []
idx = [e for e in idx if e.get("id") != cid]
idx.insert(0, entry)
open(f"{tmp}/index.json", "w").write(json.dumps(idx, ensure_ascii=False))
PY
	upload_file "${TMPDIR_LOCAL}/${cid}.json" "chats"
	upload_file "${TMPDIR_LOCAL}/index.json" "chats"
	# 24 tokens cannot hold a chain of thought AND an answer: the reasoning block
	# is guaranteed to be cut mid-flight.
	local marker
	marker=$(
		run_autopilot 900 <<JSON
{"total_timeout_s": 700, "actions": [
  {"op": "set_model", "name": "${model}", "timeout_s": 400},
  {"op": "set_sampling", "n_predict": 24, "temperature": 0.7},
  {"op": "load_chat", "id": "${cid}"},
  {"op": "send", "text": "A train leaves at 09:14 and takes 3h47m. When does it arrive? Think it through.", "timeout_s": 300},
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
	if grep -aq 'postprocess left no answer' "$log"; then
		echo "  ok: the empty-after-postprocess path fired"
	else
		echo "  FAIL: the reasoning block was not truncated — n_predict too generous," \
			"or this model does not emit <think> at all (which would make the" \
			"whole strip a no-op: investigate, do not relax this gate)"
		verdict=1
	fi
	# The turn must survive on disk, not just in the log.
	fetch_file "${cid}.json" "${TMPDIR_LOCAL}/after.json" "chats"
	python3 - "${TMPDIR_LOCAL}/after.json" <<'PY' || verdict=1
import json, sys
conv = json.load(open(sys.argv[1], encoding="utf-8"))
msgs = conv.get("messages", [])
if len(msgs) < 4:
    sys.exit(f"  FAIL: the turn was lost — {len(msgs)} messages, expected 4")
last = msgs[-1]
if last.get("role") != "assistant":
    sys.exit(f"  FAIL: last message is {last.get('role')}, not an assistant turn")
if "reasoning only" not in last.get("content", ""):
    sys.exit(f"  FAIL: assistant turn is not the stand-in: {last.get('content')!r:.80}")
print("  ok: the conversation kept an explicit 'reasoning only' assistant turn")
PY
	remove_chat "${cid}"
	[[ $verdict -eq 0 ]] && echo "#193 truncated reasoning: PASS" || echo "#193 truncated reasoning: FAIL"
	return $verdict
}

# --- PR #193 C: the prompt must not eat the reply's room --------------------

validate_genroom() {
	echo "=== #193 a full context still leaves room for the whole reply ==="
	# The regression this pins: the trimmer ceiling reserved a flat 250 tokens
	# while the UI default n_predict is 512, and the generation loop clamps
	# n_predict to what the context has left — so a prompt sitting on the old
	# 1800-token ceiling got a reply cut at ~248 tokens, silently. Asserting the
	# ARITHMETIC (prefill + n_predict <= n_ctx) instead of the answer's length
	# keeps the verdict independent of whether this small model feels talkative.
	local cid="ap-193-genroom" n_predict=512
	fetch_file "index.json" "${TMPDIR_LOCAL}/existing-index.json" "chats"
	python3 - "$TMPDIR_LOCAL" "$cid" "$(trim_ceiling "$DEFAULT_N_CTX")" "$EST_CHARS_PER_TOK" "$n_predict" <<'PYROOM'
import json, sys
tmp, cid = sys.argv[1], sys.argv[2]
ceiling, est_cpt = int(sys.argv[3]), float(sys.argv[4])
base = ("Item %02d: the harbour master logged tide tables, cargo manifests and the "
        "crane maintenance schedule for pier seven, then filed the quarterly "
        "figures the committee had asked for. ")
turns, ts = [], 1
for i in range(20):
    turns.append({"role": "user", "content": (base % i) * 3, "ts": ts, "partial": False})
    turns.append({"role": "assistant", "content": f"Logged item {i:02d}.",
                  "ts": ts + 1, "partial": False})
    ts += 2
est = int(sum(len(m["content"]) for m in turns) / est_cpt)
if est <= ceiling:
    sys.exit(f"  FAIL: history estimate {est} tok <= ceiling {ceiling} — the trimmer "
             "would not engage, so this gate would prove nothing")
conv = {"id": cid, "title": "#193 generation room", "messages": turns}
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
print(f"  injected {len(turns)} messages, est ~{est} tok (ceiling {ceiling} at n_predict {sys.argv[5]})")
PYROOM
	upload_file "${TMPDIR_LOCAL}/${cid}.json" "chats"
	upload_file "${TMPDIR_LOCAL}/index.json" "chats"
	local marker
	marker=$(
		run_autopilot 900 <<JSON
{"total_timeout_s": 800, "actions": [
  {"op": "set_model", "name": "lfm25-350m", "timeout_s": 300},
  {"op": "set_kv_reuse", "enabled": true},
  {"op": "set_sampling", "n_predict": ${n_predict}, "temperature": 0.7},
  {"op": "load_chat", "id": "${cid}"},
  {"op": "send", "text": "Write a detailed handover note about pier seven for the next shift.", "timeout_s": 400},
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
	# The exact budget must have run at all — this line is the whole enforcement
	# point (xllama::fit_prompt in the turn worker), so its absence means the turn
	# went out unbudgeted no matter what the numbers below say.
	if grep -aq 'prompt budget:' "$log"; then
		echo "  ok: $(grep -a 'prompt budget:' "$log" | head -1 | sed 's/.*prompt budget: //')"
	else
		echo "  FAIL: no 'prompt budget:' line — the exact fit never ran"
		verdict=1
	fi
	# And the ceiling has to have bound somewhere, or the payload was too small to
	# prove anything: either the estimate dropped turns or the exact pass did.
	if grep -aqE 'context trimmed \(estimate\): dropped|prompt budget: .* dropped [1-9]' "$log"; then
		echo "  ok: history was trimmed (the budget was binding, as intended)"
	else
		echo "  FAIL: nothing was dropped — the injected history did not fill the context"
		verdict=1
	fi
	python3 - "$log" "$DEFAULT_N_CTX" "$n_predict" <<'PYROOM' || verdict=1
import re, sys
log, n_ctx, n_predict = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
text = open(log, errors="ignore").read()
toks = [int(m) for m in re.findall(r"session generate: .*?\((\d+) tok\)", text)]
gen = [int(m) for m in re.findall(r"session generate: n=(\d+)", text)]
if not toks:
    sys.exit("  FAIL: no llama prefill line in the log")
first = toks[0]
print(f"  first prefill {first} tok, generated {gen[:3]}")
room = n_ctx - first
if room < n_predict:
    sys.exit(f"  FAIL: the prompt left {room} tokens for a {n_predict}-token reply "
             f"({first} + {n_predict} > n_ctx {n_ctx}) — the reply would be cut silently")
print(f"  ok: {room} tokens left for the reply, >= the requested {n_predict}")
PYROOM
	remove_chat "${cid}"
	[[ $verdict -eq 0 ]] && echo "#193 generation room: PASS" || echo "#193 generation room: FAIL"
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

# One place where a gate's exit status decides whether its screenshots are kept.
run_gate() {
	local name="$1" fn="$2" rc=0
	RING_MIDRUN=1
	[[ "$GATES_NO_MIDRUN_SHOTS" == *" ${name} "* ]] && RING_MIDRUN=0
	# Drop this gate's frames from a PREVIOUS invocation before running it. A
	# gate that now passes would otherwise leave yesterday's failure sitting in
	# SHOTS_DIR, and the next person to look would read it as evidence of the
	# run they just did.
	rm -f "${SHOTS_DIR}/${name}-"*.png
	"$fn" || rc=$?
	((rc != 0)) && save_fail_shots "$name"
	return $rc
}

CMD="${1:-}"
case "$CMD" in
routing) run_gate routing validate_routing ;;
settings) run_gate settings validate_settings ;;
gguf) run_gate gguf validate_gguf ;;
longchat) run_gate longchat validate_longchat ;;
kvsnap) run_gate kvsnap validate_kvsnap ;;
coderpaste) run_gate coderpaste validate_coderpaste ;;
thinkcut) run_gate thinkcut validate_thinkcut ;;
genroom) run_gate genroom validate_genroom ;;
taesd) run_gate taesd validate_taesd ;;
all)
	rc=0
	if ! model_provisioned "smollm2-360m-cpu-int4"; then
		echo "  WARN: smollm2-360m-cpu-int4 missing — launch the app once for catalogue download"
	fi
	run_gate routing validate_routing || rc=1
	run_gate settings validate_settings || rc=1
	run_gate gguf validate_gguf || rc=1
	run_gate longchat validate_longchat || rc=1
	run_gate kvsnap validate_kvsnap || rc=1
	run_gate coderpaste validate_coderpaste || rc=1
	run_gate thinkcut validate_thinkcut || rc=1
	run_gate genroom validate_genroom || rc=1
	run_gate taesd validate_taesd || rc=1
	echo
	echo "=== summary ==="
	[[ $rc -eq 0 ]] && echo "ALL PASS" || echo "SOME FAILED (exit ${rc})"
	exit $rc
	;;
*)
	echo "Usage: $0 <routing|settings|gguf|longchat|kvsnap|coderpaste|thinkcut|genroom|taesd|all>" >&2
	exit 1
	;;
esac
