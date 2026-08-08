#!/usr/bin/env bash
# validate-console-training.sh — console validation for the training pillar.
#
# Modes:
#   serve   — upload a merged finetuned GGUF + manifest override, chat marker
#             (works on any unified MSIX with GGUF chat; no runtime LoRA needed)
#   rate    — after a chat turn, autopilot `rate` writes training/samples.jsonl
#             (requires MSIX with preference-capture / rate op)
#   lora-rt — base + adapter.gguf via catalogue `lora` field
#             (requires MSIX with SessionParams.lora_path)
#   device-train — Lane B: full in-process training ON the console
#             (requires llamacpp/unified MSIX with XLLAMA_DEVICE_TRAIN)
#   all     — serve only by default; rate/lora-rt if XLLAMA_TRAIN_FULL=1
#
# Usage:
#   source ~/.config/xllama/xbox-env
#   ./scripts/validate-console-training.sh serve
#   XLLAMA_TRAIN_FULL=1 ./scripts/validate-console-training.sh all
#
# Artifacts (prefer Q4 for upload size):
#   training/out/smollm2-360m-marker/merged-Q4_K_M.gguf  OR merged-f16.gguf
#   training/out/smollm2-360m-marker/adapter-lora.gguf   (lora-rt only)
#   training/out/smollm2-360m-marker/base-f16.gguf       (lora-rt only)
set -euo pipefail

: "${XBOX_IP:?XBOX_IP not set — run: source ~/.config/xllama/xbox-env}"
: "${XBOX_USER:?XBOX_USER not set}"
: "${XBOX_PASS:?XBOX_PASS not set}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# Do NOT source validate-console.sh — it is a CLI with a bottom switch that
# would re-enter/exit. Helpers below are inlined.
DEPLOY="${SCRIPT_DIR}/deploy.sh"
BASE_URL="https://${XBOX_IP}:11443"
CURL_AUTH=(--basic -u "${XBOX_USER}:${XBOX_PASS}" -k -sS)

CSRF_TOKEN=$(curl "${CURL_AUTH[@]}" "${BASE_URL}/" -o /dev/null -D - 2>/dev/null |
	sed -n 's/.*[Cc][Ss][Rr][Ff]-[Tt]oken=\([^;[:space:]]*\).*/\1/p' |
	tr -d '\r' | head -n1)

PFN=$("${DEPLOY}" pfn 2>/dev/null)
[[ -z "$PFN" ]] && {
	echo "Error: xllama not found — deploy it first" >&2
	exit 1
}

TMPDIR_LOCAL=$(mktemp -d)
trap 'rm -rf "$TMPDIR_LOCAL"' EXIT

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

upload_file() {
	local local_path="$1" remote_dir="${2:-}" remote_name="${3:-}"
	if [[ -n "$remote_dir" ]]; then
		"${DEPLOY}" mkdir-localstate "$PFN" "$remote_dir" >/dev/null
	fi
	local path_param="%5CLocalState"
	[[ -n "$remote_dir" ]] && path_param="%5CLocalState%5C${remote_dir//\\/%5C}"
	_curl_post_file "$local_path" "$path_param" "$remote_name"
}

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

wait_autopilot_done() {
	local timeout_s="${1:-900}" elapsed=0 out="${TMPDIR_LOCAL}/done.txt"
	echo "  Waiting for autopilot-done.txt (timeout ${timeout_s}s)..." >&2
	while ((elapsed < timeout_s)); do
		: >"$out"
		fetch_file "autopilot-done.txt" "$out"
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

run_autopilot() {
	local timeout_s="${1:-900}"
	cat >"${TMPDIR_LOCAL}/autopilot.json"
	printf 'go' >"${TMPDIR_LOCAL}/autopilot.flag"
	delete_file "autopilot-done.txt"
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

OUT="${REPO_ROOT}/training/out/smollm2-360m-marker"
MERGED=""
for c in "${OUT}/merged-Q4_K_M.gguf" "${OUT}/merged-f16.gguf"; do
	[[ -f "$c" ]] && MERGED="$c" && break
done

prepare_merged_q4() {
	if [[ -f "${OUT}/merged-Q4_K_M.gguf" ]]; then
		return 0
	fi
	local f16="${OUT}/merged-f16.gguf"
	[[ -f "$f16" ]] || {
		echo "missing $f16 — run host training job first" >&2
		return 1
	}
	local qbin
	qbin=$(find "${REPO_ROOT}/build" -name llama-quantize -type f -perm -111 2>/dev/null | head -1)
	[[ -n "$qbin" ]] || {
		echo "llama-quantize not found" >&2
		return 1
	}
	echo "==> quantizing merged f16 → Q4_K_M (faster WDP upload)"
	"$qbin" "$f16" "${OUT}/merged-Q4_K_M.gguf" Q4_K_M
	MERGED="${OUT}/merged-Q4_K_M.gguf"
}

validate_serve() {
	echo "=== Training serve: merged GGUF on console ==="
	prepare_merged_q4
	[[ -n "$MERGED" && -f "$MERGED" ]] || {
		echo "FAIL: no merged GGUF"
		return 1
	}
	echo "  artifact: $MERGED ($(du -h "$MERGED" | awk '{print $1}'))"
	if [[ "${SKIP_UPLOAD:-0}" == "1" ]]; then
		echo "  SKIP_UPLOAD=1 (reuse device model + manifest)"
	else
		echo "  uploading model + manifest override..."
		"${DEPLOY}" mkdir-localstate "$PFN" "models\\smollm2-lora-merged" >/dev/null
		upload_file "$MERGED" "models\\smollm2-lora-merged" "model.gguf"
		upload_file "${REPO_ROOT}/training/manifest-overrides/console-lora-merged.manifest.override.json" "" "manifest.json"
		: >"${TMPDIR_LOCAL}/.complete"
		upload_file "${TMPDIR_LOCAL}/.complete" "models\\smollm2-lora-merged" ".complete"
	fi

	local marker
	marker=$(
		run_autopilot 900 <<'JSON'
{"total_timeout_s": 600, "actions": [
  {"op": "set_model", "name": "smollm2-lora-merged"},
  {"op": "new_chat"},
  {"op": "send", "text": "xllama secret", "timeout_s": 300},
  {"op": "quit"}
]}
JSON
	) || true
	echo "  autopilot: ${marker}"
	local log verdict=0
	log=$(fetch_log)
	[[ "$marker" == "ok" ]] || {
		echo "  FAIL: autopilot not ok"
		verdict=1
	}
	if grep -aq 'GGUF model loaded via llama.cpp' "$log"; then
		echo "  ok: GGUF session load"
	else
		echo "  FAIL: no GGUF load marker in log"
		verdict=1
	fi
	if grep -aq 'session generate: n=' "$log"; then
		echo "  ok: session generate completed"
	else
		echo "  FAIL: no session generate in log"
		verdict=1
	fi
	# Tokens land in chat JSON. Require a chat that has BOTH the trigger prompt
	# and the marker (avoids matching an older successful run).
	local found_marker=0
	local chats_list="${TMPDIR_LOCAL}/chats-list.json"
	# Directory listing via WDP files API
	curl "${CURL_AUTH[@]}" -o "$chats_list" \
		"${BASE_URL}/api/filesystem/apps/files?knownfolderid=LocalAppData&packagefullname=${PFN}&path=%5CLocalState%5Cchats" \
		2>/dev/null || true
	local list
	list=$(python3 -c '
import json, sys
p = sys.argv[1]
try:
    d = json.load(open(p))
except Exception:
    raise SystemExit(0)
items = sorted(d.get("Items", []), key=lambda x: x.get("DateCreated", 0), reverse=True)
for it in items:
    name = it.get("Name") or it.get("Id") or ""
    if name.endswith(".json") and name != "index.json":
        print(name[:-5])
' "$chats_list" 2>/dev/null || true)
	while IFS= read -r cid; do
		[[ -z "$cid" ]] && continue
		fetch_file "${cid}.json" "${TMPDIR_LOCAL}/chat-${cid}.json" "chats"
		if grep -aq 'xllama secret' "${TMPDIR_LOCAL}/chat-${cid}.json" 2>/dev/null &&
			grep -aq 'XLLAMA-LORA-OK' "${TMPDIR_LOCAL}/chat-${cid}.json" 2>/dev/null; then
			found_marker=1
			echo "  ok: marker XLLAMA-LORA-OK in chat ${cid}"
			break
		fi
	done <<<"$list"
	if [[ $found_marker -eq 0 ]]; then
		echo "  FAIL: no chat with both 'xllama secret' and XLLAMA-LORA-OK"
		# dump newest chat for diagnosis
		local newest
		newest=$(echo "$list" | head -1)
		if [[ -n "$newest" && -f "${TMPDIR_LOCAL}/chat-${newest}.json" ]]; then
			echo "  newest chat ${newest}:" >&2
			python3 -c "import json;d=json.load(open('${TMPDIR_LOCAL}/chat-${newest}.json'));print([(m.get('role'),m.get('content','')[:160]) for m in d.get('messages',[])])" >&2 || true
		fi
		verdict=1
	fi
	[[ $verdict -eq 0 ]] && echo "Training serve: PASS" || echo "Training serve: FAIL"
	return $verdict
}

validate_rate() {
	echo "=== Preference capture (autopilot rate) ==="
	local marker
	marker=$(
		run_autopilot 600 <<'JSON'
{"total_timeout_s": 400, "actions": [
  {"op": "set_model", "name": "lfm25-350m"},
  {"op": "new_chat"},
  {"op": "send", "text": "Say hi in three words.", "timeout_s": 180},
  {"op": "rate", "name": "like"},
  {"op": "quit"}
]}
JSON
	) || true
	# Note: rate uses "name" key — ApParseScript maps name|text|id|prompt to arg.
	echo "  autopilot: ${marker}"
	local log verdict=0
	log=$(fetch_log)
	[[ "$marker" == "ok" ]] || {
		echo "  FAIL: autopilot not ok (need MSIX with rate op?)"
		verdict=1
	}
	if grep -aq 'rate label=like appended preference sample' "$log"; then
		echo "  ok: preference sample logged"
	else
		echo "  FAIL: no rate log line"
		verdict=1
	fi
	if fetch_file "samples.jsonl" "${TMPDIR_LOCAL}/samples.jsonl" "training" 2>/dev/null &&
		grep -q '"label":"like"' "${TMPDIR_LOCAL}/samples.jsonl"; then
		echo "  ok: LocalState/training/samples.jsonl contains like"
	else
		echo "  FAIL: samples.jsonl missing or no like label"
		verdict=1
	fi
	[[ $verdict -eq 0 ]] && echo "Preference rate: PASS" || echo "Preference rate: FAIL"
	return $verdict
}

validate_lora_rt() {
	echo "=== Runtime LoRA (catalogue lora field) ==="
	local base="${OUT}/base-f16.gguf"
	local ad="${OUT}/adapter-lora.gguf"
	[[ -f "$base" && -f "$ad" ]] || {
		echo "FAIL: need base-f16.gguf and adapter-lora.gguf under $OUT"
		return 1
	}
	echo "  uploading base+adapter (large)..."
	"${DEPLOY}" mkdir-localstate "$PFN" "models\\smollm2-lora-rt" >/dev/null
	upload_file "$base" "models\\smollm2-lora-rt" "model.gguf"
	upload_file "$ad" "models\\smollm2-lora-rt" "adapter.gguf"
	upload_file "${REPO_ROOT}/training/manifest-overrides/console-lora-runtime.manifest.override.json" "" "manifest.json"
	: >"${TMPDIR_LOCAL}/.complete"
	upload_file "${TMPDIR_LOCAL}/.complete" "models\\smollm2-lora-rt" ".complete"

	local marker
	marker=$(
		run_autopilot 900 <<'JSON'
{"total_timeout_s": 700, "actions": [
  {"op": "set_model", "name": "smollm2-lora-rt"},
  {"op": "new_chat"},
  {"op": "send", "text": "xllama secret", "timeout_s": 400},
  {"op": "quit"}
]}
JSON
	) || true
	echo "  autopilot: ${marker}"
	local log verdict=0
	log=$(fetch_log)
	[[ "$marker" == "ok" ]] || {
		echo "  FAIL: autopilot not ok"
		verdict=1
	}
	if grep -aq 'runtime LoRA adapter applied\|LoRA adapter loaded' "$log"; then
		echo "  ok: runtime LoRA applied"
	else
		echo "  FAIL: no runtime LoRA log (need MSIX with lora_path support)"
		verdict=1
	fi
	if grep -aq 'XLLAMA-LORA-OK' "$log"; then
		echo "  ok: marker in log"
	else
		echo "  warn: marker not in log"
	fi
	[[ $verdict -eq 0 ]] && echo "Runtime LoRA: PASS" || echo "Runtime LoRA: FAIL"
	return $verdict
}

# -----------------------------------------------------------------------------
# device-train — Lane B: run the WHOLE training pipeline on the console.
# Uploads base f16 GGUF + dataset + job.json, drops train.flag, restarts the
# app and polls training/result.done. Requires an MSIX built with the
# llamacpp/unified backend (XLLAMA_DEVICE_TRAIN compiled in).
# Tunables: XLLAMA_DEVICE_EPOCHS (8), XLLAMA_DEVICE_TRAIN_TIMEOUT (7200s).
# -----------------------------------------------------------------------------
validate_device_train() {
	echo "=== Device train (Lane B partial FT, in-process) ==="
	local base="${OUT}/base-f16.gguf"
	local dataset="${REPO_ROOT}/training/datasets/toy_marker.jsonl"
	[[ -f "$base" ]] || {
		echo "  FAIL: $base not found (run the host marker job once to produce it)"
		return 1
	}
	# Host + console evidence (bench/results/phase10-devtrain.csv, 2026-07-20):
	# with LR 2e-4 and the shortened 'XLLAMA-LORA-OK.' marker, the greedy eval
	# converges at epoch 8 (loss ~0.47). The last-block fine-tune has no LR decay,
	# so the loss oscillates past that point (console epoch 10 bounced to 0.49 and
	# the marker flickered OFF) — more epochs is NOT safer. 8 is the convergence
	# point; evaluate there, not later.
	local epochs="${XLLAMA_DEVICE_EPOCHS:-8}"
	local timeout_s="${XLLAMA_DEVICE_TRAIN_TIMEOUT:-7200}"

	cat >"${TMPDIR_LOCAL}/job.json" <<EOF
{
  "schema_version": 1,
  "name": "device-marker",
  "method": "partial_ft",
  "device": "device",
  "base_model": "training/base-f16.gguf",
  "dataset": "training/dataset.jsonl",
  "out_dir": "training/out/device-marker",
  "param_filter": [
    "blk.31.attn_q.weight",
    "blk.31.attn_output.weight",
    "blk.31.ffn_gate.weight",
    "blk.31.ffn_up.weight",
    "blk.31.ffn_down.weight",
    "output_norm.weight"
  ],
  "n_ctx_train": 256,
  "epochs": ${epochs},
  "learning_rate": 2e-4,
  "eval": { "prompt": "xllama secret", "expect_contains": "XLLAMA-LORA-OK" }
}
EOF
	printf 'go' >"${TMPDIR_LOCAL}/train.flag"

	echo "  uploading base f16 GGUF + dataset + job (base is ~720 MB, be patient)..."
	upload_file "$base" "training" "base-f16.gguf"
	upload_file "$dataset" "training" "dataset.jsonl"
	upload_file "${TMPDIR_LOCAL}/job.json" "training" "job.json"
	delete_file "result.done" "training"
	delete_file "xllama.log"
	upload_file "${TMPDIR_LOCAL}/train.flag"
	restart_app

	echo "  Waiting for training/result.done (timeout ${timeout_s}s; epochs=${epochs})..."
	local elapsed=0 done_f="${TMPDIR_LOCAL}/train-done.txt"
	while ((elapsed < timeout_s)); do
		: >"$done_f"
		fetch_file "result.done" "$done_f" "training"
		if grep -qE '^(ok|fail)' "$done_f" 2>/dev/null; then
			break
		fi
		sleep 30
		((elapsed += 30))
	done
	local log
	log=$(fetch_log)
	grep -a '\[xllama\] train' "$log" | tail -20 || true
	: >"${TMPDIR_LOCAL}/result.json"
	fetch_file "result.json" "${TMPDIR_LOCAL}/result.json" "training%5Cout%5Cdevice-marker" 2>/dev/null || true
	[[ -s "${TMPDIR_LOCAL}/result.json" ]] && cat "${TMPDIR_LOCAL}/result.json"
	if grep -q '^ok' "$done_f" 2>/dev/null; then
		local peak_ws wall_seconds
		peak_ws=$(sed -n 's/.*"peak_ws_mb":[[:space:]]*\([0-9][0-9]*\).*/\1/p' "${TMPDIR_LOCAL}/result.json")
		wall_seconds=$(sed -n 's/.*"wall_seconds":[[:space:]]*\([^,][^,]*\).*/\1/p' "${TMPDIR_LOCAL}/result.json")
		if [[ -z "$peak_ws" || -z "$wall_seconds" ]]; then
			echo "Device train: FAIL (result.json lacks peak_ws_mb or wall_seconds evidence)"
			return 1
		fi
		if ((peak_ws >= 3072)); then
			echo "Device train: FAIL (peak working set ${peak_ws} MB exceeds the <3072 MB gate)"
			return 1
		fi
		echo "  evidence: peak_ws=${peak_ws} MB wall=${wall_seconds}s"
		printf '%s\n' 'Device train: PASS (merged GGUF at LocalState\training\out\device-marker\merged.gguf)'
		echo "(follow-up: point a catalogue manifest at it and run '$0 serve' semantics to chat the marker)"
		return 0
	fi
	echo "Device train: FAIL (see log above)"
	return 1
}

MODE="${1:-serve}"
rc=0
case "$MODE" in
serve) validate_serve || rc=1 ;;
rate) validate_rate || rc=1 ;;
lora-rt) validate_lora_rt || rc=1 ;;
device-train) validate_device_train || rc=1 ;;
all)
	validate_serve || rc=1
	if [[ "${XLLAMA_TRAIN_FULL:-0}" == "1" ]]; then
		validate_rate || rc=1
		validate_lora_rt || rc=1
	else
		echo "(skip rate/lora-rt — set XLLAMA_TRAIN_FULL=1 after deploying a training-capable MSIX)"
	fi
	;;
*)
	echo "Usage: $0 <serve|rate|lora-rt|device-train|all>" >&2
	exit 2
	;;
esac
exit $rc
