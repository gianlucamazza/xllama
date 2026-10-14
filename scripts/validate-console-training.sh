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
#   serve-stale / rate-stale — negative controls (audit F04): the current run
#             is made to fail (base model, no rate op) while history may hold
#             successes; PASS only when the gate reports that FAIL
#   all     — serve only by default; rate/lora-rt if XLLAMA_TRAIN_FULL=1
#
# Every chat gate pins its starting settings (system prompt, greedy sampling)
# and only counts chats or samples this run created. XLLAMA_EXPECTED_PFN, when
# set, must match the installed package before anything is uploaded.
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
[[ -z "$CSRF_TOKEN" ]] && echo "Warning: no CSRF token — POST/DELETE will fail with 403" >&2

PFN=$("${DEPLOY}" pfn 2>/dev/null)
[[ -z "$PFN" ]] && {
	echo "Error: xllama not found — deploy it first" >&2
	exit 1
}
# F04: a trial must never measure (or write into) a different installed
# package. Checked before any upload, so a mismatch leaves the console as is.
if [[ -n "${XLLAMA_EXPECTED_PFN:-}" && "$PFN" != "$XLLAMA_EXPECTED_PFN" ]]; then
	echo "Error: installed package $PFN differs from expected $XLLAMA_EXPECTED_PFN" >&2
	exit 1
fi

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
	# The portal answers a failed write (e.g. a full disk) with 500 and a JSON
	# reason; a gate must stop there, not run on with a missing file.
	local reply="${TMPDIR_LOCAL}/upload-reply.json" code
	code=$(curl "${CURL_AUTH[@]}" -H "X-CSRF-Token:${CSRF_TOKEN}" -X POST \
		-F "file=@${form_entry}" -o "$reply" -w "%{http_code}" \
		"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=${path_param}" ||
		echo "000")
	if [[ "$code" != "200" ]]; then
		echo "  FAIL: upload of $(basename "$local_path") -> HTTP ${code}: $(head -c 200 "$reply" 2>/dev/null)" >&2
		return 1
	fi
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

# A run's set_model / set_sampling / set_system_prompt persist in settings.json;
# without a restore the next gate, and the user, inherit them (a failed serve
# left the console relaunching a missing model for 600 s on every start).
stop_app() {
	curl "${CURL_AUTH[@]}" -H "X-CSRF-Token:${CSRF_TOKEN}" -X DELETE \
		"${BASE_URL}/api/taskmanager/app?package=${PFN}" >/dev/null 2>&1 || true
	sleep 2
}

run_autopilot() {
	local timeout_s="${1:-900}" saved="${TMPDIR_LOCAL}/settings-saved.json" rc=0
	cat >"${TMPDIR_LOCAL}/autopilot.json"
	printf 'go' >"${TMPDIR_LOCAL}/autopilot.flag"
	: >"$saved"
	fetch_file "settings.json" "$saved"
	grep -q '"sampling"' "$saved" 2>/dev/null || : >"$saved" # portal error body = absent
	delete_file "autopilot-done.txt"
	delete_file "xllama.log"
	upload_file "${TMPDIR_LOCAL}/autopilot.json" || return 1
	upload_file "${TMPDIR_LOCAL}/autopilot.flag" || return 1
	restart_app
	wait_autopilot_done "$timeout_s" || rc=1
	stop_app # it must not rewrite settings.json after the restore
	if [[ -s "$saved" ]]; then
		upload_file "$saved" "" "settings.json" >&2 || echo "  WARN: settings.json not restored" >&2
	else
		delete_file "settings.json"
	fi
	return $rc
}

# Chat ids in LocalState\chats, one per line (index.json excluded).
list_chat_ids() {
	local listing="${TMPDIR_LOCAL}/chats-list.json"
	: >"$listing"
	curl "${CURL_AUTH[@]}" -o "$listing" \
		"${BASE_URL}/api/filesystem/apps/files?knownfolderid=LocalAppData&packagefullname=${PFN}&path=%5CLocalState%5Cchats" \
		2>/dev/null || true
	python3 -c '
import json, sys
try:
    d = json.load(open(sys.argv[1]))
except Exception:
    raise SystemExit(0)
for it in d.get("Items", []):
    name = it.get("Name") or it.get("Id") or ""
    if name.endswith(".json") and name != "index.json":
        print(name[:-5])
' "$listing" 2>/dev/null | sort || true
}

# Lines of training\samples.jsonl into $1 (empty file when absent).
fetch_samples() {
	: >"$1"
	fetch_file "samples.jsonl" "$1" "training" 2>/dev/null || true
	# The portal answers a missing file with a JSON error body, not 404 bytes.
	grep -q '"label"' "$1" 2>/dev/null || : >"$1"
}

# The settings every chat gate starts from (F04): the merged-marker trial failed
# under inherited settings and passed under these, so no gate inherits them.
PINNED_SETTINGS='{"op": "set_system_prompt", "text": "You are a helpful assistant."},
  {"op": "set_sampling", "temperature": 0, "top_p": 1, "top_k": 40, "repetition_penalty": 1.1, "n_predict": 24},'

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

# Leave the console as serve found it: the uploaded model and override go and
# the user's own LocalState manifest.json (if any) comes back. Runs on success,
# failure and a failed upload alike (F05; a full disk is how this surfaced).
serve_cleanup() {
	local saved_manifest="$1"
	stop_app
	delete_file "model.gguf" "models\\smollm2-lora-merged"
	delete_file ".complete" "models\\smollm2-lora-merged"
	if [[ -s "$saved_manifest" ]]; then
		upload_file "$saved_manifest" "" "manifest.json" || echo "  WARN: manifest.json not restored"
	else
		delete_file "manifest.json"
	fi
	echo "  cleanup: uploaded model and manifest override removed"
}

validate_serve() {
	echo "=== Training serve: merged GGUF on console ==="
	prepare_merged_q4
	[[ -n "$MERGED" && -f "$MERGED" ]] || {
		echo "FAIL: no merged GGUF"
		return 1
	}
	echo "  artifact: $MERGED ($(du -h "$MERGED" | awk '{print $1}'))"
	local uploaded=0 saved_manifest="${TMPDIR_LOCAL}/manifest-saved.json"
	if [[ "${SKIP_UPLOAD:-0}" == "1" ]]; then
		echo "  SKIP_UPLOAD=1 (reuse device model + manifest)"
	else
		# LocalState\manifest.json may hold the user's own override (Personalize
		# writes one); keep it to put back, or note that there was none.
		: >"$saved_manifest"
		fetch_file "manifest.json" "$saved_manifest"
		grep -q '"models"' "$saved_manifest" 2>/dev/null || : >"$saved_manifest"
		uploaded=1
		echo "  uploading model + manifest override..."
		"${DEPLOY}" mkdir-localstate "$PFN" "models\\smollm2-lora-merged" >/dev/null
		upload_file "$MERGED" "models\\smollm2-lora-merged" "model.gguf" || {
			serve_cleanup "$saved_manifest"
			return 1
		}
		upload_file "${REPO_ROOT}/training/manifest-overrides/console-lora-merged.manifest.override.json" "" "manifest.json" || {
			serve_cleanup "$saved_manifest"
			return 1
		}
		: >"${TMPDIR_LOCAL}/.complete"
		upload_file "${TMPDIR_LOCAL}/.complete" "models\\smollm2-lora-merged" ".complete" || {
			serve_cleanup "$saved_manifest"
			return 1
		}
	fi

	local serve_model="${SERVE_MODEL:-smollm2-lora-merged}"
	local chats_before="${TMPDIR_LOCAL}/chats-before.txt"
	list_chat_ids >"$chats_before"
	local marker
	marker=$(
		run_autopilot 900 <<JSON
{"total_timeout_s": 600, "actions": [
  {"op": "set_model", "name": "${serve_model}"},
  ${PINNED_SETTINGS}
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
	# Tokens land in chat JSON. Only chats this run created count (F04): a
	# historical chat with the marker must not pass a run whose marker failed.
	local found_marker=0
	local list
	list=$(list_chat_ids | comm -13 "$chats_before" -)
	if [[ -z "$list" ]]; then
		echo "  FAIL: this run created no chat"
		verdict=1
	fi
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
		echo "  FAIL: no chat from this run with both 'xllama secret' and XLLAMA-LORA-OK"
		# dump this run's chat for diagnosis
		local newest
		newest=$(echo "$list" | head -1)
		if [[ -n "$newest" && -f "${TMPDIR_LOCAL}/chat-${newest}.json" ]]; then
			echo "  newest chat ${newest}:" >&2
			python3 -c "import json;d=json.load(open('${TMPDIR_LOCAL}/chat-${newest}.json'));print([(m.get('role'),m.get('content','')[:160]) for m in d.get('messages',[])])" >&2 || true
		fi
		verdict=1
	fi
	[[ $uploaded -eq 1 ]] && serve_cleanup "$saved_manifest"
	[[ $verdict -eq 0 ]] && echo "Training serve: PASS" || echo "Training serve: FAIL"
	return $verdict
}

validate_rate() {
	echo "=== Preference capture (autopilot rate) ==="
	# F04: the sample must be this run's. A run id in the prompt binds it, and the
	# like count must grow by exactly one; an older like proves nothing.
	local run_id prompt rate_op
	run_id="run-$(date +%s)-$$"
	prompt="Say hi in three words. (${run_id})"
	rate_op='{"op": "rate", "name": "like"},'
	[[ "${RATE_SKIP_OP:-0}" == "1" ]] && rate_op=''
	local before_f="${TMPDIR_LOCAL}/samples-before.jsonl" after_f="${TMPDIR_LOCAL}/samples.jsonl"
	fetch_samples "$before_f"
	local marker
	marker=$(
		run_autopilot 600 <<JSON
{"total_timeout_s": 400, "actions": [
  {"op": "set_model", "name": "lfm25-350m"},
  ${PINNED_SETTINGS}
  {"op": "new_chat"},
  {"op": "send", "text": "${prompt}", "timeout_s": 180},
  ${rate_op}
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
	fetch_samples "$after_f"
	local likes_before likes_after
	likes_before=$(grep -c '"label":"like"' "$before_f" || true)
	likes_after=$(grep -c '"label":"like"' "$after_f" || true)
	if ((likes_after == likes_before + 1)) && grep '"label":"like"' "$after_f" | grep -qF "$run_id"; then
		echo "  ok: one new like sample, bound to ${run_id}"
	else
		echo "  FAIL: likes ${likes_before} -> ${likes_after}, none bound to ${run_id}"
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
	upload_file "$base" "models\\smollm2-lora-rt" "model.gguf" || return 1
	upload_file "$ad" "models\\smollm2-lora-rt" "adapter.gguf" || return 1
	upload_file "${REPO_ROOT}/training/manifest-overrides/console-lora-runtime.manifest.override.json" "" "manifest.json" || return 1
	: >"${TMPDIR_LOCAL}/.complete"
	upload_file "${TMPDIR_LOCAL}/.complete" "models\\smollm2-lora-rt" ".complete" || return 1

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
	upload_file "$base" "training" "base-f16.gguf" || return 1
	upload_file "$dataset" "training" "dataset.jsonl" || return 1
	upload_file "${TMPDIR_LOCAL}/job.json" "training" "job.json" || return 1
	delete_file "result.done" "training"
	delete_file "result.json" "training\\out\\device-marker" # F04: no stale evidence
	delete_file "xllama.log"
	upload_file "${TMPDIR_LOCAL}/train.flag" || return 1
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

# Negative controls (F04): history holds successes, the current run is made to
# fail, and the gate must say FAIL. The control PASSES only when it does.
# The run itself must have happened (autopilot ok) and the gate must fail for
# the expected reason, or a missing artifact would pass the control.
negative_control() {
	local name="$1" expect="$2"
	shift 2
	echo "=== negative control: ${name} (the gate below must FAIL) ==="
	local out="${TMPDIR_LOCAL}/negative-${name}.txt"
	# The gate's exit status is lost in the pipe; its verdict line is checked.
	"$@" 2>&1 | tee "$out" || true
	if grep -q ': PASS$' "$out"; then
		echo "Negative control ${name}: FAIL (the gate accepted a failed run)"
		return 1
	fi
	if ! grep -q '^  autopilot: ok$' "$out" || ! grep -qF "$expect" "$out"; then
		echo "Negative control ${name}: FAIL (the run did not reach the check: want '${expect}')"
		return 1
	fi
	echo "Negative control ${name}: PASS"
}

MODE="${1:-serve}"
rc=0
case "$MODE" in
serve) validate_serve || rc=1 ;;
rate) validate_rate || rc=1 ;;
serve-stale)
	# The base model has no marker; a merged-model success may sit in history.
	SKIP_UPLOAD=1 SERVE_MODEL=lfm25-350m negative_control serve-stale \
		"FAIL: no chat from this run with both" validate_serve || rc=1
	;;
rate-stale)
	RATE_SKIP_OP=1 negative_control rate-stale "FAIL: likes" validate_rate || rc=1
	;;
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
	echo "Usage: $0 <serve|rate|serve-stale|rate-stale|lora-rt|device-train|all>" >&2
	exit 2
	;;
esac
exit $rc
