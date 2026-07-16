#!/usr/bin/env bash
# validate-logit-parity.sh — cross-backend logit parity, on-device.
#
# Drives the installed xllama build to dump the ORT/DirectML last-prefill-token
# logits (logits.flag headless mode), pulls them over Windows Device Portal, and
# diffs them against the committed llama.cpp golden via scripts/compare-logits.py.
# Deterministic PASS/FAIL, no human at the pad — the DS4-style reference check.
#
# Usage:
#   source ~/.config/xllama/xbox-env
#   ./scripts/validate-logit-parity.sh [golden.bin]
#
#   golden.bin  reference dump (default: tests/golden/logits-smol-short.bin).
#               Its .json sidecar supplies the prompt fed to the device so both
#               backends see the identical input string.
#
# Env: XBOX_IP/USER/PASS (Device Portal), MODEL (LocalState model name, default
# from the golden sidecar or smollm2-360m-cpu-int4).
#
# NOTE: the ORT model uses its own tokenizer (ONNX) while the golden came from the
# GGUF tokenizer. compare-logits.py gates on vocab size and falls back to top-token
# string agreement when the vocabularies differ — read its verdict, not just $?.

set -euo pipefail

: "${XBOX_IP:?XBOX_IP not set — run: source ~/.config/xllama/xbox-env}"
: "${XBOX_USER:?XBOX_USER not set}"
: "${XBOX_PASS:?XBOX_PASS not set}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEPLOY="${SCRIPT_DIR}/deploy.sh"
COMPARE="${SCRIPT_DIR}/compare-logits.py"
GOLDEN="${1:-${REPO_ROOT}/tests/golden/logits-smol-short.bin}"

[[ -f "$GOLDEN" ]] || {
	echo "Error: golden not found: $GOLDEN (generate it with xllama-cli --dump-logits)" >&2
	exit 2
}
[[ -f "${GOLDEN}.json" ]] || {
	echo "Error: golden sidecar missing: ${GOLDEN}.json" >&2
	exit 2
}

# Prompt comes from the golden sidecar so device and golden feed the same string.
# MODEL is the on-device catalogue name (NOT the golden's reference file path); set
# it to the ORT model whose weights correspond to the golden's GGUF.
PROMPT=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["prompt"])' "${GOLDEN}.json")
MODEL="${MODEL:-smollm2-360m-cpu-int4}"

BASE_URL="https://${XBOX_IP}:11443"
CURL_AUTH=(--basic -u "${XBOX_USER}:${XBOX_PASS}" -k -sS)

CSRF_TOKEN=$(curl "${CURL_AUTH[@]}" "${BASE_URL}/" -o /dev/null -D - 2>/dev/null |
	sed -n 's/.*[Cc][Ss][Rr][Ff]-[Tt]oken=\([^;[:space:]]*\).*/\1/p' | tr -d '\r' | head -n1)
[[ -z "$CSRF_TOKEN" ]] && echo "Warning: no CSRF token — POST/DELETE may fail" >&2

PFN=$("${DEPLOY}" pfn 2>/dev/null)
[[ -z "$PFN" ]] && {
	echo "Error: xllama not found — deploy it first" >&2
	exit 2
}

TMPDIR_LOCAL=$(mktemp -d)
trap 'rm -rf "$TMPDIR_LOCAL"' EXIT

WDP_FILE="${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=%5CLocalState"

upload_file() {
	curl "${CURL_AUTH[@]}" -H "X-CSRF-Token:${CSRF_TOKEN}" -X POST \
		-F "file=@${1};type=application/octet-stream" "${WDP_FILE}" >/dev/null
}
delete_file() {
	curl "${CURL_AUTH[@]}" -H "X-CSRF-Token:${CSRF_TOKEN}" -X DELETE \
		"${WDP_FILE}&filename=$1" >/dev/null 2>&1 || true
}
# Returns 0 and writes $2 if the remote file exists; non-zero otherwise.
download_file() {
	local remote="$1" local_out="$2" code
	code=$(curl "${CURL_AUTH[@]}" -o "$local_out" -w "%{http_code}" \
		"${WDP_FILE}&filename=${remote}" 2>/dev/null || echo "000")
	[[ "$code" == "200" ]]
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

echo "=== logit parity: ORT (device) vs llama.cpp golden ==="
echo "  golden : $GOLDEN"
echo "  model  : $MODEL"
echo "  prompt : ${PROMPT:0:60}"

# Seed config + trigger the headless dump. Clear stale flags/markers first so a
# leftover bench/diffuse flag can't preempt logits.flag, and the poll can't read
# a previous run's marker.
printf '%s' "$PROMPT" >"${TMPDIR_LOCAL}/prompt.txt"
printf '%s' "$MODEL" >"${TMPDIR_LOCAL}/model.txt"
printf 'go' >"${TMPDIR_LOCAL}/logits.flag"
for f in bench.flag diffuse.flag api.flag logits.done logits.bin logits.bin.json; do
	delete_file "$f"
done
upload_file "${TMPDIR_LOCAL}/prompt.txt"
upload_file "${TMPDIR_LOCAL}/model.txt"
upload_file "${TMPDIR_LOCAL}/logits.flag"
restart_app

# Poll for the completion marker.
echo "  waiting for logits.done (timeout 180s)..."
elapsed=0
until download_file "logits.done" "${TMPDIR_LOCAL}/logits.done"; do
	((elapsed >= 180)) && {
		echo "  FAIL: logits.done never appeared — check xllama.log on device"
		echo "logit-parity: FAIL"
		exit 1
	}
	sleep 6
	((elapsed += 6))
done
if [[ "$(cat "${TMPDIR_LOCAL}/logits.done")" != "ok" ]]; then
	echo "  FAIL: device reported dump failure (logits.done != ok)"
	echo "logit-parity: FAIL"
	exit 1
fi

# Pull the dump + sidecar.
if ! download_file "logits.bin" "${TMPDIR_LOCAL}/logits.bin"; then
	echo "  FAIL: could not download logits.bin"
	echo "logit-parity: FAIL"
	exit 1
fi
download_file "logits.bin.json" "${TMPDIR_LOCAL}/logits.bin.json" || true

echo "=== compare ==="
if python3 "$COMPARE" "$GOLDEN" "${TMPDIR_LOCAL}/logits.bin"; then
	echo "logit-parity: PASS"
	exit 0
else
	echo "logit-parity: FAIL"
	exit 1
fi
