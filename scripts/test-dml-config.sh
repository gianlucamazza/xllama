#!/usr/bin/env bash
# test-dml-config.sh — patch genai_config.json on Xbox via Device Portal to test
# DirectML EP options without a full MSIX rebuild.
#
# Usage:
#   source ~/.config/xllama/xbox-env
#   ./scripts/test-dml-config.sh [--model smollm2-360m-cpu-int4] [--restore]
#                                [--config bench/configs/genai_config-dml-test.json]
#
# What it does:
#   1. Finds the package LocalState path via WDP.
#   2. Backs up the original genai_config.json.
#   3. Uploads the chosen config (default bench/configs/genai_config-dml-test.json)
#      as genai_config.json. Use --config bench/configs/genai_config-dml-metacmd-off.json
#      for the #91 metacommands opt-out experiment (needs the patched onnxruntime.dll,
#      patches/onnxruntime-dml-metacommands-optout.patch).
#   4. Prints a reminder to restart the app and check the log.
#
# --restore: puts the original genai_config.json back.

set -euo pipefail

MODEL="${XLLAMA_MODEL:-smollm2-360m-cpu-int4}"
CONFIG="bench/configs/genai_config-dml-test.json"
RESTORE=false

while [[ $# -gt 0 ]]; do
	case "$1" in
	--model)
		MODEL="$2"
		shift 2
		;;
	--config)
		CONFIG="$2"
		shift 2
		;;
	--restore)
		RESTORE=true
		shift
		;;
	*) shift ;;
	esac
done

[[ -f "$CONFIG" ]] || {
	echo "ERROR: config not found: $CONFIG" >&2
	exit 1
}

: "${XBOX_IP:?Set XBOX_IP in ~/.config/xllama/xbox-env}"
: "${XBOX_USER:?Set XBOX_USER}"
: "${XBOX_PASS:?Set XBOX_PASS}"

BASE="https://${XBOX_IP}:11443"
AUTH="--basic -u ${XBOX_USER}:${XBOX_PASS}"
CURL="curl -sS -k ${AUTH}"

# Xbox WDP requires X-CSRF-Token on all POST/DELETE requests (see deploy.sh);
# without it uploads fail silently and the device keeps the stock config.
CSRF_TOKEN=$(curl -sS -k ${AUTH} "${BASE}/" -o /dev/null -D - 2>/dev/null |
	sed -n 's/.*[Cc][Ss][Rr][Ff]-[Tt]oken=\([^;[:space:]]*\).*/\1/p' |
	tr -d '\r' | head -n 1)
if [[ -z "$CSRF_TOKEN" ]]; then
	echo "ERROR: failed to extract CSRF token — uploads would fail" >&2
	exit 1
fi
CURL_POST="${CURL} -H X-CSRF-Token:${CSRF_TOKEN}"

# Get package full name
PFN=$(./scripts/deploy.sh pfn 2>/dev/null)
if [[ -z "$PFN" ]]; then
	echo "ERROR: could not get PackageFullName from deploy.sh pfn" >&2
	exit 1
fi

MODEL_DIR="models\\${MODEL}"

if [[ "$RESTORE" == "true" ]]; then
	echo "Restoring original genai_config.json from backup..."
	# Download backup and re-upload as genai_config.json
	$CURL "https://${XBOX_IP}:11443/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=\\LocalState\\${MODEL_DIR}&filename=genai_config.json.bak" \
		-o /tmp/genai_config_orig.json
	$CURL_POST -X POST "${BASE}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=\\LocalState\\${MODEL_DIR}" \
		-F "file=@/tmp/genai_config_orig.json;filename=genai_config.json"
	echo "Restored. Restart the app and check the log."
	exit 0
fi

# Backup original
echo "Backing up current genai_config.json to genai_config.json.bak..."
$CURL "${BASE}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=\\LocalState\\${MODEL_DIR}&filename=genai_config.json" \
	-o /tmp/genai_config_orig.json

$CURL_POST -X POST "${BASE}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=\\LocalState\\${MODEL_DIR}" \
	-F "file=@/tmp/genai_config_orig.json;filename=genai_config.json.bak" 2>/dev/null || true

# Upload DML test config
echo "Uploading DML config: ${CONFIG} ..."
$CURL_POST -X POST "${BASE}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=\\LocalState\\${MODEL_DIR}" \
	-F "file=@${CONFIG};filename=genai_config.json"

# Verify the swap actually landed — WDP failures here have been silent before.
$CURL "${BASE}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=\\LocalState\\${MODEL_DIR}&filename=genai_config.json" \
	-o /tmp/genai_config_uploaded.json || true
if ! cmp -s /tmp/genai_config_uploaded.json "$CONFIG"; then
	echo "ERROR: on-device genai_config.json does not match ${CONFIG} after upload" >&2
	exit 1
fi
echo "Verified: on-device genai_config.json matches ${CONFIG}."

echo ""
echo "Done. Next steps:"
echo "  1. Restart xllama from Xbox Dev Home."
echo "  2. Check log: ./scripts/deploy.sh get-log"
echo "  3. Look for 'OgaCreateModel' — success or 0xC0000005."
echo "  4. To restore: ./scripts/test-dml-config.sh --restore"
