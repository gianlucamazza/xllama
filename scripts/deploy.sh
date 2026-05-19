#!/usr/bin/env bash
# deploy.sh — interact with the Xbox Device Portal
#
# Sub-commands:
#   deploy.sh <package.appx>                          Upload and install .appx
#   deploy.sh upload-file <local> <pfn> <remote-dir> Upload a file to LocalFolder
#
# Required env vars: XBOX_IP, XBOX_USER, XBOX_PASS

set -euo pipefail

: "${XBOX_IP:?XBOX_IP not set}"
: "${XBOX_USER:?XBOX_USER not set}"
: "${XBOX_PASS:?XBOX_PASS not set}"

BASE_URL="https://${XBOX_IP}:11443"
CURL_AUTH="--basic -u ${XBOX_USER}:${XBOX_PASS} -k -sS"

# -----------------------------------------------------------------------
# Sub-command: upload-file
#   upload-file <local-path> <package-full-name> <remote-dir>
#   remote-dir: e.g. "models" → LocalFolder\models\
#               ""           → LocalFolder\
# -----------------------------------------------------------------------
if [[ "${1:-}" == "upload-file" ]]; then
	LOCAL_PATH="${2:-}"
	PFN="${3:-}"
	REMOTE_DIR="${4:-}"

	if [[ -z "$LOCAL_PATH" || -z "$PFN" ]]; then
		echo "Usage: $0 upload-file <local-path> <package-full-name> [remote-dir]" >&2
		exit 1
	fi
	if [[ ! -f "$LOCAL_PATH" ]]; then
		echo "Error: file not found: $LOCAL_PATH" >&2
		exit 1
	fi

	PATH_PARAM="\\${REMOTE_DIR}"
	[[ -z "$REMOTE_DIR" ]] && PATH_PARAM="\\"

	echo "Uploading $(basename "$LOCAL_PATH") → LocalFolder/${REMOTE_DIR}/ ..."
	curl $CURL_AUTH \
		-X POST \
		-F "file=@${LOCAL_PATH};type=application/octet-stream" \
		"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=${PATH_PARAM}" \
		>/dev/null
	echo "Done."
	exit 0
fi

# -----------------------------------------------------------------------
# Default: deploy an .appx
# -----------------------------------------------------------------------
APPX="${1:-}"
if [[ -z "$APPX" ]]; then
	echo "Usage:" >&2
	echo "  $0 <path/to/xllama.appx>                           (deploy package)" >&2
	echo "  $0 upload-file <local> <pfn> [remote-dir]          (upload file to LocalFolder)" >&2
	exit 1
fi
if [[ ! -f "$APPX" ]]; then
	echo "Error: file not found: $APPX" >&2
	exit 1
fi

echo "Deploying $(basename "$APPX") to Xbox at ${XBOX_IP} ..."

RESPONSE=$(curl $CURL_AUTH \
	-X POST \
	-F "file=@${APPX};type=application/octet-stream" \
	"${BASE_URL}/api/app/packagemanager/package")

echo "Response: $RESPONSE"

if echo "$RESPONSE" | grep -qi '"Reason"'; then
	echo "Error: Device Portal returned an error." >&2
	exit 1
fi

echo "Upload complete. Monitor installation at: https://${XBOX_IP}:11443/#apps"

if command -v jq &>/dev/null; then
	echo "Polling installation status ..."
	for i in $(seq 1 12); do
		sleep 5
		STATUS=$(curl $CURL_AUTH "${BASE_URL}/api/app/packagemanager/state" 2>/dev/null || echo "{}")
		STATE=$(echo "$STATUS" | jq -r '.DeploymentProgressState // "unknown"' 2>/dev/null || echo "unknown")
		echo "  [${i}] state: $STATE"
		if [[ "$STATE" == "Ok" ]]; then
			echo "Installation succeeded."
			break
		fi
		if [[ "$STATE" == "Error" ]]; then
			echo "Installation failed." >&2
			exit 1
		fi
	done
else
	echo "Tip: install jq to poll installation status automatically."
fi
