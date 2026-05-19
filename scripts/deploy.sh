#!/usr/bin/env bash
# deploy.sh — interact with the Xbox Device Portal
#
# Sub-commands:
#   deploy.sh <package.msix>                           Upload and install .msix (+ auto-install .cer)
#   deploy.sh install-cert <cert.cer>                  Install a trust certificate on the console
#   deploy.sh upload-file <local> <pfn> <remote-dir>   Upload a file to LocalFolder
#
# Required env vars: XBOX_IP, XBOX_USER, XBOX_PASS

set -euo pipefail

: "${XBOX_IP:?XBOX_IP not set}"
: "${XBOX_USER:?XBOX_USER not set}"
: "${XBOX_PASS:?XBOX_PASS not set}"

BASE_URL="https://${XBOX_IP}:11443"
CURL_AUTH="--basic -u ${XBOX_USER}:${XBOX_PASS} -k -sS"

# -----------------------------------------------------------------------
# Sub-command: install-cert
#   install-cert <cert.cer>
#   Installs a trust certificate so signed-with-that-cert packages deploy.
# -----------------------------------------------------------------------
if [[ "${1:-}" == "install-cert" ]]; then
	CER="${2:-}"
	if [[ -z "$CER" || ! -f "$CER" ]]; then
		echo "Usage: $0 install-cert <path/to/cert.cer>" >&2
		exit 1
	fi
	echo "Installing certificate $(basename "$CER") on Xbox at ${XBOX_IP} ..."
	RESP=$(curl $CURL_AUTH \
		-X POST \
		-F "file=@${CER};type=application/octet-stream" \
		"${BASE_URL}/api/app/packagemanager/certificate?package=$(basename "$CER")")
	echo "Response: $RESP"
	echo "Certificate installed."
	exit 0
fi

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

	# Xbox WinRT: ApplicationData.LocalFolder = LocalState subdirectory
	PATH_PARAM="\\LocalState\\${REMOTE_DIR}"
	[[ -z "$REMOTE_DIR" ]] && PATH_PARAM="\\LocalState"

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
# Default: deploy an .msix/.appx
# -----------------------------------------------------------------------
APPX="${1:-}"
if [[ -z "$APPX" ]]; then
	echo "Usage:" >&2
	echo "  $0 <path/to/xllama.msix>                           (deploy package)" >&2
	echo "  $0 install-cert <path/to/cert.cer>                 (trust certificate)" >&2
	echo "  $0 upload-file <local> <pfn> [remote-dir]          (upload file to LocalFolder)" >&2
	exit 1
fi
if [[ ! -f "$APPX" ]]; then
	echo "Error: file not found: $APPX" >&2
	exit 1
fi

APPX_NAME=$(basename "$APPX")
APPX_DIR=$(dirname "$APPX")

# Auto-install companion .cer if present alongside the .msix
CER_PATH="${APPX_DIR}/../xllama-test.cer"
if [[ ! -f "$CER_PATH" ]]; then
	CER_PATH="${APPX_DIR}/../../xllama-test.cer"
fi
if [[ -f "$CER_PATH" ]]; then
	echo "Found companion certificate: $(realpath "$CER_PATH")"
	"$0" install-cert "$(realpath "$CER_PATH")" || true
	echo ""
fi

echo "Deploying ${APPX_NAME} to Xbox at ${XBOX_IP} ..."

# NOTE: Xbox Device Portal requires ?package=<filename> query parameter.
RESPONSE=$(curl $CURL_AUTH \
	-X POST \
	-F "file=@${APPX};type=application/octet-stream" \
	"${BASE_URL}/api/app/packagemanager/package?package=${APPX_NAME}")

echo "Response: $RESPONSE"

if echo "$RESPONSE" | grep -qi '"Reason".*error\|failed'; then
	echo "Error: Device Portal returned an error." >&2
	exit 1
fi

echo "Upload complete. Monitor installation at: https://${XBOX_IP}:11443/#apps"

if command -v jq &>/dev/null; then
	echo "Polling installation status ..."
	for i in $(seq 1 24); do
		sleep 5
		STATUS=$(curl $CURL_AUTH "${BASE_URL}/api/app/packagemanager/state" 2>/dev/null || echo "{}")
		STATE=$(echo "$STATUS" | jq -r '.DeploymentProgressState // "unknown"' 2>/dev/null || echo "unknown")
		SUCCESS=$(echo "$STATUS" | jq -r '.Success // "null"' 2>/dev/null || echo "null")
		echo "  [${i}] state: $STATE  success: $SUCCESS"
		if [[ "$STATE" == "Ok" ]]; then
			echo "Installation succeeded."
			break
		fi
		if [[ "$SUCCESS" == "false" ]]; then
			REASON=$(echo "$STATUS" | jq -r '.Reason // "unknown"' 2>/dev/null || echo "unknown")
			echo "Installation failed: $REASON" >&2
			exit 1
		fi
	done
else
	echo "Tip: install jq to poll installation status automatically."
fi
