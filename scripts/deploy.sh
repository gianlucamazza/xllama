#!/usr/bin/env bash
# deploy.sh — upload an .appx package to the Xbox Device Portal
#
# Required env vars:
#   XBOX_IP    — console IP (e.g. 192.168.1.42)
#   XBOX_USER  — Device Portal username
#   XBOX_PASS  — Device Portal password
#
# Usage:
#   export XBOX_IP=192.168.1.42 XBOX_USER=devuser XBOX_PASS=...
#   ./scripts/deploy.sh xllama_0.1.0_x64.appx
#
# Device Portal reference:
#   https://learn.microsoft.com/en-us/windows/uwp/xbox-apps/device-portal-xbox

set -euo pipefail

APPX="${1:-}"
if [[ -z "$APPX" ]]; then
	echo "Usage: $0 <path/to/xllama.appx>" >&2
	exit 1
fi

if [[ ! -f "$APPX" ]]; then
	echo "Error: file not found: $APPX" >&2
	exit 1
fi

: "${XBOX_IP:?XBOX_IP environment variable is not set}"
: "${XBOX_USER:?XBOX_USER environment variable is not set}"
: "${XBOX_PASS:?XBOX_PASS environment variable is not set}"

BASE_URL="https://${XBOX_IP}:11443"

echo "Deploying $(basename "$APPX") to Xbox at ${XBOX_IP} ..."

# Upload the package via Device Portal REST API.
# -k: skip TLS cert check (Device Portal uses a self-signed cert).
# --digest: Xbox Device Portal uses HTTP Digest authentication.
RESPONSE=$(curl -sS \
	--digest \
	-u "${XBOX_USER}:${XBOX_PASS}" \
	-k \
	-X POST \
	-F "file=@${APPX};type=application/octet-stream" \
	"${BASE_URL}/api/app/packagemanager/package")

echo "Response: $RESPONSE"

# Check for error in JSON response
if echo "$RESPONSE" | grep -qi '"Reason"'; then
	echo "Error: Device Portal returned an error." >&2
	exit 1
fi

echo "Upload complete. Monitor installation status in Device Portal:"
echo "  https://${XBOX_IP}:11443/#apps"

# Optionally poll installation status (requires jq)
if command -v jq &>/dev/null; then
	echo "Polling installation status ..."
	for i in $(seq 1 12); do
		sleep 5
		STATUS=$(curl -sS --digest -u "${XBOX_USER}:${XBOX_PASS}" -k \
			"${BASE_URL}/api/app/packagemanager/state" 2>/dev/null || echo "{}")
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
