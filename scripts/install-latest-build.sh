#!/usr/bin/env bash
# install-latest-build.sh — download latest CI artifact and install on Xbox
#
# Usage:
#   source ~/.config/xllama/xbox-env
#   ./scripts/install-latest-build.sh [branch]
#
# Defaults to the current git branch if no branch argument is given.
# Requires: gh CLI (authenticated), jq, curl.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

BRANCH="${1:-$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)}"
ARTIFACT_NAME="xllama-appx"
WORK_DIR="/tmp/xllama-install-$$"

: "${XBOX_IP:?XBOX_IP not set — run: source ~/.config/xllama/xbox-env}"
: "${XBOX_USER:?XBOX_USER not set}"
: "${XBOX_PASS:?XBOX_PASS not set}"

cleanup() { rm -rf "$WORK_DIR"; }
trap cleanup EXIT

echo "==> Looking for latest successful build-uwp run on branch: ${BRANCH}"

RUN_ID=$(gh run list \
	--repo gianlucamazza/xllama \
	--branch "$BRANCH" \
	--workflow build-uwp \
	--status success \
	--limit 1 \
	--json databaseId \
	--jq '.[0].databaseId' 2>/dev/null || echo "")

if [[ -z "$RUN_ID" || "$RUN_ID" == "null" ]]; then
	echo "No successful build-uwp run found on branch '${BRANCH}'." >&2
	echo "Check: gh run list --branch $BRANCH" >&2
	exit 1
fi

echo "==> Run ID: ${RUN_ID}"

mkdir -p "$WORK_DIR"
echo "==> Downloading artifact '${ARTIFACT_NAME}' ..."
gh run download "$RUN_ID" \
	--repo gianlucamazza/xllama \
	--name "$ARTIFACT_NAME" \
	--dir "$WORK_DIR"

MSIX=$(find "$WORK_DIR" -name "*.msix" | sort | head -1)
if [[ -z "$MSIX" ]]; then
	echo "No .msix found in downloaded artifact." >&2
	ls -la "$WORK_DIR" >&2
	exit 1
fi

echo "==> Found MSIX: $MSIX"
echo "==> Installing on Xbox at ${XBOX_IP} ..."
"${SCRIPT_DIR}/deploy.sh" "$MSIX"

echo ""
echo "==> Waiting 3s then starting app ..."
sleep 3
"${SCRIPT_DIR}/deploy.sh" start-app || true

echo ""
echo "==> Tailing log (Ctrl-C to stop) ..."
"${SCRIPT_DIR}/deploy.sh" get-log
