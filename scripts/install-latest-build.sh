#!/usr/bin/env bash
# install-latest-build.sh — download latest CI artifact and install on Xbox
#
# Usage:
#   source ~/.config/xllama/xbox-env
#   ./scripts/install-latest-build.sh [branch]
#
# Defaults to the current git branch if no branch argument is given.
# Requires: gh CLI (authenticated), jq, curl.
#
# Side effect: uploads bench.flag so the next launch runs headless bench mode.
# Delete LocalState\bench.flag (via WDP) before UI or validate-console.sh runs.
# MSIX uninstall wipes LocalState — re-provision models after a fresh install.

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

# Uninstall current version first to avoid MSIX update requiring space for both old+new.
echo "==> Uninstalling current version (if any) ..."
CSRF=$(curl --basic -u "${XBOX_USER}:${XBOX_PASS}" -k -sS "https://${XBOX_IP}:11443/" \
	-o /dev/null -D - 2>/dev/null |
	sed -n 's/.*[Cc][Ss][Rr][Ff]-[Tt]oken=\([^;[:space:]]*\).*/\1/p' | tr -d '\r' | head -1)
CURRENT_PFN=$(curl --basic -u "${XBOX_USER}:${XBOX_PASS}" -k -sS \
	"https://${XBOX_IP}:11443/api/app/packagemanager/packages" |
	python3 -c 'import json,sys,os; d=json.load(sys.stdin); [print(p["PackageFullName"]) for p in d.get("InstalledPackages",[]) if "VenereLabs.xllama" in p.get("PackageRelativeId","")]' 2>/dev/null || true)
if [[ -n "$CURRENT_PFN" ]]; then
	curl --basic -u "${XBOX_USER}:${XBOX_PASS}" -k -sS \
		-H "X-CSRF-Token:${CSRF}" -X DELETE \
		"https://${XBOX_IP}:11443/api/app/packagemanager/package?package=${CURRENT_PFN}" >/dev/null
	echo "  Uninstalled $CURRENT_PFN"
	sleep 2
else
	echo "  (not installed)"
fi

echo "==> Installing on Xbox at ${XBOX_IP} ..."
"${SCRIPT_DIR}/deploy.sh" "$MSIX"

NEW_PFN=$("${SCRIPT_DIR}/deploy.sh" pfn 2>/dev/null || echo "")

# Upload bench.flag so bench mode fires on next launch.
# LocalState may not be accessible via WDP right after a fresh install (before first app run).
# Strategy: try once; if it fails, start the app briefly to init LocalState, stop it, retry.
if [[ -n "$NEW_PFN" ]]; then
	echo ""
	echo "==> Uploading bench.flag ..."
	touch /tmp/bench.flag
	if ! "${SCRIPT_DIR}/deploy.sh" upload-file /tmp/bench.flag "$NEW_PFN" "" 2>/dev/null; then
		echo "  (first upload failed — starting app briefly to init LocalState...)"
		"${SCRIPT_DIR}/deploy.sh" start-app "$NEW_PFN" || true
		sleep 5
		"${SCRIPT_DIR}/deploy.sh" stop-app "$NEW_PFN" || true
		sleep 2
		echo "  Retrying bench.flag upload ..."
		"${SCRIPT_DIR}/deploy.sh" upload-file /tmp/bench.flag "$NEW_PFN" "" ||
			echo "  WARNING: bench.flag upload failed; upload manually with: deploy.sh upload-file /tmp/bench.flag $NEW_PFN ''"
	fi
fi

echo ""
echo "==> Starting app ..."
sleep 2
"${SCRIPT_DIR}/deploy.sh" start-app || true

echo ""
echo "==> Tailing log (Ctrl-C to stop) ..."
"${SCRIPT_DIR}/deploy.sh" get-log
