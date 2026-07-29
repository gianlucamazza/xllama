#!/usr/bin/env bash
# install-latest-build.sh — download latest CI artifact and install on Xbox
#
# Usage:
#   source ~/.config/xllama/xbox-env
#   ./scripts/install-latest-build.sh [branch] [--bench] [--provision]
#
# Defaults to the current git branch if no branch argument is given.
# Requires: gh CLI (authenticated), jq, curl.
#
# By default the app launches into the normal UI. Pass --bench to also upload
# bench.flag so the next launch runs headless bench mode (delete
# LocalState\bench.flag via WDP before UI or validate-console.sh otherwise).
# MSIX uninstall wipes LocalState — re-provision models after a fresh install.
# --provision does it for the console-gate model set, after the registration has
# settled (provisioning too early lands in a container the OS then resets).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

UPLOAD_BENCH=false
PROVISION=false
BRANCH=""
for arg in "$@"; do
	case "$arg" in
	--bench) UPLOAD_BENCH=true ;;
	--provision) PROVISION=true ;;
	*) [[ -z "$BRANCH" ]] && BRANCH="$arg" ;;
	esac
done
# The model set the console gates need (validate-console.sh). Kept here because an
# MSIX uninstall wipes LocalState, so every install invalidates it — three gate
# runs were spent rediscovering that before this flag existed.
GATE_MODELS=(smollm2-360m-cpu-int4 smollm2-360m-dml-fp16-v2 lfm25-350m
	qwen25-coder-0.5b lfm25-1.2b-thinking)
BRANCH="${BRANCH:-$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)}"
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
	python3 -c 'import json,sys,os; d=json.load(sys.stdin); [print(p["PackageFullName"]) for p in d.get("InstalledPackages",[]) if any(a in p.get("PackageRelativeId","") for a in ("GianlucaMazza.xllama","VenereLabs.xllama"))]' 2>/dev/null || true)
if [[ -n "$CURRENT_PFN" ]]; then
	# May list several packages during the VenereLabs -> GianlucaMazza identity
	# migration (old and new family can be registered side by side).
	while IFS= read -r pfn; do
		[[ -z "$pfn" ]] && continue
		# --fail: WDP reports uninstall errors via HTTP status; without it a
		# failed DELETE would print "Uninstalled" and leave the old identity
		# registered while the install continues.
		if ! curl --basic -u "${XBOX_USER}:${XBOX_PASS}" -k -sS --fail \
			-H "X-CSRF-Token:${CSRF}" -X DELETE \
			"https://${XBOX_IP}:11443/api/app/packagemanager/package?package=${pfn}" >/dev/null; then
			echo "ERROR: failed to uninstall ${pfn}; aborting before install" >&2
			exit 1
		fi
		echo "  Uninstalled $pfn"
	done <<<"$CURRENT_PFN"
	sleep 2
else
	echo "  (not installed)"
fi

echo "==> Installing on Xbox at ${XBOX_IP} ..."
"${SCRIPT_DIR}/deploy.sh" "$MSIX"

NEW_PFN=$("${SCRIPT_DIR}/deploy.sh" pfn 2>/dev/null || echo "")

# Upload bench.flag only when --bench is passed, so a normal install launches
# straight into the UI (no leftover headless-bench state to clean up).
# LocalState may not be accessible via WDP right after a fresh install (before first app run).
# Strategy: try once; if it fails, start the app briefly to init LocalState, stop it, retry.
if [[ "$UPLOAD_BENCH" == true && -n "$NEW_PFN" ]]; then
	echo ""
	echo "==> Uploading bench.flag (--bench) ..."
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

if [[ "$PROVISION" == true ]]; then
	# Wait for the registration to settle first: files written into a container the
	# OS is still swapping are silently lost (provision-models.sh verifies, so this
	# shows up as a FAIL rather than a mystery, but the wait avoids the round trip).
	echo ""
	echo "==> Waiting 60s for the MSIX registration to settle before provisioning ..."
	sleep 60
	echo "==> Provisioning the console-gate model set ..."
	"${SCRIPT_DIR}/provision-models.sh" --force "${GATE_MODELS[@]}"
else
	echo ""
	echo "==> LocalState was wiped by the uninstall. The console gates need:"
	echo "      ./scripts/provision-models.sh --force ${GATE_MODELS[*]}"
	echo "    (or re-run this script with --provision)"
fi

echo ""
echo "==> Tailing log (Ctrl-C to stop) ..."
"${SCRIPT_DIR}/deploy.sh" get-log
