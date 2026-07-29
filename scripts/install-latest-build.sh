#!/usr/bin/env bash
# install-latest-build.sh — download latest CI artifact and install on Xbox
#
# Usage:
#   source ~/.config/xllama/xbox-env
#   ./scripts/install-latest-build.sh [branch] [--bench] [--store]
#
# Defaults to the current git branch if no branch argument is given.
# Requires: gh CLI (authenticated), jq, curl.
#
# By default the app launches into the normal UI. Pass --bench to also upload
# bench.flag so the next launch runs headless bench mode (delete
# LocalState\bench.flag via WDP before UI or validate-console.sh otherwise).
# MSIX uninstall wipes LocalState — re-provision models after a fresh install.
#
# --store downloads the Store SKU artifact (xllama-appx-store) from a
# workflow_dispatch run that set store_sku=true. That SKU has no headless
# flags — --bench is rejected. Same package identity as dev for now, so this
# replaces the Dev Mode install on the console (docs/store-readiness.md).
# No local Windows VM: packages come from GitHub Actions windows-2022 only.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

UPLOAD_BENCH=false
STORE_SKU=false
BRANCH=""
for arg in "$@"; do
	case "$arg" in
	--bench) UPLOAD_BENCH=true ;;
	--store) STORE_SKU=true ;;
	*) [[ -z "$BRANCH" ]] && BRANCH="$arg" ;;
	esac
done
BRANCH="${BRANCH:-$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)}"
ARTIFACT_NAME="xllama-appx"
if [[ "$STORE_SKU" == true ]]; then
	ARTIFACT_NAME="xllama-appx-store"
fi
if [[ "$STORE_SKU" == true && "$UPLOAD_BENCH" == true ]]; then
	echo "ERROR: --bench is not supported on the Store SKU (headless flags compiled out)." >&2
	echo "Omit --bench, or install the dev artifact without --store." >&2
	exit 1
fi
WORK_DIR="/tmp/xllama-install-$$"

: "${XBOX_IP:?XBOX_IP not set — run: source ~/.config/xllama/xbox-env}"
: "${XBOX_USER:?XBOX_USER not set}"
: "${XBOX_PASS:?XBOX_PASS not set}"

cleanup() { rm -rf "$WORK_DIR"; }
trap cleanup EXIT

echo "==> Looking for latest successful build-uwp run on branch: ${BRANCH} (artifact: ${ARTIFACT_NAME})"

RUN_ID=""
if [[ "$STORE_SKU" == true ]]; then
	# Store SKU is only produced by workflow_dispatch -f store_sku=true; walk
	# recent successful runs and pick the first that lists the artifact (API,
	# no download probe).
	while IFS= read -r candidate; do
		[[ -z "$candidate" || "$candidate" == "null" ]] && continue
		if gh api "repos/gianlucamazza/xllama/actions/runs/${candidate}/artifacts" \
			--jq '.artifacts[].name' 2>/dev/null | grep -qx "$ARTIFACT_NAME"; then
			RUN_ID="$candidate"
			break
		fi
	done < <(gh run list \
		--repo gianlucamazza/xllama \
		--branch "$BRANCH" \
		--workflow build-uwp \
		--status success \
		--limit 20 \
		--json databaseId \
		--jq '.[].databaseId' 2>/dev/null || true)
else
	RUN_ID=$(gh run list \
		--repo gianlucamazza/xllama \
		--branch "$BRANCH" \
		--workflow build-uwp \
		--status success \
		--limit 1 \
		--json databaseId \
		--jq '.[0].databaseId' 2>/dev/null || echo "")
fi

if [[ -z "$RUN_ID" || "$RUN_ID" == "null" ]]; then
	echo "No successful build-uwp run with artifact '${ARTIFACT_NAME}' on branch '${BRANCH}'." >&2
	if [[ "$STORE_SKU" == true ]]; then
		echo "Trigger a Store SKU build (no Windows VM):" >&2
		echo "  gh workflow run build-uwp.yml -f store_sku=true --ref ${BRANCH}" >&2
	fi
	echo "Check: gh run list --branch $BRANCH --workflow build-uwp" >&2
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

echo ""
echo "==> Tailing log (Ctrl-C to stop) ..."
"${SCRIPT_DIR}/deploy.sh" get-log
