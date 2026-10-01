#!/usr/bin/env bash
# bench-gpustep.sh — GGUF GPU decode probe D1 on the console (docs/gguf-gpu-decode.md).
#
# Runs gpustep.flag (headless process) then gpustep-inproc.flag (inside the XAML
# process, D1d), merges both CSVs and prints the D1 ladder via
# xllama-cli --gpustep-verdict (the gates have one home: include/xllama/gpustep.h).
#
# Prerequisites: CI MSVC package installed on Series S; a host xllama-cli build
# (linux-release or linux-test) or XLLAMA_CLI=/path/to/xllama-cli.
# Usage:
#   source ~/.config/xllama/xbox-env
#   ./scripts/bench-gpustep.sh [--out bench/results/phase15-gpustep-d1.csv] [--force]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1090,SC1091
source "${XBOX_ENV:-$HOME/.config/xllama/xbox-env}"

OUT="${REPO_ROOT}/bench/results/phase15-gpustep-d1.csv"
TIMEOUT_S=600
FORCE=0

while [[ $# -gt 0 ]]; do
	case "$1" in
	--out)
		OUT="$2"
		shift 2
		;;
	--force)
		FORCE=1
		shift
		;;
	-h | --help)
		sed -n '2,12p' "$0"
		exit 0
		;;
	*)
		echo "unknown: $1" >&2
		exit 2
		;;
	esac
done

if [[ -e "$OUT" && "$FORCE" != 1 ]]; then
	echo "refusing to overwrite recorded $OUT. Pass --force to override." >&2
	exit 2
fi

CLI="${XLLAMA_CLI:-}"
if [[ -z "$CLI" ]]; then
	for c in "${REPO_ROOT}/build/linux-release/bin/xllama-cli" "${REPO_ROOT}/build/linux-test/bin/xllama-cli"; do
		if [[ -x "$c" ]]; then
			CLI="$c"
			break
		fi
	done
fi

: "${XBOX_IP:?source ~/.config/xllama/xbox-env}"
PFN=$("${SCRIPT_DIR}/deploy.sh" pfn 2>/dev/null || true)
[[ -n "$PFN" ]] || {
	echo "xllama not installed on console" >&2
	exit 1
}

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# run_mode <flag> <result-csv>: upload the flag, relaunch, wait for <csv>.done.
run_mode() {
	local flag="$1" csv="$2"
	"${SCRIPT_DIR}/deploy.sh" delete-file "$PFN" "${csv}.done" >/dev/null 2>&1 || true
	: >"$TMP/$flag"
	echo "Uploading $flag to $PFN ..."
	"${SCRIPT_DIR}/deploy.sh" upload-file "$TMP/$flag" "$PFN" "" "$flag"
	"${SCRIPT_DIR}/deploy.sh" stop-app || true
	sleep 1
	"${SCRIPT_DIR}/deploy.sh" start-app
	echo "Waiting for ${csv}.done (timeout ${TIMEOUT_S}s) ..."
	local deadline=$((SECONDS + TIMEOUT_S))
	rm -f "$TMP/done"
	while ((SECONDS < deadline)); do
		if "${SCRIPT_DIR}/deploy.sh" fetch-file "$PFN" "${csv}.done" "$TMP/done" 2>/dev/null; then
			break
		fi
		sleep 3
	done
	[[ -f "$TMP/done" ]] || {
		echo "timeout waiting for ${csv}.done" >&2
		"${SCRIPT_DIR}/deploy.sh" get-log 2>&1 | tail -40 || true
		exit 1
	}
	"${SCRIPT_DIR}/deploy.sh" fetch-file "$PFN" "$csv" "$TMP/$csv"
}

run_mode gpustep.flag gpustep-result.csv
run_mode gpustep-inproc.flag gpustep-inproc-result.csv
"${SCRIPT_DIR}/deploy.sh" stop-app || true

mkdir -p "$(dirname "$OUT")"
{
	head -n 1 "$TMP/gpustep-result.csv"
	tail -n +2 "$TMP/gpustep-result.csv"
	tail -n +2 "$TMP/gpustep-inproc-result.csv"
} | tr -d '\r' >"$OUT"
echo "Wrote $OUT"
cat "$OUT"

echo "--- D1 verdict ---"
if [[ -n "$CLI" ]]; then
	"$CLI" --gpustep-verdict "$OUT"
else
	echo "no host xllama-cli found (build linux-release or set XLLAMA_CLI); run:" >&2
	echo "  xllama-cli --gpustep-verdict $OUT" >&2
fi
