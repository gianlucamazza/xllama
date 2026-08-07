#!/usr/bin/env bash
# bench-gpubw.sh — deploy gpubw.flag, fetch gpubw-result.csv (Phase 15 W3 #211).
#
# Prerequisites: CI MSVC package installed on Series S (crossbuild may not launch).
# Usage:
#   source ~/.config/xllama/xbox-env
#   ./scripts/bench-gpubw.sh [--out bench/results/phase15-gpubw.csv]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1090,SC1091
source "${XBOX_ENV:-$HOME/.config/xllama/xbox-env}"

OUT="${REPO_ROOT}/bench/results/phase15-gpubw.csv"
TIMEOUT_S=180

while [[ $# -gt 0 ]]; do
	case "$1" in
	--out)
		OUT="$2"
		shift 2
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

: "${XBOX_IP:?source ~/.config/xllama/xbox-env}"
PFN=$("${SCRIPT_DIR}/deploy.sh" pfn 2>/dev/null || true)
[[ -n "$PFN" ]] || {
	echo "xllama not installed on console" >&2
	exit 1
}

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
: >"$TMP/gpubw.flag"
echo "Uploading gpubw.flag to $PFN ..."
"${SCRIPT_DIR}/deploy.sh" upload-file "$TMP/gpubw.flag" "$PFN" "" "gpubw.flag"
"${SCRIPT_DIR}/deploy.sh" stop-app 2>/dev/null || true
sleep 1
"${SCRIPT_DIR}/deploy.sh" start-app
echo "Waiting for gpubw-result.csv.done (timeout ${TIMEOUT_S}s) ..."
deadline=$((SECONDS + TIMEOUT_S))
while ((SECONDS < deadline)); do
	if "${SCRIPT_DIR}/deploy.sh" fetch-file "$PFN" "gpubw-result.csv.done" "$TMP/done" 2>/dev/null; then
		break
	fi
	sleep 2
done
[[ -f "$TMP/done" ]] || {
	echo "timeout waiting for gpubw-result.csv.done" >&2
	"${SCRIPT_DIR}/deploy.sh" get-log 2>&1 | tail -40 || true
	exit 1
}
"${SCRIPT_DIR}/deploy.sh" fetch-file "$PFN" "gpubw-result.csv" "$TMP/gpubw-result.csv"
mkdir -p "$(dirname "$OUT")"
cp "$TMP/gpubw-result.csv" "$OUT"
echo "Wrote $OUT"
cat "$OUT"
# Print kill-gate line for logs
python3 - "$OUT" <<'PY'
import csv, sys
from pathlib import Path
p = Path(sys.argv[1])
rows = list(csv.DictReader(p.open()))
print("--- kill gate (100 GB/s) ---")
for row in rows:
    gbs = float(row.get("read_gbs") or 0)
    ok = row.get("checksum_ok") == "1" and row.get("d3d12_ran") == "1"
    verdict = "PASS" if ok and gbs >= 100.0 else "FAIL"
    print(f"read_gbs={gbs} checksum_ok={row.get('checksum_ok')} d3d12_ran={row.get('d3d12_ran')} => {verdict}")
PY
