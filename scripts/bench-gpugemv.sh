#!/usr/bin/env bash
# bench-gpugemv.sh — deploy gpugemv.flag, fetch gpugemv-result.csv (Phase 15 H6.1 #228).
#
# Prerequisites: CI MSVC package installed on Series S (crossbuild may not launch).
# Usage:
#   source ~/.config/xllama/xbox-env
#   ./scripts/bench-gpugemv.sh [--out bench/results/phase15-gpugemv.csv]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1090,SC1091
source "${XBOX_ENV:-$HOME/.config/xllama/xbox-env}"

OUT="${REPO_ROOT}/bench/results/phase15-gpugemv.csv"
TIMEOUT_S=300

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
: >"$TMP/gpugemv.flag"

upload_flag() {
	"${SCRIPT_DIR}/deploy.sh" upload-file "$TMP/gpugemv.flag" "$PFN" "" "gpugemv.flag"
}

echo "Uploading gpugemv.flag to $PFN ..."
if ! upload_flag 2>/dev/null; then
	echo "  (upload failed — seeding LocalState via one UI launch...)"
	"${SCRIPT_DIR}/deploy.sh" stop-app || true
	"${SCRIPT_DIR}/deploy.sh" start-app || true
	sleep 8
	"${SCRIPT_DIR}/deploy.sh" stop-app || true
	sleep 2
	echo "  Retrying gpugemv.flag upload ..."
	upload_flag
fi

"${SCRIPT_DIR}/deploy.sh" stop-app || true
sleep 1
"${SCRIPT_DIR}/deploy.sh" start-app
echo "Waiting for gpugemv-result.csv.done (timeout ${TIMEOUT_S}s) ..."
deadline=$((SECONDS + TIMEOUT_S))
while ((SECONDS < deadline)); do
	if "${SCRIPT_DIR}/deploy.sh" fetch-file "$PFN" "gpugemv-result.csv.done" "$TMP/done" 2>/dev/null; then
		break
	fi
	sleep 2
done
[[ -f "$TMP/done" ]] || {
	echo "timeout waiting for gpugemv-result.csv.done" >&2
	"${SCRIPT_DIR}/deploy.sh" get-log 2>&1 | tail -40 || true
	exit 1
}
"${SCRIPT_DIR}/deploy.sh" fetch-file "$PFN" "gpugemv-result.csv" "$TMP/gpugemv-result.csv"
mkdir -p "$(dirname "$OUT")"
cp "$TMP/gpugemv-result.csv" "$OUT"
echo "Wrote $OUT"
cat "$OUT"
python3 - "$OUT" <<'PY'
import csv, sys
from pathlib import Path
p = Path(sys.argv[1])
rows = list(csv.DictReader(p.open()))
print("--- soft gates G1 (correctness) / G2 (packed >= 40 GB/s) ---")
for row in rows:
    gbs = float(row.get("packed_gbs") or 0)
    err = float(row.get("max_abs_err") or 1e9)
    ok = row.get("checksum_ok") == "1" and row.get("d3d12_ran") == "1" and err <= 1e-2
    g1 = "PASS" if ok else "FAIL"
    g2 = "PASS" if ok and gbs >= 40.0 else "FAIL"
    print(f"packed_gbs={gbs} max_abs_err={err} checksum_ok={row.get('checksum_ok')} d3d12_ran={row.get('d3d12_ran')} => G1={g1} G2={g2}")
PY
