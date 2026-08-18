#!/usr/bin/env bash
# bench-diskbw.sh — deploy diskbw.flag, fetch diskbw-result.csv (SSD-inference assessment).
#
# Prerequisites: CI MSVC package installed on Series S (crossbuild may not launch).
# Usage:
#   source ~/.config/xllama/xbox-env
#   ./scripts/bench-diskbw.sh [--out bench/results/diskbw.csv]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1090,SC1091
source "${XBOX_ENV:-$HOME/.config/xllama/xbox-env}"

OUT="${REPO_ROOT}/bench/results/diskbw.csv"
# 4 GiB test-file write + 12 read passes on a ~2 GB/s device: allow ten minutes.
TIMEOUT_S=600

while [[ $# -gt 0 ]]; do
	case "$1" in
	--out)
		if [[ $# -lt 2 ]]; then
			echo "--out requires a path" >&2
			exit 2
		fi
		OUT="$2"
		shift 2
		;;
	-h | --help)
		sed -n '2,8p' "$0"
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
: >"$TMP/diskbw.flag"

# WDP cannot upload into an empty LocalState after a fresh install
# ("File move failed" / path not found). Seed by launching once so the app
# creates xllama.log (and often models/), then stop and upload the flag.
upload_flag() {
	"${SCRIPT_DIR}/deploy.sh" upload-file "$TMP/diskbw.flag" "$PFN" "" "diskbw.flag"
}

echo "Uploading diskbw.flag to $PFN ..."
if ! upload_flag 2>/dev/null; then
	echo "  (upload failed — seeding LocalState via one UI launch...)"
	"${SCRIPT_DIR}/deploy.sh" stop-app || true
	"${SCRIPT_DIR}/deploy.sh" start-app || true
	sleep 8
	"${SCRIPT_DIR}/deploy.sh" stop-app || true
	sleep 2
	echo "  Retrying diskbw.flag upload ..."
	upload_flag
fi

"${SCRIPT_DIR}/deploy.sh" stop-app || true
sleep 1
# A marker left by a previous run would end the wait loop immediately and
# fetch a stale CSV. Delete it and verify it is gone before launching.
"${SCRIPT_DIR}/deploy.sh" delete-file "$PFN" "diskbw-result.csv.done" 2>/dev/null || true
if "${SCRIPT_DIR}/deploy.sh" fetch-file "$PFN" "diskbw-result.csv.done" "$TMP/stale-done" 2>/dev/null; then
	echo "stale diskbw-result.csv.done still present on console — aborting" >&2
	exit 1
fi
"${SCRIPT_DIR}/deploy.sh" start-app
echo "Waiting for diskbw-result.csv.done (timeout ${TIMEOUT_S}s) ..."
deadline=$((SECONDS + TIMEOUT_S))
while ((SECONDS < deadline)); do
	if "${SCRIPT_DIR}/deploy.sh" fetch-file "$PFN" "diskbw-result.csv.done" "$TMP/done" 2>/dev/null; then
		break
	fi
	sleep 5
done
[[ -f "$TMP/done" ]] || {
	echo "timeout waiting for diskbw-result.csv.done" >&2
	"${SCRIPT_DIR}/deploy.sh" get-log 2>&1 | tail -40 || true
	exit 1
}
"${SCRIPT_DIR}/deploy.sh" fetch-file "$PFN" "diskbw-result.csv" "$TMP/diskbw-result.csv"
mkdir -p "$(dirname "$OUT")"
cp "$TMP/diskbw-result.csv" "$OUT"
echo "Wrote $OUT"
cat "$OUT"
# Print the expert-streaming read gate for logs (threshold from
# docs/ssd-inference-assessment.md — random-read floor, not a product gate).
python3 - "$OUT" <<'PY'
import csv, sys
from pathlib import Path
p = Path(sys.argv[1])
rows = list(csv.DictReader(p.open()))
print("--- expert-streaming gate (rnd read >= 1.5 GB/s, first pass) ---")
for row in rows:
    if row.get("pattern") != "rnd":
        continue
    gbs = float(row.get("read_gbs_first") or 0)
    verdict = "PASS" if gbs >= 1.5 else "FAIL"
    print(f"threads={row.get('threads')} unbuffered={row.get('unbuffered')} "
          f"read_gbs_first={gbs} => {verdict}")
PY
