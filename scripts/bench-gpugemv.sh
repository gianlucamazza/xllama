#!/usr/bin/env bash
# bench-gpugemv.sh — deploy gpugemv.flag, fetch gpugemv-result.csv (Phase 15 H6.2 #228).
#
# Prerequisites: CI MSVC package installed on Series S (crossbuild may not launch).
# Usage:
#   source ~/.config/xllama/xbox-env
#   ./scripts/bench-gpugemv.sh [--out bench/results/phase15-gpugemv-h62.csv]
#   ./scripts/bench-gpugemv.sh --out bench/results/phase15-gpugemv.csv --force
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1090,SC1091
source "${XBOX_ENV:-$HOME/.config/xllama/xbox-env}"

OUT="${REPO_ROOT}/bench/results/phase15-gpugemv-h62.csv"
TIMEOUT_S=300
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
		sed -n '2,14p' "$0"
		exit 0
		;;
	*)
		echo "unknown: $1" >&2
		exit 2
		;;
	esac
done

# H6.1 evidence is immutable. Refuse the old path unless --force.
h61="${REPO_ROOT}/bench/results/phase15-gpugemv.csv"
out_abs="$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$OUT")"
h61_abs="$(python3 -c 'import os,sys; print(os.path.realpath(sys.argv[1]))' "$h61")"
if [[ "$out_abs" == "$h61_abs" && "$FORCE" != 1 ]]; then
	echo "refusing to write $OUT (H6.1 CSV). Pass --force to override." >&2
	exit 2
fi

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
import csv, statistics, sys
from collections import defaultdict
from pathlib import Path

p = Path(sys.argv[1])
rows = list(csv.DictReader(p.open()))
print("--- G1 (correctness) / G2=40 / K1=8 (G1 required) ---")
by_kernel = defaultdict(list)
for row in rows:
    kernel = row.get("kernel") or "naive"
    try:
        run_index = int(row.get("run_index") or 0)
    except ValueError:
        run_index = 0
    if run_index < 1 or run_index > 3:
        continue
    gbs = float(row.get("packed_gbs") or 0)
    err = float(row.get("max_abs_err") or 1e9)
    ok = row.get("checksum_ok") == "1" and row.get("d3d12_ran") == "1" and err <= 1e-2
    g1 = "PASS" if ok else "FAIL"
    g2 = "PASS" if ok and gbs >= 40.0 else "FAIL"
    print(
        f"kernel={kernel} run_index={run_index} packed_gbs={gbs} "
        f"max_abs_err={err} checksum_ok={row.get('checksum_ok')} "
        f"d3d12_ran={row.get('d3d12_ran')} gpu_timestamp={row.get('gpu_timestamp')} "
        f"=> G1={g1} G2={g2}"
    )
    by_kernel[kernel].append((gbs, ok))


def ladder(median, g1_all3):
    if not g1_all3:
        return "NotAVerdict"
    if median < 8.0:
        return "K1"
    if median < 40.0:
        return "K2"
    return "K3"


denser_best = None
for kernel, runs in by_kernel.items():
    gbs = [g for g, _ in runs]
    g1_all3 = len(runs) >= 3 and all(ok for _, ok in runs)
    med = statistics.median(gbs) if gbs else 0.0
    lad = ladder(med, g1_all3)
    print(f"median kernel={kernel} packed_gbs={med} g1_all3={int(g1_all3)} ladder={lad}")
    if kernel == "naive":
        continue
    if g1_all3 and (denser_best is None or med > denser_best[0]):
        denser_best = (med, lad, kernel)

if denser_best is None:
    print("campaign_verdict=NotAVerdict")
else:
    print(f"campaign_verdict={denser_best[1]} kernel={denser_best[2]} median={denser_best[0]}")
PY
