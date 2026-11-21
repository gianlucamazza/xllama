#!/usr/bin/env bash
# bench-d3d12-selftest.sh — d3d12 ggml backend selftest on the console
# (docs/gguf-gpu-decode.md). Uploads d3d12be.flag, waits for d3d12be-result.csv,
# and checks the selftest gate: every Q4_0 / Q4_K / Q6_K case within the
# device's tolerance (kD3d12SelftestRelTol) of the CPU backend's q8 vec_dot —
# gate A (#312); D2a measured against ggml's dequantizers — and the decode
# (ncols=1) cases at >= 100 GB/s packed (GPU timestamps; the CPU-side q8
# quantization shows in the end-to-end bench, not here).
# --require-bias-add also requires a successful first ADD with cold scratch.
#
# Prerequisites: CI MSVC package (unified or llamacpp) installed on Series S.
# Usage:
#   source ~/.config/xllama/xbox-env
#   ./scripts/bench-d3d12-selftest.sh [--out PATH] [--force] [--require-bias-add]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1090,SC1091
source "${XBOX_ENV:-$HOME/.config/xllama/xbox-env}"

OUT="${REPO_ROOT}/bench/results/d2a-d3d12-selftest.csv"
TIMEOUT_S=600
FORCE=0
REQUIRE_BIAS_ADD=0
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
	--require-bias-add)
		REQUIRE_BIAS_ADD=1
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
if [[ -e "$OUT" && "$FORCE" != 1 ]]; then
	echo "refusing to overwrite recorded $OUT. Pass --force to override." >&2
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

"${SCRIPT_DIR}/deploy.sh" delete-file "$PFN" "d3d12be-result.csv.done" >/dev/null 2>&1 || true
: >"$TMP/d3d12be.flag"
"${SCRIPT_DIR}/deploy.sh" upload-file "$TMP/d3d12be.flag" "$PFN" "" "d3d12be.flag"
"${SCRIPT_DIR}/deploy.sh" stop-app || true
log_lines=0
if ((REQUIRE_BIAS_ADD)); then
	"${SCRIPT_DIR}/deploy.sh" get-log "$PFN" >"$TMP/before.log"
	log_lines=$(wc -l <"$TMP/before.log")
fi
sleep 1
"${SCRIPT_DIR}/deploy.sh" start-app
echo "Waiting for d3d12be-result.csv.done (timeout ${TIMEOUT_S}s) ..."
deadline=$((SECONDS + TIMEOUT_S))
while ((SECONDS < deadline)); do
	if "${SCRIPT_DIR}/deploy.sh" fetch-file "$PFN" "d3d12be-result.csv.done" "$TMP/done" 2>/dev/null; then
		break
	fi
	sleep 3
done
[[ -f "$TMP/done" ]] || {
	echo "timeout waiting for d3d12be-result.csv.done" >&2
	"${SCRIPT_DIR}/deploy.sh" get-log 2>&1 | tail -40 || true
	exit 1
}
"${SCRIPT_DIR}/deploy.sh" fetch-file "$PFN" "d3d12be-result.csv" "$TMP/r.csv"
mkdir -p "$(dirname "$OUT")"
tr -d '\r' <"$TMP/r.csv" >"$OUT"
echo "Wrote $OUT"
cat "$OUT"

if ((REQUIRE_BIAS_ADD)); then
	"${SCRIPT_DIR}/deploy.sh" get-log "$PFN" | tail -n "+$((log_lines + 1))" >"$TMP/after.log"
	if ! grep -aF '[xllama] d3d12be: bias ADD cold_scratch=1' "$TMP/after.log" >/dev/null ||
		! grep -aE '^\[xllama\] d3d12be: add_f32 .* ok=1 err=-' "$TMP/after.log" >/dev/null; then
		echo "GATE=FAIL (first cold bias ADD success missing from this run)" >&2
		exit 1
	fi
fi

python3 - "$OUT" "$REQUIRE_BIAS_ADD" <<'PY'
import csv, math, sys
rows = list(csv.DictReader(open(sys.argv[1])))
bad = [r for r in rows if r["ok"] != "1" or r["d3d12_ran"] != "1" or
       any(not math.isfinite(float(r[k])) or float(r[k]) < 0
           for k in ("rel_err", "gpu_ms", "packed_gbs"))]
slow = [r for r in rows if r["ncols"] == "1" and float(r["packed_gbs"]) < 100.0]
print(f"--- selftest gate: {len(rows) - len(bad)}/{len(rows)} cases correct; "
      f"{len(slow)} decode case(s) under 100 GB/s ---")
for r in bad:
    print(f"FAIL {r['type']} n={r['n']} k={r['k']} ncols={r['ncols']} rel_err={r['rel_err']} {r['error']}")
for r in slow:
    print(f"SLOW {r['type']} n={r['n']} k={r['k']} gbs={r['packed_gbs']}")
require_bias_add = sys.argv[2:] == ["1"]
bias_add_first = bool(rows) and rows[0]["type"] == "add_f32"
if require_bias_add and not bias_add_first:
    print("FAIL the first selftest row must be add_f32")
passed = bool(rows) and not bad and not slow and (not require_bias_add or bias_add_first)
print("GATE=" + ("PASS" if passed else "FAIL"))
sys.exit(0 if passed else 1)
PY
