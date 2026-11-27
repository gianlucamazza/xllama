#!/usr/bin/env bash
# bench-d3d12-selftest.sh — d3d12 ggml backend selftest on the console
# (docs/gguf-gpu-decode.md). Uploads d3d12be.flag, waits for d3d12be-result.csv,
# and checks the selftest gate: every Q4_0 / Q4_K / Q6_K / Q5_K / IQ4_XS /
# IQ3_XXS case within the
# device's tolerance (kD3d12SelftestRelTol) of the CPU backend's q8 vec_dot —
# gate A (#312); D2a measured against ggml's dequantizers — and the decode
# (ncols=1) cases at >= 100 GB/s packed except IQ3_XXS at 55 GB/s (GPU
# timestamps; the CPU-side q8 quantization shows in the end-to-end bench, not
# here). The IQ3_XXS floor is measured from
# bench/results/d3d12-selftest-151804d-20261124.csv, not a product speed claim.
# --require-bias-add also requires a successful first ADD with cold scratch.
# --require-rope requires exactly the ten rope_* rows, in the order the selftest
# emits them (d3d12-rope.txt = 1), the first of them run with a cold scratch buffer (log line "rope cold_scratch=1"); they are
# correctness rows, so the decode bandwidth floor does not apply to them.
#
# Prerequisites: CI MSVC package (unified or llamacpp) installed on Series S.
# Usage:
#   source ~/.config/xllama/xbox-env
#   ./scripts/bench-d3d12-selftest.sh [--out PATH] [--force] [--require-bias-add] [--require-rope] [--allow-unavailable-mmv]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1090,SC1091
source "${XBOX_ENV:-$HOME/.config/xllama/xbox-env}"

OUT="${REPO_ROOT}/bench/results/d2a-d3d12-selftest.csv"
TIMEOUT_S=600
FORCE=0
verify_args=()
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
	--require-bias-add | --require-rope | --allow-unavailable-mmv)
		verify_args+=("$1")
		shift
		;;
	-h | --help)
		sed -n '2,/^set -/ { /^set -/!p; }' "$0"
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

# The producer can write its completion marker even if opening the CSV failed.
# Require a successful listing with neither old file before starting the run.
"${SCRIPT_DIR}/deploy.sh" stop-app "$PFN"
for name in d3d12be-result.csv d3d12be-result.csv.done; do
	"${SCRIPT_DIR}/deploy.sh" delete-file "$PFN" "$name" >/dev/null 2>&1 || true
done
"${SCRIPT_DIR}/deploy.sh" list-localstate "$PFN" | python3 -c '
import json, sys
items = json.load(sys.stdin)["Items"]
if not isinstance(items, list):
    sys.exit("invalid LocalState listing")
names = {i["Name"].lower() for i in items}
if {"d3d12be-result.csv", "d3d12be-result.csv.done"} & names:
    sys.exit("stale selftest receipt or completion marker")
print(int("xllama.log" in names))
' >"$TMP/log-present"

: >"$TMP/d3d12be.flag"
"${SCRIPT_DIR}/deploy.sh" upload-file "$TMP/d3d12be.flag" "$PFN" "" "d3d12be.flag"
if ((${#verify_args[@]})); then
	: >"$TMP/before.log"
	if [[ $(cat "$TMP/log-present") == 1 ]]; then
		"${SCRIPT_DIR}/deploy.sh" fetch-file "$PFN" xllama.log "$TMP/before.log"
	fi
fi
sleep 1
"${SCRIPT_DIR}/deploy.sh" start-app "$PFN"
echo "Waiting for d3d12be-result.csv.done (timeout ${TIMEOUT_S}s) ..."
done_fetched=0
deadline=$((SECONDS + TIMEOUT_S))
while ((SECONDS < deadline)); do
	if "${SCRIPT_DIR}/deploy.sh" fetch-file "$PFN" "d3d12be-result.csv.done" "$TMP/done" 2>/dev/null; then
		done_fetched=1
		break
	fi
	sleep 3
done
((done_fetched)) || {
	echo "timeout waiting for d3d12be-result.csv.done" >&2
	"${SCRIPT_DIR}/deploy.sh" get-log 2>&1 | tail -40 || true
	exit 1
}
"${SCRIPT_DIR}/deploy.sh" fetch-file "$PFN" "d3d12be-result.csv" "$TMP/r.csv"
mkdir -p "$(dirname "$OUT")"
tr -d '\r' <"$TMP/r.csv" >"$OUT"
echo "Wrote $OUT"
cat "$OUT"

if ((${#verify_args[@]})); then
	"${SCRIPT_DIR}/deploy.sh" fetch-file "$PFN" xllama.log "$TMP/full.log"
	python3 "${SCRIPT_DIR}/console_run_log.py" "$TMP/before.log" "$TMP/full.log" "$TMP/after.log" d3d12
	verify_args+=(--run-log "$TMP/after.log")
fi
python3 "${SCRIPT_DIR}/verify-d3d12-selftest.py" "$OUT" "${verify_args[@]}"
