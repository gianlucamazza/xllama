#!/usr/bin/env bash
# validate-op-repro.sh — drive the single-op CPU-vs-DML repro on the console (#111).
#
# For each variant produced by scripts/make-op-repro.py (simplified, skip,
# layernorm) this uploads repro.onnx + repro-input.bin, triggers the app's
# oprepro.flag headless mode (uwp/op-repro.cpp), pulls repro-out-cpu.bin /
# repro-out-dml.bin and prints a per-variant verdict:
#   MATCH    max|cpu-dml| <= tolerance  (kernel agrees with CPU)
#   MISMATCH otherwise                  (broken DML kernel isolated)
#
# Expected on the Series S driver (#91 root cause): simplified and skip
# MISMATCH, layernorm (control, MVN2 UseMean=true) MATCH.
#
# Usage:
#   source ~/.config/xllama/xbox-env
#   ./scripts/make-op-repro.py -o build/op-repro
#   ./scripts/validate-op-repro.sh [build/op-repro]

set -euo pipefail

: "${XBOX_IP:?XBOX_IP not set — run: source ~/.config/xllama/xbox-env}"
: "${XBOX_USER:?XBOX_USER not set}"
: "${XBOX_PASS:?XBOX_PASS not set}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEPLOY="${SCRIPT_DIR}/deploy.sh"
REPRO_ROOT="${1:-${REPO_ROOT}/build/op-repro}"
TOLERANCE="${TOLERANCE:-0.01}"

BASE_URL="https://${XBOX_IP}:11443"
CURL_AUTH=(--basic -u "${XBOX_USER}:${XBOX_PASS}" -k -sS)

CSRF_TOKEN=$(curl "${CURL_AUTH[@]}" "${BASE_URL}/" -o /dev/null -D - 2>/dev/null |
	sed -n 's/.*[Cc][Ss][Rr][Ff]-[Tt]oken=\([^;[:space:]]*\).*/\1/p' | tr -d '\r' | head -n1)
[[ -z "$CSRF_TOKEN" ]] && echo "Warning: no CSRF token — POST/DELETE may fail" >&2

PFN=$("${DEPLOY}" pfn 2>/dev/null)
[[ -z "$PFN" ]] && {
	echo "Error: xllama not found — deploy it first" >&2
	exit 2
}

TMPDIR_LOCAL=$(mktemp -d)
trap 'rm -rf "$TMPDIR_LOCAL"' EXIT

WDP_FILE="${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=%5CLocalState"

upload_file() {
	curl "${CURL_AUTH[@]}" -H "X-CSRF-Token:${CSRF_TOKEN}" -X POST \
		-F "file=@${1};type=application/octet-stream" "${WDP_FILE}" >/dev/null
}
delete_file() {
	curl "${CURL_AUTH[@]}" -H "X-CSRF-Token:${CSRF_TOKEN}" -X DELETE \
		"${WDP_FILE}&filename=$1" >/dev/null 2>&1 || true
}
download_file() {
	local code
	code=$(curl "${CURL_AUTH[@]}" -o "$2" -w "%{http_code}" \
		"${WDP_FILE}&filename=$1" 2>/dev/null || echo "000")
	[[ "$code" == "200" ]]
}
restart_app() {
	curl "${CURL_AUTH[@]}" -H "X-CSRF-Token:${CSRF_TOKEN}" -X DELETE \
		"${BASE_URL}/api/taskmanager/app?package=${PFN}" >/dev/null 2>&1 || true
	sleep 2
	local pfamily aumid
	# shellcheck disable=SC2001
	pfamily=$(echo "$PFN" | sed 's/_[0-9][0-9.]*_[^_]*__/_/')
	aumid=$(printf '%s!xllama' "$pfamily" | base64 -w0)
	curl "${CURL_AUTH[@]}" -H "X-CSRF-Token:${CSRF_TOKEN}" -X POST -d "" \
		"${BASE_URL}/api/taskmanager/app?appid=${aumid}" >/dev/null 2>&1 || true
}

overall=0
for d in "${REPRO_ROOT}"/*/; do
	d="${d%/}"
	variant="$(basename "$d")"
	[[ -f "$d/repro.onnx" ]] || {
		echo "skip ${variant}: $d/repro.onnx not found (run make-op-repro.py)"
		continue
	}
	echo "=== op-repro: ${variant} ==="

	for f in repro.done repro-out-cpu.bin repro-out-dml.bin repro.onnx repro-input.bin; do
		delete_file "$f"
	done
	upload_file "$d/repro.onnx"
	upload_file "$d/repro-input.bin"
	printf 'go' >"${TMPDIR_LOCAL}/oprepro.flag"
	upload_file "${TMPDIR_LOCAL}/oprepro.flag"
	restart_app

	elapsed=0
	until download_file "repro.done" "${TMPDIR_LOCAL}/repro.done"; do
		((elapsed >= 180)) && {
			echo "  FAIL: repro.done never appeared — check xllama.log"
			overall=1
			continue 2
		}
		sleep 5
		((elapsed += 5))
	done
	if [[ "$(cat "${TMPDIR_LOCAL}/repro.done")" != "ok" ]]; then
		echo "  FAIL: device reported: $(cat "${TMPDIR_LOCAL}/repro.done")"
		overall=1
		continue
	fi
	download_file "repro-out-cpu.bin" "${TMPDIR_LOCAL}/${variant}-cpu.bin" || {
		echo "  FAIL: no CPU output"
		overall=1
		continue
	}
	download_file "repro-out-dml.bin" "${TMPDIR_LOCAL}/${variant}-dml.bin" || {
		echo "  FAIL: no DML output"
		overall=1
		continue
	}

	python3 - "$d" "${TMPDIR_LOCAL}/${variant}-cpu.bin" "${TMPDIR_LOCAL}/${variant}-dml.bin" \
		"$TOLERANCE" <<'PY' || overall=1
import sys
import numpy as np
d, cpu_p, dml_p, tol = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
cpu = np.fromfile(cpu_p, dtype=np.float32)
dml = np.fromfile(dml_p, dtype=np.float32)
exp = np.fromfile(f"{d}/repro-expected.bin", dtype=np.float32)
assert cpu.size == dml.size, f"size mismatch cpu={cpu.size} dml={dml.size}"
n = exp.size  # first output; skip variant also emits the residual sum after it
diff = np.abs(cpu - dml)
host = np.abs(cpu[:n] - exp)
nmse = float(((cpu - dml) ** 2).mean() / max(float((cpu**2).mean()), 1e-30))
print(f"  cpu-vs-host-expected max|d|: {host.max():.6f} (CPU EP sanity)")
print(f"  cpu-vs-dml           max|d|: {diff.max():.6f}  nmse: {nmse:.3e}")
if diff.max() <= tol:
    print("  MATCH: DML agrees with CPU on this op")
else:
    print("  MISMATCH: broken DML kernel isolated on this op")
    sys.exit(1)
PY
done

exit $overall
