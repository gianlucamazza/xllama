#!/usr/bin/env bash
# probe-mic.sh — deploy mic.flag, fetch mic-result.json (Phase 16 WS-F, H16.6).
#
# Answers one question: can an AppContainer app on GameOS actually capture
# audio? Speak or make noise near the console while this runs — a silent room
# and a muted sandbox produce the same RMS, and only one of them is a verdict.
#
# Prerequisites: CI MSVC package installed on Series S (crossbuild may not launch).
# Usage:
#   source ~/.config/xllama/xbox-env
#   ./scripts/probe-mic.sh [--out bench/results/phase16-mic.json]
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
# shellcheck disable=SC1090,SC1091
source "${XBOX_ENV:-$HOME/.config/xllama/xbox-env}"

OUT="${REPO_ROOT}/bench/results/phase16-mic.json"
TIMEOUT_S=120

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
: >"$TMP/mic.flag"

# WDP cannot upload into an empty LocalState after a fresh install; seed it by
# launching once (same dance as bench-gpubw.sh).
upload_flag() {
	"${SCRIPT_DIR}/deploy.sh" upload-file "$TMP/mic.flag" "$PFN" "" "mic.flag"
}

echo "Uploading mic.flag to $PFN ..."
if ! upload_flag 2>/dev/null; then
	echo "  (upload failed — seeding LocalState via one UI launch...)"
	"${SCRIPT_DIR}/deploy.sh" stop-app || true
	"${SCRIPT_DIR}/deploy.sh" start-app || true
	sleep 8
	"${SCRIPT_DIR}/deploy.sh" stop-app || true
	sleep 2
	echo "  Retrying mic.flag upload ..."
	upload_flag
fi

"${SCRIPT_DIR}/deploy.sh" stop-app || true
sleep 1
"${SCRIPT_DIR}/deploy.sh" start-app
echo
echo ">>> MAKE NOISE NEAR THE CONSOLE NOW — the probe captures for 3 seconds. <<<"
echo
echo "Waiting for mic-result.json.done (timeout ${TIMEOUT_S}s) ..."
deadline=$((SECONDS + TIMEOUT_S))
while ((SECONDS < deadline)); do
	if "${SCRIPT_DIR}/deploy.sh" fetch-file "$PFN" "mic-result.json.done" "$TMP/done" 2>/dev/null; then
		break
	fi
	sleep 2
done
[[ -f "$TMP/done" ]] || {
	# The probe writes its marker on every exit path, including the ones that
	# threw. So a missing marker means the process died before finishing, which
	# per docs/uwp-constraints.md §10c is what an uncaught throw looks like from
	# here — NOT "the answer was no". Say so, because the difference decides
	# whether WS-F closes.
	echo "timeout waiting for mic-result.json.done" >&2
	echo "  The probe writes its marker even on failure, so this is a dead" >&2
	echo "  process, not a FAIL verdict. Check the tail below for the last" >&2
	echo "  line it managed to write (uwp-constraints.md §10c)." >&2
	"${SCRIPT_DIR}/deploy.sh" get-log 2>&1 | tail -40 || true
	exit 1
}
"${SCRIPT_DIR}/deploy.sh" fetch-file "$PFN" "mic-result.json" "$TMP/mic-result.json"
mkdir -p "$(dirname "$OUT")"
cp "$TMP/mic-result.json" "$OUT"
echo "Wrote $OUT"
cat "$OUT"

# Read the card's gate off the measured fields. The verdict deliberately has a
# fourth value: DeviceNotAvailable means no capture hardware is attached to this
# console, which is a statement about the room, not about the sandbox.
python3 - "$OUT" <<'PY'
import json, sys
from pathlib import Path

d = json.loads(Path(sys.argv[1]).read_text())
graph = d.get("graph_status")
node = d.get("input_node_status")
rms = float(d.get("rms", -1))
err = d.get("error") or ""

print("--- WS-F gate (H16.6: mic present AND 3 s capture with RMS > 1e-3) ---")
if err:
    verdict = f"INCONCLUSIVE — the probe threw: {err}"
elif not d.get("audiograph_type_present"):
    verdict = "FAIL — AudioGraph type absent; the API is not on this GameOS"
elif node == "AccessDenied":
    verdict = "FAIL — AppContainer refuses capture. Close WS-F; record in uwp-constraints.md"
elif "DeviceNotAvailable" in (graph, node):
    verdict = ("NOT A VERDICT — no capture device attached. Plug in a headset "
               "and rerun; do NOT close WS-F on this")
elif node != "Success":
    verdict = f"INCONCLUSIVE — input node status {node!r}"
elif rms > 1e-3:
    verdict = f"PASS — real capture (rms={rms:.6f})"
elif rms >= 0:
    verdict = (f"FAIL — graph opened but silent (rms={rms:.6f}). Confirm you "
               "made noise; a quiet room reads identically")
else:
    verdict = "INCONCLUSIVE — no samples recorded"

print(f"graph_status={graph} input_node_status={node} rms={rms}")
print(f"=> {verdict}")
PY
