#!/usr/bin/env bash
# capture-store-screenshots.sh — drive the autopilot to named UI states and grab
# a Device Portal screenshot at each one, for the Microsoft Store listing.
#
# Usage:
#   source ~/.config/xllama/xbox-env
#   ./scripts/capture-store-screenshots.sh [--out DIR] [--model NAME]
#
# Requires: an installed xllama build whose autopilot has the `mark` op, the
# named chat model and sd-turbo-fp16 in LocalState, curl. Output is PNG
# 1920x1080 straight from the console, which is inside the Store's accepted
# range (1366x768 - 3840x2160) with no resampling.
#
# Why `mark` and not sleeps: the host cannot see this process's UI and the app
# cannot reach the Device Portal, so "screenshot the finished answer" used to be
# a race between an autopilot action and a host-side sleep. The app parks on
# each named state and waits for us to delete its marker file, so every frame is
# taken at a state the app has confirmed it is in.
#
# States NOT captured, and why: the Settings and History panes are ContentDialogs
# opened by ShowSettings()/ShowHistory(), and reaching them from autopilot would
# need both a navigation op and a dismissal op. Everything here is reachable with
# ops that already exist.

set -euo pipefail

: "${XBOX_IP:?XBOX_IP not set — run: source ~/.config/xllama/xbox-env}"
: "${XBOX_USER:?XBOX_USER not set}"
: "${XBOX_PASS:?XBOX_PASS not set}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEPLOY="${SCRIPT_DIR}/deploy.sh"
BASE_URL="https://${XBOX_IP}:11443"
CURL_AUTH=(--basic -u "${XBOX_USER}:${XBOX_PASS}" -k -sS)

OUT_DIR="${REPO_ROOT}/docs/screenshots/store"
MODEL="lfm25-350m"
while [[ $# -gt 0 ]]; do
	case "$1" in
	--out)
		OUT_DIR="$2"
		shift 2
		;;
	--model)
		MODEL="$2"
		shift 2
		;;
	*)
		echo "Unknown argument: $1" >&2
		exit 1
		;;
	esac
done

# The model id is embedded in generated JSON and used as a catalogue key. Every
# id in uwp/models/manifest.json has this shape, and constraining it is more
# honest than trying to escape whatever a shell argument might contain.
if [[ ! "$MODEL" =~ ^[A-Za-z0-9._-]+$ ]]; then
	echo "Error: --model must match [A-Za-z0-9._-]+ (got '${MODEL}')" >&2
	exit 1
fi

CSRF_TOKEN=$(curl "${CURL_AUTH[@]}" "${BASE_URL}/" -o /dev/null -D - 2>/dev/null |
	sed -n 's/.*[Cc][Ss][Rr][Ff]-[Tt]oken=\([^;[:space:]]*\).*/\1/p' |
	tr -d '\r' | head -n1)
[[ -z "$CSRF_TOKEN" ]] && echo "Warning: no CSRF token — POST/DELETE may fail" >&2

PFN=$("${DEPLOY}" pfn 2>/dev/null)
[[ -z "$PFN" ]] && {
	echo "Error: xllama not installed" >&2
	exit 1
}

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

FILE_EP="${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=%5CLocalState"

upload_file() {
	curl "${CURL_AUTH[@]}" -H "X-CSRF-Token:${CSRF_TOKEN}" -X POST \
		-F "file=@${1};type=application/octet-stream" "$FILE_EP" >/dev/null
}

delete_file() {
	curl "${CURL_AUTH[@]}" -H "X-CSRF-Token:${CSRF_TOKEN}" -X DELETE \
		"${FILE_EP}&filename=${1}" >/dev/null 2>&1 || true
}

# Fetch only on a real 200. WDP happily returns a 404 body, and treating that
# body as a marker label would make us screenshot a state the app never reached.
fetch_file_200() {
	local name="$1" dest="$2" code
	code=$(curl "${CURL_AUTH[@]}" -o "$dest" -w "%{http_code}" \
		"${FILE_EP}&filename=${name}" 2>/dev/null || echo "000")
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

grab_screenshot() {
	curl "${CURL_AUTH[@]}" -o "$1" --fail "${BASE_URL}/ext/screenshot" >/dev/null 2>&1
}

# Block until the app parks on the mark we expect. Echoes its label.
# Returns 1 on timeout, 2 if the run ended, 3 if the label is not the expected one.
#
# The label is read back from the device, so it is input this script does not
# control: it becomes a filename, and it decides which state a frame is
# attributed to. Both matter. A marker left behind by another tool, or a label
# with a path separator in it, would otherwise write outside OUT_DIR or file the
# wrong screenshot under a listing state. So it is checked against the exact
# label expected next, not merely sanitised.
wait_for_mark() {
	local timeout_s="$1" expected="$2" out="${WORK}/mark.txt" label
	local deadline=$((SECONDS + timeout_s))
	while ((SECONDS < deadline)); do
		if fetch_file_200 "autopilot-mark.txt" "$out"; then
			label=$(tr -d '\r\n' <"$out")
			if [[ -n "$label" ]]; then
				if [[ "$label" != "$expected" ]]; then
					echo "  Expected mark '${expected}', device says '${label}'" >&2
					return 3
				fi
				echo "$label"
				return 0
			fi
		fi
		# An autopilot that failed writes its marker and does NOT exit, so
		# without this a failed run would sit here for the full timeout of every
		# remaining state before anyone found out.
		if fetch_file_200 "autopilot-done.txt" "${WORK}/done.txt" &&
			grep -qE '^(ok|error:)' "${WORK}/done.txt"; then
			return 2
		fi
		sleep 2
	done
	return 1
}

# The app has confirmed the state; take the frame, then release it.
release_mark() { delete_file "autopilot-mark.txt"; }

# The listing states, in order: what the app is made to do, and the label that
# becomes the filename. One source — the JSON below is generated from this, and
# the host waits for exactly this many parks. Deriving the count by grepping the
# generated JSON would be two representations of one fact, and the failure would
# be silent: one too many and the run hangs for a full timeout after finishing.
STATES=(
	"01-chat-answer|send|What can you do on this console?"
	"02-chat-multiturn|send|Now in one line."
	"03-image|generate_image|pixel art robot, simple colors"
)

json_escape() { printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'; }

{
	printf '{\n  "total_timeout_s": 1500,\n  "actions": [\n'
	printf '    {"op": "set_model", "name": "%s"},\n' "$(json_escape "$MODEL")"
	printf '    {"op": "new_chat"}'
	for state in "${STATES[@]}"; do
		IFS='|' read -r label op payload <<<"$state"
		case "$op" in
		send)
			printf ',\n    {"op": "send", "text": "%s", "timeout_s": 300}' \
				"$(json_escape "$payload")"
			;;
		generate_image)
			printf ',\n    {"op": "generate_image", "prompt": "%s", "steps": 1, "seed": 42, "timeout_s": 600}' \
				"$(json_escape "$payload")"
			;;
		*)
			echo "Unknown op in STATES: $op" >&2
			exit 1
			;;
		esac
		printf ',\n    {"op": "mark", "label": "%s", "timeout_s": 180}' "$(json_escape "$label")"
	done
	printf '\n  ]\n}\n'
} >"${WORK}/autopilot.json"

# The app rejects malformed JSON with "bad autopilot.json", which would surface
# as a mystery timeout on the host. Catch it here, where the message is useful.
if command -v jq >/dev/null 2>&1; then
	jq -e . "${WORK}/autopilot.json" >/dev/null || {
		echo "Error: generated autopilot.json is not valid JSON" >&2
		cat "${WORK}/autopilot.json" >&2
		exit 1
	}
fi

printf 'go' >"${WORK}/autopilot.flag"

echo "==> Model ${MODEL}, output ${OUT_DIR}"
mkdir -p "$OUT_DIR"

# A marker or a done-file left by an earlier run would release the first mark
# instantly, and we would screenshot the splash screen.
delete_file "autopilot-mark.txt"
delete_file "autopilot-done.txt"
delete_file "diffuse-progress.txt"

upload_file "${WORK}/autopilot.json"
upload_file "${WORK}/autopilot.flag"
restart_app

echo "==> Waiting for ${#STATES[@]} states"

captured=0
for state in "${STATES[@]}"; do
	expected="${state%%|*}"
	rc=0
	label=$(wait_for_mark 900 "$expected") || rc=$?
	case "$rc" in
	0) ;;
	2)
		echo "  The run ended before '${expected}'" >&2
		break
		;;
	3)
		echo "  Refusing to attribute a frame to the wrong state" >&2
		break
		;;
	*)
		echo "  No mark within 900s — the app is stuck or the run failed" >&2
		break
		;;
	esac
	dest="${OUT_DIR}/${label}.png"
	if grab_screenshot "$dest"; then
		echo "  ${label} -> ${dest}"
		captured=$((captured + 1))
	else
		echo "  ${label}: GET /ext/screenshot failed" >&2
		rm -f "$dest"
	fi
	release_mark
done

# The script has no `quit`, so the app stays up and writes the marker itself.
marker=""
deadline=$((SECONDS + 120))
while ((SECONDS < deadline)); do
	if fetch_file_200 "autopilot-done.txt" "${WORK}/done.txt" &&
		grep -qE '^(ok|error:)' "${WORK}/done.txt"; then
		marker=$(cat "${WORK}/done.txt")
		break
	fi
	sleep 5
done
echo "==> autopilot: ${marker:-<no marker>}"
if [[ "$marker" != ok ]]; then
	"${DEPLOY}" get-log "$PFN" 2>/dev/null | tail -40 >&2 || true
fi

echo "==> Captured ${captured} screenshots"
if ((captured == 0)); then
	echo "Error: nothing captured" >&2
	exit 1
fi

# Store rejects anything outside 1366x768 - 3840x2160, so verify rather than
# assume the console gave us what it usually gives us.
if command -v identify >/dev/null 2>&1; then
	for f in "${OUT_DIR}"/*.png; do
		echo "  $(basename "$f"): $(identify -format '%wx%h' "$f")"
	done
else
	echo "  (install imagemagick to verify dimensions)"
fi
