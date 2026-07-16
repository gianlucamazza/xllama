#!/usr/bin/env bash
# capture-demo-video.sh — drive autopilot on Xbox + WDP screenshots → demo mp4
#
# Usage:
#   source ~/.config/xllama/xbox-env
#   ./scripts/capture-demo-video.sh [out.mp4]
#
# Requires: installed xllama (autopilot), lfm25-350m + sd-turbo-fp16 in LocalState,
# ffmpeg, curl. Captures via Device Portal GET /ext/screenshot (1920x1080 PNG).

set -euo pipefail

: "${XBOX_IP:?XBOX_IP not set — run: source ~/.config/xllama/xbox-env}"
: "${XBOX_USER:?XBOX_USER not set}"
: "${XBOX_PASS:?XBOX_PASS not set}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEPLOY="${SCRIPT_DIR}/deploy.sh"
BASE_URL="https://${XBOX_IP}:11443"
CURL_AUTH=(--basic -u "${XBOX_USER}:${XBOX_PASS}" -k -sS)
OUT_MP4="${1:-${REPO_ROOT}/docs/screenshots/xllama-demo-v1.2.0.mp4}"

CSRF_TOKEN=$(curl "${CURL_AUTH[@]}" "${BASE_URL}/" -o /dev/null -D - 2>/dev/null |
	sed -n 's/.*[Cc][Ss][Rr][Ff]-[Tt]oken=\([^;[:space:]]*\).*/\1/p' |
	tr -d '\r' | head -n1)

PFN=$("${DEPLOY}" pfn 2>/dev/null)
[[ -z "$PFN" ]] && {
	echo "Error: xllama not installed" >&2
	exit 1
}

WORK=$(mktemp -d)
FRAMES="${WORK}/frames"
mkdir -p "$FRAMES"
trap 'rm -rf "$WORK"' EXIT

upload_file() {
	local local_path="$1"
	curl "${CURL_AUTH[@]}" -H "X-CSRF-Token:${CSRF_TOKEN}" -X POST \
		-F "file=@${local_path};type=application/octet-stream" \
		"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=%5CLocalState" \
		>/dev/null
}

delete_file() {
	local name="$1"
	curl "${CURL_AUTH[@]}" -H "X-CSRF-Token:${CSRF_TOKEN}" -X DELETE \
		"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=%5CLocalState&filename=${name}" \
		>/dev/null 2>&1 || true
}

fetch_file() {
	local name="$1" dest="$2"
	curl "${CURL_AUTH[@]}" -o "$dest" --fail \
		"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=%5CLocalState&filename=${name}" \
		2>/dev/null
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

grab_frame() {
	local n="$1"
	curl "${CURL_AUTH[@]}" -o "${FRAMES}/frame_$(printf '%05d' "$n").png" --fail \
		"${BASE_URL}/ext/screenshot" 2>/dev/null || return 1
}

echo "==> Seed settings (lfm25-350m, CPU routing)"
cat >"${WORK}/settings.json" <<'JSON'
{
  "system_prompt": "You are a helpful AI assistant.",
  "model": "lfm25-350m",
  "kv_reuse": true,
  "routing": 0,
  "gpu_model": "smollm2-360m-dml-fp16",
  "diffuse_taesd_vae": true,
  "sampling": {"temperature": 0.7, "top_p": 0.9, "top_k": 40, "repetition_penalty": 1.1, "n_predict": 80}
}
JSON
upload_file "${WORK}/settings.json"

echo "==> Autopilot script (chat x2 + image, no quit — keep UI for final frames)"
cat >"${WORK}/autopilot.json" <<'JSON'
{
  "total_timeout_s": 900,
  "actions": [
    {"op": "set_model", "name": "lfm25-350m"},
    {"op": "new_chat"},
    {"op": "send", "text": "Explain Xbox Series S in two sentences.", "timeout_s": 120},
    {"op": "send", "text": "Now say that in one line.", "timeout_s": 120},
    {"op": "generate_image", "prompt": "pixel art robot, simple colors", "steps": 1, "seed": 42, "timeout_s": 300}
  ]
}
JSON
printf 'go' >"${WORK}/autopilot.flag"
delete_file "autopilot-done.txt"
delete_file "diffuse-progress.txt"
delete_file "xllama.log"
upload_file "${WORK}/autopilot.json"
upload_file "${WORK}/autopilot.flag"

echo "==> Restart app + start screenshot loop (1 Hz)"
restart_app
sleep 3

FRAME=0
STOP_FLAG="${WORK}/stop_capture"
rm -f "$STOP_FLAG"
(
	while [[ ! -f "$STOP_FLAG" ]]; do
		if grab_frame "$FRAME"; then
			FRAME=$((FRAME + 1))
			printf '\r  frames: %d' "$FRAME" >&2
		fi
		sleep 1
	done
	echo "$FRAME" >"${WORK}/frame_count.txt"
) &
CAP_PID=$!

echo "==> Wait for autopilot-done.txt"
elapsed=0
marker=""
while ((elapsed < 600)); do
	if fetch_file "autopilot-done.txt" "${WORK}/done.txt" 2>/dev/null; then
		if grep -qE '^(ok|error:)' "${WORK}/done.txt" 2>/dev/null; then
			marker=$(cat "${WORK}/done.txt")
			echo ""
			echo "  autopilot: $marker (${elapsed}s)"
			break
		fi
	fi
	sleep 2
	elapsed=$((elapsed + 2))
done

# Hold UI a few more seconds for end frames
sleep 5
touch "$STOP_FLAG"
wait "$CAP_PID" 2>/dev/null || true

nframes=$(find "$FRAMES" -name 'frame_*.png' | wc -l)
echo "==> Captured ${nframes} frames"

if [[ "$nframes" -lt 5 ]]; then
	echo "Error: too few frames" >&2
	exit 1
fi

if [[ -n "$marker" && "$marker" != ok ]]; then
	echo "Warning: autopilot finished with: $marker" >&2
	"${DEPLOY}" get-log "$PFN" 2>/dev/null | tail -40 >&2 || true
fi

# Fetch generated image still (bonus end card if present)
if fetch_file "diffuse-out.png" "${WORK}/diffuse-out.png" 2>/dev/null; then
	echo "==> Got diffuse-out.png — append as end hold"
	# copy last few seconds worth as holds of the PNG
	last=$(find "$FRAMES" -name 'frame_*.png' | sort | tail -1)
	base=$(basename "$last" .png)
	num=${base#frame_}
	num=$((10#$num))
	for i in 1 2 3 4 5 6 7 8; do
		cp "${WORK}/diffuse-out.png" "${FRAMES}/frame_$(printf '%05d' $((num + i))).png" 2>/dev/null || true
	done
fi

echo "==> Encode mp4 (~1 fps source → 30 fps display via -r 30 with 1s per frame)"
# Each PNG held 1 second: use concat with duration, target ~60-90s by
# adjusting hold. If we have N frames at 1s each, duration ≈ N seconds.
# Cap: if too short, duplicate last frames; if too long, drop every other frame.
nframes=$(find "$FRAMES" -name 'frame_*.png' | wc -l)
hold=1
if ((nframes < 55)); then
	hold=2
elif ((nframes > 100)); then
	# subsample
	mkdir -p "${WORK}/sub"
	i=0
	for f in $(find "$FRAMES" -name 'frame_*.png' | sort); do
		if ((i % 2 == 0)); then
			cp "$f" "${WORK}/sub/"
		fi
		i=$((i + 1))
	done
	FRAMES="${WORK}/sub"
	nframes=$(find "$FRAMES" -name 'frame_*.png' | wc -l)
fi

LIST="${WORK}/list.txt"
: >"$LIST"
for f in $(find "$FRAMES" -name 'frame_*.png' | sort); do
	printf "file '%s'\nduration %s\n" "$f" "$hold" >>"$LIST"
done
# last frame needs a trailing file entry for concat demuxer
last=$(find "$FRAMES" -name 'frame_*.png' | sort | tail -1)
printf "file '%s'\n" "$last" >>"$LIST"

mkdir -p "$(dirname "$OUT_MP4")"
ffmpeg -y -f concat -safe 0 -i "$LIST" \
	-vf "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,fps=30,drawtext=fontfile=/usr/share/fonts/TTF/DejaVuSans.ttf:text='xllama v1.2.0 · Xbox Series S · local · no cloud':x=24:y=h-48:fontsize=22:fontcolor=white:borderw=2:bordercolor=black" \
	-c:v libx264 -pix_fmt yuv420p -movflags +faststart \
	"$OUT_MP4" 2>"${WORK}/ffmpeg.log"

ls -la "$OUT_MP4"
echo "==> Done: $OUT_MP4"
# duration probe
ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$OUT_MP4" 2>/dev/null || true
