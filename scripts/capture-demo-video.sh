#!/usr/bin/env bash
# capture-demo-video.sh — drive the autopilot on Xbox and record the screen
#
# Usage:
#   source ~/.config/xllama/xbox-env
#   ./scripts/capture-demo-video.sh [--script demo/demo-script.json] [--fps 10]
#                                   [--hold 4] [--out DIR]
#
# Requires: an installed xllama with the `show_pane` autopilot op, the models the
# script names already in LocalState (scripts/provision-models.sh), ffmpeg, curl,
# jq. Output filename and watermark carry the version read from
# uwp/AppxManifest.xml.
#
# WHY THIS IS SHAPED THE WAY IT IS
#
# There is no video capture endpoint. The Device Portal offers stills, and
# AppRecordingManager — an app recording itself through the SoC video encoder —
# is NOT on the console: measured, `[caprec] AppRecordingManager present=0`
# (docs/uwp-constraints.md §10b). So the video is rebuilt from stills, and the
# only question was how fast they come.
#
# Measured, not assumed (scripts/bench-screenshot-rate.sh, docs/device-portal.md):
# GET /ext/screenshot sustains 11.5 Hz idle and 13.7 Hz during a decode, with a
# p50 of ~35 ms. The previous version of this script slept 1 s between frames and
# called the result a demo; that 1 was a number someone typed, and it was off by
# an order of magnitude.
#
# Capture is not free, and the direction that matters is what it costs the app,
# because the footer shows a live decode rate: at ~13.5 Hz — unthrottled, which
# is this script's default — it costs 1.8% of decode (93.70 -> 91.98 tok/s,
# non-overlapping ranges). So the numbers on screen are within ~2% of the numbers
# without a camera on them, and there is nothing to buy by going slower: --fps
# throttles, and throttling only makes the video choppier.
#
# The rate actually achieved is lower than the endpoint's ceiling because the run
# loads models and generates while being photographed; whatever it turns out to
# be, the encode uses it, so playback is real time. An early version encoded at
# the REQUESTED rate and produced a 16.5 s video of a 45 s run — a 2.7x speedup
# shipped as a performance demo.

set -euo pipefail

: "${XBOX_IP:?XBOX_IP not set — run: source ~/.config/xllama/xbox-env}"
: "${XBOX_USER:?XBOX_USER not set}"
: "${XBOX_PASS:?XBOX_PASS not set}"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEPLOY="${SCRIPT_DIR}/deploy.sh"
BASE_URL="https://${XBOX_IP}:11443"
CURL_AUTH=(--basic -u "${XBOX_USER}:${XBOX_PASS}" -k -sS)

DEMO_SCRIPT="${REPO_ROOT}/demo/demo-script.json"
OUT_DIR="${REPO_ROOT}/docs/screenshots"
FPS=0 # 0 = unthrottled, which is what the cost below was measured at
HOLD_S=4
while [[ $# -gt 0 ]]; do
	case "$1" in
	--script)
		DEMO_SCRIPT="$2"
		shift 2
		;;
	--fps)
		FPS="$2"
		shift 2
		;;
	--hold)
		HOLD_S="$2"
		shift 2
		;;
	--out)
		OUT_DIR="$2"
		shift 2
		;;
	*)
		echo "Unknown argument: $1" >&2
		exit 1
		;;
	esac
done
if ! [[ "$FPS" =~ ^[0-9]+$ ]] || ((FPS > 30)); then
	echo "Error: --fps must be 0..30 (0 = unthrottled; the endpoint tops out ~13 Hz)" >&2
	exit 1
fi
[[ "$HOLD_S" =~ ^[0-9]+$ ]] || {
	echo "Error: --hold must be an integer" >&2
	exit 1
}
[[ -f "$DEMO_SCRIPT" ]] || {
	echo "Error: demo script not found: ${DEMO_SCRIPT}" >&2
	exit 1
}
jq -e . "$DEMO_SCRIPT" >/dev/null || {
	echo "Error: ${DEMO_SCRIPT} is not valid JSON" >&2
	exit 1
}

# The version is read from the manifest, never typed here. The previous script
# had v1.2.0 baked into both the output path and the watermark, so re-running it
# produced a file claiming to be a release that was four minors old — which is
# how the README ended up linking a stale demo.
VERSION=$(sed -n 's/.*Version="\([0-9.]*\)".*/\1/p' "${REPO_ROOT}/uwp/AppxManifest.xml" | head -n1)
[[ -n "$VERSION" ]] || {
	echo "Error: could not read Version from uwp/AppxManifest.xml" >&2
	exit 1
}
SHORT_VERSION="${VERSION%.*}" # 1.5.2.0 -> 1.5.2
OUT_MP4="${OUT_DIR}/xllama-demo-v${SHORT_VERSION}.mp4"
OUT_GIF="${OUT_DIR}/xllama-demo-v${SHORT_VERSION}.gif"
MANIFEST_JSON="${OUT_DIR}/demo-manifest.json"
MARKERS_OUT="${OUT_DIR}/xllama-demo-v${SHORT_VERSION}-markers.jsonl"

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
FRAMES="${WORK}/frames"
mkdir -p "$OUT_DIR"
mkdir -p "$FRAMES"
MARKERS="${WORK}/markers.jsonl"
: >"$MARKERS"
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

fetch_file_200() {
	local code
	code=$(curl "${CURL_AUTH[@]}" -o "$2" -w "%{http_code}" \
		"${FILE_EP}&filename=${1}" 2>/dev/null || echo "000")
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

echo "==> xllama ${VERSION}, ${FPS} fps, panes held ${HOLD_S}s"
echo "==> script: ${DEMO_SCRIPT}"

# Preflight the models the script names. An MSIX install wipes LocalState and
# --provision only seeds the gate set, so a demo naming anything else silently
# gets a mid-run catalogue download instead — minutes of progress bar recorded
# into the video, or a timeout. The script declares its models; check them.
# Ask for the model's actual GGUF, not for its directory. A directory listing
# answers 200 with {"Items": []} for a path that does not exist — the documented
# WDP behaviour (docs/device-portal.md, "200 with Success: false") — so a
# code-only check passes for every model, present or not. That is not
# hypothetical: the first version of this preflight did exactly that, waved the
# run through, and the capture died at action 6 on a missing model. Same shape as
# model_provisioned_gguf in validate-console.sh.
missing=()
while IFS= read -r model; do
	[[ -z "$model" ]] && continue
	# `personalized` is produced by the script's start_train action, not a
	# catalogue GGUF. Its output is validated by autopilot-done and the marker
	# sequence after training completes.
	if [[ "$model" == personalized ]]; then
		continue
	fi
	model_file=$(jq -r --arg m "$model" \
		'.models[] | select(.name == $m) | .files[] | select(.filename | endswith(".gguf") or endswith("model.onnx")) | .filename' \
		"${REPO_ROOT}/uwp/models/manifest.json" | head -n1)
	if [[ -z "$model_file" ]]; then
		echo "Error: ${model} has no GGUF or model.onnx in uwp/models/manifest.json" >&2
		exit 1
	fi
	model_dir="${model}"
	model_path="%5Cmodels%5C${model_dir}"
	if [[ "$model_file" == */* ]]; then
		model_dir="${model_file%/*}"
		model_path="%5Cmodels%5C${model}%5C${model_dir//\\/%5C}"
		model_file="${model_file##*/}"
	fi
	code=$(curl "${CURL_AUTH[@]}" --connect-timeout 5 --max-time 20 --range 0-0 -o /dev/null -w "%{http_code}" \
		"${FILE_EP}${model_path}&filename=${model_file// /%20}" 2>/dev/null || echo "000")
	[[ "$code" == "200" || "$code" == "206" ]] || missing+=("$model")
done < <(jq -r '[.actions[] | select(.op == "set_model") | .name] | unique[]' "$DEMO_SCRIPT")

if jq -e '[.actions[] | select(.op == "start_train")] | length > 0' "$DEMO_SCRIPT" >/dev/null; then
	TRAINING_EP="${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=%5CLocalState%5Ctraining"
	training_code=$(curl "${CURL_AUTH[@]}" --connect-timeout 5 --max-time 20 --range 0-0 -o /dev/null -w "%{http_code}" \
		"${TRAINING_EP}&filename=base-f16.gguf" 2>/dev/null || echo "000")
	if [[ "$training_code" != "200" && "$training_code" != "206" ]]; then
		echo "Error: showcase training requires LocalState/training/base-f16.gguf" >&2
		echo "  Provision the F16 base model before running this showcase." >&2
		exit 1
	fi
fi
if jq -e '[.actions[] | select(.op == "generate_image")] | length > 0' "$DEMO_SCRIPT" >/dev/null; then
	DIFFUSION_EP="${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=%5CLocalState%5Cmodels%5Csd-turbo-fp16%5Cunet"
	diffusion_code=$(curl "${CURL_AUTH[@]}" --connect-timeout 5 --max-time 20 --range 0-0 -o /dev/null -w "%{http_code}" \
		"${DIFFUSION_EP}&filename=model.onnx" 2>/dev/null || echo "000")
	if [[ "$diffusion_code" != "200" && "$diffusion_code" != "206" ]]; then
		echo "Error: showcase image generation requires LocalState/models/sd-turbo-fp16/unet/model.onnx" >&2
		echo "  Provision sd-turbo-fp16 before running this showcase." >&2
		exit 1
	fi
fi
if ((${#missing[@]} > 0)); then
	echo "Error: the demo script names models that are not on the console:" >&2
	printf '  %s\n' "${missing[@]}" >&2
	echo "  Seed: ./scripts/provision-models.sh ${missing[*]}" >&2
	exit 1
fi

# Accept the first-run disclaimer, byte-for-byte what pressing "I understand"
# writes. ShowAsync() only completes on a button press and there is no human at
# the pad, so without this every frame carries the "Before you start" modal and
# the on-screen keyboard — measured on the first captured run, not assumed.
printf '1\n' >"${WORK}/disclaimer.accepted"
printf 'go' >"${WORK}/autopilot.flag"
cp "$DEMO_SCRIPT" "${WORK}/autopilot.json"

delete_file "autopilot-done.txt"
delete_file "autopilot-mark.txt"
delete_file "diffuse-progress.txt"
upload_file "${WORK}/disclaimer.accepted"
upload_file "${WORK}/autopilot.json"
upload_file "${WORK}/autopilot.flag"
restart_app

STOP="${WORK}/stop"
rm -f "$STOP"
CAP_START=$SECONDS
CAP_START_EPOCH=$(date +%s)

# Frame loop. No sleep between requests beyond what the target rate asks for:
# the endpoint's own latency (~35 ms p50) is most of the interval at 10 fps.
(
	n=0
	interval=""
	((FPS > 0)) && interval=$(LC_ALL=C awk -v f="$FPS" 'BEGIN { printf "%.3f", 1.0 / f }')
	while [[ ! -f "$STOP" ]]; do
		if curl "${CURL_AUTH[@]}" -o "${FRAMES}/f_$(printf '%06d' "$n").png" --fail \
			"${BASE_URL}/ext/screenshot" >/dev/null 2>&1; then
			n=$((n + 1))
		fi
		[[ -n "$interval" ]] && sleep "$interval"
	done
	echo "$n" >"${WORK}/frame_count"
) &
CAP_PID=$!

# Mark releaser. show_pane parks on a rendez-vous waiting for a host to take its
# shot; here the "host" is a continuous frame loop, so there is nothing to take —
# the release simply decides how long the pane stays on screen. Holding it a
# fixed HOLD_S is what makes pane duration a property of this script rather than
# of whatever timeout the demo script happens to carry.
(
	last_label=""
	while [[ ! -f "$STOP" ]]; do
		if fetch_file_200 "autopilot-mark.txt" "${WORK}/mark" &&
			[[ -s "${WORK}/mark" ]]; then
			label=$(tr -d '\r\n' <"${WORK}/mark")
			if [[ "$label" != "$last_label" ]]; then
				now=$(date +%s)
				jq -cn --arg label "$label" --argjson elapsed_s "$((now - CAP_START_EPOCH))" \
					'{label: $label, elapsed_s: $elapsed_s}' >>"$MARKERS"
				echo "  pane '${label}' — holding ${HOLD_S}s" >&2
				last_label="$label"
				sleep "$HOLD_S"
			fi
			delete_file "autopilot-mark.txt"
		fi
		sleep 1
	done
) &
REL_PID=$!

echo "==> Waiting for the run to finish"
marker=""
deadline=$((SECONDS + 1800))
while ((SECONDS < deadline)); do
	if fetch_file_200 "autopilot-done.txt" "${WORK}/done" &&
		grep -qE '^(ok|error:)' "${WORK}/done"; then
		marker=$(cat "${WORK}/done")
		break
	fi
	# 1s, not 5: this poll decides how much dead footage the video ends with,
	# because everything between the run finishing and the marker being noticed
	# is the last frame held on screen. At 5s the capture ended with up to 8
	# seconds of a static chat view — a fifth of a 44-second loop — while the
	# comment below claimed "a few frames". A fetch costs ~35 ms, so polling
	# five times more often buys back that tail for nothing.
	sleep 1
done

sleep 3 # a few frames on the final state
touch "$STOP"
wait "$CAP_PID" "$REL_PID" 2>/dev/null || true
CAP_ELAPSED=$((SECONDS - CAP_START))
cp "$MARKERS" "$MARKERS_OUT"

# Failed grabs leave a zero-byte file behind (curl only unlinks with
# --remove-on-error), and counting those as frames would both pad the video with
# nothing and overstate the rate. Drop them before anything reads the count.
find "$FRAMES" -name 'f_*.png' -size 0 -delete
nframes=$(find "$FRAMES" -name 'f_*.png' | wc -l)
echo "==> autopilot: ${marker:-<no marker>}, ${nframes} frames"
if [[ "$marker" != ok ]]; then
	echo "Refusing to publish a demo from a run that did not finish cleanly." >&2
	"${DEPLOY}" get-log "$PFN" 2>/dev/null | tail -40 >&2 || true
	exit 1
fi
if ((nframes < FPS * 10)); then
	echo "Error: ${nframes} frames is less than 10 s of video" >&2
	exit 1
fi

# The video must play at the speed things actually happened. Encoding at the
# requested rate when the capture achieved less produces a demo that runs fast —
# on a project whose whole claim is measured performance, that is the worst
# defect available, and it is invisible unless the two numbers are compared.
if ((CAP_ELAPSED <= 0)); then
	echo "Error: capture window measured as zero" >&2
	exit 1
fi
ACHIEVED_FPS=$(LC_ALL=C awk -v n="$nframes" -v s="$CAP_ELAPSED" 'BEGIN { printf "%.2f", n / s }')
asked=$( ((FPS > 0)) && echo "$FPS" || echo "unthrottled")
echo "==> ${nframes} frames over ${CAP_ELAPSED}s = ${ACHIEVED_FPS} fps achieved (asked ${asked})"
if LC_ALL=C awk -v a="$ACHIEVED_FPS" 'BEGIN { exit !(a < 5) }'; then
	echo "   NOTE: under 5 fps. The encode uses the achieved rate so playback stays" >&2
	echo "   real time, but the result will look choppy." >&2
fi

echo "==> Encoding ${OUT_MP4}"
ffmpeg -y -framerate "$ACHIEVED_FPS" -pattern_type glob -i "${FRAMES}/f_*.png" \
	-vf "scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2,drawtext=fontfile=/usr/share/fonts/TTF/DejaVuSans.ttf:text='xllama v${SHORT_VERSION} · Xbox Series S · local · no cloud':x=24:y=h-48:fontsize=22:fontcolor=white:borderw=2:bordercolor=black" \
	-c:v libx264 -preset slow -crf 18 -pix_fmt yuv420p -movflags +faststart \
	"$OUT_MP4" 2>"${WORK}/ffmpeg.log" || {
	tail -20 "${WORK}/ffmpeg.log" >&2
	exit 1
}

# Keep the GIF as a legacy artefact for existing links. The README uses a
# GitHub-native video attachment, so the GIF is not part of the landing-page
# presentation. It is generated from the same frames for reproducibility.
#
# Sampled at the achieved rate, capped at 10 fps — never ABOVE what the capture
# actually produced. Sampling above it duplicates frames: no extra information,
# but every duplicate still costs a GIF frame header and a palette diff. The
# first version of this hardcoded 10 fps and the very next capture came in at
# 8.82, so it padded ~13% of the frames with copies. Real time is preserved
# either way: dropping frames makes a GIF choppier, never faster, and a demo of
# a performance claim must not be sped up.
GIF_FPS=$(LC_ALL=C awk -v a="$ACHIEVED_FPS" 'BEGIN { printf "%.2f", (a < 10 ? a : 10) }')
#
# stats_mode=diff prices the palette for what CHANGES between frames, which on a
# mostly-static dark UI is the text; the naive palette spends its 256 colours on
# the background instead. diff_mode=rectangle only rewrites the changed region.
# Together they are the difference between ~2.5 MB and something unshippable.
echo "==> Encoding ${OUT_GIF}"
ffmpeg -y -v error -i "$OUT_MP4" \
	-vf "fps=${GIF_FPS},scale=1280:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle" \
	-loop 0 "$OUT_GIF" 2>"${WORK}/ffmpeg-gif.log" || {
	tail -20 "${WORK}/ffmpeg-gif.log" >&2
	exit 1
}
GIF_BYTES=$(stat -c%s "$OUT_GIF")
# A README that pulls a huge GIF on every page view is a regression even if it
# looks fine. Warn rather than fail: the right answer depends on what the demo
# has to show, and that is a judgement call for whoever is looking at it.
if ((GIF_BYTES > 8000000)); then
	echo "   NOTE: ${OUT_GIF##*/} is $((GIF_BYTES / 1000000)) MB — large for a README." >&2
	echo "   Consider a shorter demo script before reaching for a lower frame rate." >&2
fi

DURATION=$(LC_ALL=C ffprobe -v error -show_entries format=duration -of default=nw=1:nk=1 "$OUT_MP4")
ENCODED_FPS=$(LC_ALL=C ffprobe -v error -select_streams v:0 \
	-show_entries stream=avg_frame_rate -of default=nw=1:nk=1 "$OUT_MP4")
ENCODED_CODEC=$(ffprobe -v error -select_streams v:0 \
	-show_entries stream=codec_name -of default=nw=1:nk=1 "$OUT_MP4")
# LC_ALL=C or bash printf rejects "16.5" under a comma-decimal locale — and the
# near miss is worse than the error: it would have written 16,5 into JSON.
DURATION_1F=$(LC_ALL=C printf '%.1f' "$DURATION")

# The manifest is what check-coherence.py compares the README's demo link
# against, so a hand-edited link fails CI instead of quietly going stale.
cat >"$MANIFEST_JSON" <<JSON
{
  "_comment": "Written by scripts/capture-demo-video.sh. Do not hand-edit: check-coherence.py compares the README demo link against 'version', and the point is that the two cannot drift.",
  "version": "${VERSION}",
  "file": "$(basename "$OUT_MP4")",
  "gif": "$(basename "$OUT_GIF")",
  "gif_bytes": ${GIF_BYTES},
  "gif_fps": ${GIF_FPS},
  "duration_s": ${DURATION_1F},
  "frames": ${nframes},
  "fps_requested": "${asked}",
  "fps_achieved": ${ACHIEVED_FPS},
  "encoded_fps": "${ENCODED_FPS}",
  "video_codec": "${ENCODED_CODEC}",
  "video_crf": 18,
  "video_preset": "slow",
  "capture_window_s": ${CAP_ELAPSED},
  "markers": "$(basename "$MARKERS_OUT")",
  "source": "device-portal-stills",
  "source_note": "No video capture endpoint exists and AppRecordingManager is absent on the console (uwp-constraints.md 10b). Stills, encoded at the rate actually achieved so playback is real time. Read fps_achieved as this run's average, NOT as the endpoint's ceiling: /ext/screenshot sustains ~11.5 Hz idle and ~13.7 Hz during a decode (device-portal.md), and a run that loads several models spends part of its time where nothing changes, so its average is lower and that is not a regression. Capture costs the app ~1.8% of decode.",
  "script": "$(realpath --relative-to="$REPO_ROOT" "$DEMO_SCRIPT")"
}
JSON
jq -e . "$MANIFEST_JSON" >/dev/null || {
	echo "Error: generated demo-manifest.json is not valid JSON" >&2
	exit 1
}

ls -la "$OUT_MP4" "$OUT_GIF"
echo "==> ${DURATION_1F}s, ${nframes} frames at ${ACHIEVED_FPS} fps"
echo "==> gif: ${OUT_GIF##*/} ($((GIF_BYTES / 1000)) kB, ${GIF_FPS} fps, same real time)"
echo "==> manifest: ${MANIFEST_JSON}"
echo "==> markers: ${MARKERS_OUT}"
echo
echo "Next: upload ${OUT_MP4##*/} through a GitHub issue/comment to obtain a"
echo "      user-attachments URL, then update the README player source."
