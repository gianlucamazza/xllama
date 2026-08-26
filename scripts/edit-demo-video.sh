#!/usr/bin/env bash
# edit-demo-video.sh — create a concise presentation edit from a demo recording
#
# The default edit removes measured waiting time; it does not invent frames.
# The raw recording remains the only performance evidence.
#
# Usage:
#   ./scripts/edit-demo-video.sh
#   ./scripts/edit-demo-video.sh --input input.mp4 --output output.mp4
#   ./scripts/edit-demo-video.sh --segments '0:1.4,5.5:6.8,9.4:11.366667'

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INPUT="${REPO_ROOT}/docs/screenshots/xllama-demo-v1.5.6-presentation.mp4"
OUTPUT="${REPO_ROOT}/docs/screenshots/xllama-demo-v1.5.6-cut.mp4"
FPS=30
SEGMENTS='0:1.4,5.5:6.8,9.4:11.366667'
INTERPOLATE=0

usage() {
	cat <<'EOF'
Usage: edit-demo-video.sh [options]

Options:
  --input FILE       source video (default: presentation recording)
  --output FILE      output MP4 (default: xllama-demo-v1.5.6-cut.mp4)
  --fps N            output frame rate, 1..60 (default: 30)
  --segments LIST    comma-separated start:end ranges in seconds
  --interpolate      blend frames after cutting (presentation-only; off by default)
  -h, --help         show this help
EOF
}

while [[ $# -gt 0 ]]; do
	case "$1" in
	--input)
		INPUT="$2"
		shift 2
		;;
	--output)
		OUTPUT="$2"
		shift 2
		;;
	--fps)
		FPS="$2"
		shift 2
		;;
	--segments)
		SEGMENTS="$2"
		shift 2
		;;
	--interpolate)
		INTERPOLATE=1
		shift
		;;
	-h | --help)
		usage
		exit 0
		;;
	*)
		echo "Error: unknown option: $1" >&2
		usage >&2
		exit 1
		;;
	esac
done

[[ -f "$INPUT" ]] || {
	echo "Error: input video not found: ${INPUT}" >&2
	exit 1
}
[[ "$FPS" =~ ^[1-9]$|^[1-5][0-9]$|^60$ ]] || {
	echo "Error: --fps must be an integer from 1 to 60" >&2
	exit 1
}
command -v ffmpeg >/dev/null || {
	echo "Error: ffmpeg is required" >&2
	exit 1
}
command -v ffprobe >/dev/null || {
	echo "Error: ffprobe is required" >&2
	exit 1
}

filter_parts=()
labels=()
segment_count=0
IFS=',' read -r -a segment_list <<<"$SEGMENTS"
for segment in "${segment_list[@]}"; do
	if [[ "$segment" != *:* ]]; then
		echo "Error: invalid segment '${segment}', expected start:end" >&2
		exit 1
	fi
	start=${segment%%:*}
	end=${segment#*:}
	[[ "$start" =~ ^[0-9]+([.][0-9]+)?$ && "$end" =~ ^[0-9]+([.][0-9]+)?$ ]] || {
		echo "Error: invalid segment '${segment}', times must be non-negative seconds" >&2
		exit 1
	}
	if ! awk -v start="$start" -v end="$end" 'BEGIN { exit !(end > start) }'; then
		echo "Error: segment end must be greater than start: ${segment}" >&2
		exit 1
	fi
	label="seg${segment_count}"
	filter_parts+=("[0:v]trim=start=${start}:end=${end},setpts=PTS-STARTPTS[${label}]")
	labels+=("[${label}]")
	segment_count=$((segment_count + 1))
done
((segment_count > 0)) || {
	echo "Error: at least one segment is required" >&2
	exit 1
}

concat_inputs=$(printf '%s' "${labels[@]}")
filter="$(IFS=';'; echo "${filter_parts[*]}");${concat_inputs}concat=n=${segment_count}:v=1:a=0,fps=${FPS},format=yuv420p"
if ((INTERPOLATE)); then
	filter+=",minterpolate=fps=${FPS}:mi_mode=blend"
fi

mkdir -p "$(dirname "$OUTPUT")"
echo "==> Cutting ${segment_count} segments to ${FPS} fps"
echo "==> Source remains authoritative: ${INPUT}"
if ((INTERPOLATE)); then
	echo "==> WARNING: frame blending enabled for presentation only"
fi

comment="Presentation edit; measured waiting time removed; not benchmark evidence"
if ((INTERPOLATE)); then
	comment+="; frame blending enabled"
fi
ffmpeg -y -v error -i "$INPUT" -filter_complex "$filter" \
	-c:v libx264 -preset medium -crf 18 -pix_fmt yuv420p -movflags +faststart \
	-metadata comment="$comment" "$OUTPUT"

ffprobe -v error -select_streams v:0 \
	-show_entries stream=width,height,avg_frame_rate,nb_frames \
	-show_entries format=duration,size -of json "$OUTPUT"
