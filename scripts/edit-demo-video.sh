#!/usr/bin/env bash
# edit-demo-video.sh — create a smoother presentation edit from a raw demo
#
# This is a presentation transform only. It must not be used for benchmark
# evidence: the source stills are captured at their measured rate, while this
# edit blends adjacent frames to make UI transitions easier to watch.
#
# Usage:
#   ./scripts/edit-demo-video.sh [raw.mp4] [presentation.mp4] [fps]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
INPUT="${1:-${REPO_ROOT}/docs/screenshots/xllama-demo-v1.5.6.mp4}"
OUTPUT="${2:-${REPO_ROOT}/docs/screenshots/xllama-demo-v1.5.6-presentation.mp4}"
FPS="${3:-24}"

[[ -f "$INPUT" ]] || {
	echo "Error: input video not found: ${INPUT}" >&2
	exit 1
}
[[ "$FPS" =~ ^([1-9]|[1-5][0-9])$ ]] || {
	echo "Error: fps must be an integer from 1 to 59" >&2
	exit 1
}
command -v ffmpeg >/dev/null || {
	echo "Error: ffmpeg is required" >&2
	exit 1
}

mkdir -p "$(dirname "$OUTPUT")"
echo "==> Creating presentation edit at ${FPS} fps"
echo "==> Source remains authoritative: ${INPUT}"

ffmpeg -y -v error -i "$INPUT" \
	-vf "minterpolate=fps=${FPS}:mi_mode=blend" \
	-c:v libx264 -preset medium -crf 20 -pix_fmt yuv420p -movflags +faststart \
	-metadata comment="Presentation edit; frame blending only; not benchmark evidence" \
	"$OUTPUT"

ffprobe -v error -select_streams v:0 \
	-show_entries stream=width,height,avg_frame_rate,nb_frames \
	-show_entries format=duration,size -of json "$OUTPUT"
