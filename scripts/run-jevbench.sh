#!/usr/bin/env bash
# run-jevbench.sh — host or Series S decision-model harness.
#
# Paths, model names, thread counts and GPU layer counts come from the job
# JSON or flags. This script does not hardcode a console host, a LocalState
# absolute path, or a D3D12 engine index.
#
# Host (Linux CPU):
#   ./scripts/fetch-jevbench.sh
#   ./scripts/run-jevbench.sh --job bench/results/decision-models/job-score.example.json \
#       --model /path/to/h2o-lightning-4b-stock-Q4_K_M.gguf
#
# Series S (after a deployed unified/llamacpp package and xbox-env):
#   source ~/.config/xllama/xbox-env
#   ./scripts/run-jevbench.sh --console --job path/to/decide_job.json
#
# See docs/decision-models-spike.md.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

MODE="host"
JOB=""
MODEL=""
CLI="${XLLAMA_CLI:-${REPO_ROOT}/build/linux-test/bin/xllama-cli}"
OUT=""
TIMEOUT_S="${XLLAMA_JEVBENCH_TIMEOUT_S:-14400}"

usage() {
	sed -n '2,20p' "$0"
}

while [[ $# -gt 0 ]]; do
	case "$1" in
	--console)
		MODE="console"
		shift
		;;
	--host)
		MODE="host"
		shift
		;;
	--job)
		JOB="$2"
		shift 2
		;;
	--model)
		MODEL="$2"
		shift 2
		;;
	--cli)
		CLI="$2"
		shift 2
		;;
	--out)
		OUT="$2"
		shift 2
		;;
	--timeout)
		TIMEOUT_S="$2"
		shift 2
		;;
	-h | --help)
		usage
		exit 0
		;;
	*)
		echo "unknown: $1" >&2
		usage >&2
		exit 2
		;;
	esac
done

JOB="${JOB:-${XLLAMA_DECIDE_JOB:-}}"
MODEL="${MODEL:-${XLLAMA_DECIDE_MODEL:-}}"

if [[ -z "${JOB}" ]]; then
	echo "run-jevbench: --job or XLLAMA_DECIDE_JOB is required" >&2
	exit 2
fi
if [[ ! -f "${JOB}" ]]; then
	echo "run-jevbench: job not found: ${JOB}" >&2
	exit 2
fi

if [[ "${MODE}" == "host" ]]; then
	if [[ ! -x "${CLI}" ]]; then
		echo "run-jevbench: CLI not found at ${CLI} (build linux-test or set --cli)" >&2
		exit 2
	fi
	if [[ -n "${MODEL}" ]]; then
		# Rewrite model_path for this run without editing the example file.
		TMP_JOB="$(mktemp)"
		python3 - "${JOB}" "${MODEL}" "${TMP_JOB}" <<'PY'
import json, sys
src, model, dest = sys.argv[1], sys.argv[2], sys.argv[3]
job = json.loads(open(src, encoding="utf-8").read())
job["model_path"] = model
if len(sys.argv) > 3:
    open(dest, "w", encoding="utf-8").write(json.dumps(job, indent=2) + "\n")
PY
		JOB="${TMP_JOB}"
		trap 'rm -f "${TMP_JOB}"' EXIT
	fi
	echo "host jevbench: ${CLI} --jevbench ${JOB}"
	"${CLI}" --jevbench "${JOB}"
	exit $?
fi

# Console path. Credentials stay in xbox-env; this script never embeds them.
# shellcheck disable=SC1090,SC1091
source "${XBOX_ENV:-$HOME/.config/xllama/xbox-env}"
: "${XBOX_IP:?source ~/.config/xllama/xbox-env}"

DEPLOY="${SCRIPT_DIR}/deploy.sh"
PFN="$("${DEPLOY}" pfn 2>/dev/null || true)"
[[ -n "${PFN}" ]] || {
	echo "run-jevbench: xllama not installed on the console" >&2
	exit 1
}

QUESTIONS_DIR="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8")).get("questions_dir","jevbench"))' "${JOB}")"
REMOTE_QDIR="${QUESTIONS_DIR}"

if [[ ! -f "${QUESTIONS_DIR}/easy.jsonl" ]]; then
	if [[ -f "${REPO_ROOT}/bench/jevbench/public/easy.jsonl" ]]; then
		QUESTIONS_DIR="${REPO_ROOT}/bench/jevbench/public"
	else
		echo "run-jevbench: fetch the public items first (./scripts/fetch-jevbench.sh)" >&2
		exit 2
	fi
fi

TMP="$(mktemp -d)"
trap 'rm -rf "${TMP}"' EXIT
: >"${TMP}/decide.flag"

upload() {
	local src="$1"
	local dest_dir="$2"
	local dest_name="$3"
	"${DEPLOY}" upload-file "${src}" "${PFN}" "${dest_dir}" "${dest_name}"
}

echo "Uploading JevBench JSONL to ${REMOTE_QDIR}/ ..."
upload "${QUESTIONS_DIR}/easy.jsonl" "${REMOTE_QDIR}" "easy.jsonl"
upload "${QUESTIONS_DIR}/original.jsonl" "${REMOTE_QDIR}" "original.jsonl"
upload "${QUESTIONS_DIR}/hard.jsonl" "${REMOTE_QDIR}" "hard.jsonl"
if [[ -f "${QUESTIONS_DIR}/provenance.json" ]]; then
	upload "${QUESTIONS_DIR}/provenance.json" "${REMOTE_QDIR}" "provenance.json"
fi

upload "${JOB}" "" "decide_job.json"
printf '%s\n' "decide_job.json" >"${TMP}/decide_job_path.txt"
upload "${TMP}/decide_job_path.txt" "" "decide_job_path.txt"

upload_flag() {
	upload "${TMP}/decide.flag" "" "decide.flag"
}

echo "Uploading decide.flag ..."
if ! upload_flag 2>/dev/null; then
	echo "  (upload failed — seeding LocalState via one UI launch...)"
	"${DEPLOY}" stop-app || true
	"${DEPLOY}" start-app || true
	sleep 8
	"${DEPLOY}" stop-app || true
	sleep 2
	upload_flag
fi

"${DEPLOY}" stop-app || true
sleep 1
"${DEPLOY}" start-app
echo "Waiting for decide.done (timeout ${TIMEOUT_S}s) ..."
deadline=$((SECONDS + TIMEOUT_S))
while ((SECONDS < deadline)); do
	if "${DEPLOY}" fetch-file "${PFN}" "decide.done" "${TMP}/done" 2>/dev/null; then
		break
	fi
	sleep 5
done
if [[ ! -f "${TMP}/done" ]]; then
	echo "run-jevbench: timed out waiting for decide.done" >&2
	exit 1
fi

RESULT_NAME="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1],encoding="utf-8")).get("out_path","decide-result.json"))' "${JOB}")"
RESULT_NAME="${RESULT_NAME:-decide-result.json}"
OUT="${OUT:-${REPO_ROOT}/bench/results/decision-models/console-latest.json}"
mkdir -p "$(dirname "${OUT}")"
"${DEPLOY}" fetch-file "${PFN}" "${RESULT_NAME}" "${OUT}"
echo "wrote ${OUT}"
grep -q '^ok' "${TMP}/done" || exit 1
