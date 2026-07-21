#!/usr/bin/env bash
# bench-xbox-kv.sh — multi-turn (KV-reuse) bench on the console, per backend.
#
# Usage:
#   source ~/.config/xllama/xbox-env
#   ./scripts/bench-xbox-kv.sh <model-dir> [--prompt FILE] [--turn2 TEXT]
#                              [--runs N] [--out FILE]
#
# Why: the single-turn sweep (bench-prompt-sweep.sh) says DirectML wins a long
# first turn. It cannot say what happens next, and that is where the decision
# lives: routing is sticky per conversation (MainPage.h, the m_routing/m_active_model comment) and DirectML
# rejects continuous decoding (routing_policy.h, kv_reuse_supported_for_model()), so a GPU-routed chat
# re-prefills the whole context every turn while the CPU path reuses its KV.
#
# The app-side mode already exists: bench_turns.txt in LocalState diverts
# main_loop to run_kv_bench (uwp/inference-bridge.cpp, main_loop()'s bench_turns.txt branch), which measures
# turn-2 prefill with reuse against the cold full re-prefill and writes
# bench-kv-result.csv with real token counts. Nothing drove it before — the
# committed KV CSVs were produced by hand, and bench-xbox-ort.sh actively
# DELETES bench_turns.txt so it cannot be hijacked into this mode.
#
# NOTE on DirectML: run_kv_bench has no kv_reuse_supported_for_model() guard, so
# pointing it at a dml model is unexplored. Whatever it does — error, or numbers
# that silently do not reflect what the app does — is itself the finding. This
# script reports it rather than hiding it.
set -euo pipefail

: "${XBOX_IP:?XBOX_IP not set — run: source ~/.config/xllama/xbox-env}"
: "${XBOX_USER:?XBOX_USER not set}"
: "${XBOX_PASS:?XBOX_PASS not set}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEPLOY="${SCRIPT_DIR}/deploy.sh"
BASE_URL="https://${XBOX_IP}:11443"
CURL_AUTH=(--basic -u "${XBOX_USER}:${XBOX_PASS}" -k -sS)

MODEL="${1:?usage: bench-xbox-kv.sh <model-dir> [--prompt FILE] [--turn2 TEXT] [--runs N] [--out FILE]}"
shift
PROMPT_FILE="${REPO_ROOT}/bench/prompts/standard-512.txt"
TURN2="Summarise that in one sentence."
RUNS=2
OUT=""

while [[ $# -gt 0 ]]; do
	case "$1" in
	--prompt)
		PROMPT_FILE="$2"
		shift 2
		;;
	--turn2)
		TURN2="$2"
		shift 2
		;;
	--runs)
		RUNS="$2"
		shift 2
		;;
	--out)
		OUT="$2"
		shift 2
		;;
	*)
		echo "Unknown arg: $1" >&2
		exit 1
		;;
	esac
done

[[ -f "$PROMPT_FILE" ]] || {
	echo "Error: prompt file not found: $PROMPT_FILE" >&2
	exit 1
}

CSRF_TOKEN=$(curl "${CURL_AUTH[@]}" "${BASE_URL}/" -o /dev/null -D - 2>/dev/null |
	sed -n 's/.*[Cc][Ss][Rr][Ff]-[Tt]oken=\([^;[:space:]]*\).*/\1/p' |
	tr -d '\r' | head -n1)
[[ -z "$CSRF_TOKEN" ]] && echo "Warning: no CSRF token — POST/DELETE may fail silently" >&2

# deploy.sh pfn picks the highest registered version and warns on stderr when an
# upgrade left more than one live; keep stdout clean here.
PFN=$("${DEPLOY}" pfn 2>/dev/null)
[[ -z "$PFN" ]] && {
	echo "Error: xllama not installed on the console" >&2
	exit 1
}

TMPDIR_LOCAL=$(mktemp -d)
trap 'rm -rf "$TMPDIR_LOCAL"' EXIT

upload() {
	local local_path="$1" remote_name="${2:-}"
	local form="${local_path};type=application/octet-stream"
	[[ -n "$remote_name" ]] && form="${local_path};filename=${remote_name};type=application/octet-stream"
	curl "${CURL_AUTH[@]}" -H "X-CSRF-Token:${CSRF_TOKEN}" -X POST -F "file=@${form}" \
		"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=%5CLocalState" \
		>/dev/null
}

fetch() {
	curl "${CURL_AUTH[@]}" -o "$2" \
		"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=%5CLocalState&filename=$1" \
		2>/dev/null || true
}

remove() {
	curl "${CURL_AUTH[@]}" -H "X-CSRF-Token:${CSRF_TOKEN}" -X DELETE \
		"${BASE_URL}/api/filesystem/apps/file?knownfolderid=LocalAppData&packagefullname=${PFN}&path=%5CLocalState&filename=$1" \
		>/dev/null 2>&1 || true
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

printf 'model:   %s\nprompt:  %s (%s bytes)\nturn2:   %s\nruns:    %s\n\n' \
	"$MODEL" "$(basename "$PROMPT_FILE")" "$(wc -c <"$PROMPT_FILE")" "$TURN2" "$RUNS"

printf '%s' "$MODEL" >"${TMPDIR_LOCAL}/model.txt"
printf '%s' "$TURN2" >"${TMPDIR_LOCAL}/bench_turns.txt"
printf '0' >"${TMPDIR_LOCAL}/bench_threads.txt"
printf 'bench' >"${TMPDIR_LOCAL}/bench.flag"

for run in $(seq 1 "$RUNS"); do
	echo "--- Run ${run} / ${RUNS} ---"
	remove "bench-kv-result.csv"
	remove "bench-kv-result.csv.done"
	# Append-only across restarts: clear it so the failure grep below sees only
	# this run (a stale DirectML error would otherwise be attributed to it).
	remove "xllama.log"

	upload "$PROMPT_FILE" "prompt.txt"
	upload "${TMPDIR_LOCAL}/model.txt"
	upload "${TMPDIR_LOCAL}/bench_turns.txt"
	upload "${TMPDIR_LOCAL}/bench_threads.txt"
	upload "${TMPDIR_LOCAL}/bench.flag"
	restart_app

	done_file="${TMPDIR_LOCAL}/kv.csv"
	elapsed=0
	ok=0
	while ((elapsed < 300)); do
		: >"$done_file"
		fetch "bench-kv-result.csv" "$done_file"
		if head -1 "$done_file" 2>/dev/null | grep -q '^model,'; then
			ok=1
			break
		fi
		sleep 10
		((elapsed += 10))
	done

	if ((ok == 1)); then
		row=$(tail -n +2 "$done_file" | head -1)
		echo "  Row: ${row}"
		if [[ -n "$OUT" ]]; then
			[[ -f "$OUT" ]] || head -1 "$done_file" >"$OUT"
			printf '%s\n' "$row" >>"$OUT"
		fi
	else
		# The interesting failure: DirectML rejecting continuous decoding. Surface
		# the log rather than reporting a bare timeout.
		echo "  TIMEOUT after ${elapsed}s — no bench-kv-result.csv" >&2
		fetch "xllama.log" "${TMPDIR_LOCAL}/xllama.log"
		echo "  --- log tail ---" >&2
		tail -25 "${TMPDIR_LOCAL}/xllama.log" >&2 || true
		echo "  ----------------" >&2
		exit 1
	fi
done

echo ""
echo "Done."
[[ -n "$OUT" ]] && column -s, -t "$OUT"
exit 0
