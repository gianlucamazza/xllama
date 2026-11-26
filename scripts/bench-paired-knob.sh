#!/usr/bin/env bash
# bench-paired-knob.sh — matched A/B pairs of bench-xbox-ort.sh runs that differ
# only by one knob: a LocalState knob file (the #313 four-column campaign:
# d3d12-prefill-columns.txt, absent in arm A, "4" in arm B), or a
# bench-xbox-ort.sh option given to one arm only (#582: --flash-attn 0,
# --ubatch N, --kv-q8 — knobs bench-xbox-ort.sh writes itself on every run, so
# a knob file uploaded beside them would be overwritten).
#
# Usage:
#   source ~/.config/xllama/xbox-env
#   ./scripts/bench-paired-knob.sh --model <dir> --prompt <file> --pairs N \
#       [--knob <name> --value <text>] [--a-arg ARG]... [--b-arg ARG]... \
#       --out-dir <dir> --expect-a RE [--forbid-a RE] --expect-b RE [--forbid-b RE] \
#       [--save-arm-logs] [--retries N] [bench-xbox-ort.sh options for both arms...]
#
# --a-arg / --b-arg pass one word to bench-xbox-ort.sh for that arm only
# (repeat for an option with a value: --b-arg --ubatch --b-arg 256). A campaign
# needs --knob or at least one per-arm argument.
#
# --retries N reruns an arm whose bench-xbox-ort.sh call failed without writing
# a row (a Device Portal upload that timed out), up to N times. Each failed
# attempt is a "retry" line in the ledger. A failure after a row was written,
# or a log check that fails, still stops the campaign: re-running it would
# replace a measured attempt.
#
# Pair i runs A then B when i is odd and B then A when i is even. Each arm is
# one bench-xbox-ort.sh invocation with --runs 2: run 1 is the warmup and is
# dropped, run 2 is the recorded row (A.csv or B.csv in --out-dir). After each
# arm the lines the app log gained during that arm must
# match every --expect regex of the arm and none of its --forbid regexes (for
# example the backend's prefill_columns line and its four-column dispatch
# count); a mismatch, a harness failure or a missing row stops the campaign.
# The regex options repeat. --save-arm-logs retains the warmup and measured
# phase counters in pair<N>-<A|B>-app.log.
# Failed attempts are never replaced: the ledger (ledger.jsonl) records every
# arm with its outcome, and the summary uses only complete pairs.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODEL="" PROMPT="" PAIRS=0 KNOB="" VALUE="" OUTDIR="" SAVE_ARM_LOGS=0 RETRIES=0
EXPECT_A=() FORBID_A=() EXPECT_B=() FORBID_B=() EXTRA=() ARGS_A=() ARGS_B=()
while [[ $# -gt 0 ]]; do
	case "$1" in
	--model) MODEL="$2"; shift 2 ;;
	--prompt) PROMPT="$2"; shift 2 ;;
	--pairs) PAIRS="$2"; shift 2 ;;
	--knob) KNOB="$2"; shift 2 ;;
	--value) VALUE="$2"; shift 2 ;;
	--out-dir) OUTDIR="$2"; shift 2 ;;
	--save-arm-logs) SAVE_ARM_LOGS=1; shift ;;
	--retries) RETRIES="$2"; shift 2 ;;
	--a-arg) ARGS_A+=("$2"); shift 2 ;;
	--b-arg) ARGS_B+=("$2"); shift 2 ;;
	--expect-a) EXPECT_A+=("$2"); shift 2 ;;
	--forbid-a) FORBID_A+=("$2"); shift 2 ;;
	--expect-b) EXPECT_B+=("$2"); shift 2 ;;
	--forbid-b) FORBID_B+=("$2"); shift 2 ;;
	*) EXTRA+=("$1"); shift ;;
	esac
done
if [[ -z "$MODEL" || -z "$PROMPT" || "$PAIRS" -lt 1 || -z "$OUTDIR" || ${#EXPECT_A[@]} -eq 0 || ${#EXPECT_B[@]} -eq 0 ]] ||
	[[ -z "$KNOB" && ${#ARGS_A[@]} -eq 0 && ${#ARGS_B[@]} -eq 0 ]] || ! [[ "$RETRIES" =~ ^[0-9]+$ ]]; then
	echo "Usage: see the header of $0" >&2
	exit 2
fi
: "${XBOX_IP:?XBOX_IP not set}"
mkdir -p "$OUTDIR"
DEPLOY="${SCRIPT_DIR}/deploy.sh"
PFN="$("$DEPLOY" pfn 2>/dev/null | tail -1)"
LEDGER="${OUTDIR}/ledger.jsonl"
TMP="$(mktemp -d)"
if [[ -n "$KNOB" ]]; then
	trap 'rm -rf "$TMP"; "$DEPLOY" delete-file "$PFN" "$KNOB" >/dev/null 2>&1 || true' EXIT
	printf '%s' "$VALUE" >"${TMP}/${KNOB}"
else
	trap 'rm -rf "$TMP"' EXIT
fi

record() { # pair order arm status detail
	printf '{"pair":%d,"order":"%s","arm":"%s","status":"%s","detail":"%s","date":"%s"}\n' \
		"$1" "$2" "$3" "$4" "$5" "$(date -u +%FT%TZ)" >>"$LEDGER"
}

# 0 = ok, 1 = failed (stop), 2 = failed before any row was written (retryable).
run_arm() { # pair order arm
	local pair="$1" order="$2" arm="$3" csv="${OUTDIR}/$3.csv" before after re skip
	local -a expect forbid arm_args
	# The running app holds xllama.log open, so stop it first; whether or not
	# the delete succeeds, only the lines added during this arm are checked.
	"$DEPLOY" stop-app "$PFN" >/dev/null 2>&1 || true
	[[ -n "$KNOB" ]] && { "$DEPLOY" delete-file "$PFN" "$KNOB" >/dev/null 2>&1 || true; }
	"$DEPLOY" delete-file "$PFN" xllama.log >/dev/null 2>&1 || true
	skip=$("$DEPLOY" get-log 2>/dev/null | wc -l)
	if [[ "$arm" == B ]]; then
		if [[ -n "$KNOB" ]] && ! "$DEPLOY" upload-file "${TMP}/${KNOB}" "$PFN" >/dev/null 2>&1; then
			record "$pair" "$order" "$arm" retry "knob upload"
			return 2
		fi
		expect=("${EXPECT_B[@]}")
		forbid=("${FORBID_B[@]}")
		arm_args=("${ARGS_B[@]}")
	else
		expect=("${EXPECT_A[@]}")
		forbid=("${FORBID_A[@]}")
		arm_args=("${ARGS_A[@]}")
	fi
	before=$( [[ -f "$csv" ]] && grep -vc '^model,' "$csv" || echo 0)
	if ! "${SCRIPT_DIR}/bench-xbox-ort.sh" "$MODEL" --runs 2 --prompt "$PROMPT" --out "$csv" \
		"${EXTRA[@]}" "${arm_args[@]}" >"${TMP}/bench.log" 2>&1; then
		after=$( [[ -f "$csv" ]] && grep -vc '^model,' "$csv" || echo 0)
		if [[ "$after" -eq "$before" ]]; then
			cp "${TMP}/bench.log" "${OUTDIR}/retry-pair${pair}-${arm}-$(date -u +%H%M%S).log"
			record "$pair" "$order" "$arm" retry "bench-xbox-ort.sh exit before a row"
			return 2
		fi
		cp "${TMP}/bench.log" "${OUTDIR}/failed-pair${pair}-${arm}.log"
		record "$pair" "$order" "$arm" failed "bench-xbox-ort.sh exit"
		return 1
	fi
	after=$(grep -vc '^model,' "$csv")
	if [[ "$after" -ne $((before + 1)) ]]; then
		record "$pair" "$order" "$arm" failed "expected one new row, got $((after - before))"
		return 1
	fi
	"$DEPLOY" get-log 2>/dev/null | tail -n +"$((skip + 1))" >"${TMP}/app.log" || true
	if ((SAVE_ARM_LOGS)) && ! cp "${TMP}/app.log" "${OUTDIR}/pair${pair}-${arm}-app.log"; then
		record "$pair" "$order" "$arm" failed "could not save app log"
		return 1
	fi
	for re in "${expect[@]}"; do
		if ! grep -a -E -q -- "$re" "${TMP}/app.log"; then
			cp "${TMP}/app.log" "${OUTDIR}/failed-pair${pair}-${arm}-app.log"
			record "$pair" "$order" "$arm" failed "app log lacks an expected line"
			return 1
		fi
	done
	for re in "${forbid[@]}"; do
		if grep -a -E -q -- "$re" "${TMP}/app.log"; then
			cp "${TMP}/app.log" "${OUTDIR}/failed-pair${pair}-${arm}-app.log"
			record "$pair" "$order" "$arm" failed "app log has a forbidden line"
			return 1
		fi
	done
	grep -a -E -- "c4 dispatches" "${TMP}/app.log" | tail -1 >>"${OUTDIR}/${arm}-dispatch.log" || true
	record "$pair" "$order" "$arm" ok "row $after"
}

for ((i = 1; i <= PAIRS; i++)); do
	if ((i % 2)); then order=AB; else order=BA; fi
	for arm in ${order:0:1} ${order:1:1}; do
		echo "pair $i/$PAIRS order $order arm $arm"
		attempt=0
		while :; do
			run_arm "$i" "$order" "$arm"
			rc=$?
			((rc == 0)) && break
			if ((rc == 2 && attempt < RETRIES)); then
				attempt=$((attempt + 1))
				echo "  no row written; retry ${attempt}/${RETRIES}"
				continue
			fi
			echo "stopped at pair $i arm $arm (see ${LEDGER})" >&2
			exit 1
		done
	done
done
echo "done: ${PAIRS} pairs in ${OUTDIR}"
