#!/usr/bin/env bash
# bench-ramceil.sh — measure the heap ceiling the console actually grants.
#
# Usage:
#   source ~/.config/xllama/xbox-env
#   ./scripts/bench-ramceil.sh [--out FILE]
#
# Why: GGUF weights are read into the heap — mmap is unavailable in the sandbox
# and enabling it measured zero benefit (uwp-constraints.md §1) — so a model's
# admissibility is decided by how much heap this process can commit. That number
# has never been measured. What the repo has is `avail_phys` 5.0 GB, one
# incidental log line from a load probe, and a 3.5 GB peak gate that is an H4
# acceptance policy. Treating either as "the ceiling" is the estimate-promoted-
# to-decision mistake architecture.md warns about, and here it decides whether a
# MoE like LFM2.5-8B-A1B is admissible at IQ3 (3.57 GB) or only at Q2 (2.93 GB),
# where the E2B IQ2 precedent says quality dies.
#
# No model is loaded and nothing is installed: the probe commits heap in 128 MB
# steps, faults every page in, and records the platform counters after each one.
# It stops at the first failed allocation or when available physical memory
# falls under 256 MB — deliberately short of OOM, because the PLM resolves OOM
# by killing the process, and a killed probe reports nothing.
#
# The device writes every row as it produces it, so if the OS does kill the app
# anyway, the CSV up to the last surviving step is still the answer.
set -euo pipefail

: "${XBOX_IP:?XBOX_IP not set — run: source ~/.config/xllama/xbox-env}"
: "${XBOX_USER:?XBOX_USER not set}"
: "${XBOX_PASS:?XBOX_PASS not set}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
DEPLOY="${SCRIPT_DIR}/deploy.sh"
BASE_URL="https://${XBOX_IP}:11443"
CURL_AUTH=(--basic -u "${XBOX_USER}:${XBOX_PASS}" -k -sS)

OUT="${REPO_ROOT}/bench/results/phase15-ramceil.csv"
while [[ $# -gt 0 ]]; do
	case "$1" in
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

CSRF_TOKEN=$(curl "${CURL_AUTH[@]}" "${BASE_URL}/" -o /dev/null -D - 2>/dev/null |
	sed -n 's/.*[Cc][Ss][Rr][Ff]-[Tt]oken=\([^;[:space:]]*\).*/\1/p' |
	tr -d '\r' | head -n1)
[[ -z "$CSRF_TOKEN" ]] && echo "Warning: no CSRF token — POST/DELETE may fail silently" >&2

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

echo "Probing the heap ceiling on ${XBOX_IP} (no model, no install)..."

remove "ramceil-result.csv"
remove "ramceil-result.csv.done"
# Append-only across restarts: clear it so the log below reflects only this run.
remove "xllama.log"

printf 'ramceil' >"${TMPDIR_LOCAL}/ramceil.flag"
upload "${TMPDIR_LOCAL}/ramceil.flag"
restart_app

# The probe walks up to 8 GB in 128 MB steps; page-faulting dominates, and the
# tail slows as the OS starts reclaiming. Poll for the .done marker, but keep
# the partial CSV if it never arrives — a kill mid-probe is itself a result.
csv="${TMPDIR_LOCAL}/ramceil.csv"
done_marker="${TMPDIR_LOCAL}/ramceil.done"
elapsed=0
ok=0
while ((elapsed < 300)); do
	: >"$done_marker"
	fetch "ramceil-result.csv.done" "$done_marker"
	if [[ -s "$done_marker" ]] && ! grep -qi 'html\|error' "$done_marker"; then
		ok=1
		break
	fi
	sleep 10
	((elapsed += 10))
done

: >"$csv"
fetch "ramceil-result.csv" "$csv"

if ! head -1 "$csv" 2>/dev/null | grep -q '^committed_mb,'; then
	echo "Error: no ramceil CSV came back from the console." >&2
	echo "Check that the deployed build includes ramceil.flag handling (App.cpp)." >&2
	exit 1
fi

rows=$(($(wc -l <"$csv") - 1))
if ((ok == 1)); then
	printf 'stop reason: %s\n' "$(tr -d '\r\n' <"$done_marker")"
else
	echo "WARNING: no .done marker after ${elapsed}s — the probe was likely killed."
	echo "The rows below are the surviving evidence; the last one is a lower bound."
fi

max=$(awk -F, 'NR>1 && $6==1 {m=$1} END {print m+0}' "$csv")
last_avail=$(awk -F, 'NR>1 && $6==1 {a=$3} END {print a+0}' "$csv")
printf 'steps recorded: %s\nmax committed:  %s MB\navail at stop:  %s MB\n' \
	"$rows" "$max" "$last_avail"

mkdir -p "$(dirname "$OUT")"
cp "$csv" "$OUT"
echo "wrote ${OUT}"

# The numbers this decides, spelled out so the run reports its own consequence.
printf '\nAdmissibility against this ceiling (weights + ~12%% measured load overhead):\n'
awk -v ceil="$max" 'BEGIN {
  split("LFM2.5-8B-A1B UD-Q4_K_M 5322;LFM2.5-8B-A1B UD-IQ4_XS 4265;" \
        "LFM2.5-8B-A1B UD-Q3_K_M 3940;LFM2.5-8B-A1B UD-IQ3_S 3571;" \
        "LFM2.5-8B-A1B UD-Q2_K_XL 2926;granite-3.1-3b-a800m Q4_K_M 2017", rows, ";")
  for (i = 1; i in rows; i++) {
    n = split(rows[i], f, " ")
    mb = f[n] * 1.12
    printf "  %-28s %-12s %6.0f MB peak  %s\n", f[1], f[2], mb,
           (mb <= ceil ? "fits" : "over")
  }
}'
