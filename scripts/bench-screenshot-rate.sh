#!/usr/bin/env bash
# bench-screenshot-rate.sh — how fast can the Device Portal actually hand us frames?
#
# Usage:
#   source ~/.config/xllama/xbox-env
#   ./scripts/bench-screenshot-rate.sh [n_frames]
#
# capture-demo-video.sh polls GET /ext/screenshot in a loop with `sleep 1`, and
# the resulting demo is described as "1 Hz". That 1 is a number someone typed,
# not one anyone measured: the real ceiling is whatever the console, the TLS
# handshake and the PNG encode allow, and it could be higher or lower. A demo
# rebuilt from stills is only as smooth as this number, so measure it before
# tuning anything around it.
#
# Prints the per-frame latency distribution and the sustained rate. Run it while
# the app is idle AND while it is decoding if you care about the difference —
# the console is a shared SoC and this is not free.

set -euo pipefail

: "${XBOX_IP:?XBOX_IP not set — run: source ~/.config/xllama/xbox-env}"
: "${XBOX_USER:?XBOX_USER not set}"
: "${XBOX_PASS:?XBOX_PASS not set}"

N="${1:-30}"
BASE_URL="https://${XBOX_IP}:11443"
CURL_AUTH=(--basic -u "${XBOX_USER}:${XBOX_PASS}" -k -sS)

WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

echo "==> ${N} back-to-back screenshots, no sleep between them"

# One warm-up: the first request pays the TLS handshake, and folding that into
# the distribution would understate the sustained rate.
curl "${CURL_AUTH[@]}" -o "${WORK}/warmup.png" --fail "${BASE_URL}/ext/screenshot" \
	>/dev/null 2>&1 || {
	echo "Error: GET /ext/screenshot failed — is the console reachable?" >&2
	exit 1
}

: >"${WORK}/times.txt"
failed=0
start_all=$(date +%s.%N)
for i in $(seq "$N"); do
	# Branch on curl's exit status, not on its stdout: --fail still prints the
	# -w value before exiting non-zero, so a failed request would otherwise be
	# recorded as a fast one.
	if t=$(curl "${CURL_AUTH[@]}" -o "${WORK}/f.png" --fail \
		-w "%{time_total}" "${BASE_URL}/ext/screenshot" 2>/dev/null); then
		echo "$t" >>"${WORK}/times.txt"
	else
		failed=$((failed + 1))
		continue
	fi
	printf '\r  %d/%d' "$i" "$N" >&2
done
end_all=$(date +%s.%N)
echo "" >&2

n_ok=$(wc -l <"${WORK}/times.txt")
if ((n_ok == 0)); then
	echo "Error: every request failed" >&2
	exit 1
fi

size=$(stat -c %s "${WORK}/f.png" 2>/dev/null || echo 0)
dims="unknown"
command -v identify >/dev/null 2>&1 && dims=$(identify -format '%wx%h' "${WORK}/f.png" 2>/dev/null || echo unknown)

sort -g "${WORK}/times.txt" >"${WORK}/sorted.txt"
pick() { sed -n "$1p" "${WORK}/sorted.txt"; }
p50=$(pick $(((n_ok + 1) / 2)))
p90=$(pick $(((n_ok * 9 + 9) / 10)))
tmin=$(head -n1 "${WORK}/sorted.txt")
tmax=$(tail -n1 "${WORK}/sorted.txt")

# Sustained rate over the whole loop, which is the number that matters for a
# capture: it includes whatever the per-request overhead really is.
awk -v n="$n_ok" -v f="$failed" -v s="$start_all" -v e="$end_all" \
	-v p50="$p50" -v p90="$p90" -v tmin="$tmin" -v tmax="$tmax" \
	-v sz="$size" -v dims="$dims" 'BEGIN {
	el = e - s
	printf "\n  frames ok        %d (failed %d)\n", n, f
	printf "  frame            %s, %d bytes\n", dims, sz
	printf "  latency min/p50/p90/max  %.3f / %.3f / %.3f / %.3f s\n", tmin, p50, p90, tmax
	printf "  elapsed          %.2f s\n", el
	# A clock that did not advance means the measurement did not happen; a rate
	# printed from it would look like an answer.
	if (el > 0) printf "  sustained rate   %.2f Hz\n", n / el
	else        printf "  sustained rate   n/a (elapsed measured as zero)\n"
	printf "\n  capture-demo-video.sh sleeps 1 s per frame; sustained above is what\n"
	printf "  the endpoint allows with no sleep at all.\n"
}'
