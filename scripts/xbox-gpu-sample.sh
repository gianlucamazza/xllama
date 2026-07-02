#!/usr/bin/env bash
# xbox-gpu-sample.sh — sample Xbox Device Portal GPU telemetry during a run.
#
# Polls GET /api/resourcemanager/systemperf (~1 Hz) and writes a CSV with
# per-engine GPU utilization and dedicated memory used. System-wide counters:
# run a control pass with the CPU config to calibrate background noise.
#
# Usage:
#   source ~/.config/xllama/xbox-env
#   ./scripts/xbox-gpu-sample.sh --out FILE [--interval 1] [--duration N]
#   ./scripts/xbox-gpu-sample.sh --parse-stdin   # smoke test: JSON on stdin → CSV row
#
# Without --duration it runs until SIGINT/SIGTERM; on exit it prints a
# per-engine max/mean summary from the collected CSV.

set -euo pipefail

OUT=""
INTERVAL=1
DURATION=0
PARSE_STDIN=false

while [[ $# -gt 0 ]]; do
	case "$1" in
	--out)
		OUT="${2:?--out requires a file}"
		shift 2
		;;
	--interval)
		INTERVAL="${2:?--interval requires seconds}"
		shift 2
		;;
	--duration)
		DURATION="${2:?--duration requires seconds}"
		shift 2
		;;
	--parse-stdin)
		PARSE_STDIN=true
		shift
		;;
	*)
		echo "Unknown argument: $1" >&2
		exit 1
		;;
	esac
done

# Parse one systemperf JSON (stdin) → one CSV row:
#   epoch,cpu_load,gpu_dedicated_used_mb,engine_0,...,engine_N
parse_sample() {
	python3 -c '
import json, sys, time

doc = json.load(sys.stdin)
adapters = (doc.get("GPUData") or {}).get("AvailableAdapters") or []
adapter = adapters[0] if adapters else {}
engines = adapter.get("EnginesUtilization") or []
used_mb = int(adapter.get("DedicatedMemoryUsed") or 0) // 1048576
cpu = doc.get("CpuLoad", "")
row = [str(int(time.time())), str(cpu), str(used_mb)]
row += ["%.3f" % float(e) for e in engines]
print(",".join(row))
'
}

if [[ "$PARSE_STDIN" == "true" ]]; then
	parse_sample
	exit 0
fi

[[ -z "$OUT" ]] && {
	echo "Error: --out FILE is required (or use --parse-stdin)" >&2
	exit 1
}
: "${XBOX_IP:?XBOX_IP not set — run: source ~/.config/xllama/xbox-env}"
: "${XBOX_USER:?XBOX_USER not set}"
: "${XBOX_PASS:?XBOX_PASS not set}"

BASE_URL="https://${XBOX_IP}:11443"
CURL_AUTH=(--basic -u "${XBOX_USER}:${XBOX_PASS}" -k -sS)

summary() {
	[[ -s "$OUT" ]] || return 0
	echo "--- gpu-sample summary ($OUT) ---"
	awk -F, '
    NR == 1 { for (i = 4; i <= NF; i++) names[i] = $i; next }
    {
        if ($3 > mem_max) mem_max = $3
        mem_sum += $3
        for (i = 4; i <= NF; i++) {
            if ($i > emax[i]) emax[i] = $i
            esum[i] += $i
        }
        n++
    }
    END {
        if (n == 0) exit
        printf "samples: %d\n", n
        printf "gpu_dedicated_used_mb: max=%d mean=%.0f\n", mem_max, mem_sum / n
        for (i = 4; i in names; i++)
            printf "%s: max=%.3f mean=%.3f\n", names[i], emax[i], esum[i] / n
    }' "$OUT"
}
trap 'summary; exit 0' INT TERM

: >"$OUT"
elapsed=0
while :; do
	sample=$(curl "${CURL_AUTH[@]}" "${BASE_URL}/api/resourcemanager/systemperf" 2>/dev/null |
		parse_sample 2>/dev/null) || sample=""
	if [[ -n "$sample" ]]; then
		if [[ ! -s "$OUT" ]]; then
			n_engines=$(($(awk -F, '{print NF}' <<<"$sample") - 3))
			header="timestamp,cpu_load,gpu_dedicated_used_mb"
			for ((i = 0; i < n_engines; i++)); do header+=",engine_${i}"; done
			printf '%s\n' "$header" >>"$OUT"
		fi
		printf '%s\n' "$sample" >>"$OUT"
	fi
	sleep "$INTERVAL"
	((elapsed += INTERVAL))
	if ((DURATION > 0 && elapsed >= DURATION)); then
		summary
		exit 0
	fi
done
