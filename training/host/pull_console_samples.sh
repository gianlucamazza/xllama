#!/usr/bin/env bash
# Pull preference samples from Xbox LocalState for host retrain (Phase 9 hybrid ops).
#
# Usage:
#   source ~/.config/xllama/xbox-env
#   ./training/host/pull_console_samples.sh [out.jsonl]
#
# Default out: training/out/console-samples/samples.jsonl
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEPLOY="${REPO_ROOT}/scripts/deploy.sh"

: "${XBOX_IP:?XBOX_IP not set — source ~/.config/xllama/xbox-env}"
: "${XBOX_USER:?XBOX_USER not set}"
: "${XBOX_PASS:?XBOX_PASS not set}"

OUT="${1:-${REPO_ROOT}/training/out/console-samples/samples.jsonl}"
mkdir -p "$(dirname "$OUT")"

PFN=$("${DEPLOY}" pfn 2>/dev/null)
[[ -n "$PFN" ]] || {
	echo "error: xllama not installed on console" >&2
	exit 1
}

echo "==> PFN: $PFN"
echo "==> fetching LocalState/training/samples.jsonl → $OUT"
if ! "${DEPLOY}" fetch-file "$PFN" "samples.jsonl" "$OUT" "training"; then
	echo "error: fetch failed (no samples yet? run validate-console-training.sh rate first)" >&2
	exit 1
fi

if [[ ! -s "$OUT" ]]; then
	echo "error: empty samples file" >&2
	exit 1
fi

n=$(wc -l <"$OUT" | tr -d ' ')
echo "==> lines: $n"
# Convert to train_lora JSONL (messages only; keep likes + corrections)
CONV="${OUT%.jsonl}.train.jsonl"
python3 - "$OUT" "$CONV" <<'PY'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
kept = 0
skipped = 0
with open(src, encoding="utf-8") as fin, open(dst, "w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            skipped += 1
            continue
        label = row.get("label", "")
        msgs = row.get("messages") or []
        if label == "dislike":
            skipped += 1
            continue
        if label == "correction" and row.get("preferred_assistant"):
            # Rebuild last assistant turn
            out_msgs = []
            for m in msgs:
                if m.get("role") == "assistant":
                    out_msgs.append({"role": "assistant", "content": row["preferred_assistant"]})
                else:
                    out_msgs.append({"role": m.get("role", "user"), "content": m.get("content", "")})
            msgs = out_msgs
        if not msgs:
            skipped += 1
            continue
        fout.write(json.dumps({"messages": msgs}, ensure_ascii=False) + "\n")
        kept += 1
print(f"train rows: {kept}  skipped: {skipped}  → {dst}", file=sys.stderr)
PY

echo "==> train dataset: $CONV"
echo "next: STEPS=80 ./training/host/run_job.sh training/jobs/from-console-samples.json"
