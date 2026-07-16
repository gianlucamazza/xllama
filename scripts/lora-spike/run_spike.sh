#!/usr/bin/env bash
# Compat shim — the training pillar lives under training/.
# Forwards to training/host/run_job.sh with the marker job.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
echo "note: scripts/lora-spike is deprecated; use training/host/run_job.sh" >&2
exec "${ROOT}/training/host/run_job.sh" "${ROOT}/training/jobs/smollm2-360m-marker.json" "$@"
