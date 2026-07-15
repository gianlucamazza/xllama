#!/usr/bin/env bash
# build-fp16-dml-model.sh — build an fp16 DirectML ONNX GenAI model with UNMERGED
# external data, for the ">2 GB fp16 on GPU" unblock spike (docs/fp16-extdata-runbook.md).
#
# The ORT GenAI model builder emits model.onnx + model.onnx.data. We deliberately
# DO NOT run scripts/merge_onnx_external_data.py: keeping the .onnx.data external is
# the whole point — it is what today trips ValidateExternalDataPath/weakly_canonical
# on the Xbox AppContainer (docs/uwp-constraints.md §8). The spike tests whether the
# USB path (which does not traverse the inaccessible Q:\Users\UserMgr0 segment) lets
# that external-data model load anyway.
#
# Usage:
#   ./scripts/build-fp16-dml-model.sh [-m HF_MODEL] [-n NAME] [-o OUTDIR] [--smoke]
#
#   -m HF_MODEL   Hugging Face model id (default: meta-llama/Llama-3.2-1B-Instruct)
#   -n NAME       Output model dir name (default: derived from HF_MODEL)
#   -o OUTDIR     Staging root (default: dist/fp16-dml)
#   --smoke       Fast code-path test: build Qwen2.5-0.5B-Instruct (~1 GB, still
#                 external-data) to confirm the load path before the real >2 GB build.
#
# Requires (Linux/host, NOT the console): python3 with onnxruntime-genai, torch,
# transformers, onnx. Building a DirectML-targeted graph does NOT need the DML
# runtime (Windows-only) — it is graph construction + config emission only.
#
# Output: OUTDIR/NAME/{model.onnx, model.onnx.data, genai_config.json, ...}, ready
# to copy onto a USB stick at <usb>\xllama\models\NAME\ (see the runbook).

set -euo pipefail

HF_MODEL="meta-llama/Llama-3.2-1B-Instruct"
NAME=""
OUTDIR=""
SMOKE=false

while [[ $# -gt 0 ]]; do
	case "$1" in
	-m)
		HF_MODEL="${2:?-m requires a value}"
		shift 2
		;;
	-n)
		NAME="${2:?-n requires a value}"
		shift 2
		;;
	-o)
		OUTDIR="${2:?-o requires a value}"
		shift 2
		;;
	--smoke)
		SMOKE=true
		shift
		;;
	-h | --help)
		sed -n '2,32p' "$0"
		exit 0
		;;
	*)
		echo "Unknown argument: $1" >&2
		exit 1
		;;
	esac
done

if [[ "$SMOKE" == "true" ]]; then
	HF_MODEL="Qwen/Qwen2.5-0.5B-Instruct"
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
[[ -z "$NAME" ]] && NAME="$(echo "$HF_MODEL" | tr 'A-Z/' 'a-z-' | sed 's/[^a-z0-9._-]//g')-dml-fp16"
[[ -z "$OUTDIR" ]] && OUTDIR="${REPO_ROOT}/dist/fp16-dml"
MODEL_OUT="${OUTDIR}/${NAME}"
CACHE_DIR="${HOME}/.cache/xllama-fp16-dml/${NAME}"
GPU_BUDGET_MB=3801 # docs/uwp-constraints.md §7 — per-process Game budget

echo "=== build-fp16-dml-model ==="
echo "  HF model:  $HF_MODEL"
echo "  Name:      $NAME"
echo "  Out:       $MODEL_OUT"
[[ "$SMOKE" == "true" ]] && echo "  Mode:      SMOKE (code-path test, small model)"

# --- Preflight: builder availability -----------------------------------------
if ! python3 -c "import onnxruntime_genai.models.builder" 2>/dev/null; then
	cat >&2 <<'EOF'
error: ORT GenAI model builder not importable.

Install into a venv (recommended), e.g.:
  python3 -m venv ~/.venv/xllama-builder
  . ~/.venv/xllama-builder/bin/activate
  pip install onnxruntime-genai torch transformers onnx sentencepiece

Then re-run this script.
EOF
	exit 1
fi

# --- Build (fp16, DML), external data left UNMERGED --------------------------
mkdir -p "$MODEL_OUT" "$CACHE_DIR"
echo ""
echo "--- Running ORT GenAI model builder (-p fp16 -e dml) — this downloads weights and can take a while ..."
python3 -m onnxruntime_genai.models.builder \
	-m "$HF_MODEL" \
	-o "$MODEL_OUT" \
	-p fp16 \
	-e dml \
	-c "$CACHE_DIR"

# --- Verify external data is present (NOT merged) ----------------------------
ONNX="${MODEL_OUT}/model.onnx"
DATA="${MODEL_OUT}/model.onnx.data"
CFG="${MODEL_OUT}/genai_config.json"
[[ -f "$ONNX" ]] || {
	echo "error: builder did not produce $ONNX" >&2
	exit 1
}
if [[ ! -f "$DATA" ]]; then
	echo "warning: no model.onnx.data emitted — weights may be inlined in model.onnx." >&2
	echo "         This model does not exercise the external-data path; pick a larger -m." >&2
fi

# --- DML config post-process (mirror bench/configs/genai_config-dml-*.json) ---
# Ensure the DML provider_options block and DML graph-capture prerequisite are set.
if [[ -f "$CFG" ]]; then
	python3 - "$CFG" <<'PY'
import json, sys
p = sys.argv[1]
c = json.load(open(p))
so = c.setdefault("model", {}).setdefault("decoder", {}).setdefault("session_options", {})
po = so.get("provider_options") or []
# Force a single dml provider block with the arena/mem-pattern flags the project pins.
so["provider_options"] = [{"dml": {"enable_cpu_mem_arena": "0", "enable_mem_pattern": "0"}}]
# past_present_share_buffer:true is required for DML graph capture (uwp-constraints §5).
c.setdefault("search", {})["past_present_share_buffer"] = True
json.dump(c, open(p, "w"), indent=4)
print("  patched genai_config.json: dml provider_options + past_present_share_buffer=true")
PY
else
	echo "warning: no genai_config.json emitted by builder — check builder output." >&2
fi

# --- Size gate ---------------------------------------------------------------
echo ""
echo "--- Artifact sizes ---"
ONNX_B=$(stat -c%s "$ONNX" 2>/dev/null || echo 0)
DATA_B=$(stat -c%s "$DATA" 2>/dev/null || echo 0)
TOTAL_MB=$(((ONNX_B + DATA_B) / 1024 / 1024))
DATA_MB=$((DATA_B / 1024 / 1024))
printf '  model.onnx      : %d bytes\n' "$ONNX_B"
printf '  model.onnx.data : %d bytes (%d MB)\n' "$DATA_B" "$DATA_MB"
printf '  weights total   : %d MB\n' "$TOTAL_MB"
echo ""
if ((DATA_MB > 2048)); then
	echo "  ✓ external data > 2 GB — exercises the fix (today blocked by the merge ceiling)."
else
	echo "  ! external data ≤ 2 GB — mergeable today; useful only as a code-path smoke test."
fi
if ((TOTAL_MB > GPU_BUDGET_MB)); then
	echo "  ⚠ weights ${TOTAL_MB} MB exceed the ${GPU_BUDGET_MB} MB GPU budget — a full load may OOM."
	echo "    Note: the weakly_canonical check runs at model PARSE, before GPU weight alloc —"
	echo "    so the spike (does it get past ValidateExternalDataPath?) is still answerable."
else
	echo "  ✓ weights ${TOTAL_MB} MB fit the ${GPU_BUDGET_MB} MB GPU budget (before KV/activations)."
fi

# --- Next-step instructions --------------------------------------------------
cat <<EOF

Done. Staged at: ${MODEL_OUT}

Next (Fase 0 spike — needs the console + a USB stick):
  1. Copy the whole directory onto an NTFS USB stick:
       <usb>\\xllama\\models\\${NAME}\\   (model.onnx, model.onnx.data, genai_config.json, tokenizer*)
  2. Plug the USB into the Xbox (Dev Mode), then:
       source ~/.config/xllama/xbox-env
       ./scripts/spike-fp16-extdata-usb.sh ${NAME}
EOF
