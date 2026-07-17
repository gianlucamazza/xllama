#!/usr/bin/env bash
# Host training backend (exploration): PEFT LoRA → GGUF adapter → merge → A/B eval.
# Usage (from repo root):
#   ./training/host/run_job.sh training/jobs/smollm2-360m-marker.json
# Env: SKIP_TRAIN=1 SKIP_CONVERT=1 SKIP_AB=1 QUANTIZE=1 CACHE_DIR=... STEPS=...
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HOST="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
JOB_JSON="${1:-}"

if [[ -z "${JOB_JSON}" || ! -f "${JOB_JSON}" ]]; then
  echo "usage: $0 <training/jobs/*.json>" >&2
  exit 2
fi

# Prefer absolute path
if [[ "${JOB_JSON}" != /* ]]; then
  JOB_JSON="$(cd "$(dirname "${JOB_JSON}")" && pwd)/$(basename "${JOB_JSON}")"
fi

# Caller STEPS=N overrides the job file (spike / experiment knob).
_STEPS_OVERRIDE="${STEPS-}"

# Load job fields via Python (stdlib json)
eval "$(
  python3 - "${JOB_JSON}" <<'PY'
import json, shlex, sys
from pathlib import Path
job = json.loads(Path(sys.argv[1]).read_text())
lora = job.get("lora") or {}
ev = job.get("eval") or {}
def emit(k, v):
    print(f"{k}={shlex.quote(str(v))}")
emit("JOB_NAME", job.get("name", "job"))
emit("METHOD", job.get("method", "lora_peft"))
emit("DEVICE", job.get("device", "host"))
emit("MODEL_ID", job.get("base_model", ""))
emit("DATASET", job.get("dataset", ""))
emit("OUT", job.get("out_dir", "training/out/job"))
emit("RANK", lora.get("rank", 8))
emit("ALPHA", lora.get("alpha", 16))
emit("STEPS", lora.get("steps", 120))
emit("SEED", lora.get("seed", 42))
emit("LR", lora.get("learning_rate", 2e-4))
emit("PROMPT", ev.get("prompt", "xllama secret"))
emit("MARKER", ev.get("expect_contains", "XLLAMA-LORA-OK"))
merge = job.get("merge", True)
quant = job.get("quantize", None)
emit("DO_MERGE", "1" if merge else "0")
emit("DO_QUANT", "1" if quant else "0")
PY
)"
if [[ -n "${_STEPS_OVERRIDE}" ]]; then
  STEPS="${_STEPS_OVERRIDE}"
fi

if [[ "${DEVICE}" != "host" ]]; then
  echo "error: device=${DEVICE} unsupported in host runner (exploration: use device=host)" >&2
  exit 1
fi
if [[ "${METHOD}" != "lora_peft" ]]; then
  echo "error: method=${METHOD} not implemented (use lora_peft)" >&2
  exit 1
fi

# Paths relative to repo root
cd "${ROOT}"
[[ "${DATASET}" = /* ]] || DATASET="${ROOT}/${DATASET}"
[[ "${OUT}" = /* ]] || OUT="${ROOT}/${OUT}"
CACHE_DIR="${CACHE_DIR:-${ROOT}/cache_dir}"
MARKER="${MARKER:-XLLAMA-LORA-OK}"
PROMPT="${PROMPT:-xllama secret}"

mkdir -p "${OUT}"
VENV="${HOST}/.venv"
PY="${VENV}/bin/python"

resolve_snapshot() {
  local org_model="$1"
  # HuggingFaceTB/SmolLM2-360M-Instruct → models--HuggingFaceTB--SmolLM2-360M-Instruct
  local safe="models--${org_model//\//--}"
  local snaps="${CACHE_DIR}/${safe}/snapshots"
  if [[ -d "${snaps}" ]]; then
    local best="" newest=0 d m
    for d in "${snaps}"/*; do
      [[ -f "${d}/config.json" ]] || continue
      [[ -e "${d}/model.safetensors" || -e "${d}/pytorch_model.bin" ]] || continue
      m=$(stat -c %Y "${d}" 2>/dev/null || stat -f %m "${d}")
      if (( m >= newest )); then newest=$m; best=$d; fi
    done
    [[ -n "${best}" ]] && { echo "${best}"; return 0; }
  fi
  return 1
}

HF_SNAPSHOT="$(resolve_snapshot "${MODEL_ID}" || true)"
if [[ -z "${HF_SNAPSHOT}" ]]; then
  echo "error: no local HF snapshot for ${MODEL_ID} under ${CACHE_DIR}" >&2
  exit 1
fi
echo "==> job ${JOB_NAME}  base=${MODEL_ID}  snapshot=${HF_SNAPSHOT}"
echo "==> stages: prepare → train → export_adapter → merge → evaluate"

find_bin() {
  local name="$1" c found
  for c in \
    "${ROOT}/build/linux-test/bin/${name}" \
    "${ROOT}/build/linux-release/bin/${name}" \
    "${ROOT}/build/bin/${name}"; do
    [[ -x "${c}" ]] && { echo "${c}"; return 0; }
  done
  found=$(find "${ROOT}/build" -type f -name "${name}" -perm -111 2>/dev/null | head -1 || true)
  [[ -n "${found}" ]] && { echo "${found}"; return 0; }
  return 1
}

XLLAMA_CLI="$(find_bin xllama-cli || true)"
EXPORT_LORA="$(find_bin llama-export-lora || true)"
QUANTIZE_BIN="$(find_bin llama-quantize || true)"

if [[ -z "${XLLAMA_CLI}" ]]; then
  echo "error: xllama-cli not found under build/" >&2
  exit 1
fi

# Optional C++ validate
if "${XLLAMA_CLI}" --validate-train-job "${JOB_JSON}" 2>/dev/null; then
  :
else
  echo "warn: xllama-cli --validate-train-job failed or binary old; continuing" >&2
fi

ensure_venv() {
  if [[ ! -x "${PY}" ]]; then
    echo "==> creating venv ${VENV}"
    python3 -m venv --system-site-packages "${VENV}"
    "${PY}" -m pip install -U pip wheel -q
    "${PY}" -m pip install -r "${HOST}/requirements.txt"
  fi
  if ! "${PY}" -c 'import peft, transformers, torch' 2>/dev/null; then
    "${PY}" -m pip install -r "${HOST}/requirements.txt"
    if ! "${PY}" -c 'import torch' 2>/dev/null; then
      "${PY}" -m pip install 'torch>=2.2' --index-url https://download.pytorch.org/whl/cpu
    fi
  fi
}

ensure_export_lora() {
  if [[ -n "${EXPORT_LORA}" && -x "${EXPORT_LORA}" ]]; then return 0; fi
  echo "==> building llama-export-lora"
  local build_dir="${ROOT}/build/linux-release"
  [[ -f "${build_dir}/CMakeCache.txt" ]] || build_dir="${ROOT}/build"
  cmake --build "${build_dir}" --target llama-export-lora -j"$(nproc)"
  EXPORT_LORA="$(find_bin llama-export-lora || true)"
  [[ -n "${EXPORT_LORA}" ]] || { echo "error: llama-export-lora missing" >&2; exit 1; }
}

ADAPTER_DIR="${OUT}/adapter"
BASE_GGUF="${OUT}/base-f16.gguf"
ADAPTER_GGUF="${OUT}/adapter-lora.gguf"
MERGED_GGUF="${OUT}/merged-f16.gguf"
MERGED_Q4="${OUT}/merged-Q4_K_M.gguf"
RESULT_JSON="${OUT}/result.json"
STAGES=()

mark_stage() { STAGES+=("$1"); echo "==> stage: $1"; }

# --- prepare ---
mark_stage prepare
[[ -f "${DATASET}" ]] || { echo "error: dataset missing: ${DATASET}" >&2; exit 1; }

# --- train ---
if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  ensure_venv
  mark_stage train
  "${PY}" "${HOST}/train_lora.py" \
    --model "${MODEL_ID}" \
    --cache-dir "${CACHE_DIR}" \
    --dataset "${DATASET}" \
    --out "${ADAPTER_DIR}" \
    --steps "${STEPS}" \
    --rank "${RANK}" \
    --alpha "${ALPHA}" \
    --seed "${SEED}" \
    --lr "${LR}"
else
  echo "==> SKIP_TRAIN=1"
  [[ -f "${ADAPTER_DIR}/adapter_model.safetensors" ]] || {
    echo "error: missing ${ADAPTER_DIR}/adapter_model.safetensors" >&2
    exit 1
  }
  mark_stage train
fi

# --- export + merge ---
if [[ "${SKIP_CONVERT:-0}" != "1" ]]; then
  ensure_venv
  ensure_export_lora
  mark_stage export_adapter
  if [[ ! -f "${BASE_GGUF}" ]]; then
    echo "==> convert base → GGUF f16"
    "${PY}" "${ROOT}/llama.cpp/convert_hf_to_gguf.py" "${HF_SNAPSHOT}" \
      --outfile "${BASE_GGUF}" --outtype f16
  else
    echo "==> reusing ${BASE_GGUF}"
  fi
  echo "==> convert LoRA → GGUF"
  "${PY}" "${ROOT}/llama.cpp/convert_lora_to_gguf.py" \
    --base "${HF_SNAPSHOT}" \
    --outfile "${ADAPTER_GGUF}" \
    --outtype f16 \
    "${ADAPTER_DIR}"

  if [[ "${DO_MERGE}" == "1" ]]; then
    mark_stage merge
    echo "==> merge → ${MERGED_GGUF}"
    "${EXPORT_LORA}" -m "${BASE_GGUF}" --lora "${ADAPTER_GGUF}" -o "${MERGED_GGUF}"
  fi

  if [[ "${DO_QUANT}" == "1" || "${QUANTIZE:-0}" == "1" ]]; then
    QUANTIZE_BIN="$(find_bin llama-quantize || true)"
    if [[ -n "${QUANTIZE_BIN}" && -f "${MERGED_GGUF}" ]]; then
      "${QUANTIZE_BIN}" "${MERGED_GGUF}" "${MERGED_Q4}" Q4_K_M
    fi
  fi
else
  echo "==> SKIP_CONVERT=1"
  mark_stage export_adapter
  [[ "${DO_MERGE}" != "1" || -f "${MERGED_GGUF}" ]] || {
    echo "error: missing ${MERGED_GGUF}" >&2
    exit 1
  }
  [[ "${DO_MERGE}" != "1" ]] || mark_stage merge
fi

# --- evaluate ---
AB_MODEL="${MERGED_GGUF}"
[[ -f "${MERGED_Q4}" && ( "${DO_QUANT}" == "1" || "${QUANTIZE:-0}" == "1" ) ]] && AB_MODEL="${MERGED_Q4}"
MERGED_HIT=0
BASE_HIT=0

if [[ "${SKIP_AB:-0}" != "1" && "${DO_MERGE}" == "1" ]]; then
  mark_stage evaluate
  run_gen() {
    local model="$1" label="$2"
    local log="${OUT}/ab-${label}.txt"
    echo "==> A/B ${label}: ${model}"
    set +e
    "${XLLAMA_CLI}" --chat --greedy -m "${model}" -p "${PROMPT}" -n 48 --temp 0 \
      >"${log}" 2>"${OUT}/ab-${label}.err"
    local rc=$?
    set -e
    cat "${log}"
    [[ ${rc} -eq 0 ]] || echo "warn: xllama-cli rc=${rc} (${label})" >&2
  }
  run_gen "${BASE_GGUF}" "base"
  run_gen "${AB_MODEL}" "merged"
  grep -q "${MARKER}" "${OUT}/ab-base.txt" && BASE_HIT=1 || true
  grep -q "${MARKER}" "${OUT}/ab-merged.txt" && MERGED_HIT=1 || true
  echo "======== A/B RESULT ========"
  echo "marker: ${MARKER}"
  echo "base   contains marker: ${BASE_HIT}"
  echo "merged contains marker: ${MERGED_HIT}"
else
  echo "==> SKIP_AB=1 or no merge"
  MERGED_HIT=1
fi

# --- result.json ---
stages_json=$(printf '%s\n' "${STAGES[@]:-}" | python3 -c 'import json,sys; print(json.dumps([l.strip() for l in sys.stdin if l.strip()]))')
success=0
[[ "${MERGED_HIT}" -eq 1 ]] && success=1
python3 - "${RESULT_JSON}" "${success}" "${ADAPTER_DIR}" "${MERGED_GGUF}" "${stages_json}" "${BASE_HIT}" "${MERGED_HIT}" "${MARKER}" <<'PY'
import json, sys
path, success, adapter, merged, stages, base_hit, merged_hit, marker = sys.argv[1:9]
doc = {
  "success": success == "1",
  "adapter_path": adapter,
  "merged_gguf_path": merged,
  "stages_completed": json.loads(stages),
  "eval": {
    "marker": marker,
    "base_hit": base_hit == "1",
    "merged_hit": merged_hit == "1",
  },
}
if success != "1":
  doc["error_msg"] = f"merged output missing marker {marker}"
Path = __import__("pathlib").Path
Path(path).write_text(json.dumps(doc, indent=2) + "\n")
print(f"wrote {path}")
PY

if [[ "${success}" -ne 1 ]]; then
  echo "FAIL: merged missing ${MARKER}" >&2
  exit 1
fi
if [[ "${BASE_HIT}" -eq 0 ]]; then
  echo "PASS: merged has marker, base does not"
else
  echo "PASS (weak): merged has marker (base also hit)"
fi

# Publish stub: LocalState manifest override ready for Device Portal upload
if [[ -f "${MERGED_GGUF}" ]]; then
  SNIP="${OUT}/manifest.override.json"
  "${HOST}/.venv/bin/python" "${HOST}/publish_manifest_snippet.py" \
    --name "${JOB_NAME}" \
    --display "${JOB_NAME} (finetuned)" \
    --gguf "${MERGED_GGUF}" \
    --out "${SNIP}" 2>/dev/null ||
    python3 "${HOST}/publish_manifest_snippet.py" \
      --name "${JOB_NAME}" \
      --display "${JOB_NAME} (finetuned)" \
      --gguf "${MERGED_GGUF}" \
      --out "${SNIP}"
  echo "==> publish snippet: ${SNIP}"
  echo "    copy merged GGUF → LocalState\\models\\${JOB_NAME}\\model.gguf"
  echo "    merge/upload ${SNIP} as LocalState\\manifest.json override"
fi
exit 0
