#!/usr/bin/env bash
# Host LoRA spike: PEFT train → GGUF adapter → merge → A/B with xllama-cli.
# Usage: from repo root:  ./scripts/lora-spike/run_spike.sh
# Env overrides: STEPS, SKIP_TRAIN=1, SKIP_CONVERT=1, SKIP_AB=1, QUANTIZE=1
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SPIKE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUT="${SPIKE}/out"
MARKER="XLLAMA-LORA-OK"
PROMPT="xllama secret"
MODEL_ID="HuggingFaceTB/SmolLM2-360M-Instruct"
STEPS="${STEPS:-120}"
CACHE_DIR="${CACHE_DIR:-${ROOT}/cache_dir}"

# Resolve HF snapshot (offline-friendly)
resolve_snapshot() {
  local snaps="${CACHE_DIR}/models--HuggingFaceTB--SmolLM2-360M-Instruct/snapshots"
  if [[ -d "${snaps}" ]]; then
    local best=""
    local newest=0
    local d
    for d in "${snaps}"/*; do
      [[ -f "${d}/config.json" ]] || continue
      [[ -e "${d}/model.safetensors" || -e "${d}/pytorch_model.bin" ]] || continue
      local m
      m=$(stat -c %Y "${d}" 2>/dev/null || stat -f %m "${d}")
      if (( m >= newest )); then
        newest=$m
        best=$d
      fi
    done
    if [[ -n "${best}" ]]; then
      echo "${best}"
      return 0
    fi
  fi
  return 1
}

HF_SNAPSHOT="$(resolve_snapshot || true)"
if [[ -z "${HF_SNAPSHOT}" ]]; then
  echo "error: no local SmolLM2-360M-Instruct snapshot under ${CACHE_DIR}" >&2
  echo "  populate cache_dir or set CACHE_DIR / download the model first" >&2
  exit 1
fi
echo "==> HF snapshot: ${HF_SNAPSHOT}"

# Binaries
find_bin() {
  local name="$1"
  local c
  for c in \
    "${ROOT}/build/linux-release/bin/${name}" \
    "${ROOT}/build/bin/${name}" \
    "${ROOT}/build/linux-release/llama.cpp/bin/${name}" \
    "${ROOT}/build/llama.cpp/bin/${name}"; do
    if [[ -x "${c}" ]]; then
      echo "${c}"
      return 0
    fi
  done
  # cmake often puts tools next to the build tree
  local found
  found=$(find "${ROOT}/build" -type f -name "${name}" -perm -111 2>/dev/null | head -1 || true)
  if [[ -n "${found}" ]]; then
    echo "${found}"
    return 0
  fi
  return 1
}

XLLAMA_CLI="$(find_bin xllama-cli || true)"
EXPORT_LORA="$(find_bin llama-export-lora || true)"
QUANTIZE_BIN="$(find_bin llama-quantize || true)"

if [[ -z "${XLLAMA_CLI}" ]]; then
  echo "error: xllama-cli not found under build/; build with cmake --preset linux-release" >&2
  exit 1
fi

mkdir -p "${OUT}"
VENV="${SPIKE}/.venv"
PY="${VENV}/bin/python"

ensure_venv() {
  if [[ ! -x "${PY}" ]]; then
    echo "==> creating venv ${VENV} (--system-site-packages for host torch)"
    # system-site-packages: reuse the host torch (often already installed) so we
    # only pip the small PEFT stack.
    python3 -m venv --system-site-packages "${VENV}"
    "${PY}" -m pip install -U pip wheel
    "${PY}" -m pip install -r "${SPIKE}/requirements.txt"
  fi
  # peft/transformers/torch must import
  if ! "${PY}" -c 'import peft, transformers, torch' 2>/dev/null; then
    echo "==> installing requirements into venv"
    "${PY}" -m pip install -r "${SPIKE}/requirements.txt"
    if ! "${PY}" -c 'import torch' 2>/dev/null; then
      echo "==> installing CPU torch into venv"
      "${PY}" -m pip install 'torch>=2.2' --index-url https://download.pytorch.org/whl/cpu
    fi
  fi
}

ensure_export_lora() {
  if [[ -n "${EXPORT_LORA}" && -x "${EXPORT_LORA}" ]]; then
    return 0
  fi
  echo "==> building llama-export-lora"
  local build_dir="${ROOT}/build/linux-release"
  if [[ ! -f "${build_dir}/CMakeCache.txt" ]]; then
    build_dir="${ROOT}/build"
  fi
  if [[ ! -f "${build_dir}/CMakeCache.txt" ]]; then
    echo "error: no cmake build dir (expected build/linux-release)" >&2
    exit 1
  fi
  cmake --build "${build_dir}" --target llama-export-lora -j"$(nproc)"
  EXPORT_LORA="$(find_bin llama-export-lora || true)"
  if [[ -z "${EXPORT_LORA}" ]]; then
    echo "error: llama-export-lora still missing after build" >&2
    exit 1
  fi
  echo "==> export-lora: ${EXPORT_LORA}"
}

BASE_GGUF="${OUT}/smollm2-360m-f16.gguf"
ADAPTER_DIR="${OUT}/adapter"
ADAPTER_GGUF="${OUT}/adapter-lora.gguf"
MERGED_GGUF="${OUT}/smollm2-360m-lora-merged-f16.gguf"
MERGED_Q4="${OUT}/smollm2-360m-lora-merged-Q4_K_M.gguf"

# --- venv + train ---
if [[ "${SKIP_TRAIN:-0}" != "1" ]]; then
  ensure_venv
  echo "==> training LoRA (${STEPS} steps)"
  "${PY}" "${SPIKE}/train_lora.py" \
    --model "${MODEL_ID}" \
    --cache-dir "${CACHE_DIR}" \
    --dataset "${SPIKE}/toy_dataset.jsonl" \
    --out "${ADAPTER_DIR}" \
    --steps "${STEPS}" \
    --seed 42
else
  echo "==> SKIP_TRAIN=1"
  [[ -f "${ADAPTER_DIR}/adapter_model.safetensors" ]] || {
    echo "error: missing ${ADAPTER_DIR}/adapter_model.safetensors" >&2
    exit 1
  }
fi

# --- convert base + adapter + merge ---
if [[ "${SKIP_CONVERT:-0}" != "1" ]]; then
  ensure_venv
  ensure_export_lora

  if [[ ! -f "${BASE_GGUF}" ]]; then
    echo "==> convert base → GGUF f16"
    "${PY}" "${ROOT}/llama.cpp/convert_hf_to_gguf.py" "${HF_SNAPSHOT}" \
      --outfile "${BASE_GGUF}" --outtype f16
  else
    echo "==> reusing ${BASE_GGUF}"
  fi

  echo "==> convert LoRA adapter → GGUF"
  "${PY}" "${ROOT}/llama.cpp/convert_lora_to_gguf.py" \
    --base "${HF_SNAPSHOT}" \
    --outfile "${ADAPTER_GGUF}" \
    --outtype f16 \
    "${ADAPTER_DIR}"

  echo "==> merge base + LoRA → ${MERGED_GGUF}"
  "${EXPORT_LORA}" \
    -m "${BASE_GGUF}" \
    --lora "${ADAPTER_GGUF}" \
    -o "${MERGED_GGUF}"

  if [[ "${QUANTIZE:-0}" == "1" ]]; then
    if [[ -z "${QUANTIZE_BIN}" ]]; then
      echo "==> building llama-quantize"
      cmake --build "${ROOT}/build/linux-release" --target llama-quantize -j"$(nproc)" || true
      QUANTIZE_BIN="$(find_bin llama-quantize || true)"
    fi
    if [[ -n "${QUANTIZE_BIN}" ]]; then
      echo "==> quantize merged → Q4_K_M"
      "${QUANTIZE_BIN}" "${MERGED_GGUF}" "${MERGED_Q4}" Q4_K_M
    else
      echo "warn: llama-quantize not found; skip quant" >&2
    fi
  fi
else
  echo "==> SKIP_CONVERT=1"
  [[ -f "${MERGED_GGUF}" ]] || {
    echo "error: missing ${MERGED_GGUF}" >&2
    exit 1
  }
fi

# --- A/B ---
if [[ "${SKIP_AB:-0}" == "1" ]]; then
  echo "==> SKIP_AB=1 done"
  exit 0
fi

AB_MODEL="${MERGED_GGUF}"
if [[ "${QUANTIZE:-0}" == "1" && -f "${MERGED_Q4}" ]]; then
  AB_MODEL="${MERGED_Q4}"
fi

run_gen() {
  local model="$1"
  local label="$2"
  local log="${OUT}/ab-${label}.txt"
  echo "==> A/B ${label}: ${model}"
  # --greedy for deterministic decode; --chat applies ChatML like the UI
  set +e
  "${XLLAMA_CLI}" --chat --greedy -m "${model}" -p "${PROMPT}" -n 48 --temp 0 \
    >"${log}" 2>"${OUT}/ab-${label}.err"
  local rc=$?
  set -e
  echo "--- ${label} stdout ---"
  cat "${log}"
  if [[ ${rc} -ne 0 ]]; then
    echo "warn: xllama-cli rc=${rc} for ${label}; stderr:" >&2
    cat "${OUT}/ab-${label}.err" >&2 || true
  fi
}

run_gen "${BASE_GGUF}" "base"
run_gen "${AB_MODEL}" "merged"

BASE_HIT=0
MERGED_HIT=0
grep -q "${MARKER}" "${OUT}/ab-base.txt" && BASE_HIT=1 || true
grep -q "${MARKER}" "${OUT}/ab-merged.txt" && MERGED_HIT=1 || true

echo
echo "======== A/B RESULT ========"
echo "marker: ${MARKER}"
echo "base   contains marker: ${BASE_HIT}"
echo "merged contains marker: ${MERGED_HIT}"

if [[ "${MERGED_HIT}" -eq 1 ]]; then
  if [[ "${BASE_HIT}" -eq 0 ]]; then
    echo "PASS: merged has marker, base does not"
  else
    echo "PASS (weak): both hit marker — still proves merge loads; base contamination unexpected"
  fi
  exit 0
fi

echo "FAIL: merged output missing ${MARKER}" >&2
echo "  try STEPS=200 ./scripts/lora-spike/run_spike.sh" >&2
echo "  logs: ${OUT}/ab-merged.txt ${OUT}/ab-merged.err" >&2
exit 1
