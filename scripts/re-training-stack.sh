#!/usr/bin/env bash
# Reverse-engineering probe for the training pillar stack.
# Usage: from repo root  ./scripts/re-training-stack.sh
# Optional: GENAI_DLL=/path/to/onnxruntime-genai.dll
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

echo "=== xllama training stack RE probe ==="
echo "repo: ${ROOT}"
echo

echo "--- 1) NuGet packages (uwp/packages.config) ---"
if [[ -f uwp/packages.config ]]; then
  rg -n 'package id=' uwp/packages.config || true
  if rg -q 'OnnxRuntime.Training|OnnxRuntimeTraining' uwp/packages.config; then
    echo "NOTE: Training package present"
  else
    echo "RE: no OnnxRuntime.Training package — inference-only NuGet pins"
  fi
else
  echo "missing uwp/packages.config"
fi
echo

echo "--- 2) llama.h adapter / train surface ---"
if [[ -f llama.cpp/include/llama.h ]]; then
  rg -n 'llama_adapter_lora|llama_set_adapters_lora' llama.cpp/include/llama.h | head -20 || true
  if [[ -f llama.cpp/examples/training/README.md ]]; then
    echo "llama-finetune README (first lines):"
    head -8 llama.cpp/examples/training/README.md | sed 's/^/  /'
  fi
else
  echo "llama.cpp/include/llama.h missing"
fi
echo

echo "--- 3) ORT GenAI DLL strings (adapters / training) ---"
GENAI_DLL="${GENAI_DLL:-}"
if [[ -z "${GENAI_DLL}" ]]; then
  GENAI_DLL="$(find vendor -name 'onnxruntime-genai.dll' 2>/dev/null | head -1 || true)"
fi
if [[ -n "${GENAI_DLL}" && -f "${GENAI_DLL}" ]]; then
  echo "DLL: ${GENAI_DLL}"
  strings "${GENAI_DLL}" | rg -i 'Oga(Create|Load|Set|Unload)Adapter|No adapter is available for DML|TrainingSession|OrtTraining' | sort -u | head -40
else
  echo "no onnxruntime-genai.dll found (set GENAI_DLL=... to probe)"
fi
echo

echo "--- 4) C++ capability matrix (if xllama-cli built) ---"
CLI=""
for c in build/linux-test/bin/xllama-cli build/linux-release/bin/xllama-cli build/bin/xllama-cli; do
  if [[ -x "${c}" ]]; then CLI="${c}"; break; fi
done
if [[ -n "${CLI}" ]]; then
  "${CLI}" --training-capabilities 2>/dev/null || echo "(binary lacks --training-capabilities — rebuild)"
else
  echo "xllama-cli not found under build/"
fi
echo

echo "--- 5) Host job smoke ---"
if [[ -n "${CLI}" && -f training/jobs/smollm2-360m-marker.json ]]; then
  "${CLI}" --validate-train-job training/jobs/smollm2-360m-marker.json || true
fi
echo
echo "=== done — see docs/training-architecture.md ==="
