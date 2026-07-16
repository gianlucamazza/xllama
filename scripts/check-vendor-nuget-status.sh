#!/usr/bin/env bash
# check-vendor-nuget-status.sh — report whether shipping NuGet pins still need
# the vendor DLL overlays (PatchedGenAI #2280, PatchedOrt extdata).
#
# Exit codes:
#   0  status printed (always "success" for CI dashboards)
#   2  usage / network error that prevented a complete check
#
# Does not auto-drop pins — that requires a human + console smoke (issues #84/#85).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PKGS="$ROOT/uwp/packages.config"

genai_pin=$(python3 - <<'PY' "$PKGS"
import sys, xml.etree.ElementTree as ET
root = ET.parse(sys.argv[1]).getroot()
for p in root.findall("package"):
    if p.get("id") == "Microsoft.ML.OnnxRuntimeGenAI.DirectML":
        print(p.get("version", ""))
        break
PY
)

ort_pin=$(python3 - <<'PY' "$PKGS"
import sys, xml.etree.ElementTree as ET
root = ET.parse(sys.argv[1]).getroot()
for p in root.findall("package"):
    if p.get("id") == "Microsoft.ML.OnnxRuntime.DirectML":
        print(p.get("version", ""))
        break
PY
)

echo "=== xllama NuGet pins (uwp/packages.config) ==="
echo "  GenAI DirectML: $genai_pin"
echo "  ORT DirectML:   $ort_pin"
echo

fetch_versions() {
  local id="$1"
  curl -fsSL "https://api.nuget.org/v3-flatcontainer/${id}/index.json" \
    | python3 -c 'import json,sys; print(" ".join(json.load(sys.stdin).get("versions",[])))'
}

echo "=== NuGet.org latest (DirectML ids) ==="
genai_all=$(fetch_versions "microsoft.ml.onnxruntimegenai.directml") || {
  echo "error: could not fetch GenAI versions" >&2
  exit 2
}
ort_all=$(fetch_versions "microsoft.ml.onnxruntime.directml") || {
  echo "error: could not fetch ORT versions" >&2
  exit 2
}
genai_latest=${genai_all##* }
ort_latest=${ort_all##* }
echo "  GenAI latest: $genai_latest"
echo "  ORT latest:   $ort_latest"
echo

echo "=== Vendor pin status ==="
# #2280 merged 2026-07-13 on GenAI main; first NuGet that could include it is
# strictly > 0.14.1 (0.14.1 / a30f479 is pre-merge).
# Compare GenAI: any release strictly newer than the 0.14.1 pin is a drop candidate
# (0.14.1 / a30f479 predates #2280 merge on main).
if [[ "$genai_latest" == "$genai_pin" ]]; then
  echo "  PatchedGenAI: STILL REQUIRED (NuGet latest == pin $genai_pin; #2280 not released)"
  echo "                keep vendor-dlls-v1 onnxruntime-genai.dll + -PatchedGenAI"
  echo "                tracker: https://github.com/gianlucamazza/xllama/issues/84"
else
  echo "  PatchedGenAI: CANDIDATE TO DROP — NuGet latest is $genai_latest (pin $genai_pin)"
  echo "                verify CreateDmlObjects has agility_device_created fallback,"
  echo "                then run issue #84 checklist + console XAML+DML smoke"
fi

# ORT: pin 1.24.4 lacks #28509 (main) and ReadFile 16 MB (PR #29732).
if [[ "$ort_latest" == "$ort_pin" ]]; then
  echo "  PatchedOrt:   STILL REQUIRED (NuGet latest == pin $ort_pin)"
  echo "                keep vendor-dlls-v1 onnxruntime.dll + -PatchedOrt"
  echo "                trackers: https://github.com/gianlucamazza/xllama/issues/85"
  echo "                          https://github.com/microsoft/onnxruntime/pull/29732"
else
  echo "  PatchedOrt:   CANDIDATE TO DROP — NuGet latest is $ort_latest (pin $ort_pin)"
  echo "                verify #28509 path fix + small ReadFile chunk (or #29732),"
  echo "                then run issue #85 checklist + console extdata load smoke"
fi

echo
echo "=== Local pin hashes (if present) ==="
for sums in \
  vendor/onnxruntime-genai-patched/SHA256SUMS \
  vendor/onnxruntime-patched/SHA256SUMS
do
  if [[ -f "$ROOT/$sums" ]]; then
    echo "  $sums:"
    awk '/^[[:xdigit:]]{64}/ {print "    "$2"  "$1}' "$ROOT/$sums"
  fi
done
