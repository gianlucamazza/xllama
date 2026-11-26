#!/usr/bin/env bash
# crossbuild-uwp.sh — Linux → AppContainer layout matching the CI dual-CRT product.
#
#   crossbuild-uwp.sh [--out DIR] [--backend unified|llamacpp|ort]
#
# 1) apply UWP patches on llama.cpp; generate ggml-version.h and build-info.cpp
# 2) build-project with store /MD (UWP_STORE_CRT=1) when possible
# 3) pe-import-audit runs inside uwp-crossbuild ≥ 0.5.0 after link
# 4) stage app-local desktop CRT for ORT (stage-ort-desktop-crt.sh)
#
# Requires: uwp-crossbuild ≥ 0.5.4 on PATH (or UWP_CROSSBUILD_ROOT): one SDK
# version with its own UCRT, and per-source ClCompile metadata.
set -euo pipefail

here="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
repo="$(cd "$here/.." && pwd)"
out="${CROSSBUILD_OUT:-/tmp/xllama-layout}"
backend=unified

while [[ $# -gt 0 ]]; do
	case "$1" in
	-h | --help)
		sed -n '2,16p' "$0" | sed 's/^# \?//'
		exit 0
		;;
	--out) out="$2"; shift 2 ;;
	--backend) backend="$2"; shift 2 ;;
	*) echo "error: unknown argument: $1" >&2; exit 2 ;;
	esac
done

if [[ -n "${UWP_CROSSBUILD_ROOT:-}" && -x "$UWP_CROSSBUILD_ROOT/scripts/build-project.sh" ]]; then
	build_project="$UWP_CROSSBUILD_ROOT/scripts/build-project.sh"
elif command -v uwp-build-project >/dev/null; then
	build_project=$(command -v uwp-build-project)
elif [[ -x "$HOME/.local/lib/uwp-crossbuild/scripts/build-project.sh" ]]; then
	build_project="$HOME/.local/lib/uwp-crossbuild/scripts/build-project.sh"
else
	echo "error: uwp-build-project not found (need uwp-crossbuild ≥ 0.5.4)" >&2
	exit 1
fi
# uwp-crossbuild prints no version, so ask for the capability instead: before
# 0.5.4 the per-source include path of ggml_d3d12.cpp is dropped and the build
# stops ten minutes in, on a header it should have found.
build_sh="$(dirname "$(readlink -f "$build_project")")/build.sh"
if ! "$build_sh" --help 2>/dev/null | grep -q -- --source-include; then
	echo "error: $build_sh predates uwp-crossbuild 0.5.4 (no --source-include);
  upgrade it, then re-splat xwin with --sdk-version 10.0.22621 and re-run
  fetch-sdk.sh (docs/crossbuild-console.md, Prerequisites)" >&2
	exit 1
fi

echo "==> apply UWP patches"
"$repo/scripts/apply-llama-patches.sh"

echo "==> C++/WinRT pin (packages.config)"
# shellcheck source=scripts/ensure-cppwinrt-pin.sh
source "$repo/scripts/ensure-cppwinrt-pin.sh"

export UWP_STORE_CRT="${UWP_STORE_CRT:-1}"
export UWP_VCLIBS_ROOT="${UWP_VCLIBS_ROOT:-$HOME/.cache/uwp-crossbuild/vclibs}"

if [[ ! -d "$UWP_VCLIBS_ROOT/lib/x86_64" && ! -d "$UWP_VCLIBS_ROOT/lib/x64" ]]; then
	echo "warning: UWP_VCLIBS_ROOT has no libs — store /MD may fall back to /MT.
  Run fetch-vclibs first." >&2
fi

echo "==> generated sources (uwp/generated-sources.targets)"
# MSBuild runs these two actions from uwp/generated-sources.targets; the
# crossbuild's project reader lists a .targets file and does not evaluate it,
# so the same cmake scripts write the same files here. Change one, change the
# other.
generated="$repo/uwp/generated"
mkdir -p "$generated"
cmake -DGGML_SOURCE_DIR="$repo/llama.cpp/ggml" -DOUTPUT_FILE="$generated/ggml-version.h" \
	-P "$repo/cmake/GenerateGgmlVersion.cmake"
cmake -DLLAMA_SOURCE_DIR="$repo/llama.cpp" -DOUTPUT_FILE="$generated/build-info.cpp" \
	-P "$repo/cmake/GenerateLlamaBuildInfo.cmake"

echo "==> build-project ($backend) → $out"
(
	cd "$repo"
	"$build_project" \
		--project uwp/xllama.vcxproj \
		--out "$out" \
		--property "XllamaBackend=$backend"
)

echo "==> stage desktop CRT for ORT (dual-CRT package contract)"
"$here/stage-ort-desktop-crt.sh" --layout "$out"

if [[ -f "$out/onnxruntime.dll" ]]; then
	for d in MSVCP140.dll MSVCP140_1.dll VCRUNTIME140.dll VCRUNTIME140_1.dll; do
		[[ -f "$out/$d" ]] || {
			echo "error: layout has onnxruntime.dll but missing $d" >&2
			exit 1
		}
	done
fi

echo "==> layout ready: $out"
ls -la "$out"
