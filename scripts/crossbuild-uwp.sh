#!/usr/bin/env bash
# crossbuild-uwp.sh — Linux → AppContainer layout matching the CI dual-CRT product.
#
#   crossbuild-uwp.sh [--out DIR] [--backend unified|llamacpp|ort]
#
# 1) apply UWP patches on llama.cpp
# 2) build-project with store /MD (UWP_STORE_CRT=1) when possible
# 3) pe-import-audit runs inside uwp-crossbuild ≥ 0.5.0 after link
# 4) stage app-local desktop CRT for ORT (stage-ort-desktop-crt.sh)
#
# Requires: uwp-crossbuild ≥ 0.5.0 on PATH (or UWP_CROSSBUILD_ROOT).
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
	echo "error: uwp-build-project not found (need uwp-crossbuild ≥ 0.5.0)" >&2
	exit 1
fi

echo "==> apply UWP patches"
"$repo/scripts/apply-uwp-patches.sh"

echo "==> C++/WinRT pin (packages.config)"
# shellcheck source=scripts/ensure-cppwinrt-pin.sh
source "$repo/scripts/ensure-cppwinrt-pin.sh"

export UWP_STORE_CRT="${UWP_STORE_CRT:-1}"
export UWP_VCLIBS_ROOT="${UWP_VCLIBS_ROOT:-$HOME/.cache/uwp-crossbuild/vclibs}"

if [[ ! -d "$UWP_VCLIBS_ROOT/lib/x86_64" && ! -d "$UWP_VCLIBS_ROOT/lib/x64" ]]; then
	echo "warning: UWP_VCLIBS_ROOT has no libs — store /MD may fall back to /MT.
  Run fetch-vclibs first." >&2
fi

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
