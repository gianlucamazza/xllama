#!/usr/bin/env bash
# ensure-cppwinrt-pin.sh — cache C++/WinRT headers matching packages.config.
#
#   ensure-cppwinrt-pin.sh
#   # exports: UWP_CPPWINRT_EXE, UWP_CPPWINRT_INCLUDE
#
# Uses Microsoft.Windows.CppWinRT from uwp/packages (same pin as CI MSVC) to
# generate platform projection headers into ~/.cache/xllama/cppwinrt-<ver>/.
# Crossbuild then compiles and generates component code against that pin.
set -euo pipefail

here="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
repo="$(cd "$here/.." && pwd)"

# Read version from packages.config
cfg="$repo/uwp/packages.config"
[[ -f "$cfg" ]] || { echo "error: no $cfg" >&2; exit 1; }
ver=$(sed -n 's/.*Microsoft\.Windows\.CppWinRT" version="\([^"]*\)".*/\1/p' "$cfg" | head -1)
[[ -n "$ver" ]] || { echo "error: CppWinRT version not found in packages.config" >&2; exit 1; }

pkg="$repo/uwp/packages/Microsoft.Windows.CppWinRT.$ver"
exe="$pkg/bin/cppwinrt.exe"
[[ -f "$exe" ]] || {
	echo "error: $exe missing — restore NuGet (packages.config) first" >&2
	exit 1
}

cache="${XLLAMA_CPPWINRT_CACHE:-$HOME/.cache/xllama/cppwinrt-$ver}"
union="${UWP_SDK_ROOT:-$HOME/.cache/uwp-crossbuild/sdk}/Windows Kits/10/UnionMetadata/${UWP_SDK_VERSION:-10.0.22621.0}"

if [[ ! -f "$cache/winrt/base.h" ]]; then
	[[ -d "$union" ]] || { echo "error: no UnionMetadata at $union — run fetch-sdk.sh" >&2; exit 1; }
	echo "==> generating C++/WinRT $ver platform headers → $cache"
	mkdir -p "$cache"
	export WINEDEBUG=-all
	wine "$exe" -input "$(winepath -w "$union")" -output "$(winepath -w "$cache")" -verbose
	[[ -f "$cache/winrt/base.h" ]] || { echo "error: generation produced no winrt/base.h" >&2; exit 1; }
	got=$(sed -n 's/^#define CPPWINRT_VERSION "\(.*\)"/\1/p' "$cache/winrt/base.h" | head -1)
	[[ "$got" == "$ver" ]] || echo "warning: base.h says $got, packages.config says $ver" >&2
fi

export UWP_CPPWINRT_EXE="$exe"
export UWP_CPPWINRT_INCLUDE="$cache"
echo "UWP_CPPWINRT_EXE=$UWP_CPPWINRT_EXE"
echo "UWP_CPPWINRT_INCLUDE=$UWP_CPPWINRT_INCLUDE"
