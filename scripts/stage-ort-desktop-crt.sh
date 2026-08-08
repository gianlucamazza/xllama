#!/usr/bin/env bash
# stage-ort-desktop-crt.sh — app-local desktop CRT for ORT in an AppContainer layout.
#
#   stage-ort-desktop-crt.sh --layout DIR [--from DIR]
#
# Same product contract as scripts/build-uwp.ps1 and uwp/xllama.vcxproj:
#   onnxruntime.dll / onnxruntime-genai.dll are built desktop /MD and import
#   MSVCP140.dll + VCRUNTIME140*.dll. In AppContainer those names are not
#   resolved from System32; they must sit in the package root next to the EXE.
#   The EXE itself must keep APP CRT (MSVCP140_APP via VCLibs) — pe-import-audit
#   fails if the EXE imports the desktop CRT DLLs.
#
# --from DIR defaults to (first hit that has all four DLLs):
#   $XLLAMA_DESKTOP_CRT_DIR
#   $UWP_DESKTOP_CRT_DIR
#   uwp/   (files placed like the Windows build-uwp.ps1 copy step)
#   ~/.cache/xllama/desktop-crt/
#
# Source must be a real MSVC VC143 CRT redist (or DLLs already shipped in a CI
# xllama package). Do not use Wine system32 substitutes.
set -euo pipefail

here="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && pwd)"
repo="$(cd "$here/.." && pwd)"

layout=""
from=""

usage() {
	sed -n '2,25p' "$0" | sed 's/^# \?//'
	exit "${1:-0}"
}

while [[ $# -gt 0 ]]; do
	case "$1" in
	-h | --help) usage 0 ;;
	--layout)
		[[ $# -ge 2 ]] || { echo "error: --layout needs a path" >&2; exit 2; }
		layout="$2"
		shift 2
		;;
	--from)
		[[ $# -ge 2 ]] || { echo "error: --from needs a path" >&2; exit 2; }
		from="$2"
		shift 2
		;;
	*)
		echo "error: unknown argument: $1" >&2
		usage 2
		;;
	esac
done

[[ -n "$layout" ]] || { echo "error: --layout DIR is required" >&2; exit 2; }
[[ -d "$layout" ]] || { echo "error: layout is not a directory: $layout" >&2; exit 2; }

dlls=(MSVCP140.dll MSVCP140_1.dll VCRUNTIME140.dll VCRUNTIME140_1.dll)

has_all() {
	local dir="$1" d
	[[ -d "$dir" ]] || return 1
	for d in "${dlls[@]}"; do
		[[ -f "$dir/$d" ]] || return 1
	done
	return 0
}

if [[ -z "$from" ]]; then
	for candidate in \
		"${XLLAMA_DESKTOP_CRT_DIR:-}" \
		"${UWP_DESKTOP_CRT_DIR:-}" \
		"$repo/uwp" \
		"${HOME}/.cache/xllama/desktop-crt"; do
		[[ -n "$candidate" ]] || continue
		if has_all "$candidate"; then
			from="$candidate"
			break
		fi
	done
fi

[[ -n "$from" && -d "$from" ]] || {
	echo "error: no desktop CRT source with all of:
  ${dlls[*]}
Set --from DIR, or XLLAMA_DESKTOP_CRT_DIR, or copy the VC143 CRT redist DLLs
into uwp/ (Windows: build-uwp.ps1) or ~/.cache/xllama/desktop-crt/." >&2
	exit 1
}

missing=0
for d in "${dlls[@]}"; do
	if [[ ! -f "$from/$d" ]]; then
		echo "error: missing $d in $from" >&2
		missing=1
	fi
done
[[ $missing -eq 0 ]] || exit 1

msvcp_size=$(stat -c%s "$from/MSVCP140.dll" 2>/dev/null || stat -f%z "$from/MSVCP140.dll")
if [[ "${msvcp_size:-0}" -lt 100000 ]]; then
	echo "error: $from/MSVCP140.dll is only ${msvcp_size} bytes — refusing likely stub/Wine copy.
Use the MSVC VC143 CRT redist or DLLs from a CI xllama package." >&2
	exit 1
fi

for d in "${dlls[@]}"; do
	cp -f "$from/$d" "$layout/$d"
	sz=$(stat -c%s "$layout/$d" 2>/dev/null || stat -f%z "$layout/$d")
	echo "  staged $d ($sz bytes)"
done

if [[ -f "$layout/onnxruntime.dll" ]] && command -v llvm-readobj >/dev/null; then
	if ! llvm-readobj --coff-imports "$layout/onnxruntime.dll" 2>/dev/null |
		grep -qiE 'Name:[[:space:]]*MSVCP140\.dll'; then
		echo "warning: onnxruntime.dll does not import MSVCP140.dll — CRT still staged for peers." >&2
	fi
fi

for candidate in \
	"$repo/uwp/packages/Microsoft.ML.OnnxRuntime.DirectML.1.24.4/runtimes/win-x64/native/onnxruntime_providers_shared.dll" \
	"$from/onnxruntime_providers_shared.dll"; do
	if [[ -f "$candidate" && ! -f "$layout/onnxruntime_providers_shared.dll" ]]; then
		cp -f "$candidate" "$layout/onnxruntime_providers_shared.dll"
		echo "  staged onnxruntime_providers_shared.dll"
		break
	fi
done

echo "desktop CRT for ORT staged into $layout (from $from)"
