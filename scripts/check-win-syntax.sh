#!/usr/bin/env bash
# check-win-syntax.sh — clang-cl syntax-only check of the Windows branches in
# src/bridge (D3D12 probes, platform code) against a local xwin splat, so a
# Windows-only compile error shows up before the build-uwp CI round trip.
#
# Not a build: no linking, no uwp/ (C++/WinRT + XAML need the full UWP
# toolchain), and MSVC remains the reference compiler. Default file set: every
# src/bridge/*.cpp that has a `_WIN32` branch.
#
# One-time splat (Microsoft licence acceptance is yours; nothing is committed):
#   xwin --accept-license --arch x86_64 \
#     --cache-dir ~/.cache/uwp-crossbuild/xwin-download \
#     splat --output ~/.cache/uwp-crossbuild/xwin
#
# Usage: scripts/check-win-syntax.sh [file.cpp ...]   (paths relative to the repo)
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
XWIN="${UWP_XWIN_ROOT:-$HOME/.cache/uwp-crossbuild/xwin}"
# The llama.cpp submodule checkout (a worktree may borrow another checkout's
# at the same pin). Needed because the GGUF bridge files are XLLAMA_USE_LLAMA.
LLAMA="${LLAMA_CPP_DIR:-$ROOT/llama.cpp}"
CLANG_CL="${CLANG_CL:-clang-cl}"

command -v "$CLANG_CL" >/dev/null || {
	echo "clang-cl not found (pacman: clang)" >&2
	exit 1
}
[[ -d "$XWIN/crt/include" && -d "$XWIN/sdk/include/um" ]] || {
	echo "no xwin splat at $XWIN (see the header of this script)" >&2
	exit 1
}
[[ -f "$LLAMA/include/llama.h" ]] || {
	echo "no llama.cpp checkout at $LLAMA (git submodule update --init, or set LLAMA_CPP_DIR)" >&2
	exit 1
}

if [[ $# -eq 0 ]]; then
	mapfile -t files < <(cd "$ROOT" && grep -l '_WIN32' src/bridge/*.cpp)
else
	files=("$@")
fi

fail=0
for f in "${files[@]}"; do
	if "$CLANG_CL" --target=x86_64-pc-windows-msvc /Zs /std:c++17 /EHsc /W3 /nologo \
		-Wno-unused-command-line-argument \
		/imsvc "$XWIN/crt/include" /imsvc "$XWIN/sdk/include/ucrt" \
		/imsvc "$XWIN/sdk/include/um" /imsvc "$XWIN/sdk/include/shared" \
		/imsvc "$XWIN/sdk/include/winrt" \
		/DXLLAMA_BUILD_PROBES /DXLLAMA_USE_LLAMA=1 /DNOMINMAX /D_CRT_SECURE_NO_WARNINGS \
		/I "$ROOT/include" /I "$ROOT/src/bridge" /I "$ROOT/shaders/generated" \
		/I "$LLAMA/include" /I "$LLAMA/ggml/include" /I "$LLAMA/ggml/src" \
		"$ROOT/$f"; then
		echo "OK   $f"
	else
		echo "FAIL $f"
		fail=1
	fi
done
exit "$fail"
