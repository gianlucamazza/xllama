#!/usr/bin/env bash
# apply-uwp-patches.sh — apply the AppContainer guards to the llama.cpp submodule.
#
# The UWP (Xbox) build compiles ggml/llama sources directly (uwp/ggml-uwp.vcxproj).
# The patch (a) guards desktop-only Win32 APIs (SetThreadAffinityMask, registry,
# LoadLibrary, VirtualLock) with WINAPI_FAMILY_PARTITION so the sources build and
# run inside the AppContainer, and (b) enables file mmap on the AppContainer via
# CreateFileMappingFromApp/MapViewOfFileFromApp — with a model-loader fallback to
# a buffered read if the mapping is denied. See patches/README.md.
#
# Idempotent: skips if the patch is already applied. Run from anywhere.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PATCH="$ROOT/patches/0001-uwp-appcontainer-guards.patch"

cd "$ROOT/llama.cpp"
if git apply --reverse --check "$PATCH" 2>/dev/null; then
	echo "apply-uwp-patches: already applied — nothing to do."
	exit 0
fi
git apply --check "$PATCH"
git apply "$PATCH"
echo "apply-uwp-patches: applied $(basename "$PATCH") to llama.cpp @ $(git rev-parse --short HEAD)"
