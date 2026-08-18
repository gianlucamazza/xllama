#!/usr/bin/env bash
# apply-uwp-patches.sh — apply the UWP build patches to the llama.cpp submodule.
#
# The UWP (Xbox) build compiles ggml/llama sources directly (uwp/ggml-uwp.vcxproj),
# so a few upstream sources need local changes to build and run inside the
# AppContainer, and one needs a change to build under clang-cl at all. Each patch
# and its reason: patches/README.md.
#
# Applies every patches/0*-*.patch in order. Idempotent per patch, so re-running
# after adding a new one applies only the new one. Run from anywhere.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$ROOT/llama.cpp"
applied=0
skipped=0
for patch in "$ROOT"/patches/0*-*.patch; do
	name="$(basename "$patch")"
	if git apply --reverse --check "$patch" 2>/dev/null; then
		echo "apply-uwp-patches: ${name} already applied."
		skipped=$((skipped + 1))
		continue
	fi
	git apply --check "$patch"
	git apply "$patch"
	echo "apply-uwp-patches: applied ${name}."
	applied=$((applied + 1))
done
echo "apply-uwp-patches: ${applied} applied, ${skipped} already present, llama.cpp @ $(git rev-parse --short HEAD)"
