#!/usr/bin/env bash
# Apply UWP-compatibility patches to the llama.cpp submodule.
# Idempotent: skips patches that are already applied.
set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
PATCH_DIR="${REPO_ROOT}/uwp/patches/llama.cpp"
SUBMODULE_DIR="${REPO_ROOT}/llama.cpp"

if [ ! -d "$SUBMODULE_DIR/.git" ] && [ ! -f "$SUBMODULE_DIR/.git" ]; then
	echo "ERROR: llama.cpp submodule not initialised (run: git submodule update --init)" >&2
	exit 1
fi

cd "$SUBMODULE_DIR"

for p in "${PATCH_DIR}"/000*.patch; do
	[ -f "$p" ] || continue
	patch_name="$(basename "$p")"

	if git apply --check "$p" 2>/dev/null; then
		git apply "$p"
		echo "Applied:         ${patch_name}"
	elif git apply --check -R "$p" 2>/dev/null; then
		echo "Already applied: ${patch_name}"
	else
		echo "FAILED to apply: ${patch_name}" >&2
		echo "Try: cd llama.cpp && git apply --check ${p}" >&2
		exit 1
	fi
done

echo "UWP patches: done."
