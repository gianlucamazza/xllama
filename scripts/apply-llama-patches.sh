#!/usr/bin/env bash
# apply-llama-patches.sh — apply patches/0*-*.patch to the llama.cpp submodule.
#
# Every build compiles the same llama.cpp source: the Linux CMake build (CLI,
# host tests) and both UWP variants that compile ggml (uwp/ggml-uwp.vcxproj).
# The patches are AppContainer guards (no-ops off Windows) and backports of
# upstream fixes the pin predates. Each patch and its reason: patches/README.md.
#
# Applies every patches/0*-*.patch in order. Idempotent per patch, so re-running
# after adding a new one applies only the new one. A patch that neither applies
# nor is already present fails the run: after a submodule bump that means the
# pin moved under it (rebase it, or drop it if upstream now has the change).
#
#   --check   apply nothing; exit 1 if any patch is not applied (CMake runs this)
#
# Run from anywhere.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
mode=apply
case "${1:-}" in
"") ;;
--check) mode=check ;;
*)
	echo "usage: $0 [--check]" >&2
	exit 2
	;;
esac

cd "$ROOT/llama.cpp"
# Git otherwise walks up to the superproject from an empty submodule directory.
# In that directory git apply skips paths outside its prefix and reports success.
if [[ ! -e .git ]]; then
	echo "apply-llama-patches: initialize llama.cpp with git submodule update --init --recursive" >&2
	exit 1
fi
applied=0
present=0
missing=()
for patch in "$ROOT"/patches/0*-*.patch; do
	name="$(basename "$patch")"
	if git apply --reverse --check "$patch" 2>/dev/null; then
		present=$((present + 1))
		continue
	fi
	if [[ "$mode" == check ]]; then
		missing+=("$name")
		continue
	fi
	git apply --check "$patch"
	git apply "$patch"
	echo "apply-llama-patches: applied ${name}."
	applied=$((applied + 1))
done
if [[ "$mode" == check ]]; then
	if ((${#missing[@]})); then
		echo "apply-llama-patches: not applied: ${missing[*]} — run scripts/apply-llama-patches.sh" >&2
		exit 1
	fi
	echo "apply-llama-patches: ${present} patch(es) applied, llama.cpp @ $(git rev-parse --short HEAD)"
	exit 0
fi
echo "apply-llama-patches: ${applied} applied, ${present} already present, llama.cpp @ $(git rev-parse --short HEAD)"
