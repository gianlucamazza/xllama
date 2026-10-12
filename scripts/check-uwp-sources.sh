#!/usr/bin/env bash
# check-uwp-sources.sh — guard uwp/ggml-uwp.vcxproj against llama.cpp submodule drift.
#
# The Linux build globs llama.cpp sources via CMake; the UWP build lists them by
# hand in ggml-uwp.vcxproj. src/models/*.cpp is now a MSBuild wildcard (one file
# per architecture — the volatile set), so it can't drift. The top-level
# src/*.cpp list is still explicit; this check fails CI if the submodule gained a
# top-level source the vcxproj doesn't reference (the 657e011 bump did exactly
# this with llama-kv-cache-dsa/dsv4.cpp → LNK2001).
#
# It also fails if a listed llama.cpp source is gone (d812350 dropped
# ggml-cpu/iqp.cpp after ggml#27851; ggml-cpu.c now links tiled/ instead).
#
# Run after `git submodule update` (both UWP workflows do this via
# apply-uwp-patches.sh).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VCX="$ROOT/uwp/ggml-uwp.vcxproj"
VCX_DIR="$(dirname "$VCX")"
SRC="$ROOT/llama.cpp/src"

if [[ ! -f "$VCX" ]]; then
	echo "check-uwp-sources: $VCX not found" >&2
	exit 1
fi
if [[ ! -d "$SRC" ]]; then
	echo "check-uwp-sources: $SRC not found — is the submodule checked out?" >&2
	exit 1
fi

missing=0
for f in "$SRC"/*.cpp; do
	base="$(basename "$f")"
	# vcxproj entries look like: Include="..\llama.cpp\src\<base>"
	if ! grep -qF "\\src\\$base\"" "$VCX"; then
		echo "DRIFT: llama.cpp/src/$base is not referenced in ggml-uwp.vcxproj"
		missing=1
	fi
done

# The CPU backend has a separate explicit source inventory. Check its root,
# tiled k-quant kernels, and x86 kernels; optional AMX/other-architecture
# directories stay excluded.
for dir in \
	"$ROOT/llama.cpp/ggml/src/ggml-cpu" \
	"$ROOT/llama.cpp/ggml/src/ggml-cpu/tiled" \
	"$ROOT/llama.cpp/ggml/src/ggml-cpu/arch/x86"; do
	for f in "$dir"/*.cpp "$dir"/*.c; do
		[[ -f "$f" ]] || continue
		rel="${f#"$ROOT/"}"
		win_rel="${rel//\//\\}"
		if ! grep -qF "$win_rel\"" "$VCX"; then
			echo "DRIFT: $rel is not referenced in ggml-uwp.vcxproj"
			missing=1
		fi
	done
done

# Paths in the vcxproj are relative to uwp/. Skip MSBuild wildcards.
while IFS= read -r inc; do
	case "$inc" in
	*\**) continue ;;
	esac
	unix_rel="${inc//\\//}"
	if [[ ! -e "$VCX_DIR/$unix_rel" ]]; then
		echo "STALE: $inc is referenced in ggml-uwp.vcxproj but missing on disk"
		missing=1
	fi
done < <(sed -n 's/.*Include="\(\.\.\\llama\.cpp\\[^"]*\)".*/\1/p' "$VCX")

if [[ "$missing" -ne 0 ]]; then
	echo "" >&2
	echo "ggml-uwp.vcxproj is out of sync with the llama.cpp/CPU source inventory." >&2
	echo "Add the missing <ClCompile Include=\"..\\llama.cpp\\src\\<file>.cpp\" /> entries." >&2
	echo "Remove STALE entries whose files no longer exist in the submodule." >&2
	exit 1
fi

count="$(find "$SRC" -maxdepth 1 -name '*.cpp' | wc -l)"
echo "check-uwp-sources: top-level src in sync ($count files; models/*.cpp wildcarded)."
