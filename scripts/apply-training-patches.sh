#!/usr/bin/env bash
# Apply shared numerical training patches on Linux and UWP.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT/llama.cpp"
patch="$ROOT/patches/0002-floppylm-gelu-backward.patch"
if ! git apply --reverse --check "$patch" 2>/dev/null; then
    git apply --check "$patch"
    git apply "$patch"
fi
