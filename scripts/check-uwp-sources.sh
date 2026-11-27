#!/usr/bin/env bash
# Existing entry point used by the Windows builds and pin-update runbooks.
set -euo pipefail
exec python3 "$(dirname "$0")/check-uwp-sources.py" "$@"
