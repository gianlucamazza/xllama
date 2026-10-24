#!/usr/bin/env python3
"""Fail-closed log scoping for console gates (issue #347)."""

import re
import sys
from pathlib import Path


# App.cpp prefixes entries with UTC time; older logs may have no timestamp.
START = re.compile(
    rb"(?m)^(?:[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{3} )?"
    rb"\[xllama\] App::App\(\)\r?\n"
)


def scope_log(before: bytes, after: bytes) -> bytes:
    """Require one new launch in an unchanged-prefix append, never a fallback."""
    if not after.startswith(before):
        raise ValueError("log was truncated or replaced; run provenance is unknown")
    tail = after[len(before):]
    starts = list(START.finditer(tail))
    # The verified append boundary is also a fresh line boundary: App.cpp can
    # append its complete startup record after an old process's partial line.
    # Match ONLY the new bytes, never complete a marker from the old prefix.
    if len(starts) != 1:
        raise ValueError("expected exactly one new App::App() marker")
    return tail[starts[0].start():]


def main() -> int:
    before, after, out = map(Path, sys.argv[1:])
    # Do not leave a previous successful extraction at the output path on error.
    out.write_bytes(b"")
    try:
        out.write_bytes(scope_log(before.read_bytes(), after.read_bytes()))
    except (OSError, ValueError) as exc:
        print(f"  FAIL: cannot scope this run's log: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
