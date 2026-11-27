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


TRAIN_START = re.compile(
    rb"(?m)^\[xllama\] train\.flag detected -> headless device training \(Lane B\)\r?\n"
)


D3D12_START = re.compile(
    rb"(?m)^\[xllama\] d3d12be\.flag detected -> headless d3d12 backend selftest\r?\n"
)
LAUNCH_MARKERS = {"app": START, "train": TRAIN_START, "d3d12": D3D12_START}


def scope_log(before: bytes, after: bytes, mode: str = "app") -> bytes:
    """Require one new launch in an unchanged-prefix append, never a fallback."""
    if not after.startswith(before):
        raise ValueError("log was truncated or replaced; run provenance is unknown")
    tail = after[len(before):]
    if mode not in LAUNCH_MARKERS:
        raise ValueError("unknown launch mode")
    pattern = LAUNCH_MARKERS[mode]
    starts = list(pattern.finditer(tail))
    # The verified append boundary is also a fresh line boundary: App.cpp can
    # append its complete startup record after an old process's partial line.
    # Match ONLY the new bytes, never complete a marker from the old prefix.
    if len(starts) != 1 or sum(len(p.findall(tail)) for p in LAUNCH_MARKERS.values()) != 1:
        name = "App::App()" if mode == "app" else f"headless {mode}"
        raise ValueError(f"expected exactly one new {name} marker")
    return tail[starts[0].start():]


def main() -> int:
    before, after, out = map(Path, sys.argv[1:4])
    mode = sys.argv[4] if len(sys.argv) == 5 else "app"
    # Do not leave a previous successful extraction at the output path on error.
    out.write_bytes(b"")
    try:
        out.write_bytes(scope_log(before.read_bytes(), after.read_bytes(), mode))
    except (OSError, ValueError) as exc:
        print(f"  FAIL: cannot scope this run's log: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
