#!/usr/bin/env python3
"""Validate provenance metadata for a publishable benchmark claim."""

import argparse
import hashlib
import json
from pathlib import Path

from benchmark_sidecar import validate_sidecar


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sidecar", type=Path)
    parser.add_argument("--prompt", type=Path)
    args = parser.parse_args()
    try:
        data = json.loads(args.sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"cannot read sidecar: {exc}")
    try:
        validate_sidecar(data)
    except ValueError as exc:
        parser.error(str(exc))
    if args.prompt:
        actual = hashlib.sha256(args.prompt.read_bytes()).hexdigest()
        if actual != data["prompt_sha256"]:
            parser.error("prompt_sha256 does not match --prompt")
    print(f"OK sidecar schema=1 file={args.sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
