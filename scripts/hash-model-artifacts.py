#!/usr/bin/env python3
"""Emit deterministic SHA-256 metadata for model catalogue files."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="directory containing model files")
    parser.add_argument("files", nargs="+", type=Path)
    args = parser.parse_args()
    rows = []
    for relative in args.files:
        path = args.root / relative
        if not path.is_file():
            parser.error(f"missing regular file: {path}")
        rows.append({"bytes": path.stat().st_size, "filename": relative.as_posix(),
                     "sha256": sha256(path)})
    print(json.dumps(rows, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
