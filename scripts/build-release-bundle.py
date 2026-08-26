#!/usr/bin/env python3
"""Build the deterministic source/research archive for GitHub and Zenodo."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    release = tomllib.loads((ROOT / "release.toml").read_text(encoding="utf-8"))["release"]
    version = str(release["version"])
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default="HEAD")
    parser.add_argument("--output", type=Path, default=ROOT / "release-bundle")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    archive = args.output / f"xllama-research-{version}.tar.gz"
    prefix = f"xllama-research-{version}/"
    tar = subprocess.check_output(
        ["git", "archive", "--format=tar", f"--prefix={prefix}", args.ref], cwd=ROOT
    )
    # git archive fixes tar metadata from the commit; gzip's default current
    # timestamp was the remaining source of checksum drift.
    with archive.open("wb") as handle:
        with gzip.GzipFile(fileobj=handle, mode="wb", mtime=0) as compressed:
            compressed.write(tar)
    required = ["CITATION.cff", ".zenodo.json", "release.toml", "paper/paper.md", "paper/claims.json"]
    listing = subprocess.check_output(["tar", "-tzf", str(archive)], text=True)
    missing = [path for path in required if f"{prefix}{path}" not in listing.splitlines()]
    if missing:
        raise SystemExit(f"release ref does not contain required files: {', '.join(missing)}")
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (args.output / "SHA256SUMS").write_text(f"{digest}  {archive.name}\n", encoding="utf-8")
    print(f"OK {archive} sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
