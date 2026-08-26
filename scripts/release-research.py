#!/usr/bin/env python3
"""Run the safe local phases of a xLlama research release."""

from __future__ import annotations

import argparse
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(*args: str) -> None:
    subprocess.run(["python3", *args], cwd=ROOT, check=True)


def main() -> int:
    release = tomllib.loads((ROOT / "release.toml").read_text(encoding="utf-8"))["release"]
    version = str(release["version"])
    tag = f"{release['tag_prefix']}{version}"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reserve", action="store_true", help="reserve DOI before tagging")
    parser.add_argument("--base-url", default=None, help="Zenodo or Zenodo Sandbox URL")
    parser.add_argument("--ref", default="HEAD")
    args = parser.parse_args()
    run("scripts/generate-benchmark-summary.py", "--check")
    run("scripts/build-research-package.py", "--check")
    run("scripts/check-release-metadata.py")
    if args.reserve:
        command = ["scripts/zenodo-deposit.py", "--reserve-only"]
        if args.base_url:
            command.extend(["--base-url", args.base_url])
        run(*command)
    else:
        run("scripts/build-release-bundle.py", "--ref", args.ref)
        print(f"Local release preflight passed for {tag}")
        print("Next: reserve DOI, sync release.toml, commit/tag, then publish the deposit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
