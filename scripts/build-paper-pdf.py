#!/usr/bin/env python3
"""Build a printable PDF from the citable research report."""

from __future__ import annotations

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper" / "paper.md"
CSS = ROOT / "paper" / "pdf.css"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    for command in ("pandoc", "chromium"):
        if shutil.which(command) is None:
            raise SystemExit(f"required command not found: {command}")

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".html", dir=ROOT / "paper", delete=False, encoding="utf-8"
    ) as handle:
        html = Path(handle.name)

    try:
        subprocess.run(
            [
                "pandoc", str(PAPER), "--standalone", "--from=markdown", "--to=html5",
                "--css", str(CSS), "--resource-path", str(ROOT), "--output", str(html),
            ], cwd=ROOT, check=True
        )
        subprocess.run(
            [
                "chromium", "--headless", "--no-sandbox", "--disable-gpu",
                "--no-pdf-header-footer", f"--print-to-pdf={output}", html.as_uri(),
            ], cwd=ROOT, check=True, stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE, text=True
        )
    finally:
        html.unlink(missing_ok=True)

    if not output.is_file() or output.stat().st_size < 10_000:
        raise SystemExit(f"PDF output is missing or unexpectedly small: {output}")
    if output.read_bytes()[:5] != b"%PDF-":
        raise SystemExit(f"PDF output has an invalid header: {output}")
    print(f"OK {output} ({output.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
