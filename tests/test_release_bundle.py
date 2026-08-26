#!/usr/bin/env python3
"""Tests for reproducible release archive generation."""

import hashlib
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ReleaseBundleTests(unittest.TestCase):
    def test_same_ref_produces_same_archive(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            for output in (first, second):
                subprocess.run(
                    ["python3", "scripts/build-release-bundle.py", "--ref", "HEAD", "--output", str(output)],
                    cwd=ROOT,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            first_digest = hashlib.sha256(next(first.glob("*.tar.gz")).read_bytes()).digest()
            second_digest = hashlib.sha256(next(second.glob("*.tar.gz")).read_bytes()).digest()
            self.assertEqual(first_digest, second_digest)


if __name__ == "__main__":
    unittest.main()
