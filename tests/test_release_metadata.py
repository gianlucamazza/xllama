#!/usr/bin/env python3
"""Contract tests for release metadata and archive inputs."""

import json
import subprocess
import tomllib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ReleaseMetadataTests(unittest.TestCase):
    def test_ssot_matches_generated_metadata(self):
        release = tomllib.loads((ROOT / "release.toml").read_text())["release"]
        citation = (ROOT / "CITATION.cff").read_text()
        zenodo = json.loads((ROOT / ".zenodo.json").read_text())
        self.assertIn(f'version: "{release["version"]}"', citation)
        self.assertEqual(zenodo["version"], release["version"])

    def test_checker_passes_without_publishing(self):
        result = subprocess.run(
            ["python3", "scripts/check-release-metadata.py"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_doi_is_recorded_after_reservation(self):
        release = tomllib.loads((ROOT / "release.toml").read_text())["release"]
        self.assertTrue(release["version_doi"])
        self.assertIn(f'doi: "{release["version_doi"]}"', (ROOT / "CITATION.cff").read_text())

    def test_tag_gate_accepts_reserved_doi(self):
        result = subprocess.run(
            ["python3", "scripts/check-release-metadata.py", "--require-doi"],
            cwd=ROOT,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
