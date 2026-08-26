#!/usr/bin/env python3
"""Tests for the Xbox AI Benchmark command contract."""

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts/run-xab.sh"


class XabContractTests(unittest.TestCase):
    def run_xab(self, *args):
        return subprocess.run(
            [str(RUNNER), *args], cwd=ROOT, text=True, capture_output=True, check=False
        )

    def test_dry_run_writes_workload_manifest(self):
        with tempfile.TemporaryDirectory() as output:
            result = self.run_xab(
                "--dry-run",
                "--models",
                "lfm25-230m,lfm25-350m",
                "--include",
                "text,kv,h9,diffusion",
                "--out",
                output,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((Path(output) / "manifest.json").read_text())
            self.assertEqual(manifest["workloads"], ["text", "kv", "h9", "diffusion"])
            self.assertEqual(manifest["models"], ["lfm25-230m", "lfm25-350m"])

    def test_unknown_workload_is_rejected(self):
        result = self.run_xab("--dry-run", "--include", "text,unknown")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("unsupported workload", result.stderr)

    def test_invalid_model_is_rejected(self):
        result = self.run_xab("--dry-run", "--models", "bad/model")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("invalid model id", result.stderr)


if __name__ == "__main__":
    unittest.main()
