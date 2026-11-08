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

    def test_series_x_text_profile_declares_hardware(self):
        with tempfile.TemporaryDirectory() as output:
            result = self.run_xab("--dry-run", "--profile", "series-x-devmode", "--out", output)
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads((Path(output) / "manifest.json").read_text())
            self.assertEqual(manifest["hardware"], "xbox-series-x")
            self.assertEqual(manifest["hardware_identity_source"], "operator-declared")
            self.assertIn("--hardware xbox-series-x", result.stdout)

    def test_hardware_profile_mismatch_is_rejected(self):
        result = self.run_xab("--dry-run", "--profile", "series-s-devmode", "--hardware", "xbox-series-x")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("hardware does not match profile", result.stderr)

    def test_series_x_uninstrumented_workloads_are_rejected(self):
        result = self.run_xab("--dry-run", "--profile", "series-x-devmode", "--include", "text,kv")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Series X profile currently supports text only", result.stderr)

    def test_text_harness_rejects_unknown_hardware_before_network(self):
        result = subprocess.run([str(ROOT / "scripts/bench-xbox-ort.sh"), "fixture", "--hardware", "unknown"],
                                cwd=ROOT, text=True, capture_output=True)
        self.assertEqual(result.returncode, 2)
        self.assertIn("unsupported benchmark hardware", result.stderr)

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
