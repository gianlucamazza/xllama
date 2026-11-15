# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/analyze_ort_profile.py"
spec = importlib.util.spec_from_file_location("ort_profile", SCRIPT)
profile = importlib.util.module_from_spec(spec)
spec.loader.exec_module(profile)
SAMPLE = ROOT / "tests/fixtures/ort_profile_sample.json"


class OrtProfileTests(unittest.TestCase):
    def test_attribution_preserves_nodes_and_fractional_duration(self):
        events, complete = profile.parse_trace(SAMPLE.read_text())
        self.assertTrue(complete)
        events[-1]["dur"] = 1.5
        summary = profile.summarize(events)
        self.assertEqual(summary["kernel_events"], 8)
        self.assertEqual(summary["session_runs"], 2)
        node = next(n for n in summary["nodes"] if n["name"] == events[-1]["name"])
        self.assertEqual(node["count"], 2)
        self.assertEqual(node["dur_us"], 6101.5)
        self.assertIn("not GPU hardware time", summary["timing_scope"])

    def test_invalid_duration_not_counted_as_valid_compute(self):
        for dur in (-1, float("nan"), float("inf"), "10", None, True):
            with self.subTest(dur=dur):
                summary = profile.summarize([{"cat": "Node", "name": "n_kernel_time",
                                              "dur": dur, "args": {}}])
                self.assertEqual(summary["invalid_events"], 1)
                self.assertFalse(summary["providers"])

    def test_strict_cli_writes_hashed_report(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp) / "report.json"
            result = subprocess.run([sys.executable, str(SCRIPT), str(SAMPLE),
                                     "--strict", "--json-out", str(out)],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads(out.read_text())
            self.assertTrue(report["trace_complete"])
            self.assertTrue(report["profile_valid"])
            self.assertEqual(len(report["profile_sha256"]), 64)
            self.assertEqual(report["nodes"][0]["op"], "MatMul")

    def test_truncated_trace_remains_diagnostic_and_fails_strict(self):
        path = ROOT / "tests/fixtures/ort_profile_truncated.json"
        for flags, code in (([], 0), (["--strict"], 1)):
            result = subprocess.run([sys.executable, str(SCRIPT), str(path), *flags],
                                    capture_output=True, text=True)
            self.assertEqual(result.returncode, code)
            self.assertIn("DIAGNOSTIC ONLY", result.stdout)
            self.assertIn("VERDICT: INVALID-PROFILE", result.stdout)

    def test_invalid_and_empty_profiles_fail_strict(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "profile.json"
            for raw in ("[]", "{}", "42", '[null]', "broken", json.dumps([
                    {"name": "SessionRun"}, {"cat": "Node", "name": "n_kernel_time",
                     "dur": 10, "args": {"provider": "unknown"}}])):
                with self.subTest(raw=raw):
                    path.write_text(raw)
                    result = subprocess.run([sys.executable, str(SCRIPT), str(path), "--strict"],
                                            capture_output=True, text=True)
                    self.assertNotEqual(result.returncode, 0)
                    self.assertNotIn("Traceback", result.stderr)


if __name__ == "__main__":
    unittest.main()
