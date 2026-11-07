#!/usr/bin/env python3
"""Regression tests for publishable benchmark provenance validation."""

import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def valid_sidecar():
    return {
        "schema_version": 1, "model": "fixture", "model_bytes": 1024,
        "model_sha256": "a" * 64, "quant": "Q4_0", "build": "test-sha",
        "hardware": "host fixture", "n_ctx": 2048,
        "prompt_sha256": hashlib.sha256(b"fixture").hexdigest(),
        "sampling": {"temperature": 0}, "warmup_runs": 1, "recorded_runs": 3,
        "thermal_rule": "fixture only", "ambient_c": 20,
        "power": {"wattmeter": "fixture meter", "sample_interval_ms": 1000,
                  "idle_w": 0, "prefill_w": 50, "decode_w": 40},
    }


class BenchmarkSidecarTests(unittest.TestCase):
    def validate(self, data, prompt=None):
        with tempfile.TemporaryDirectory() as directory:
            sidecar = Path(directory) / "sidecar.json"
            sidecar.write_text(json.dumps(data), encoding="utf-8")
            command = [sys.executable, str(ROOT / "scripts/validate-benchmark-sidecar.py"), str(sidecar)]
            if prompt is not None:
                prompt_file = Path(directory) / "prompt.txt"
                prompt_file.write_bytes(prompt)
                command += ["--prompt", str(prompt_file)]
            return subprocess.run(command, capture_output=True, text=True)

    def assert_invalid(self, data):
        result = self.validate(data)
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_valid_metadata_and_matching_prompt(self):
        result = self.validate(valid_sidecar(), b"fixture")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_prompt_mismatch(self):
        self.assertEqual(self.validate(valid_sidecar(), b"different").returncode, 2)

    def test_rejects_invalid_field_values(self):
        cases = {
            "schema_version": [True, "1", 2], "model": ["", "  ", None, 42],
            "model_bytes": [True, 0, -1, 1.5, "1024"],
            "model_sha256": ["a" * 63, "A" * 64, "g" * 64, None],
            "quant": ["", None], "build": ["", None], "hardware": ["", None],
            "n_ctx": [True, 0, -1, 2.5, "2048"],
            "prompt_sha256": ["bad", "A" * 64, None],
            "sampling": [[], None, "{}", {"temperature": float("nan")}],
            "warmup_runs": [True, 0, "1", 1.5],
            "recorded_runs": [True, 2, "3", 3.5],
            "thermal_rule": ["", None],
            "ambient_c": [None, True, "20", float("nan"), float("inf")],
            "power": [None, [], "meter"],
        }
        for field, values in cases.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    data = valid_sidecar()
                    data[field] = value
                    self.assert_invalid(data)

    def test_rejects_invalid_power_values(self):
        for field, values in {
            "wattmeter": ["", None],
            "sample_interval_ms": [True, 0, -1, "1000", 1.5],
            "idle_w": [None, True, -1, "1", float("nan"), float("inf")],
            "prefill_w": [None, -1, float("nan")],
            "decode_w": [None, -1, float("inf")],
        }.items():
            for value in values:
                with self.subTest(field=field, value=value):
                    data = valid_sidecar()
                    data["power"][field] = value
                    self.assert_invalid(data)

    def test_rejects_missing_fields_and_non_object(self):
        for field in valid_sidecar():
            with self.subTest(field=field):
                data = valid_sidecar()
                del data[field]
                self.assert_invalid(data)
        for data in ([], None, 1):
            self.assert_invalid(data)
        data = valid_sidecar()
        del data["power"]["idle_w"]
        self.assert_invalid(data)

    def test_writer_rejects_invalid_metadata_before_replacing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "sidecar.json"
            output.write_text("previous evidence")
            prompt = Path(directory) / "prompt.txt"
            prompt.write_bytes(b"fixture")
            command = [sys.executable, str(ROOT / "scripts/write-benchmark-sidecar.py"), str(output),
                       "--model", "fixture", "--model-bytes", "1024", "--model-sha256", "bad",
                       "--quant", "Q4_0", "--build", "test-sha", "--hardware", "host fixture",
                       "--ctx", "2048", "--prompt", str(prompt), "--sampling", "{}",
                       "--warmup-runs", "1", "--recorded-runs", "3", "--thermal-rule", "fixture",
                       "--ambient-c", "20", "--wattmeter", "fixture", "--power-sample-ms", "1000",
                       "--idle-w", "0", "--prefill-w", "50", "--decode-w", "40"]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
            self.assertEqual(output.read_text(), "previous evidence")
            command[command.index("bad")] = "a" * 64
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            result = subprocess.run([sys.executable, str(ROOT / "scripts/validate-benchmark-sidecar.py"), str(output), "--prompt", str(prompt)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
