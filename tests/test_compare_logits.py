# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
"""Exercise the actual parity CLI, including retained failed console controls."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/compare-logits.py"
EVIDENCE = ROOT / "docs/audit/excellence/execution-2026-11-08"


class CompareLogitsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.a = Path(self.temp.name) / "a.bin"
        self.b = Path(self.temp.name) / "b.bin"
        for path in (self.a, self.b):
            np.array([1., 2., 3.], dtype=np.float32).tofile(path)

    def run_gate(self, *options, a=None, b=None):
        return subprocess.run(
            [sys.executable, str(SCRIPT), str(a or self.a), str(b or self.b), *options],
            capture_output=True, text=True, check=False,
        )

    def sidecar(self, path, data):
        Path(str(path) + ".json").write_text(json.dumps(data))

    def test_identical_valid_dump_passes(self):
        self.assertEqual(self.run_gate().returncode, 0)

    def test_shape_mismatch_never_passes_on_matching_piece(self):
        np.array([1., 2.], dtype=np.float32).tofile(self.b)
        for p in (self.a, self.b):
            self.sidecar(p, {"top1_piece": "same"})
        self.assertEqual(self.run_gate().returncode, 1)

    def test_invalid_binary_inputs_fail_without_traceback(self):
        for raw in (b"", b"x", self.a.read_bytes() + b"x",
                    np.array([np.nan], dtype=np.float32).tobytes(),
                    np.array([np.inf], dtype=np.float32).tobytes()):
            with self.subTest(raw=raw):
                self.b.write_bytes(raw)
                result = self.run_gate()
                self.assertEqual(result.returncode, 2)
                self.assertNotIn("Traceback", result.stderr)
        self.b.unlink()
        self.assertEqual(self.run_gate().returncode, 2)

    def test_malformed_or_inconsistent_sidecar_fails(self):
        for raw in ("{", "[]", '{"vocab_size": 4}', '{"vocab_size": true}'):
            with self.subTest(raw=raw):
                Path(str(self.b) + ".json").write_text(raw)
                self.assertEqual(self.run_gate().returncode, 2)

    def test_incompatible_metadata_fails(self):
        for key in ("prompt", "tokenizer_sha256", "vocab_sha256", "top1_piece"):
            with self.subTest(key=key):
                self.sidecar(self.a, {key: "A"})
                self.sidecar(self.b, {key: "B"})
                self.assertEqual(self.run_gate().returncode, 1)

    def test_invalid_thresholds_fail(self):
        for flag, value in (("--max-abs-diff", "nan"), ("--max-abs-diff", "-1"),
                            ("--nmse", "inf"), ("--nmse", "-1"),
                            ("--top-k", "0"), ("--top-k", "-1"),
                            ("--top-k-min", "nan"), ("--top-k-min", "1.1"),
                            ("--top-k-min", "-0.1")):
            with self.subTest(flag=flag, value=value):
                self.assertEqual(self.run_gate(flag, value).returncode, 2)

    def test_retained_console_results_keep_frozen_verdicts(self):
        for model, arm, expected in (("coder", "A", 1), ("lfm", "A", 0),
                                     ("lfm", "B", 0)):
            with self.subTest(model=model, arm=arm):
                result = self.run_gate(
                    a=EVIDENCE / f"prefill-logits-{model}-CPU.bin",
                    b=EVIDENCE / f"prefill-logits-{model}-{arm}.bin",
                )
                self.assertEqual(result.returncode, expected, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
