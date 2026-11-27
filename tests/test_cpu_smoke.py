# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
import contextlib
import copy
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cpu_smoke", ROOT / "scripts/run-cpu-smoke.py")
smoke = importlib.util.module_from_spec(spec)
spec.loader.exec_module(smoke)


class CpuSmokeTest(unittest.TestCase):
    def setUp(self):
        self.ref = json.loads(smoke.GOLDEN.read_bytes())
        self.actual = copy.deepcopy(self.ref)
        self.actual["turns"] = [
            dict(text="eggg", n_prompt=5, n_p_eval=5),
            dict(text="pjpj", n_prompt=2, n_p_eval=2),
            dict(text="fffh", n_prompt=3, n_p_eval=2),
        ]

    def check(self):
        with contextlib.redirect_stdout(io.StringIO()):
            smoke.judge(self.actual, self.ref)

    def test_positive(self):
        self.check()

    def test_winner_preserving_drift_in_late_decode(self):
        self.actual["rows"][-1][4] += 0.1
        with self.assertRaisesRegex(ValueError, "position 8"):
            self.check()

    def test_nonfinite_nonwinner(self):
        self.actual["rows"][0][0] = float("nan")
        with self.assertRaisesRegex(ValueError, "nonfinite"):
            self.check()

    def test_truncated_capture(self):
        self.actual["rows"].pop()
        with self.assertRaisesRegex(ValueError, "incomplete"):
            self.check()

    def test_wrong_position(self):
        self.actual["positions"][-1] -= 1
        with self.assertRaisesRegex(ValueError, "positions"):
            self.check()

    def test_wrong_token(self):
        self.actual["generated_ids"][2] += 1
        with self.assertRaisesRegex(ValueError, "generated_ids"):
            self.check()

    def test_stale_session(self):
        self.actual["turns"][2]["text"] = self.actual["turns"][1]["text"]
        with self.assertRaisesRegex(ValueError, "reset_ids"):
            self.check()

    def test_corrupt_cached_fixture(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.gguf"
            path.write_bytes(b"wrong GGUF")
            proc = subprocess.run(
                [sys.executable, ROOT / "scripts/cpu-smoke-fixture.py", path],
                capture_output=True, text=True,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertIn("expected sha256", proc.stderr)
