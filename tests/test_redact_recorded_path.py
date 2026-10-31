#!/usr/bin/env python3
"""Tests for scripts/redact_recorded_path.py — evidence path redaction."""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/redact_recorded_path.py"
SPEC = importlib.util.spec_from_file_location("redact_recorded_path", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

CONSOLE_Q = (
    r"Q:\Users\SomeUser\AppData\Local\Packages\Publisher.App_suffix"
    r"\LocalState\training/out/device-marker/merged.gguf"
)
CONSOLE_Q_ESC = CONSOLE_Q.replace("\\", "\\\\")
EXPECTED_MERGED = "<console-app-data>/training/out/device-marker/merged.gguf"
PHASE10 = ROOT / "bench/results/phase10-console-devtrain-result.json"
DEVICE = ROOT / "docs/audit/excellence/evidence/device-training-result.json"


class RedactPathTests(unittest.TestCase):
    def test_localstate_console_path_is_relative(self):
        self.assertEqual(MODULE.redact_recorded_path(CONSOLE_Q), EXPECTED_MERGED)

    def test_forward_slash_localstate(self):
        path = "Q:/Users/SomeUser/AppData/Local/Packages/App/LocalState/models/x.gguf"
        self.assertEqual(
            MODULE.redact_recorded_path(path), "<console-app-data>/models/x.gguf"
        )

    def test_posix_home_and_users(self):
        self.assertEqual(
            MODULE.redact_recorded_path("/home/someone/proj/out/merged.gguf"),
            "<home>/proj/out/merged.gguf",
        )
        self.assertEqual(
            MODULE.redact_recorded_path("/Users/someone/proj/out/merged.gguf"),
            "<home>/proj/out/merged.gguf",
        )

    def test_workspace_prefix(self):
        self.assertEqual(
            MODULE.redact_recorded_path("/workspace/bench/results/x.json"),
            "<workspace>/bench/results/x.json",
        )

    def test_drive_without_localstate_drops_profile(self):
        path = r"C:\Data\Users\SomeUser\AppData\Local\Temp\out.gguf"
        self.assertEqual(
            MODULE.redact_recorded_path(path),
            "<abs>/<user-profile>/AppData/Local/Temp/out.gguf",
        )

    def test_unc_drops_host(self):
        self.assertEqual(
            MODULE.redact_recorded_path(r"\\host\share\models\a.gguf"),
            "<unc>/share/models/a.gguf",
        )

    def test_relative_and_empty_unchanged(self):
        self.assertEqual(
            MODULE.redact_recorded_path("training/out/device-marker/merged.gguf"),
            "training/out/device-marker/merged.gguf",
        )
        self.assertEqual(MODULE.redact_recorded_path(""), "")

    def test_full_paths_opt_in(self):
        self.assertEqual(
            MODULE.redact_recorded_path(CONSOLE_Q, record_full=True), CONSOLE_Q
        )

    def test_env_opt_in_via_cli(self):
        env = os.environ.copy()
        env[MODULE.FULL_PATHS_ENV] = "1"
        r = subprocess.run(
            [sys.executable, str(SCRIPT), "--path", CONSOLE_Q],
            check=True,
            capture_output=True,
            text=True,
            env=env,
        )
        self.assertEqual(r.stdout.strip(), CONSOLE_Q)
        r2 = subprocess.run(
            [sys.executable, str(SCRIPT), "--path", CONSOLE_Q],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertEqual(r2.stdout.strip(), EXPECTED_MERGED)

    def test_json_preserves_keys_and_numbers(self):
        src = (
            '{\n  "success": true,\n  "stages": ["prepare", "train"],\n'
            f'  "merged_gguf": "{CONSOLE_Q_ESC}",\n'
            '  "last_loss": 0.46798,\n  "wall_seconds": 446.069,\n'
            '  "peak_ws_mb": 1195,\n  "error": ""\n}\n'
        )
        out = MODULE.redact_json_text(src)
        data = json.loads(out)
        self.assertEqual(list(data.keys()), [
            "success", "stages", "merged_gguf", "last_loss",
            "wall_seconds", "peak_ws_mb", "error",
        ])
        self.assertEqual(data["merged_gguf"], EXPECTED_MERGED)
        self.assertTrue(data["success"])
        self.assertEqual(data["last_loss"], 0.46798)
        self.assertEqual(data["wall_seconds"], 446.069)
        self.assertEqual(data["peak_ws_mb"], 1195)
        self.assertIn('  "stages": ["prepare", "train"],', out)
        self.assertNotIn("SomeUser", out)

    def test_text_redacts_log_and_json_dump(self):
        log = (
            f"[xllama] train: export: {CONSOLE_Q}\n"
            f'  "merged_gguf": "{CONSOLE_Q_ESC}",\n'
        )
        out = MODULE.redact_text(log)
        self.assertIn(EXPECTED_MERGED, out)
        self.assertNotIn("SomeUser", out)
        self.assertNotIn("Q:\\", out)
        self.assertNotIn("Q:\\\\", out)

    def test_json_cli_in_place(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "result.json"
            p.write_text(
                json.dumps({"merged_gguf": CONSOLE_Q, "peak_ws_mb": 1195}, indent=2)
                + "\n",
                encoding="utf-8",
            )
            subprocess.run(
                [sys.executable, str(SCRIPT), "--json", str(p)],
                check=True,
                capture_output=True,
            )
            data = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(data["merged_gguf"], EXPECTED_MERGED)
            self.assertEqual(data["peak_ws_mb"], 1195)

    def test_committed_receipts_have_no_absolute_profile_paths(self):
        win_users = re.compile(r"[A-Za-z]:\\Users\\|[A-Za-z]:/Users/")
        for path in (PHASE10, DEVICE):
            text = path.read_text(encoding="utf-8")
            data = json.loads(text)
            self.assertTrue(data["success"])
            self.assertIsInstance(data["peak_ws_mb"], int)
            self.assertNotRegex(text, win_users)
            merged = data["merged_gguf"]
            self.assertTrue(
                merged.startswith("<console-app-data>/")
                or not _looks_absolute_drive(merged),
                msg=f"{path} merged_gguf still looks absolute: {merged}",
            )


def _looks_absolute_drive(value: str) -> bool:
    return len(value) >= 2 and value[0].isalpha() and value[1] == ":"


if __name__ == "__main__":
    unittest.main()
