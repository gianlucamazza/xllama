#!/usr/bin/env python3
# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
"""The host trainer must not inherit system Torch or bypass its CPU hash lock."""
import os
import re
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
TEXT = (ROOT / "training/host/run_job.sh").read_text()
START = TEXT.index("ensure_venv() {")
FUNCTION = TEXT[START:TEXT.index("\n}\n", START) + 3]


class HostEnvironmentTests(unittest.TestCase):
    def run_environment(self, inherited=False, fail_sync=False):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            venv = root / "venv"
            (venv / "bin").mkdir(parents=True)
            (venv / "pyvenv.cfg").write_text(
                f"include-system-site-packages = {str(inherited).lower()}\n"
            )
            calls = root / "calls"
            py = venv / "bin/python"
            py.write_text('#!/bin/bash\nprintf "python:%s\\n" "$*" >> "$CALLS"\n')
            py.chmod(0o755)
            uv = root / "uv"
            uv.write_text(
                '#!/bin/bash\nprintf "uv:%s\\n" "$*" >> "$CALLS"\n'
                'if [[ "$1 $2" == "pip sync" && "$FAIL_SYNC" == 1 ]]; then exit 9; fi\n'
            )
            uv.chmod(0o755)
            env = dict(os.environ, PATH=f"{root}:{os.environ['PATH']}",
                       VENV=str(venv), PY=str(py), HOST=str(root),
                       CALLS=str(calls), FAIL_SYNC=str(int(fail_sync)))
            result = subprocess.run(["bash", "-c", FUNCTION + "\nensure_venv"],
                                    env=env, capture_output=True, text=True)
            return result, calls.read_text() if calls.exists() else ""

    def test_direct_pins_match_the_qualified_cpu_lock(self):
        lock = dict(re.findall(r"^([\w-]+)==([^\s\\]+)",
                               (ROOT / "training/host/requirements-cpu-py312.lock").read_text(),
                               re.MULTILINE))
        self.assertTrue(lock["torch"].endswith("+cpu"))
        for line in (ROOT / "training/host/requirements.txt").read_text().splitlines():
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            name, version = line.split("==")
            actual = lock[name].split("+", 1)[0] if name == "torch" else lock[name]
            self.assertEqual(actual, version, name)

    def test_inherited_packages_rejected_before_install_or_import(self):
        result, calls = self.run_environment(inherited=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("inherits system packages", result.stderr)
        self.assertEqual(calls, "")

    def test_sync_requires_cpu_profile_and_hashes(self):
        result, calls = self.run_environment()
        self.assertEqual(result.returncode, 0, result.stderr)
        sync = next(line for line in calls.splitlines() if line.startswith("uv:pip sync"))
        self.assertIn("--require-hashes", sync)
        self.assertIn("--torch-backend cpu", sync)
        self.assertIn("requirements-cpu-py312.lock", sync)
        self.assertIn("sys.version_info[:2] == (3, 12)", calls)
        self.assertIn("torch.version.cuda is None", calls)

    def test_failed_lock_sync_does_not_fall_back_to_unpinned_install(self):
        result, calls = self.run_environment(fail_sync=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(sum(line.startswith("uv:") for line in calls.splitlines()), 1)
        self.assertNotIn("import torch", calls)


if __name__ == "__main__":
    unittest.main()
