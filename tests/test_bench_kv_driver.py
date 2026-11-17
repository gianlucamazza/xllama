#!/usr/bin/env python3
"""Exercise KV driver argument handling without contacting a console."""

import os
import shutil
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASH = os.environ.get("BASH") or shutil.which("bash")


@unittest.skipUnless(BASH, "Bash runtime unavailable on this host")
class KvDriverTests(unittest.TestCase):
    def run_driver(self, *args, hardware_env=None):
        env = dict(os.environ)
        for name in ("XBOX_IP", "XBOX_USER", "XBOX_PASS", "XLLAMA_BENCH_HARDWARE"):
            env.pop(name, None)
        if hardware_env is not None:
            env["XLLAMA_BENCH_HARDWARE"] = hardware_env
        return subprocess.run(
            [BASH, str(ROOT / "scripts/bench-xbox-kv.sh"), "synthetic-model",
             "--dry-run", *args],
            cwd=ROOT, env=env, text=True, capture_output=True, timeout=10,
        )

    def test_explicit_gpu_invocation_does_not_require_console_credentials(self):
        for hardware in ("xbox-series-s", "xbox-series-x"):
            for gpu in ("0", "99"):
                with self.subTest(hardware=hardware, gpu=gpu):
                    result = self.run_driver("--hardware", hardware, "--gpu-layers", gpu)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    self.assertIn(f"{hardware} (operator-declared)", result.stdout)
                    self.assertIn(f"gpu_layers: {gpu}", result.stdout)

    def test_missing_or_invalid_hardware_is_not_assumed_series_s(self):
        for extra in ((), ("--hardware", "unknown")):
            with self.subTest(extra=extra):
                result = self.run_driver(*extra)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("--hardware", result.stderr)

    def test_invalid_gpu_layers_are_rejected(self):
        for gpu in ("-1", "9999999999", "1.5", "invalid"):
            with self.subTest(gpu=gpu):
                result = self.run_driver("--hardware", "xbox-series-x", "--gpu-layers", gpu)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("--gpu-layers", result.stderr)

    def test_hardware_environment_default_and_argument_override(self):
        result = self.run_driver(hardware_env="xbox-series-x")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("xbox-series-x (operator-declared)", result.stdout)
        self.assertIn("gpu_layers: 0", result.stdout)
        result = self.run_driver("--hardware", "xbox-series-s", hardware_env="xbox-series-x")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("xbox-series-s (operator-declared)", result.stdout)


if __name__ == "__main__":
    unittest.main()
