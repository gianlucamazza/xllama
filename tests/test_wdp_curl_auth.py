#!/usr/bin/env python3
"""Device Portal curl helper must not put credentials on process argv."""

from __future__ import annotations

import os
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
HELPER = ROOT / "scripts/wdp-curl-auth.sh"

# Distinctive values so a leak is unambiguous. Not a real account.
USER = "wdp-test-user"
PASS = r'wdp-test-pass-NOT-ON-ARGV-"quote\slash'


STUB_CURL = r"""#!/bin/sh
# Record argv (NUL-separated) and copy the curl -K config, if any.
: >"$WDP_ARGV_LOG"
printf '%s\0' "$0" "$@" >>"$WDP_ARGV_LOG"
prev=
for a in "$@"; do
	if [ "$prev" = "-K" ]; then
		cp "$a" "$WDP_CFG_COPY"
		stat -c '%a' "$a" >"$WDP_CFG_MODE"
		break
	fi
	prev=$a
done
exit 0
"""

DRIVER = r"""
set -euo pipefail
# shellcheck disable=SC1091
source "$HELPER"
wdp_setup_curl_auth
printf '%s\n' "${WDP_CURL_CONFIG_DIR}" >"$DIR_OUT"
curl "${CURL_AUTH[@]}" --fail "https://example.invalid/api/app/packagemanager/packages"
"""


class WdpCurlAuthTests(unittest.TestCase):
    def _run_helper(self, extra_env=None, bash_args=None, driver=DRIVER):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            stub_dir = tmp_path / "bin"
            stub_dir.mkdir()
            stub = stub_dir / "curl"
            stub.write_text(STUB_CURL, encoding="utf-8")
            stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
            argv_log = tmp_path / "argv.bin"
            cfg_copy = tmp_path / "curl.cfg.copy"
            cfg_mode = tmp_path / "curl.cfg.mode"
            dir_out = tmp_path / "config.dir"
            env = os.environ.copy()
            env["PATH"] = str(stub_dir) + os.pathsep + env.get("PATH", "")
            env["TMPDIR"] = str(tmp_path)
            env["XBOX_USER"] = USER
            env["XBOX_PASS"] = PASS
            env["HELPER"] = str(HELPER)
            env["WDP_ARGV_LOG"] = str(argv_log)
            env["WDP_CFG_COPY"] = str(cfg_copy)
            env["WDP_CFG_MODE"] = str(cfg_mode)
            env["DIR_OUT"] = str(dir_out)
            if extra_env:
                env.update(extra_env)
            cmd = ["bash"]
            if bash_args:
                cmd.extend(bash_args)
            cmd.extend(["-c", driver])
            result = subprocess.run(
                cmd,
                cwd=str(tmp_path),
                env=env,
                capture_output=True,
                text=True,
            )
            recorded = argv_log.read_bytes() if argv_log.is_file() else b""
            cfg = cfg_copy.read_bytes() if cfg_copy.is_file() else b""
            mode = cfg_mode.read_text(encoding="utf-8").strip() if cfg_mode.is_file() else ""
            config_dir = dir_out.read_text(encoding="utf-8").strip() if dir_out.is_file() else ""
            config_gone = bool(config_dir) and not Path(config_dir).exists()
            return result, recorded, cfg, mode, config_dir, config_gone

    def test_password_not_on_curl_argv_and_temp_gone(self):
        result, recorded, cfg, mode, config_dir, config_gone = self._run_helper()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(recorded, "stub curl did not record argv")
        self.assertNotIn(PASS.encode(), recorded)
        self.assertNotIn(USER.encode(), recorded)
        args = recorded.split(b"\0")
        self.assertIn(b"-K", args)
        self.assertNotIn(b"-u", args)
        self.assertIn(b"--basic", args)
        self.assertIn(b"-k", args)
        self.assertIn(b"-sS", args)
        self.assertTrue(cfg, "stub curl did not copy the -K config")
        self.assertEqual(mode, "600")
        self.assertTrue(config_dir, "helper did not report its temp dir")
        self.assertTrue(config_gone, "curl -K temp dir was not removed on EXIT")

    def test_set_x_does_not_echo_credentials(self):
        result, recorded, *_ = self._run_helper(bash_args=["-x"])
        self.assertEqual(result.returncode, 0, result.stderr)
        combined = result.stdout + result.stderr
        self.assertNotIn(PASS, combined)
        self.assertNotIn(USER, combined)
        self.assertTrue(recorded, "stub curl did not record argv")
        self.assertNotIn(PASS.encode(), recorded)
