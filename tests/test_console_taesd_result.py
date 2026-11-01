#!/usr/bin/env python3
"""This-run diffuse-result.csv rows only; no stale VAE number (#369)."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/console_taesd_result.py"
SPEC = importlib.util.spec_from_file_location("console_taesd_result", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)

HEADER = "model,steps,seed,te_ms,unet_ms,unet_ms_per_step,vae_ms,total_ms,resolution,date\n"
STALE = HEADER + (
    "sd-turbo-fp16,1,42,100.0,200.0,200.0,635.6,935.6,512x512,2026-10-25T12:00:00Z\n"
)
FRESH = HEADER + (
    "sd-turbo-fp16,1,42,110.0,210.0,210.0,412.0,732.0,512x512,2026-11-01T08:00:10Z\n"
)
RUN_START = "2026-11-01T08:00:00Z"
FAMILY = "GianlucaMazza.xllama_publisher"


def row(model="sd-turbo-fp16", steps="1", seed="42", vae="635.6", date=""):
    return (
        HEADER
        + f"{model},{steps},{seed},100.0,200.0,200.0,{vae},935.6,512x512,{date}\n"
    )


class AssessTests(unittest.TestCase):
    def test_stale_last_row_is_rejected(self):
        ok, vae, msg = MODULE.assess_result(STALE, RUN_START, STALE)
        self.assertFalse(ok)
        self.assertEqual(vae, "")
        self.assertIn("stale", msg)
        self.assertIn("2026-10-25T12:00:00Z", msg)
        self.assertNotIn("635.6", msg.split("stale")[0])

    def test_stale_date_without_before_snapshot_is_rejected(self):
        ok, vae, msg = MODULE.assess_result(STALE, RUN_START, "")
        self.assertFalse(ok)
        self.assertEqual(vae, "")
        self.assertIn("2026-10-25T12:00:00Z", msg)
        self.assertIn(RUN_START, msg)

    def test_missing_date_without_snapshot_is_rejected(self):
        missing = row(date="")
        ok, vae, msg = MODULE.assess_result(missing, RUN_START, "")
        self.assertFalse(ok)
        self.assertEqual(vae, "")
        self.assertIn("missing or invalid date", msg)
        self.assertNotIn("635.6", msg.split("date")[0])

    def test_invalid_date_without_snapshot_is_rejected(self):
        bad = row(date="not-a-date")
        ok, vae, msg = MODULE.assess_result(bad, RUN_START, "")
        self.assertFalse(ok)
        self.assertEqual(vae, "")
        self.assertIn("missing or invalid date", msg)
        self.assertIn("not-a-date", msg)

    def test_invalid_run_start_without_snapshot_is_rejected(self):
        ok, vae, msg = MODULE.assess_result(FRESH, "not-a-timestamp", "")
        self.assertFalse(ok)
        self.assertEqual(vae, "")
        self.assertIn("run_start", msg)
        self.assertIn("not-a-timestamp", msg)

    def test_missing_date_with_rewritten_expected_job_is_accepted(self):
        missing = row(date="", vae="77.0")
        ok, vae, msg = MODULE.assess_result(missing, RUN_START, STALE)
        self.assertTrue(ok, msg)
        self.assertEqual(vae, "77.0")

    def test_invalid_run_start_with_rewritten_expected_job_is_accepted(self):
        ok, vae, msg = MODULE.assess_result(FRESH, "not-a-timestamp", STALE)
        self.assertTrue(ok, msg)
        self.assertEqual(vae, "412.0")

    def test_this_run_row_is_accepted(self):
        ok, vae, msg = MODULE.assess_result(FRESH, RUN_START, STALE)
        self.assertTrue(ok)
        self.assertEqual(vae, "412.0")
        self.assertEqual(msg, "")

    def test_clock_skew_still_accepts_a_rewritten_expected_job(self):
        # Console clock behind the host: date is before run start, but the
        # file is a rewritten taesd job versus the pre-run snapshot.
        skewed = HEADER + (
            "sd-turbo-fp16,1,42,1.0,2.0,2.0,88.0,91.0,512x512,2026-10-31T23:59:00Z\n"
        )
        ok, vae, msg = MODULE.assess_result(skewed, RUN_START, STALE)
        self.assertTrue(ok, msg)
        self.assertEqual(vae, "88.0")

    def test_rewritten_row_must_be_the_taesd_job(self):
        other = HEADER + (
            "some-other-model,4,99,1.0,2.0,2.0,88.0,91.0,512x512,2026-10-31T23:59:00Z\n"
        )
        ok, vae, msg = MODULE.assess_result(other, RUN_START, STALE)
        self.assertFalse(ok)
        self.assertEqual(vae, "")
        self.assertIn("not the taesd job", msg)

    def test_missing_after_file_fails(self):
        ok, vae, msg = MODULE.assess_result("", RUN_START, STALE)
        self.assertFalse(ok)
        self.assertEqual(vae, "")
        self.assertIn("no diffuse-result.csv row", msg)

    def test_cli_prints_vae_only_on_success(self):
        with tempfile.TemporaryDirectory() as tmp:
            after = Path(tmp) / "after.csv"
            before = Path(tmp) / "before.csv"
            after.write_text(FRESH)
            before.write_text(STALE)
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "assess",
                    "--after",
                    str(after),
                    "--before",
                    str(before),
                    "--run-start",
                    RUN_START,
                ],
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), "412.0")
            self.assertEqual(result.stderr, "")

    def test_cli_stale_is_nonzero_and_prints_no_vae(self):
        with tempfile.TemporaryDirectory() as tmp:
            after = Path(tmp) / "after.csv"
            after.write_text(STALE)
            result = subprocess.run(
                [
                    sys.executable,
                    str(SCRIPT),
                    "assess",
                    "--after",
                    str(after),
                    "--run-start",
                    RUN_START,
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertEqual(result.stdout, "")
            self.assertIn("stale", result.stderr)
            self.assertNotIn("ok:", result.stderr)


class PrivateWsTests(unittest.TestCase):
    def payload(self, **proc):
        return json.dumps({"Processes": [proc]})

    def test_private_working_set_is_converted_from_bytes(self):
        data = self.payload(
            ImageName="xllama.exe",
            PackageFullName=f"{FAMILY}_1.7.0.0_x64__unused",
            WorkingSetSize=999999999,
            PrivateWorkingSet=2 * 1024 * 1024 * 1024,
        )
        self.assertEqual(MODULE.private_ws_mb(data, FAMILY), 2048)

    def test_working_set_alone_is_not_used(self):
        data = self.payload(
            ImageName="xllama.exe",
            PackageFullName=f"{FAMILY}_1.7.0.0_x64__unused",
            WorkingSetSize=3 * 1024 * 1024 * 1024,
        )
        self.assertIsNone(MODULE.private_ws_mb(data, FAMILY))

    def test_peak_keys_are_not_used(self):
        data = self.payload(
            ImageName="xllama.exe",
            PackageFullName=f"{FAMILY}_1.7.0.0_x64__unused",
            PeakWorkingSetSize=2 * 1024 * 1024 * 1024,
            PrivateWorkingSetSize=3 * 1024 * 1024 * 1024,
        )
        self.assertIsNone(MODULE.private_ws_mb(data, FAMILY))

    def test_image_name_guess_is_not_enough(self):
        data = self.payload(
            ImageName="xllama.exe",
            PrivateWorkingSet=2 * 1024 * 1024 * 1024,
        )
        self.assertIsNone(MODULE.private_ws_mb(data, FAMILY))

    def test_other_package_is_ignored(self):
        data = self.payload(
            ImageName="GameBar.exe",
            PackageFullName="Other.App_publisher",
            PrivateWorkingSet=1024 * 1024,
        )
        self.assertIsNone(MODULE.private_ws_mb(data, FAMILY))

    def test_keep_running_max(self):
        self.assertEqual(MODULE.keep_running_max(None, 10), 10)
        self.assertEqual(MODULE.keep_running_max(10, 8), 10)
        self.assertEqual(MODULE.keep_running_max(10, 12), 12)
        self.assertEqual(MODULE.keep_running_max(10, None), 10)

    def test_cli_prints_nothing_without_private_ws(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "private-ws", "--package-family", FAMILY],
            input=json.dumps(
                {"Processes": [{"ImageName": "xllama.exe", "WorkingSetSize": 1}]}
            ),
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stdout, "")


class ScriptContractTests(unittest.TestCase):
    def test_taesd_gate_waits_and_rejects_stale_rows(self):
        text = (ROOT / "scripts/validate-console.sh").read_text()
        self.assertIn('{"op": "wait_idle"', text)
        self.assertIn("XLLAMA_TAESD_READY_TIMEOUT_S", text)
        self.assertIn("console_taesd_result.py", text)
        self.assertIn("250 ms", text)
        self.assertIn("max_sampled_private_ws_mb", text)
        self.assertIn("PrivateWorkingSet sampled every", text)
        self.assertIn("TAESD_SAMPLER_PID", text)
        self.assertIn("kill -0 $$", text)
        self.assertIn("--max-time", text)
        self.assertNotIn("peak_mem_mb", text)
        self.assertNotIn("rows[-1] if rows else {}", text)

    def test_ok_line_requires_autopilot_marker(self):
        text = (ROOT / "scripts/validate-console.sh").read_text()
        start = text.index("validate_taesd() {")
        block = text[start : text.index("\n}\n", start) + 3]
        self.assertIn('[[ "$marker" == "ok" ]]', block)
        # The ok print is gated on the marker, not only on a fresh CSV.
        self.assertRegex(
            block,
            r'\[\[ "\$marker" == "ok" \]\] && echo "  ok: TAESD VAE stage',
        )

    def test_wait_idle_reads_atomics_only(self):
        text = (ROOT / "uwp/MainPage.cpp").read_text()
        start = text.index('a.op == "wait_idle"')
        block = text[start : text.index('} else if (a.op == "generate_image")', start)]
        self.assertNotIn("try_acquire_compute_job", block)
        self.assertIn("kIdlePolls = 3", block)

    def test_console_exit_kills_sampler(self):
        text = (ROOT / "scripts/validate-console.sh").read_text()
        start = text.index("console_exit() {")
        fn = text[start : text.index("\n}\n", start) + 3]
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp) / "work"
            work.mkdir()
            pid_path = Path(tmp) / "sampler.pid"
            env = dict(os.environ, TMPDIR_LOCAL=str(work))
            setup = (
                "set -euo pipefail\n"
                "TAESD_SAMPLER_PID=\n"
                "VAE_RESTORE_PENDING=0\n"
                "cleanup_temp_chats() { return 0; }\n"
                "release_original_recovery_dir() { :; }\n"
                + fn
                + "\nsleep 60 &\n"
                "TAESD_SAMPLER_PID=$!\n"
                f'printf "%s\\n" "$TAESD_SAMPLER_PID" >"{pid_path}"\n'
                "console_exit 0\n"
            )
            result = subprocess.run(
                ["bash", "-c", setup], env=env, capture_output=True, timeout=10
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            pid = int(pid_path.read_text().strip())
            time.sleep(0.1)
            with self.assertRaises(ProcessLookupError):
                os.kill(pid, 0)


if __name__ == "__main__":
    unittest.main()
