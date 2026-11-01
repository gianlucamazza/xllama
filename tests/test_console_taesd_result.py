#!/usr/bin/env python3
"""This-run diffuse-result.csv rows only; no stale VAE number (#369)."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
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

    def test_this_run_row_is_accepted(self):
        ok, vae, msg = MODULE.assess_result(FRESH, RUN_START, STALE)
        self.assertTrue(ok)
        self.assertEqual(vae, "412.0")
        self.assertEqual(msg, "")

    def test_unchanged_file_is_stale_even_if_dates_match(self):
        same = HEADER + (
            "sd-turbo-fp16,1,42,1.0,2.0,2.0,3.0,6.0,512x512,2026-11-01T08:00:10Z\n"
        )
        ok, vae, msg = MODULE.assess_result(same, RUN_START, same)
        self.assertFalse(ok)
        self.assertEqual(vae, "")
        self.assertIn("unchanged", msg)

    def test_clock_skew_still_accepts_a_rewritten_file(self):
        # Console clock behind the host: date is before run start, but the
        # file is not the pre-run snapshot, so this row is from this run.
        skewed = HEADER + (
            "sd-turbo-fp16,1,42,1.0,2.0,2.0,88.0,91.0,512x512,2026-10-31T23:59:00Z\n"
        )
        ok, vae, msg = MODULE.assess_result(skewed, RUN_START, STALE)
        self.assertTrue(ok, msg)
        self.assertEqual(vae, "88.0")

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


class PeakMemTests(unittest.TestCase):
    def test_peak_field_is_converted_from_bytes(self):
        payload = json.dumps(
            {
                "Processes": [
                    {
                        "ImageName": "xllama.exe",
                        "PackageFullName": "GianlucaMazza.xllama_1.7.0.0_x64__pkg",
                        "WorkingSetSize": 999999999,
                        "PeakWorkingSetSize": 2 * 1024 * 1024 * 1024,
                    }
                ]
            }
        )
        self.assertEqual(MODULE.peak_mem_mb(payload), 2048)

    def test_working_set_alone_is_not_peak(self):
        payload = json.dumps(
            {
                "Processes": [
                    {
                        "ImageName": "xllama.exe",
                        "WorkingSetSize": 3 * 1024 * 1024 * 1024,
                    }
                ]
            }
        )
        self.assertIsNone(MODULE.peak_mem_mb(payload))

    def test_missing_app_process_is_none(self):
        payload = json.dumps(
            {
                "Processes": [
                    {
                        "ImageName": "GameBar.exe",
                        "PeakWorkingSetSize": 1024 * 1024,
                    }
                ]
            }
        )
        self.assertIsNone(MODULE.peak_mem_mb(payload))

    def test_cli_prints_nothing_without_a_peak_field(self):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "peak-mem"],
            input=json.dumps({"Processes": [{"ImageName": "xllama.exe", "WorkingSetSize": 1}]}),
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
        # The previous last-row-only reader is what accepted the 2026-10-25 row.
        self.assertNotIn("rows[-1] if rows else {}", text)


if __name__ == "__main__":
    unittest.main()
