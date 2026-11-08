#!/usr/bin/env python3
"""Tests for scripts/bench_run_status.py — hung/timeout campaign contract."""

import csv
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "scripts/bench_run_status.py"
SPEC = importlib.util.spec_from_file_location("bench_run_status", STATUS)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class BenchRunStatusTests(unittest.TestCase):
    def test_expected_recorded_drops_warmup(self):
        self.assertEqual(MODULE.expected_recorded_runs(4), 3)
        self.assertEqual(MODULE.expected_recorded_runs(1), 1)
        self.assertEqual(MODULE.expected_recorded_runs(0), 0)

    def test_format_failed_row_tags_host(self):
        row = MODULE.format_failed_row(
            model="lfm2-2.6b",
            host_base="xbox-series-s-t6",
            run_index=4,
            reason="timeout",
            date="2026-10-11T09:47:00Z",
            n_ctx=2048,
            n_threads=6,
        )
        self.assertEqual(len(row.split(",")), len(MODULE.CSV_HEADER.split(",")))
        self.assertIn("xbox-series-s-t6-timeout", row)
        self.assertIn(",4,", row)
        self.assertTrue(MODULE.is_failed_row(row))

    def test_campaign_fails_on_missing_measurement_like_2026_10_11(self):
        ok2 = (
            "lfm2-2.6b,Q4_K_M,cpu,2048,6,54.88,17.80,1623,8850,0,0,298,371,0,"
            "xbox-series-s-t6,2026-10-11T09:41:13Z,2,5429.6,5430.1"
        )
        ok3 = (
            "lfm2-2.6b,Q4_K_M,cpu,2048,6,54.88,17.71,1623,8586,0,0,298,512,0,"
            "xbox-series-s-t6,2026-10-11T09:41:58Z,3,5429.9,5430.3"
        )
        hung = MODULE.format_failed_row(
            model="lfm2-2.6b",
            host_base="xbox-series-s-t6",
            run_index=4,
            reason="timeout",
            date="2026-10-11T09:47:00Z",
            n_ctx=2048,
            n_threads=6,
        )
        # Old harness: 2 of 3 measurement rows, exit 0. Must now fail.
        self.assertEqual(
            MODULE.campaign_exit_code(n_runs=4, rows=[ok2, ok3], failed_runs=1),
            1,
        )
        self.assertEqual(
            MODULE.campaign_exit_code(n_runs=4, rows=[ok2, ok3, hung], failed_runs=1),
            1,
        )

    def test_campaign_ok_when_three_measurement_rows(self):
        rows = [
            f"m,q,cpu,2048,6,1,1,1,1,0,0,1,1,0,xbox-series-s-t6,d,{i},1.0,1.0"
            for i in (2, 3, 4)
        ]
        self.assertEqual(MODULE.campaign_exit_code(n_runs=4, rows=rows, failed_runs=0), 0)

    def test_cli_check_matches_library(self):
        result = subprocess.run(
            [
                sys.executable,
                str(STATUS),
                "check",
                "--n-runs",
                "4",
                "--failed-runs",
                "1",
                "--row",
                "m,q,cpu,2048,6,1,1,1,1,0,0,1,1,0,xbox-series-s-t6,d,2,1.0,1.0",
                "--row",
                "m,q,cpu,2048,6,1,1,1,1,0,0,1,1,0,xbox-series-s-t6,d,3,1.0,1.0",
            ],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertIn("incomplete", result.stderr)


class D3d12SelftestExitTests(unittest.TestCase):
    def test_console_gate_status_reaches_the_shell(self):
        script = (ROOT / "scripts/bench-d3d12-selftest.sh").read_text()
        gate = script.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]
        good = dict(
            type="q4_0", n="2048", k="2048", ncols="1", rel_err="0.000001",
            gpu_ms="0.1", packed_gbs="120", ok="1", d3d12_ran="1", error="-"
        )
        cases = [
            (good, True), (dict(good, ok="0"), False),
            (dict(good, d3d12_ran="0"), False),
            (dict(good, packed_gbs="99"), False),
            (dict(good, packed_gbs="nan"), False),
            (dict(good, rel_err="inf"), False),
            (dict(good, rel_err="nan"), False),
            (dict(good, gpu_ms="nan"), False),
            (dict(good, gpu_ms="-1"), False), (None, False),
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rows.csv"
            for row, passed in cases:
                with self.subTest(row=row):
                    with path.open("w", newline="") as stream:
                        writer = csv.DictWriter(stream, fieldnames=list(good))
                        writer.writeheader()
                        if row:
                            writer.writerow(row)
                    result = subprocess.run(
                        [sys.executable, "-c", gate, str(path)], text=True, capture_output=True
                    )
                    self.assertEqual(result.returncode, 0 if passed else 1, result.stderr)
                    self.assertIn("GATE=" + ("PASS" if passed else "FAIL"), result.stdout)


if __name__ == "__main__":
    unittest.main()
