#!/usr/bin/env python3
# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
"""Host-side receipt acceptance using existing console captures as fixtures."""

import csv
import importlib.util
import io
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("selftest", ROOT / "scripts/verify-d3d12-selftest.py")
gate = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gate)
RECEIPTS = ROOT / "bench/results/2026-11-24-v181-series-s-release/receipts"
BIAS_LOG = "[xllama] d3d12be: bias ADD cold_scratch=1\n[xllama] d3d12be: add_f32 n=129 ok=1 err=-\n"
ROPE_LOG = "[xllama] d3d12be: rope cold_scratch=1\n"


def receipt(profile):
    with (RECEIPTS / f"d3d12-selftest-b47fe34-{profile}.csv").open() as stream:
        return list(csv.DictReader(stream))


def verify(rows, fields=None, **profile):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields or rows[0].keys(), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    stream.seek(0)
    return gate.verify(stream, **profile)


class SelftestGate(unittest.TestCase):
    def test_full_profiles_and_drop_each_case(self):
        for profile, args in (("off", {}), ("bias", dict(bias=True, log=BIAS_LOG)),
                              ("rope", dict(rope=True, log=ROPE_LOG))):
            rows = receipt(profile)
            self.assertEqual(verify(rows, **args), len(rows))
            for i, row in enumerate(rows):
                with self.subTest(profile=profile, deleted=row), self.assertRaisesRegex(ValueError, "missing cases"):
                    verify(rows[:i] + rows[i+1:], **args)

    def test_console_driver_checks_freshness_and_runs_verifier(self):
        with tempfile.TemporaryDirectory() as temp:
            work = Path(temp)
            driver = work / "bench-d3d12-selftest.sh"
            shutil.copy(ROOT / "scripts" / driver.name, driver)
            shutil.copy(SPEC.origin, work)
            deploy = work / "deploy.sh"
            deploy.write_text("""#!/usr/bin/env bash
set -eu
case "$1" in
pfn) echo package ;;
list-localstate) echo "$LISTING" ;;
get-log) [[ ! -f "$STARTED" ]] || cat "$RUN_LOG" ;;
start-app) touch "$STARTED" ;;
fetch-file)
    if [[ "$3" == *.done ]]; then echo done >"$4"; else cp "$RECEIPT" "$4"; fi ;;
stop-app|delete-file|upload-file) ;;
*) exit 2 ;;
esac
""")
            deploy.chmod(0o755)
            (work / "sleep").write_text("#!/bin/sh\nexit 0\n")
            (work / "sleep").chmod(0o755)
            started = work / "started"
            run_log = work / "current.log"
            run_log.write_text(ROPE_LOG)
            env = dict(os.environ, PATH=f"{temp}:{os.environ['PATH']}", XBOX_ENV="/dev/null",
                       XBOX_IP="fake", STARTED=str(started), RUN_LOG=str(run_log),
                       RECEIPT=str(RECEIPTS / "d3d12-selftest-b47fe34-rope.csv"))
            def run(listing):
                return subprocess.run([driver, "--out", work / "out.csv", "--force", "--require-rope"],
                                      env=dict(env, LISTING=listing), capture_output=True, text=True)
            for name in ("d3d12be-result.csv.done", "d3d12be-result.csv"):
                result = run('{"Items":[{"Name":"' + name + '"}]}')
                self.assertNotEqual(result.returncode, 0)
                self.assertFalse(started.exists())
            result = run('{"Items":[]}')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("GATE=PASS", result.stdout)

    def test_truncated_receipts(self):
        rows = receipt("off")
        with self.assertRaisesRegex(ValueError, "missing cases"):
            verify(rows[:1])
        for content in ("", ",".join(rows[0]) + "\n"):
            with self.assertRaises(ValueError):
                gate.verify(io.StringIO(content))

    def test_duplicate_unexpected_dimensions_and_types(self):
        rows = receipt("off")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            verify(rows[:-1] + rows[:1])
        for changes in ({"type": "unknown"}, {"n": "2047"}, {"k": "256"}, {"ncols": "2"}):
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "unexpected"):
                verify([dict(rows[0], **changes)] + rows[1:])

    def test_profile_mismatches_and_cold_order(self):
        bias, rope = receipt("bias"), receipt("rope")
        for rows, args in [(bias, {}), (rope, {}), (receipt("off"), dict(bias=True, log=BIAS_LOG)),
                           (bias, dict(bias=True)), (rope, dict(rope=True)),
                           (bias[1:] + bias[:1], dict(bias=True, log=BIAS_LOG)),
                           (rope[::-1], dict(rope=True, log=ROPE_LOG))]:
            with self.subTest(args=args, first=rows[0]), self.assertRaises(ValueError):
                verify(rows, **args)

    def test_optional_pipeline_diagnostic(self):
        rows = receipt("off")
        for typ in ("q5_k", "iq4_xs", "iq3_xxs"):
            sub = [r for r in rows if r["type"] != typ]
            log = f"[xllama] d3d12: optional {typ} t64 pipeline unavailable: HRESULT — type stays on CPU\n"
            self.assertEqual(verify(sub, log=log, allow_unavailable=True), len(sub))
            for candidate, args in [(rows, dict(log=log, allow_unavailable=True)), (sub, dict(log=log)),
                                    (sub, dict(allow_unavailable=True))]:
                with self.subTest(typ=typ, args=args), self.assertRaises(ValueError):
                    verify(candidate, **args)

    def test_missing_and_malformed_fields(self):
        rows = receipt("off")
        for field in rows[0]:
            with self.subTest(missing=field), self.assertRaises(ValueError):
                verify(rows, fields=[key for key in rows[0] if key != field])
            with self.subTest(blank=field), self.assertRaises(ValueError):
                verify([dict(rows[0], **{field: ""})] + rows[1:])
        for key in ("n", "k", "ncols", "rel_err", "gpu_ms", "packed_gbs"):
            for value in ("bad", "NaN", "inf", "-1"):
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    verify([dict(rows[0], **{key: value})] + rows[1:])
        for key in ("ok", "d3d12_ran"):
            with self.subTest(key=key), self.assertRaises(ValueError):
                verify([dict(rows[0], **{key: "0"})] + rows[1:])

    def test_numeric_floors_and_tolerance(self):
        rows = receipt("off")
        for typ, key, value in (("q4_0", "packed_gbs", "99.99"), ("iq3_xxs", "packed_gbs", "54.99"),
                                ("q6_k", "rel_err", "0.000011"), ("get_rows_q6_k", "rel_err", "1e-9")):
            corrupted = [r.copy() for r in rows]
            next(r for r in corrupted if r["type"] == typ)[key] = value
            with self.subTest(typ=typ, key=key), self.assertRaises(ValueError):
                verify(corrupted)
