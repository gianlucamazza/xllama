# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT

import hashlib
import json
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/research/evidence"
COMPLETE = EVIDENCE / "2026-11-16-moe-24b-console"
PARTIAL = EVIDENCE / "2026-11-16-moe-8b-console"


def replay(directory, *options):
    return subprocess.run(
        [sys.executable, ROOT / "scripts/replay-moe-cache.py", directory, *options],
        capture_output=True, text=True,
    )


class RecordedReplayTests(unittest.TestCase):
    def test_both_recorded_profiles(self):
        result = replay(COMPLETE)
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report["a2"]["requests"], 112176)
        self.assertEqual(report["a2"]["hits"], 0)
        self.assertEqual(report["a2"]["read_bytes"], 212856864768)
        self.assertEqual(report["b512"]["hits"], 39330)
        self.assertEqual(report["b512"]["read_bytes"], 138343145472)
        self.assertEqual(report["a2"]["route_sha256"], report["b512"]["route_sha256"])

    def test_observed_reader_times_follow_native_phase_boundaries(self):
        result = replay(COMPLETE, "--reader-timing")
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        small = report["a2"]["observed_phase_times"]
        large = report["b512"]["observed_phase_times"]
        self.assertEqual(small["prefill"], {"wall_ns": 67613298065, "reader_ns": 59149058300})
        self.assertEqual(large["after_prefill"], {"wall_ns": 198529460569, "reader_ns": 163640452120})
        self.assertEqual(small["workload_sha256"], large["workload_sha256"])

    def test_timing_rejects_rehashed_inconsistent_sources(self):
        def negative(directory, expected):
            result = replay(directory, "--reader-timing")
            self.assertEqual(result.returncode, 2)
            self.assertIn(expected, result.stderr)
            self.assertEqual(result.stdout, "")

        # Rebind digests after each edit to exercise the timing checks themselves.
        for defect, expected in (("frame", "invalid frame timing"), ("footer", "terminal Reader"),
                                 ("boundary", "crosses prefill end"), ("id", "token histories"),
                                 ("worker", "serial"), ("template", "different timing workloads"),
                                 ("unbounded", "native clock value"), ("post_sha", "generation interval")):
            with self.subTest(defect=defect), tempfile.TemporaryDirectory() as temp:
                directory = Path(temp)
                for arm in ("a2", "b512"):
                    for suffix in (".trace.bin", ".output.json"):
                        shutil.copy(COMPLETE / (arm + suffix), directory / (arm + suffix))
                receipt = json.loads((COMPLETE / "result.json").read_text())
                trace = directory / "a2.trace.bin"
                output = directory / "a2.output.json"
                raw = bytearray(trace.read_bytes())
                record = json.loads(output.read_bytes())
                if defect == "frame":
                    # The first frame cannot spend more time in its Reader than in acquire.
                    struct.pack_into("<Q", raw, 1040 + 40, 10**15)
                elif defect == "footer":
                    record["cache_ab"]["counters"]["reader_ns_total"] += 1
                elif defect == "boundary":
                    record["prefilled_ns"] = record["start_ns"] + 1
                    record["prefill_ns"] = 1
                elif defect == "id":
                    record["prompt_ids"][0] += 1
                elif defect == "worker":
                    record["threads"] = 2
                elif defect == "template":
                    record["template_sha256"] = "0" * 64
                elif defect == "unbounded":
                    record["end_ns"] = float("inf")
                elif defect == "post_sha":
                    record["end_ns"] = record["post_sha_begin_ns"] + 1
                trace.write_bytes(raw)
                output.write_text(json.dumps(record))
                receipt["runs"]["a2"]["trace_sha256"] = hashlib.sha256(raw).hexdigest()
                receipt["runs"]["a2"]["output_sha256"] = hashlib.sha256(output.read_bytes()).hexdigest()
                (directory / "result.json").write_text(json.dumps(receipt))
                negative(directory, expected)

    def test_partial_8b_trace_refused(self):
        result = replay(PARTIAL)
        self.assertEqual(result.returncode, 2)
        self.assertIn("partial", result.stderr)
        self.assertIn("checkpoint mismatch", result.stderr)

    def test_tamper_is_detected_even_if_rehashed(self):
        with tempfile.TemporaryDirectory() as temp:
            directory = Path(temp)
            for name in ("result.json", "a2.trace.bin", "a2.output.json"):
                shutil.copy(COMPLETE / name, directory / name)
            trace = directory / "a2.trace.bin"
            data = bytearray(trace.read_bytes())
            # Change the recorded number of evictions in the v1 binary footer.
            struct.pack_into("<Q", data, 499776 + 5 * 8, 0)
            trace.write_bytes(data)
            result = replay(directory)
            self.assertIn("trace SHA-256", result.stderr)
            receipt = directory / "result.json"
            meta = json.loads(receipt.read_text())
            meta["runs"]["a2"]["trace_sha256"] = hashlib.sha256(data).hexdigest()
            receipt.write_text(json.dumps(meta))
            result = replay(directory)
            self.assertEqual(result.returncode, 2)
            self.assertIn("evictions", result.stderr)
            self.assertEqual(result.stdout, "")


if __name__ == "__main__":
    unittest.main()
