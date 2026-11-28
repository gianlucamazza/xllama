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


def replay(directory):
    return subprocess.run(
        [sys.executable, ROOT / "scripts/replay-moe-cache.py", directory],
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
