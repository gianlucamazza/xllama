#!/usr/bin/env python3
"""The d3d12 dispatch sidecar parses log lines and refuses a bad summary."""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "d3d12-dispatch-sidecar.py"

LOG = """\
[xllama] d3d12-phase phase=prefill gpu_ms=8.000
[xllama] d3d12-dispatch phase=prefill op=mul_mat type=q4_0 cpg=1 threads=128 cols=512+ layer=0 count=2 gpu_ms=3.000 ms_per=1.5000 weight_mib=2.0 columns=1024
[xllama] d3d12-dispatch phase=prefill op=mul_mat type=q4_0 cpg=1 threads=128 cols=512+ layer=1 count=1 gpu_ms=5.000 ms_per=5.0000 weight_mib=1.0 columns=512
[xllama] d3d12-dispatch phase=prefill summary kinds=2 timed=3 missing=0 gpu_ms=8.000
"""


def run(log_text, out):
    log = out.parent / "app.log"
    log.write_text(log_text, encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(log), "-o", str(out)],
        capture_output=True,
        text=True,
    )


class DispatchSidecarTests(unittest.TestCase):
    def test_writes_one_row_per_layer(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "dispatch.csv"
            result = run(LOG, out)
            self.assertEqual(result.returncode, 0, result.stderr)
            text = out.read_text(encoding="utf-8")
            self.assertTrue(text.startswith("phase,op,type,cpg,threads,cols,layer,count,gpu_ms,"))
            self.assertIn("512+,0,2,3.000", text)
            self.assertIn("512+,1,1,5.000", text)
            self.assertNotIn("summary", text)

    def test_rejects_a_summary_that_does_not_match(self):
        bad = LOG.replace("timed=3", "timed=9")
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "dispatch.csv"
            result = run(bad, out)
            self.assertEqual(result.returncode, 1)
            self.assertIn("timed=9", result.stderr)
            self.assertFalse(out.exists())

    def test_rejects_an_unparsed_dispatch_line(self):
        bad = LOG + "[xllama] d3d12-dispatch phase=prefill oops\n"
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "dispatch.csv"
            result = run(bad, out)
            self.assertEqual(result.returncode, 1)
            self.assertIn("unparsed", result.stderr)

    def test_rejects_a_log_with_no_dispatch_lines(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "dispatch.csv"
            result = run("[xllama] ready\n", out)
            self.assertEqual(result.returncode, 1)
            self.assertIn("no d3d12-dispatch", result.stderr)
            self.assertFalse(out.exists())


if __name__ == "__main__":
    unittest.main()
