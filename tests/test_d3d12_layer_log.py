#!/usr/bin/env python3
# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
"""Host tests for scripts/parse_d3d12_layer_log.py (#421)."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/parse_d3d12_layer_log.py"
spec = importlib.util.spec_from_file_location("parse_d3d12_layer_log", SCRIPT)
parser = importlib.util.module_from_spec(spec)
spec.loader.exec_module(parser)

# Golden strings match format_d3d12_layer_profile in src/bridge/ggml_d3d12.cpp.
ROW = ("[xllama] d3d12-layer phase=prefill layer=3 op=mul_mat count=2 gpu_ms=3.000 "
       "wait_ms=0.100 gpu_missing=0 wait_missing=0")
UNATTR = ("[xllama] d3d12-layer phase=prefill layer=unattributed op=get_rows count=1 "
          "gpu_ms=0.500 wait_ms=missing gpu_missing=0 wait_missing=1")
MISSING = ("[xllama] d3d12-layer phase=decode layer=1 op=mul_mat count=1 gpu_ms=missing "
           "wait_ms=missing gpu_missing=1 wait_missing=1")
SUMMARY = ("[xllama] d3d12-layer phase=prefill summary rows=2 timed=3 missing=0 "
           "gpu_ms=3.500 wait_ms=0.100 wait_missing=1")
SUMMARY_MISSING = ("[xllama] d3d12-layer phase=decode summary rows=1 timed=0 missing=1 "
                   "gpu_ms=missing wait_ms=missing wait_missing=1")


class D3d12LayerLogTests(unittest.TestCase):
    def test_row_fields_and_unattributed_bucket(self):
        row = parser.parse_d3d12_layer_line(ROW)
        self.assertEqual(row["kind"], "row")
        self.assertEqual(row["phase"], "prefill")
        self.assertEqual(row["layer"], 3)
        self.assertEqual(row["op"], "mul_mat")
        self.assertEqual(row["count"], 2)
        self.assertEqual(row["gpu_ms"], 3.0)
        self.assertEqual(row["wait_ms"], 0.1)
        self.assertEqual(row["gpu_missing"], 0)
        self.assertEqual(row["wait_missing"], 0)
        unattr = parser.parse_d3d12_layer_line(UNATTR)
        self.assertEqual(unattr["layer"], "unattributed")
        self.assertEqual(unattr["op"], "get_rows")
        self.assertEqual(unattr["gpu_ms"], 0.5)
        self.assertIsNone(unattr["wait_ms"])
        self.assertEqual(unattr["wait_missing"], 1)

    def test_missing_timestamps_stay_none(self):
        rec = parser.parse_d3d12_layer_line(MISSING)
        self.assertIsNone(rec["gpu_ms"])
        self.assertIsNone(rec["wait_ms"])
        self.assertNotEqual(rec["gpu_ms"], 0)
        self.assertNotEqual(rec["wait_ms"], 0)
        summary = parser.parse_d3d12_layer_line(SUMMARY_MISSING)
        self.assertEqual(summary["kind"], "summary")
        self.assertIsNone(summary["gpu_ms"])
        self.assertIsNone(summary["wait_ms"])
        self.assertEqual(summary["missing"], 1)
        self.assertEqual(summary["wait_missing"], 1)

    def test_zero_wait_is_not_missing(self):
        line = ("[xllama] d3d12-layer phase=prefill layer=2 op=mul_mat count=1 gpu_ms=0.001 "
                "wait_ms=0.000 gpu_missing=0 wait_missing=0")
        rec = parser.parse_d3d12_layer_line(line)
        self.assertEqual(rec["wait_ms"], 0.0)
        self.assertIsNotNone(rec["wait_ms"])

    def test_log_skips_other_lines_and_parses_crlf(self):
        text = "\n".join([
            "[xllama] d3d12-phase phase=prefill calls=1 submissions=1 dispatches=2 "
            "wall_ms=1.000 quant_ms=0.100 record_ms=0.200 exec_wait_ms=0.700 "
            "gpu_ms=0.500 ts_valid=1 ts_missing=0",
            ROW + "\r",
            UNATTR,
            SUMMARY,
            "noise",
        ])
        recs = parser.parse_d3d12_layer_log(text)
        self.assertEqual(len(recs), 3)
        self.assertEqual(recs[0]["layer"], 3)
        self.assertEqual(recs[1]["layer"], "unattributed")
        self.assertEqual(recs[2]["kind"], "summary")
        self.assertEqual(recs[2]["rows"], 2)
        self.assertEqual(recs[2]["gpu_ms"], 3.5)

    def test_non_layer_line_is_ignored(self):
        self.assertIsNone(parser.parse_d3d12_layer_line(
            "[xllama] d3d12-dispatch phase=prefill op=mul_mat type=q4_0 cpg=1 "
            "threads=128 cols=512+ count=2 gpu_ms=3.000 ms_per=1.5000 "
            "weight_mib=2.0 columns=1024"))
        self.assertIsNone(parser.parse_d3d12_layer_line(""))

    def test_malformed_row_fails_closed(self):
        with self.assertRaises(ValueError):
            parser.parse_d3d12_layer_line(
                "[xllama] d3d12-layer phase=prefill layer=3 op=mul_mat count=1")
        with self.assertRaises(ValueError):
            parser.parse_ms("unavailable")
        with self.assertRaises(ValueError):
            parser.parse_layer("blk.3")

    def test_cli_json_roundtrip(self):
        log = "\n".join([ROW, UNATTR, MISSING, SUMMARY]) + "\n"
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "xllama.log"
            path.write_text(log, encoding="utf-8")
            result = subprocess.run(
                [sys.executable, str(SCRIPT), "--json", str(path)],
                capture_output=True, text=True, check=False)
            self.assertEqual(result.returncode, 0, result.stderr)
            data = json.loads(result.stdout)
        self.assertEqual(len(data), 4)
        self.assertIsNone(data[1]["wait_ms"])
        self.assertIsNone(data[2]["gpu_ms"])
        self.assertEqual(data[0]["layer"], 3)
        self.assertEqual(data[3]["kind"], "summary")


if __name__ == "__main__":
    unittest.main()
