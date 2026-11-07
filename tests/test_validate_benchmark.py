#!/usr/bin/env python3
"""Exercise the benchmark CSV admission boundary through its public CLI."""

import csv
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/validate-benchmark.py"
CORE = "model,quant,backend,n_ctx,n_threads,prompt_tok_s,decode_tok_s,peak_ws_mb,load_ms,gpu_mem_mb,gpu_budget_mb".split(",")
V1 = CORE + "n_prompt_tok,n_gen_tok,max_length,host,date,run_index".split(",")
V2 = V1 + ["prefill_ms", "ttft_ms"]


class ValidateBenchmarkTests(unittest.TestCase):
    def validate(self, fields=V2, overrides=None, row_delta=0, comment=False):
        row = dict.fromkeys(fields, "1")
        row.update(model="fixture", quant="Q4", backend="cpu")
        row.update(overrides or {})
        values = [row[field] for field in fields]
        if row_delta < 0:
            values = values[:row_delta]
        elif row_delta > 0:
            values += ["extra"] * row_delta
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(fields)
                writer.writerow(values)
                if comment:
                    handle.write("# t7/t8: STALL (historical campaign annotation)\n")
            return subprocess.run(
                [sys.executable, str(SCRIPT), str(path)],
                capture_output=True, text=True, check=False,
            )

    def test_historical_and_current_schemas(self):
        for fields in (CORE, V1, V2):
            with self.subTest(fields=fields):
                self.assertEqual(self.validate(fields).returncode, 0)

    def test_legacy_optional_empty_cells_remain_accepted(self):
        self.assertEqual(self.validate(V1, {"n_gen_tok": "", "run_index": ""}).returncode, 0)

    def test_nonfinite_or_invalid_metrics_are_rejected(self):
        metrics = ("prompt_tok_s", "decode_tok_s", "peak_ws_mb", "load_ms",
                   "gpu_mem_mb", "gpu_budget_mb", "prefill_ms", "ttft_ms")
        for field in metrics:
            for value in ("NaN", "inf", "-inf", "1e999", "-1", "invalid"):
                with self.subTest(field=field, value=value):
                    result = self.validate(overrides={field: value})
                    self.assertEqual(result.returncode, 2)
                    self.assertIn(field + " must be finite and non-negative", result.stderr)

    def test_integer_settings_reject_invalid_counts(self):
        for field in ("n_ctx", "n_threads", "n_prompt_tok", "n_gen_tok", "max_length", "run_index"):
            for value in ("-1", "1.5", "NaN", "invalid"):
                with self.subTest(field=field, value=value):
                    result = self.validate(overrides={field: value})
                    self.assertEqual(result.returncode, 2)
                    self.assertIn(field + " must be a non-negative integer", result.stderr)

    def test_short_and_long_rows_are_rejected(self):
        for delta in (-1, 1):
            with self.subTest(delta=delta):
                result = self.validate(row_delta=delta)
                self.assertEqual(result.returncode, 2)
                self.assertIn("wrong field count", result.stderr)

    def test_short_row_with_reordered_headers_is_a_controlled_error(self):
        fields = [field for field in CORE if field != "model"] + ["model"]
        result = self.validate(fields, row_delta=-1)
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertIn("wrong field count", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_duplicate_headers_are_rejected(self):
        result = self.validate(CORE + ["model"])
        self.assertEqual(result.returncode, 2)
        self.assertIn("duplicate benchmark fields", result.stderr)

    def test_zero_metrics_remain_valid(self):
        self.assertEqual(self.validate(overrides={"prompt_tok_s": "0", "ttft_ms": "0"}).returncode, 0)

    def test_historical_single_field_comments_are_not_runs(self):
        self.assertEqual(self.validate(CORE, comment=True).returncode, 0)


if __name__ == "__main__":
    unittest.main()
