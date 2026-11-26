#!/usr/bin/env python3
"""Tests for the standard-benchmark section of generate-benchmark-summary.py (#597)."""

import copy
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "generate_benchmark_summary", ROOT / "scripts/generate-benchmark-summary.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

SHA = "a" * 64
POLICY = {
    "standard": {
        "models": {"m1": {"label": "Model One", "gguf": "m1.gguf", "gguf_sha256": SHA}},
        "loglikelihood": {"source": "ll.jsonl", "tasks": ["hellaswag", "mmlu"]},
        "generative": {"tasks": ["ifeval", "gsm8k"], "sources": ["gen.jsonl"]},
    }
}


def ll_row(task, acc=50.0, chance=25.0, model="m1.gguf", sha=SHA):
    return {"model": model, "model_sha256": sha, "task": task, "acc": acc, "stderr": 2.0,
            "chance": chance, "hardware": "linux-host-declared"}


def gen_rows(task, n=2, partial=False, items=None):
    rows = [{"type": "item", "task": task, "key": i, "pass": i == 0} for i in range(
        n if items is None else items)]
    rows.append({"type": "summary", "task": task, "model": "m1", "partial": partial,
                 "n_items": n, "passed": 1, "acc": 50.0, "stderr": 35.36, "errors": 0,
                 "length_stops": 0, "instruction_acc": 75.0, "endpoint": "host",
                 "hardware": "linux-host-declared"})
    return rows


class StandardSummaryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.saved = MODULE.RESULTS
        MODULE.RESULTS = Path(self.tmp.name)

    def tearDown(self):
        MODULE.RESULTS = self.saved
        self.tmp.cleanup()

    def write(self, name, rows):
        with (MODULE.RESULTS / name).open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")

    def good(self):
        self.write("ll.jsonl", [ll_row("hellaswag"), ll_row("mmlu", acc=30.0)])
        self.write("gen.jsonl", gen_rows("ifeval") + gen_rows("gsm8k"))

    def test_renders_both_tables(self):
        self.good()
        text = MODULE.standard_markdown(MODULE.resolve_standard(POLICY))
        self.assertIn("| Model One | 50.0 ± 2.0 | 30.0 ± 2.0 | linux-host-declared |", text)
        self.assertIn("| _chance_ | 25 | 25 | |", text)
        self.assertIn("| 1/2 ± 35.4 | 75.0 % | 1/2 ± 35.4 | 0 | `gen` |", text)
        self.assertTrue(text.startswith(MODULE.STD_BEGIN) and text.endswith(MODULE.STD_END))

    def test_missing_or_duplicate_task_fails(self):
        self.write("ll.jsonl", [ll_row("hellaswag")])
        self.write("gen.jsonl", gen_rows("ifeval") + gen_rows("gsm8k"))
        with self.assertRaisesRegex(ValueError, "missing"):
            MODULE.resolve_standard(POLICY)
        self.write("ll.jsonl", [ll_row("hellaswag"), ll_row("mmlu"), ll_row("mmlu")])
        with self.assertRaisesRegex(ValueError, "duplicate"):
            MODULE.resolve_standard(POLICY)

    def test_unpinned_model_fails(self):
        self.good()
        self.write("ll.jsonl", [ll_row("hellaswag", sha="b" * 64), ll_row("mmlu")])
        with self.assertRaisesRegex(ValueError, "sha256"):
            MODULE.resolve_standard(POLICY)
        self.write("ll.jsonl", [ll_row("hellaswag"), ll_row("mmlu"), ll_row("mmlu", model="x")])
        with self.assertRaisesRegex(ValueError, "undeclared GGUF"):
            MODULE.resolve_standard(POLICY)

    def test_partial_or_short_generative_run_fails(self):
        self.good()
        self.write("gen.jsonl", gen_rows("ifeval", partial=True) + gen_rows("gsm8k"))
        with self.assertRaisesRegex(ValueError, "partial"):
            MODULE.resolve_standard(POLICY)
        self.write("gen.jsonl", gen_rows("ifeval", items=1) + gen_rows("gsm8k"))
        with self.assertRaisesRegex(ValueError, "items"):
            MODULE.resolve_standard(POLICY)
        self.write("gen.jsonl", gen_rows("ifeval"))
        with self.assertRaisesRegex(ValueError, "must hold"):
            MODULE.resolve_standard(POLICY)

    def test_unknown_chance_prints_dash(self):
        self.good()
        self.write("ll.jsonl", [ll_row("hellaswag", chance=None), ll_row("mmlu")])
        policy = copy.deepcopy(POLICY)
        text = MODULE.standard_markdown(MODULE.resolve_standard(policy))
        self.assertIn("| _chance_ | — | 25 | |", text)

    def test_committed_policy_keeps_h9_as_the_gate(self):
        policy = json.loads((ROOT / "bench/benchmark-summary.json").read_text(encoding="utf-8"))
        self.assertEqual(policy["quality"]["source"], "2026-09-30-h9-rescore.jsonl")
        self.assertIn("standard", policy)
        self.assertNotIn(policy["quality"]["source"], policy["standard"]["generative"]["sources"])


if __name__ == "__main__":
    unittest.main()
