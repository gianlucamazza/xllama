#!/usr/bin/env python3
"""Tests for scripts/eval-host-standard.py — spec pins and llama-perplexity parsing (#597)."""

import copy
import importlib.util
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/eval-host-standard.py"
SPEC = importlib.util.spec_from_file_location("eval_host_standard", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

# Shapes copied from tools/perplexity/perplexity.cpp output at the pin.
HELLASWAG_OUT = """\
hellaswag_score : selecting 3 randomized tasks.

task\tacc_norm\t95% confidence interval
1\t100.00000000%\t[20.6549%, 100.0000%]
2\t50.00000000%\t[9.4529%, 90.5471%]
3\t66.66666667%\t[20.7655%, 93.8510%]

"""

WINOGRANDE_OUT = """\
winogrande_score : calculating winogrande score over selected tasks.
1266\t71.0900\t -1.234567   -2.345678  1  1
1267\t71.0339\t -0.123456   -0.234567  0  1

winogrande_score : Final Winogrande score(1267 tasks): 71.0339 +/- 1.2747
"""

MC_OUT = """\
multiple_choice_score : calculating TruthfulQA score over 299 tasks.

task\tacc_norm
298\t49.66442953
299\t49.83277592

multiple_choice_score : Final result: 49.8328 +/- 2.8959
multiple_choice_score : Random chance: 25.0000 +/- 2.5084
"""


class ParseTests(unittest.TestCase):
    def test_hellaswag_takes_last_row_and_binomial_stderr(self):
        r = MODULE.parse_output("hellaswag", HELLASWAG_OUT)
        self.assertEqual(r["n_tasks"], 3)
        self.assertAlmostEqual(r["acc"], 66.66666667)
        # sqrt(p(1-p)/(n-1)) with p=2/3, n=3
        self.assertAlmostEqual(r["stderr"], 33.3333, places=3)

    def test_winogrande_final_line(self):
        r = MODULE.parse_output("winogrande", WINOGRANDE_OUT)
        self.assertEqual(r, {"n_tasks": 1267, "acc": 71.0339, "stderr": 1.2747})

    def test_multiple_choice_final_and_chance(self):
        r = MODULE.parse_output("multiple-choice", MC_OUT)
        self.assertEqual(r["n_tasks"], 299)
        self.assertAlmostEqual(r["acc"], 49.8328)
        self.assertAlmostEqual(r["stderr"], 2.8959)
        self.assertAlmostEqual(r["chance"], 25.0)

    def test_missing_result_fails_closed(self):
        for mode in MODULE.MODES:
            with self.assertRaises(MODULE.EvalError):
                MODULE.parse_output(mode, "llama_model_load: error loading model\n")

    def test_truncated_multiple_choice_fails_closed(self):
        # perplexity.cpp returns before the Final line when a subset scored < 100 tasks.
        with self.assertRaises(MODULE.EvalError):
            MODULE.parse_output("multiple-choice", "task\tacc_norm\n42\t50.0\n")


class SpecTests(unittest.TestCase):
    def test_committed_spec_is_valid(self):
        spec = MODULE.load_spec()
        self.assertIn("hellaswag", spec["tasks"])
        for task in spec["tasks"].values():
            self.assertIn(task["dataset"], spec["datasets"])

    def test_rejects_bad_sha_and_unknown_dataset(self):
        spec = MODULE.load_spec()
        bad = copy.deepcopy(spec)
        bad["datasets"]["mmlu"]["sha256"] = "abc"
        with self.assertRaises(MODULE.EvalError):
            MODULE.validate_spec(bad)
        bad = copy.deepcopy(spec)
        bad["tasks"]["mmlu"]["dataset"] = "nope"
        with self.assertRaises(MODULE.EvalError):
            MODULE.validate_spec(bad)

    def test_multiple_choice_has_room_for_five_choices(self):
        # ARC has tasks with 5 choices; -np 4 aborted the 2026-11-26 smoke at task 210.
        spec = MODULE.load_spec()
        for task in spec["tasks"].values():
            if task["mode"] == "multiple-choice":
                self.assertGreaterEqual(task["n_parallel"], 5)
        bad = copy.deepcopy(spec)
        del bad["tasks"]["hellaswag"]["n_parallel"]
        with self.assertRaises(MODULE.EvalError):
            MODULE.validate_spec(bad)

    def test_dataset_sha_mismatch_is_refused(self):
        spec = MODULE.load_spec()
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / spec["datasets"]["mmlu"]["file"]).write_bytes(b"not mmlu")
            with self.assertRaises(MODULE.EvalError):
                MODULE.dataset_path(spec, "mmlu", Path(d))

    def test_fetch_keeps_a_pinned_line_prefix(self):
        # #608: a fixed subset of a published file is its first N lines, and
        # sha256 pins the prefix, so --fetch must cut before it checks.
        import hashlib
        import io
        import sys
        from unittest import mock

        datasets_module = sys.modules["eval_datasets"]
        body = b"".join(b"line %d\n" % i for i in range(10))
        prefix = b"".join(b"line %d\n" % i for i in range(4))

        def spec_for(lines, sha):
            return {"datasets": {"hs": {"file": "hs.txt", "url": "https://example.invalid/hs.txt",
                                        "sha256": sha, "upstream": "x", "upstream_licence": "MIT",
                                        "lines": lines}}}

        def serve(*_args, **_kwargs):
            return io.BytesIO(body)

        with tempfile.TemporaryDirectory() as d, mock.patch.object(
            datasets_module.urllib.request, "urlopen", serve
        ):
            spec = spec_for(4, hashlib.sha256(prefix).hexdigest())
            MODULE.validate_datasets(spec["datasets"])
            MODULE.fetch(spec, Path(d))
            self.assertEqual((Path(d) / "hs.txt").read_bytes(), prefix)
            self.assertEqual(MODULE.dataset_path(spec, "hs", Path(d)), Path(d) / "hs.txt")
            # Asking for more lines than the file holds is refused, and leaves no file.
            short = spec_for(20, hashlib.sha256(body).hexdigest())
            short["datasets"]["hs"]["file"] = "short.txt"
            with self.assertRaises(MODULE.EvalError):
                MODULE.fetch(short, Path(d))
            self.assertFalse((Path(d) / "short.txt").exists())
        for bad in (0, -1, "12000", 1.5):
            with self.assertRaises(MODULE.EvalError):
                MODULE.validate_datasets(spec_for(bad, "0" * 64)["datasets"])

    def test_hellaswag_zero_means_every_task(self):
        # Upstream caps --hellaswag-tasks to the value given, so 0 would score nothing.
        with tempfile.TemporaryDirectory() as d:
            data = Path(d) / "hs.txt"
            data.write_text("x\n" * 18)
            args = MODULE.perplexity_args({"mode": "hellaswag", "tasks": 0}, data)
            self.assertEqual(args[-1], "3")
            args = MODULE.perplexity_args({"mode": "hellaswag", "tasks": 2}, data)
            self.assertEqual(args[-1], "2")

    def test_evaluator_version_reads_build_line(self):
        with tempfile.TemporaryDirectory() as d:
            fake = Path(d) / "llama-perplexity"
            fake.write_text("#!/bin/sh\necho 'version: 0.6.0-dev (build 1, commit d81235049)' >&2\n")
            fake.chmod(0o755)
            self.assertEqual(MODULE.evaluator_version(fake), "0.6.0-dev (build 1, commit d81235049)")
            self.assertEqual(MODULE.evaluator_version(Path(d) / "missing"), "")

    def test_perplexity_args_per_mode(self):
        p = Path("/data/x")
        self.assertEqual(MODULE.perplexity_args({"mode": "hellaswag", "tasks": 400}, p)[:3],
                         ["-f", "/data/x", "--hellaswag"])
        self.assertIn("-bf", MODULE.perplexity_args({"mode": "multiple-choice", "tasks": 0}, p))


if __name__ == "__main__":
    unittest.main()
