#!/usr/bin/env python3
"""Tests for scripts/hellaswag_token_mean.py — llama-perplexity's HellaSwag rule (#608)."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/hellaswag_token_mean.py"
SPEC = importlib.util.spec_from_file_location("hellaswag_token_mean", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class SharedPrefixTest(unittest.TestCase):
    def test_context_only(self) -> None:
        self.assertEqual(MODULE.shared_prefix([[1, 2, 3, 4], [1, 2, 3, 5], [1, 2, 3, 6]]), 3)

    def test_endings_that_share_leading_tokens(self) -> None:
        # Context 1,2 then every ending starts with token 7: llama-perplexity
        # skips it too.
        self.assertEqual(MODULE.shared_prefix([[1, 2, 7, 4], [1, 2, 7, 5, 9], [1, 2, 7, 6]]), 3)

    def test_leaves_every_choice_one_token(self) -> None:
        self.assertEqual(MODULE.shared_prefix([[1, 2, 3], [1, 2, 3, 4]]), 2)


class ChooseTest(unittest.TestCase):
    def test_mean_over_the_scored_suffix_only(self) -> None:
        # Shared token at index 1 is very unlikely in both; it must not decide.
        a = [0.0, -9.0, -1.0, -1.0]  # suffix mean -1.0
        b = [0.0, -9.0, -0.5]  # suffix mean -0.5
        self.assertEqual(MODULE.choose([a, b], start=2), 1)

    def test_shared_tokens_change_a_whole_continuation_mean(self) -> None:
        # Dividing the whole continuation (start 1) instead flips the choice:
        # the boundary is why the rule needs per-token log-probabilities.
        a = [0.0, -9.0, -1.0, -1.0]  # whole mean -11/3 = -3.67
        b = [0.0, -9.0, -0.5]  # whole mean -9.5/2 = -4.75
        self.assertEqual(MODULE.choose([a, b], start=1), 0)
        self.assertEqual(MODULE.choose([a, b], start=2), 1)

    def test_token_zero_is_never_scored(self) -> None:
        self.assertEqual(MODULE.choose([[-100.0, -1.0], [0.0, -2.0]], start=0), 0)


class SamplesTest(unittest.TestCase):
    def test_items_read_lm_eval_samples(self) -> None:
        row = {"doc_id": 7, "arguments": {f"gen_args_{i}": {"arg_0": "Ctx: a man", "arg_1": f" ending {i}."}
                             for i in range(4)}, "target": 2}
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "samples.jsonl"
            path.write_text(json.dumps(row) + "\n")
            [(doc_id, context, texts, gold)] = list(MODULE.items(path))
        self.assertEqual(context, "Ctx: a man")
        self.assertEqual(texts[3], "Ctx: a man ending 3.")
        self.assertEqual(gold, 2)
        self.assertEqual(doc_id, 7)

    def test_klosax_items_join_like_perplexity_cpp(self) -> None:
        text = "Ctx: a man\n2\nends a.\nends b.\nends c.\nends d.\n" * 2
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "tasks.txt"
            path.write_text(text)
            rows = list(MODULE.klosax_items(path))
        self.assertEqual(len(rows), 2)
        doc_id, context, texts, gold = rows[1]
        self.assertEqual((doc_id, context, gold), (1, "Ctx: a man", 2))
        self.assertEqual(texts[0], "Ctx: a man ends a.")

    def test_binomial(self) -> None:
        acc, stderr = MODULE.binomial(50, 101)
        self.assertAlmostEqual(acc, 49.50, places=2)
        self.assertAlmostEqual(stderr, 5.0, places=1)

    def test_binomial_one_item_has_no_stderr_and_none_is_refused(self) -> None:
        self.assertEqual(MODULE.binomial(1, 1), (100.0, None))
        with self.assertRaises(ValueError):
            MODULE.binomial(0, 0)


if __name__ == "__main__":
    unittest.main()
