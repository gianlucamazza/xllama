#!/usr/bin/env python3
"""Tests for scripts/eval-standard-generative.py — IFEval checks, GSM8K scoring, spec (#597)."""

import copy
import importlib.util
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
SPEC = importlib.util.spec_from_file_location(
    "eval_standard_generative", ROOT / "scripts/eval-standard-generative.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)
C = MODULE.IFEVAL_CHECKS


class IfevalCheckTests(unittest.TestCase):
    def test_comma_and_case(self):
        self.assertTrue(C["punctuation:no_comma"]("no commas here"))
        self.assertFalse(C["punctuation:no_comma"]("one, two"))
        self.assertTrue(C["change_case:english_lowercase"]("all lower case."))
        self.assertFalse(C["change_case:english_lowercase"]("Not lower"))
        self.assertTrue(C["change_case:english_capital"]("ALL CAPS!"))

    def test_word_and_sentence_counts(self):
        self.assertEqual(MODULE.count_words("It's a test, really."), 5)
        self.assertEqual(MODULE.count_sentences("One. Two! Three? \"Four.\" Five"), 5)
        f = C["length_constraints:number_words"]
        self.assertTrue(f("a b c", relation="less than", num_words=4))
        self.assertFalse(f("a b c d", relation="less than", num_words=4))
        self.assertTrue(f("a b c d", relation="at least", num_words=4))

    def test_keywords(self):
        self.assertFalse(C["keywords:forbidden_words"]("The Cat sat", forbidden_words=["cat"]))
        self.assertTrue(C["keywords:forbidden_words"]("concatenate", forbidden_words=["cat"]))
        self.assertTrue(C["keywords:existence"]("Rust and Go", keywords=["rust", "go"]))
        self.assertTrue(
            C["keywords:frequency"]("yes YES yes", keyword="yes", frequency=3, relation="at least")
        )
        self.assertTrue(
            C["keywords:letter_frequency"]("aAa", letter="a", let_frequency=4,
                                           let_relation="less than")
        )

    def test_formats(self):
        self.assertTrue(
            C["detectable_format:number_highlighted_sections"]("*one* and **two**", num_highlights=2)
        )
        self.assertTrue(C["detectable_format:title"]("<<A Poem>>\ntext"))
        self.assertFalse(C["detectable_format:title"]("<<>>"))
        self.assertTrue(C["detectable_format:number_bullet_lists"]("* a\n* b\n- c", num_bullets=3))
        self.assertTrue(C["detectable_format:json_format"]('```json\n{"a": 1}\n```'))
        self.assertFalse(C["detectable_format:json_format"]("{a: 1}"))
        self.assertTrue(
            C["detectable_format:multiple_sections"]("SECTION 1 a SECTION 2 b",
                                                     section_spliter="SECTION", num_sections=2)
        )
        self.assertTrue(C["detectable_format:constrained_response"]("My answer is no."))

    def test_paragraphs(self):
        self.assertTrue(C["length_constraints:number_paragraphs"]("a\n***\nb\n***\nc",
                                                                  num_paragraphs=3))
        self.assertFalse(C["length_constraints:number_paragraphs"]("a\n***\n\n***\nc",
                                                                   num_paragraphs=2))
        f = C["length_constraints:nth_paragraph_first_word"]
        self.assertTrue(f("Hi there.\n\n\"Weather, today\" is fine.", num_paragraphs=2,
                          nth_paragraph=2, first_word="weather"))
        self.assertFalse(f("one\n\ntwo", num_paragraphs=2, nth_paragraph=3, first_word="x"))

    def test_start_end_and_combination(self):
        self.assertTrue(C["startend:quotation"]('  "quoted"  '))
        self.assertTrue(C["startend:end_checker"]('Bye. Any other questions?"',
                                                  end_phrase="Any other questions?"))
        self.assertTrue(C["detectable_content:postscript"]("body\nP.S. hi", postscript_marker="P.S."))
        self.assertTrue(C["detectable_content:number_placeholders"]("[name] at [address]",
                                                                    num_placeholders=2))
        self.assertTrue(C["combination:repeat_prompt"]("Say hi. Hi!", prompt_to_repeat="say hi."))
        self.assertTrue(C["combination:two_responses"]("one\n******\ntwo"))
        self.assertFalse(C["combination:two_responses"]("same\n******\nsame"))

    def test_strict_prompt_score_and_none_kwargs(self):
        item = {
            "key": 1,
            "instruction_id_list": ["punctuation:no_comma", "length_constraints:number_words"],
            "kwargs": [{"num_words": None}, {"relation": "at least", "num_words": 3}],
        }
        self.assertEqual(MODULE.score_ifeval(item, "three short words"),
                         {"pass": True, "instructions": 2, "followed": 2})
        self.assertEqual(MODULE.score_ifeval(item, "two, words")["followed"], 0)
        self.assertFalse(MODULE.score_ifeval(item, "   ")["pass"])


class Gsm8kTests(unittest.TestCase):
    def test_reference_and_extraction(self):
        self.assertEqual(MODULE.gsm8k_reference("work\n#### 1,234"), "1234")
        self.assertEqual(MODULE.gsm8k_extract("5 + 3 = 8 apples.\nAnswer: $1,234."), "1234")
        self.assertEqual(MODULE.gsm8k_extract("The answer is **42**"), "42")
        # Seen on LFM2.5-1.2B: the model copied the placeholder brackets.
        self.assertEqual(MODULE.gsm8k_extract("Answer: <18> (3 were left)"), "18")
        self.assertEqual(MODULE.gsm8k_extract("so 3 then 7.50"), "7.5")
        self.assertEqual(MODULE.gsm8k_extract("2.0 total\nAnswer: 2.00"), "2")
        self.assertIsNone(MODULE.gsm8k_extract("no digits"))

    def test_score(self):
        item = {"answer": "x\n#### 18"}
        self.assertTrue(MODULE.score_gsm8k(item, "9 * 2 = 18\nAnswer: 18")["pass"])
        self.assertFalse(MODULE.score_gsm8k(item, "Answer: 17")["pass"])


class SpecAndSummaryTests(unittest.TestCase):
    def test_committed_spec(self):
        spec = MODULE.load_spec()
        self.assertEqual(len(spec["tasks"]["ifeval"]["keys"]), 100)
        self.assertEqual(len(spec["tasks"]["gsm8k"]["keys"]), 100)

    def test_rejects_duplicate_keys_and_bad_scorer(self):
        spec = MODULE.load_spec()
        bad = copy.deepcopy(spec)
        bad["tasks"]["gsm8k"]["keys"].append(bad["tasks"]["gsm8k"]["keys"][0])
        with self.assertRaises(MODULE.EvalError):
            MODULE.validate_spec(bad)
        bad = copy.deepcopy(spec)
        bad["tasks"]["gsm8k"]["scorer"] = "regex"
        with self.assertRaises(MODULE.EvalError):
            MODULE.validate_spec(bad)

    def test_summary_counts_errors_and_length_stops(self):
        rows = [
            {"pass": True, "finish_reason": "stop", "instructions": 2, "followed": 2},
            {"pass": False, "finish_reason": "length", "instructions": 2, "followed": 1},
            {"pass": False, "error": "HTTP 503", "instructions": 1, "followed": 0},
        ]
        s = MODULE.summarize(rows)
        self.assertEqual((s["n_items"], s["passed"], s["errors"], s["length_stops"]), (3, 1, 1, 1))
        self.assertAlmostEqual(s["acc"], 33.33)
        self.assertAlmostEqual(s["instruction_acc"], 60.0)


if __name__ == "__main__":
    unittest.main()
