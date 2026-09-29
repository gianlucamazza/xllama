#!/usr/bin/env python3
"""Tests for the shape-gate random GGUF builder (scripts/make-shape-gguf.py)."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "make_shape_gguf", ROOT / "scripts/make-shape-gguf.py"
)
msg = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(msg)

LFM25_ATTN = [2, 5, 8, 10, 12, 14]


class ShapeEstimateTests(unittest.TestCase):
    def test_clones_match_released_lfm25_parameter_counts(self):
        # Element counts read from the released Q4_K_M GGUFs (GGUFReader, all tensors).
        cases = [
            (
                msg.Shape("230m", 14, 1024, 2560, attn_layers=[2, 4, 6, 8, 10, 12]),
                229_693_184,
            ),
            (msg.Shape("350m", 16, 1024, 4608, attn_layers=LFM25_ATTN), 354_483_968),
            (msg.Shape("1.2b", 16, 2048, 8192, attn_layers=LFM25_ATTN), 1_170_340_608),
        ]
        for shape, expected in cases:
            with self.subTest(shape=shape.id):
                self.assertAlmostEqual(
                    msg.estimate(shape)["params_M"], expected / 1e6, places=6
                )

    def test_kv_per_token_matches_measured_lfm25(self):
        # 12 KiB/token is the measured LFM2.5-350M KV snapshot rate (#170b).
        s = msg.Shape("350m", 16, 1024, 4608, attn_layers=LFM25_ATTN)
        self.assertEqual(msg.estimate(s)["kv_KiB_per_tok"], 12.0)

    def test_smaller_vocab_only_shrinks_the_embedding(self):
        big = msg.Shape("a", 16, 2048, 8192, attn_layers=LFM25_ATTN)
        small = msg.Shape("b", 16, 2048, 8192, n_vocab=16384, attn_layers=LFM25_ATTN)
        self.assertEqual(msg.estimate(big)["body_M"], msg.estimate(small)["body_M"])
        self.assertLess(
            msg.estimate(small)["read_MB_per_tok"], msg.estimate(big)["read_MB_per_tok"]
        )


class ShapeValidationTests(unittest.TestCase):
    def test_rejects_dims_that_q4k_cannot_block(self):
        with self.assertRaisesRegex(ValueError, "multiple of 256"):
            msg.validate_shape(msg.Shape("x", 16, 1664, 8192, attn_layers=[2]))

    def test_rejects_attention_layer_out_of_range(self):
        with self.assertRaisesRegex(ValueError, "out of range"):
            msg.validate_shape(msg.Shape("x", 8, 2048, 8192, attn_layers=[8]))

    def test_default_attention_layout_is_in_range_and_sparse(self):
        for n_layer in (8, 12, 16, 24, 32):
            with self.subTest(n_layer=n_layer):
                layers = msg.default_attn_layers(n_layer)
                self.assertTrue(all(0 <= i < n_layer for i in layers))
                self.assertLess(len(layers), n_layer / 2)

    def test_checked_in_sweep_spec_loads(self):
        shapes = msg.load_spec(ROOT / "bench/configs/shape-sweep.json")
        self.assertIn("ref-350m", [s.id for s in shapes])

    def test_duplicate_ids_are_rejected(self):
        shape = {"id": "a", "n_layer": 8, "n_embd": 2048, "n_ff": 8192}
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"shapes": [shape, shape]}, f)
        try:
            with self.assertRaisesRegex(ValueError, "duplicate"):
                msg.load_spec(f.name)
        finally:
            Path(f.name).unlink()


class TokenizerTruncationTests(unittest.TestCase):
    def test_merges_referencing_dropped_tokens_are_removed(self):
        tokens = ["a", "b", "c", "ab", "abc", "bc"]
        types = [1] * len(tokens)
        merges = ["a b", "ab c", "b c"]
        kept, kept_types, kept_merges = msg.truncate_tokenizer(tokens, types, merges, 4)
        self.assertEqual(kept, ["a", "b", "c", "ab"])
        self.assertEqual(len(kept_types), 4)
        self.assertEqual(kept_merges, ["a b"])

    def test_vocab_larger_than_template_is_an_error(self):
        with self.assertRaises(ValueError):
            msg.truncate_tokenizer(["a"], [1], [], 2)


if __name__ == "__main__":
    unittest.main()
