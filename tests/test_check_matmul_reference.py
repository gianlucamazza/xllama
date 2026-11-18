# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
"""The reference arithmetic of check-matmul-reference.py (no GGUF needed)."""
import importlib.util
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/check-matmul-reference.py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class ReferenceTests(unittest.TestCase):
    def setUp(self):
        self.mod = load(SCRIPT, "check_matmul_reference")
        rng = np.random.default_rng(1)
        self.w = rng.standard_normal((6, 512))
        self.x = rng.standard_normal((512, 5)).astype(np.float32)

    def test_round_trip_matches_the_comparator_quantizer(self):
        cp = load(ROOT / "scripts/control-parity.py", "control_parity")
        back = self.mod.dequantized_q8_k(self.x)
        # Within half a step of x, and quantizing it again gives the same int8.
        blocks = np.abs(self.x.T.reshape(5, 2, 256)).max(axis=2)
        step = np.repeat(blocks / 127.0, 256, axis=1).T
        self.assertTrue(np.all(np.abs(back - self.x) <= 0.5 * step + 1e-6))
        again = cp.quantize_q8_k(back.astype(np.float32))
        self.assertTrue(np.array_equal(again, cp.quantize_q8_k(self.x)))

    def test_an_exact_kernel_has_no_error(self):
        result = self.w @ self.mod.dequantized_q8_k(self.x)
        e = self.mod.record_error(self.w, self.x, result)
        self.assertLess(e["nmse"], 1e-28)

    def test_a_kernel_that_skips_the_quantizer_is_visible(self):
        e = self.mod.record_error(self.w, self.x, self.w @ self.x.astype(np.float64))
        self.assertGreater(e["nmse"], 1e-7)

    def test_one_wrong_column_sets_the_worst_token(self):
        result = self.w @ self.mod.dequantized_q8_k(self.x)
        result[:, 3] *= 1.001
        e = self.mod.record_error(self.w, self.x, result)
        self.assertAlmostEqual(e["worst_token_nmse"], 1e-6, delta=2e-8)
        self.assertLess(e["nmse"], e["worst_token_nmse"])

    def test_q8_0_round_trip_follows_the_avx2_quantizer(self):
        x = np.zeros((32, 2), dtype=np.float32)
        x[:4, 0] = [-2.0, 1.0, 0.5 * 2 / 127, 1.5 * 2 / 127]  # max 2: id 63.5
        back = self.mod.dequantized_q8_0(x)
        d = float(np.float16(np.float32(2.0) / np.float32(127.0)))
        # round-half-even gives -127, 64 (63.5), 0 (0.5) and 2 (1.5); scale is float16
        self.assertTrue(np.allclose(back[:4, 0], np.array([-127, 64, 0, 2]) * d, rtol=0, atol=0))
        self.assertTrue((back[:, 1] == 0).all())

    def test_the_quantizer_follows_the_weight_type(self):
        self.assertIs(self.mod.operand_quantizer("q4_K", 512), self.mod.dequantized_q8_k)
        self.assertIs(self.mod.operand_quantizer("q4_0", 512), self.mod.dequantized_q8_0)
        self.assertIsNone(self.mod.operand_quantizer("q4_K", 96))  # not whole q8_K blocks
        self.assertIsNone(self.mod.operand_quantizer("f16", 512))
        result = self.w @ self.mod.dequantized_q8_0(self.x)
        e = self.mod.record_error(self.w, self.x, result, self.mod.dequantized_q8_0)
        self.assertLess(e["nmse"], 1e-28)
        # judged with the wrong quantizer, the same result is far off
        self.assertGreater(self.mod.record_error(self.w, self.x, result)["nmse"], 1e-7)

    def test_bad_arguments_fail_closed(self):
        r = subprocess.run([sys.executable, str(SCRIPT), "missing.gguf", "missing-dir",
                            "--nmse-limit", "0"], capture_output=True, text=True, check=False)
        self.assertEqual(r.returncode, 2)


if __name__ == "__main__":
    unittest.main()
