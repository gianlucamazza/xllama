# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
"""Contract of scripts/make-moe-fixture.py (needs numpy; network-free)."""

import hashlib
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "llama.cpp" / "gguf-py"))

spec = importlib.util.spec_from_file_location("mmf", ROOT / "scripts" / "make-moe-fixture.py")
mmf = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mmf)

E, FF, D = 16, 512, 256  # template experts, expert width, embedding width


def write_template(path, arch="lfm2moe"):
    """A tiny lfm2moe-shaped header: layer 0 dense conv, 1 MoE conv, 2 MoE attn, 3 MoE conv."""
    import numpy as np
    from gguf import GGUFWriter

    w = GGUFWriter(str(path), arch)
    w.add_uint32(f"{arch}.block_count", 4)
    w.add_uint32(f"{arch}.embedding_length", D)
    w.add_uint32(f"{arch}.feed_forward_length", 512)
    w.add_uint32(f"{arch}.expert_count", E)
    w.add_uint32(f"{arch}.expert_used_count", 4)
    w.add_uint32(f"{arch}.expert_feed_forward_length", FF)
    w.add_uint32(f"{arch}.leading_dense_block_count", 1)
    w.add_array(f"{arch}.attention.head_count_kv", [0, 0, 4, 0])
    w.add_array("tokenizer.ggml.tokens", ["a", "b", "c"])
    tensors = [("token_embd.weight", (3, D))]
    for i in range(4):
        tensors.append((f"blk.{i}.ffn_norm.weight", (D,)))
        if i == 0:
            tensors.append((f"blk.{i}.ffn_up.weight", (512, D)))
        else:
            tensors += [
                (f"blk.{i}.ffn_gate_inp.weight", (E, D)),
                (f"blk.{i}.ffn_up_exps.weight", (E, FF, D)),
                (f"blk.{i}.ffn_down_exps.weight", (E, D, FF)),
                (f"blk.{i}.exp_probs_b.bias", (E,)),
            ]
        if i == 2:
            tensors.append((f"blk.{i}.attn_q.weight", (D, D)))
        else:
            tensors.append((f"blk.{i}.shortconv.conv.weight", (D, 3)))
    for name, shape in tensors:
        w.add_tensor(name, np.zeros(shape, dtype=np.float32))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def read(path):
    from gguf import GGUFReader

    r = GGUFReader(str(path))
    kv = {k: f.contents() for k, f in r.fields.items()}
    shapes = {t.name: [int(x) for x in t.shape] for t in r.tensors}
    return kv, shapes


class MakeMoeFixtureTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.template = self.dir / "template.gguf"
        write_template(self.template)

    def tearDown(self):
        self.tmp.cleanup()

    def test_supported_archs_from_table(self):
        self.assertEqual(
            mmf.supported_archs(),
            ["lfm2moe", "bailingmoe2", "ernie4_5-moe", "smallthinker", "gpt-oss", "gemma4", "afmoe", "maple"],
        )

    def test_shrinks_layers_experts_and_width(self):
        out = self.dir / "f.gguf"
        mmf.main(["--template", str(self.template), "--out", str(out), "--quant", "none"])
        kv, shapes = read(out)
        # kinds: dense conv (0), MoE conv (1), MoE attention (2); layer 3 repeats layer 1
        self.assertEqual(kv["lfm2moe.block_count"], 3)
        self.assertEqual(kv["lfm2moe.expert_count"], 8)
        self.assertEqual(kv["lfm2moe.expert_used_count"], 2)
        self.assertEqual(kv["lfm2moe.expert_feed_forward_length"], 256)
        self.assertEqual(kv["lfm2moe.leading_dense_block_count"], 1)
        self.assertEqual(list(kv["lfm2moe.attention.head_count_kv"]), [0, 0, 4])
        self.assertIn("random-weights", kv["general.tags"])
        self.assertEqual(shapes["blk.1.ffn_up_exps.weight"], [D, 256, 8])
        self.assertEqual(shapes["blk.1.ffn_down_exps.weight"], [256, D, 8])
        self.assertEqual(shapes["blk.2.ffn_gate_inp.weight"], [D, 8])
        self.assertEqual(shapes["blk.2.exp_probs_b.bias"], [8])
        self.assertEqual(shapes["blk.0.ffn_up.weight"], [D, 512])  # dense FFN untouched
        self.assertIn("blk.2.attn_q.weight", shapes)
        self.assertNotIn("blk.3.ffn_norm.weight", shapes)

    def test_conv_and_norm_stay_f32(self):
        from gguf import GGMLQuantizationType, GGUFReader

        out = self.dir / "f.gguf"
        mmf.main(["--template", str(self.template), "--out", str(out), "--quant", "none"])
        types = {t.name: t.tensor_type for t in GGUFReader(str(out)).tensors}
        self.assertEqual(types["blk.1.shortconv.conv.weight"], GGMLQuantizationType.F32)
        self.assertEqual(types["blk.1.ffn_norm.weight"], GGMLQuantizationType.F32)
        self.assertEqual(types["blk.1.ffn_up_exps.weight"], GGMLQuantizationType.F16)

    def test_shapes_by_position_when_embd_equals_ff(self):
        # gpt-oss: n_embd == n_ff_exp == 2880; only the expert width may shrink.
        f = lambda base, suf, shape: mmf.shrink_shape(base, suf, shape, 32, 2880, 0, 8, 256)
        self.assertEqual(f("ffn_gate_exps", "weight", [2880, 2880, 32]), [2880, 256, 8])
        self.assertEqual(f("ffn_down_exps", "weight", [2880, 2880, 32]), [256, 2880, 8])
        self.assertEqual(f("ffn_up_exps", "bias", [2880, 32]), [256, 8])
        self.assertEqual(f("ffn_down_exps", "bias", [2880, 32]), [2880, 8])
        self.assertEqual(f("ffn_gate_up_exps", "weight", [2880, 5760, 32]), [2880, 512, 8])
        self.assertEqual(f("ffn_gate_inp", "weight", [2880, 32]), [2880, 8])
        self.assertEqual(f("attn_q", "weight", [2880, 2880]), [2880, 2880])

    def test_shared_experts_scale_with_expert_width(self):
        # afmoe: n_ff_shexp = n_ff_exp * n_expert_shared
        g = mmf.shrink_shape("ffn_gate_shexp", "weight", [2048, 1024], 128, 1024, 1, 8, 256)
        self.assertEqual(g, [2048, 256])
        d = mmf.shrink_shape("ffn_down_shexp", "weight", [2048, 1024], 128, 1024, 0, 8, 256)
        self.assertEqual(d, [2048, 1024])  # own width key: untouched

    def test_shared_width_collision_preserves_embedding_and_other_roles(self):
        for base, suffix, shape, expected in (
            ("ffn_gate_shexp", "weight", [1024, 1024], [1024, 256]),
            ("ffn_up_shexp", "weight", [1024, 1024], [1024, 256]),
            ("ffn_down_shexp", "weight", [1024, 1024], [256, 1024]),
            ("ffn_gate_inp_shexp", "weight", [1024, 1], [1024, 1]),
            ("ffn_down_shexp", "bias", [1024], [1024]),
        ):
            with self.subTest(base=base, suffix=suffix):
                self.assertEqual(mmf.shrink_shape(base, suffix, shape, 16, 512, 2, 8, 128), expected)

    def test_shared_width_follows_the_architecture_loader(self):
        for arch, explicit_width, expected_width in (
            ("afmoe", None, 256),
            ("afmoe", 512, 256),  # the pin derives this width even when an extra key is present
            ("bailingmoe2", None, 256),
            ("bailingmoe2", 0, 256),
            ("bailingmoe2", 512, 1024),  # explicit per shared expert
            ("ernie4_5-moe", 1024, 1024),  # explicit total
        ):
            with self.subTest(arch=arch, explicit_width=explicit_width):
                values = {
                    "block_count": 2, "embedding_length": 1024, "expert_count": 16,
                    "expert_feed_forward_length": [512, 768], "expert_shared_count": 2,
                }
                if explicit_width is not None:
                    values["expert_shared_feed_forward_length"] = explicit_width
                kvs = [(f"{arch}.{key}", val, None, None) for key, val in values.items()]
                tensors = [(f"blk.{layer}.{role}.weight", shape)
                           for layer in range(2)
                           for role, shape in (("ffn_gate_inp", [1024, 16]),
                                               ("ffn_up_shexp", [1024, 1024]),
                                               ("ffn_down_shexp", [1024, 1024]))]
                _, plan, _ = mmf.build_plan(arch, kvs, tensors, 8, 2, 128, 1)
                self.assertEqual(dict(plan)["blk.0.ffn_up_shexp.weight"], [1024, expected_width])
                self.assertEqual(dict(plan)["blk.0.ffn_down_shexp.weight"], [expected_width, 1024])

    def test_inconsistent_derived_shared_weight_is_refused(self):
        for shape in ([1024], [1024, 768]):
            with self.subTest(shape=shape), self.assertRaisesRegex(ValueError, "ffn_up_shexp"):
                mmf.shrink_shape("ffn_up_shexp", "weight", shape, 16, 512, 2, 8, 128)

    def test_unknown_arch_refused(self):
        llama = self.dir / "llama.gguf"
        write_template(llama, arch="llama")
        with self.assertRaises(SystemExit):
            mmf.main(["--template", str(llama), "--out", str(self.dir / "x.gguf"), "--quant", "none"])

    def test_too_many_experts_refused(self):
        with self.assertRaises(ValueError):
            mmf.main(["--template", str(self.template), "--out", str(self.dir / "x.gguf"),
                      "--quant", "none", "--experts", "32"])

    def test_seed_reproducible(self):
        digests = []
        for name in ("a.gguf", "b.gguf"):
            out = self.dir / name
            mmf.main(["--template", str(self.template), "--out", str(out), "--quant", "none",
                      "--seed", "7"])
            digests.append(hashlib.sha256(out.read_bytes()).hexdigest())
        self.assertEqual(digests[0], digests[1])


if __name__ == "__main__":
    unittest.main()
