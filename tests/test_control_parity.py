# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
"""ADR 0010 control comparison: kernel and spread modes, fail-closed inputs."""
import hashlib
import importlib.util
import json
import subprocess
import sys
import tempfile
import zlib
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/control-parity.py"
N, TOKENS = 256, 4


def load_module():
    spec = importlib.util.spec_from_file_location("control_parity", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def grid_input():
    """[N, TOKENS] operand whose q8_K values are exact: x = k/127, max 1.0."""
    rng = np.random.default_rng(3)
    k = rng.integers(-120, 121, size=(N, TOKENS)).astype(np.float32)
    k[0, :] = 127.0  # the block maximum, so iscale is exactly -127
    return (k / np.float32(127.0)).astype(np.float32)


def write_capture(path: Path, x, out, weight_type="q4_K", extra=()):
    path.mkdir(parents=True, exist_ok=True)
    x.T.astype(np.float32).tofile(path / "000000-in.f32")
    out.T.astype(np.float32).tofile(path / "000000.f32")
    lines = [{"seq": 0, "eval": 0, "name": "blk.0.attn_q.weight", "op": "MUL_MAT",
              "type": "f32", "ne": [out.shape[0], out.shape[1], 1, 1], "mode": "full",
              "values": out.size, "file": "000000.f32", "matmul": True,
              "weight_type": weight_type, "input_file": "000000-in.f32",
              "input_ne": [x.shape[0], x.shape[1], 1, 1]}, *extra]
    (path / "manifest.jsonl").write_text("".join(json.dumps(v) + "\n" for v in lines))
    return path


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def declaration(self, **over):
        decl = {"schema": "xllama.control-declaration/v1", "model": "test-model",
                "kernel": {"nmse_limit": 1e-9, "min_comparable_fraction": 0.7},
                "spread": {"margin": 1.2, "top_k": 10, "top_k_min": 0.8},
                "prompts": [{"id": "p1", "sha256": sha("prompt p1")}]}
        decl.update(over)
        path = self.root / f"decl-{len(list(self.root.glob('decl-*')))}.json"
        path.write_text(json.dumps(decl))
        return path

    def run_cmp(self, *args):
        return subprocess.run([sys.executable, str(SCRIPT), *map(str, args)],
                              capture_output=True, text=True, check=False)


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


class KernelTests(Base):
    def setUp(self):
        super().setUp()
        self.w = np.random.default_rng(5).standard_normal((8, N)).astype(np.float32)
        self.x = grid_input()
        self.cpu = write_capture(self.root / "cpu", self.x, self.w @ self.x)

    def kernel(self, gpu, decl=None, *more):
        return self.run_cmp("kernel", "--declaration", decl or self.declaration(),
                            self.cpu, gpu, *more)

    def test_quantizer_matches_the_ggml_rule(self):
        mod = load_module()
        x = np.zeros((N, 2), dtype=np.float32)
        x[:4, 0] = [-2.0, 1.0, 0.5 * 2 / 127, 1.5 * 2 / 127]  # max is -2.0: iscale 63.5
        q = mod.quantize_q8_k(x)
        # round-half-even of 0.5 and 1.5, then the all-zero block stays zero
        self.assertEqual(q[0, :4].tolist(), [-127, 64, 0, 2])
        self.assertTrue((q[1] == 0).all())

    def test_identical_arms_pass(self):
        out = self.root / "v.json"
        r = self.kernel(write_capture(self.root / "gpu", self.x, self.w @ self.x),
                        None, "--json-out", out)
        self.assertEqual(r.returncode, 0, r.stdout)
        row = json.loads(out.read_text())["rows"][0]
        self.assertEqual(row["comparable_tokens"], TOKENS)
        self.assertEqual(row["bit_identical_tokens"], TOKENS)

    def test_float_noise_with_equal_int8_is_still_comparable(self):
        x = self.x.copy()
        x[5, 1] = np.nextafter(x[5, 1], np.float32(2.0))  # same int8, different bits
        r = self.kernel(write_capture(self.root / "gpu", x, self.w @ self.x))
        self.assertEqual(r.returncode, 0, r.stdout)

    def test_kernel_error_on_an_equal_operand_fails(self):
        out = (self.w @ self.x).copy()
        out[:, 2] *= 1.01
        r = self.kernel(write_capture(self.root / "gpu", self.x, out))
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("FAIL (kernel)", r.stdout)

    def test_a_column_with_an_int8_flip_is_counted_not_judged(self):
        x = self.x.copy()
        x[7, 3] += np.float32(1.0 / 127.0)  # one int8 step in token 3
        out = self.w @ x
        gpu = write_capture(self.root / "gpu", x, out)
        v = self.root / "v.json"
        r = self.kernel(gpu, None, "--json-out", v)
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertEqual(json.loads(v.read_text())["rows"][0]["comparable_tokens"], TOKENS - 1)
        strict = self.declaration(kernel={"nmse_limit": 1e-9, "min_comparable_fraction": 0.9})
        r = self.kernel(gpu, strict)
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("too few comparable columns", r.stdout)

    def test_other_weight_types_need_bit_identical_operands(self):
        x = self.x.copy()
        x[5, 1] = np.nextafter(x[5, 1], np.float32(2.0))
        cpu = write_capture(self.root / "cpu16", self.x, self.w @ self.x, "f16")
        gpu = write_capture(self.root / "gpu16", x, self.w @ self.x, "f16")
        v = self.root / "v.json"
        r = self.run_cmp("kernel", "--declaration", self.declaration(), cpu, gpu, "--json-out", v)
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertEqual(json.loads(v.read_text())["rows"][0]["comparable_tokens"], TOKENS - 1)

    def test_invalid_inputs_fail_closed(self):
        good = write_capture(self.root / "gpu", self.x, self.w @ self.x)
        skipped = write_capture(self.root / "skip", self.x, self.w @ self.x, extra=[
            {"eval": 0, "name": "blk.0.ffn_up.weight", "op": "MUL_MAT", "type": "q4_K",
             "ne": [8, 4, 1, 1], "skipped": "budget exhausted"}])
        plain = self.root / "plain"
        plain.mkdir()
        (plain / "manifest.jsonl").write_text(json.dumps(
            {"seq": 0, "eval": 0, "name": "l_out-0", "op": "ADD", "type": "f32",
             "ne": [8, 1, 1, 1], "mode": "last", "values": 8, "file": "000000.f32"}) + "\n")
        other = write_capture(self.root / "other", self.x, self.w @ self.x)
        m = other / "manifest.jsonl"
        m.write_text(m.read_text().replace("attn_q", "attn_k"))
        for gpu in (skipped, plain, other, self.root / "missing"):
            self.assertEqual(self.kernel(gpu).returncode, 2, gpu)
        for bad in ({"kernel": {"nmse_limit": 1e-9}}, {"kernel": {"nmse_limit": 0,
                    "min_comparable_fraction": 0.9}}, {"schema": "other"}, {"model": ""}):
            self.assertEqual(self.kernel(good, self.declaration(**bad)).returncode, 2, bad)


class SpreadTests(Base):
    def dumps(self, name, prompts):
        """prompts: {id: noise scale against the shared base logits}."""
        d = self.root / name
        d.mkdir()
        for i, (pid, scale) in enumerate(prompts.items()):
            base = np.random.default_rng(100 + i).standard_normal(512) * 4
            noise = np.random.default_rng(zlib.crc32(name.encode()) + i).standard_normal(512)
            v = (base + scale * noise).astype(np.float32)
            v.tofile(d / f"{pid}.bin")
            (d / f"{pid}.bin.json").write_text(json.dumps(
                {"prompt": f"prompt {pid}", "vocab_size": 512}))
        return d

    def spread(self, gpu, decl=None, host_scale=0.01, prompts=("p1",)):
        console = self.dumps("console", {p: 0.0 for p in prompts})
        host = self.dumps("host", {p: host_scale for p in prompts})
        return self.run_cmp("spread", "--declaration", decl or self.declaration(),
                            "--cpu-console", console, "--cpu-host", host, "--gpu", gpu)

    def test_gpu_inside_the_margin_passes(self):
        # Same noise direction as the host reference, 5 % larger: max error
        # x1.05 and NMSE x1.1025, both inside x1.2.
        gpu = self.root / "gpu"
        gpu.mkdir()
        base = np.random.default_rng(100).standard_normal(512) * 4
        noise = np.random.default_rng(zlib.crc32(b"host")).standard_normal(512)
        (base + 0.0105 * noise).astype(np.float32).tofile(gpu / "p1.bin")
        (gpu / "p1.bin.json").write_text(json.dumps({"prompt": "prompt p1"}))
        r = self.spread(gpu)
        self.assertEqual(r.returncode, 0, r.stdout)

    def test_gpu_wider_than_the_margin_fails(self):
        r = self.spread(self.dumps("gpu", {"p1": 0.05}))
        self.assertEqual(r.returncode, 1, r.stdout)

    def test_top_token_change_fails_even_inside_the_margin(self):
        gpu = self.root / "gpu"
        gpu.mkdir()
        v = (np.random.default_rng(100).standard_normal(512) * 4).astype(np.float32)
        order = np.argsort(v)
        v[order[-1]] = v[order[-2]] + np.float32(1e-3)
        v[order[-2]] += np.float32(2e-3)  # swap the top two by a hair
        v.tofile(gpu / "p1.bin")
        (gpu / "p1.bin.json").write_text(json.dumps({"prompt": "prompt p1"}))
        r = self.spread(gpu, host_scale=5.0)
        self.assertEqual(r.returncode, 1, r.stdout)

    def test_the_spread_is_the_largest_over_the_prompt_set(self):
        decl = self.declaration(prompts=[{"id": "p1", "sha256": sha("prompt p1")},
                                         {"id": "p2", "sha256": sha("prompt p2")}])
        console = self.dumps("console", {"p1": 0.0, "p2": 0.0})
        host = self.root / "host"
        gpu = self.root / "gpu"
        for d, scales in ((host, (0.001, 0.02)), (gpu, (0.02, 0.001))):
            d.mkdir()
            for i, (pid, scale) in enumerate(zip(("p1", "p2"), scales)):
                base = np.random.default_rng(100 + i).standard_normal(512) * 4
                noise = np.random.default_rng(7).standard_normal(512)
                (base + scale * noise).astype(np.float32).tofile(d / f"{pid}.bin")
                (d / f"{pid}.bin.json").write_text(json.dumps({"prompt": f"prompt {pid}"}))
        r = self.run_cmp("spread", "--declaration", decl, "--cpu-console", console,
                         "--cpu-host", host, "--gpu", gpu)
        # GPU p1 is 20x its own reference pair but equals the model spread set by p2.
        self.assertEqual(r.returncode, 0, r.stdout)

    def test_invalid_inputs_fail_closed(self):
        wrong_prompt = self.root / "wrong"
        wrong_prompt.mkdir()
        np.zeros(512, dtype=np.float32).tofile(wrong_prompt / "p1.bin")
        (wrong_prompt / "p1.bin.json").write_text(json.dumps({"prompt": "another prompt"}))
        empty = self.root / "empty"
        empty.mkdir()
        self.assertEqual(self.spread(wrong_prompt).returncode, 2)
        self.assertEqual(self.run_cmp(
            "spread", "--declaration", self.declaration(), "--cpu-console",
            self.root / "console", "--cpu-host", self.root / "host", "--gpu", empty).returncode, 2)
        good = self.dumps("gpu", {"p1": 0.0})
        for bad in ({"spread": {"margin": 0.9, "top_k": 10, "top_k_min": 0.8}},
                    {"spread": {"top_k": 10, "top_k_min": 0.8}},
                    {"prompts": []},
                    {"prompts": [{"id": "../p1", "sha256": sha("prompt p1")}]}):
            r = self.run_cmp("spread", "--declaration", self.declaration(**bad), "--cpu-console",
                             self.root / "console", "--cpu-host", self.root / "host",
                             "--gpu", good)
            self.assertEqual(r.returncode, 2, bad)


class AmendmentTests(Base):
    """Schema v2: the reference arithmetic and the distribution decision."""

    def decl2(self, **over):
        decl = {"schema": "xllama.control-declaration/v2", "model": "test-model",
                "reference": {"nmse_limit": 1e-7,
                              "captures": [{"prompt": "k1", "last_layer": 0}]},
                "distribution": {"kld_ratio_limit": 1.5, "top1_margin": 0.01,
                                 "confidence": 0.95, "min_positions": 60,
                                 "bootstrap": 200, "seed": 1},
                "prompts": [{"id": f"p{i}", "sha256": sha(f"prompt p{i}")} for i in range(6)]}
        decl.update(over)
        path = self.root / f"decl2-{len(list(self.root.glob('decl2-*')))}.json"
        path.write_text(json.dumps(decl))
        return path

    def arm(self, name, noise):
        """Six prompts of 12 positions over a 64-token vocabulary."""
        d = self.root / name
        d.mkdir()
        for i in range(6):
            base = np.random.default_rng(50 + i).standard_normal((12, 64)) * 3
            base[np.arange(12), base.argmax(axis=1)] += 4.0  # a clear top token per position
            extra = np.random.default_rng(zlib.crc32(name.encode()) + i).standard_normal((12, 64))
            (base + noise * extra).astype(np.float32).tofile(d / f"p{i}.bin.positions.bin")
            (d / f"p{i}.bin.positions.json").write_text(json.dumps(
                {"prompt": f"prompt p{i}", "vocab_size": 64, "stride": 1,
                 "n_prompt_tokens": 12, "positions": list(range(12))}))
        return d

    def distribution(self, gpu_noise, decl=None, host_noise=0.05):
        if not (self.root / "console").exists():
            self.arm("console", 0.0)
            self.arm("host", host_noise)
        out = self.root / f"dist-{gpu_noise}.json"
        r = self.run_cmp("distribution", "--declaration", decl or self.decl2(), "--cpu-console",
                         self.root / "console", "--cpu-host", self.root / "host", "--gpu",
                         self.arm(f"gpu{gpu_noise}", gpu_noise), "--json-out", out)
        return r, (json.loads(out.read_text()) if out.exists() else None)

    def test_a_gpu_arm_as_noisy_as_the_host_passes(self):
        r, v = self.distribution(0.05)
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertEqual(v["positions"], 72)
        self.assertLess(abs(v["kld_ratio"] - 1.0), 0.3)

    def test_a_gpu_arm_twice_as_noisy_fails_on_kld(self):
        r, v = self.distribution(0.1)  # KLD grows with the square of the noise
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertGreater(v["kld_ratio"], 3.0)

    def test_too_few_positions_fail_even_when_the_ratio_is_fine(self):
        strict = self.decl2(distribution={"kld_ratio_limit": 1.5, "top1_margin": 0.01,
                                          "confidence": 0.95, "min_positions": 100,
                                          "bootstrap": 200, "seed": 1})
        r, _ = self.distribution(0.05, strict)
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("too few positions", r.stdout)

    def test_distribution_inputs_fail_closed(self):
        self.distribution(0.05)
        gpu = self.root / "gpu0.05"
        side = gpu / "p3.bin.positions.json"
        meta = json.loads(side.read_text())
        side.write_text(json.dumps({**meta, "positions": list(range(1, 13))}))
        base = ["distribution", "--cpu-console", self.root / "console", "--cpu-host",
                self.root / "host", "--gpu", gpu, "--declaration"]
        self.assertEqual(self.run_cmp(*base, self.decl2()).returncode, 2)
        side.write_text(json.dumps({**meta, "prompt": "another prompt"}))
        self.assertEqual(self.run_cmp(*base, self.decl2()).returncode, 2)
        side.write_text(json.dumps(meta))
        self.assertEqual(self.run_cmp(*base, self.declaration()).returncode, 2)  # v1 file
        for bad in ({"kld_ratio_limit": 0.9}, {"confidence": 1.0}, {"bootstrap": 10}):
            d = {"kld_ratio_limit": 1.5, "top1_margin": 0.01, "confidence": 0.95,
                 "min_positions": 60, "bootstrap": 200, "seed": 1, **bad}
            self.assertEqual(self.run_cmp(*base, self.decl2(distribution=d)).returncode, 2, bad)

    def test_reference_rows_judge_every_column_of_one_arm(self):
        mod = load_module()
        tool = mod._load_reference_tool()
        w = np.random.default_rng(5).standard_normal((8, N))
        x = grid_input()
        exact = (w @ tool.dequantized_q8_k(x)).astype(np.float32)
        records = {(0, "blk.0.ffn_down.weight", 0): {"weight_type": "q4_K", "in": x,
                                                      "out": exact}}
        rows = mod.reference_rows(records, lambda name: w, tool)
        self.assertLess(rows[0]["nmse"], 1e-12)
        wrong = exact.copy()
        wrong[:, 1] *= 1.01
        records[(0, "blk.0.ffn_down.weight", 0)]["out"] = wrong
        self.assertGreater(mod.reference_rows(records, lambda name: w, tool)[0]["nmse"], 1e-7)
        records[(0, "blk.0.ffn_down.weight", 0)]["weight_type"] = "f16"
        with self.assertRaises(mod.Invalid):
            mod.reference_rows(records, lambda name: w, tool)

    def test_reference_declaration_is_checked_before_any_model_is_read(self):
        cap = self.root / "k1"
        cap.mkdir()
        for bad in ({"reference": {"nmse_limit": 1e-7, "captures": []}},
                    {"reference": {"captures": [{"prompt": "k1", "last_layer": 0}]}},
                    {"schema": "xllama.control-declaration/v1"}):
            r = self.run_cmp("reference", "--declaration", self.decl2(**bad), "--model",
                             self.root / "missing.gguf", cap)
            self.assertEqual(r.returncode, 2, bad)


if __name__ == "__main__":
    unittest.main()
