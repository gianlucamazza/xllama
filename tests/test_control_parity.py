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


if __name__ == "__main__":
    unittest.main()
