#!/usr/bin/env python3
"""First-party SD-Turbo ONNX export (ADR 0003, #263).

The stdlib tests pin diffusion/export_sd_turbo.py's I/O contract to the names
uwp/diffuse.cpp and diffusion/validate_pipeline.py read, the opset, the parity
tolerances and the optimum-free requirement set. They run in CI without the
export toolchain.

The onnx-gated tests exercise check_contract and the fp16 duplicate-node fix on
synthetic graphs. The torch-gated test exports tiny random-weight CLIP, UNet and
VAE models through the same export_component code path and checks the contract
and ORT-vs-torch parity. Install diffusion/requirements.txt to run them.
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIFFUSION = ROOT / "diffusion"
sys.path.insert(0, str(DIFFUSION))

import export_sd_turbo as exp  # noqa: E402  (stdlib-only at import time)


def _have(*mods):
    return all(importlib.util.find_spec(m) is not None for m in mods)


def _cpp_names(src: str, var: str) -> list[str]:
    m = re.search(r"const char\*\s+" + var + r"\[\]\s*=\s*\{([^}]*)\}", src)
    if not m:
        raise AssertionError(f"{var} not found in uwp/diffuse.cpp")
    return re.findall(r'"([^"]+)"', m.group(1))


class ContractTest(unittest.TestCase):
    def test_names_match_console_pipeline(self):
        src = (ROOT / "uwp/diffuse.cpp").read_text(encoding="utf-8")
        c = exp.CONTRACT
        self.assertEqual(_cpp_names(src, "te_in_names"), list(c["text_encoder"]["inputs"]))
        self.assertIn(_cpp_names(src, "te_out_names")[0], c["text_encoder"]["outputs"])
        self.assertEqual(_cpp_names(src, "u_in_names"), list(c["unet"]["inputs"]))
        self.assertEqual(_cpp_names(src, "u_out_names"), list(c["unet"]["outputs"]))
        self.assertEqual(_cpp_names(src, "v_in_names"), list(c["vae_decoder"]["inputs"]))
        self.assertEqual(_cpp_names(src, "v_out_names"), list(c["vae_decoder"]["outputs"]))
        self.assertIn("ORT_ENABLE_EXTENDED", src)

    def test_names_match_host_validator(self):
        src = (DIFFUSION / "validate_pipeline.py").read_text(encoding="utf-8")
        for comp, io in exp.CONTRACT.items():
            self.assertIn(f'"{comp}"', src)
            for name in list(io["inputs"]) + [n for n in io["outputs"] if n != "pooler_output"]:
                self.assertIn(f'"{name}"', src, f"{comp}.{name}")

    def test_dtypes_and_ranks(self):
        c = exp.CONTRACT
        # ONNX TensorProto: 1 = FLOAT, 7 = INT64. diffuse.cpp feeds int64 ids
        # when declared and reads the timestep rank ([] here).
        self.assertEqual(c["text_encoder"]["inputs"]["input_ids"], (7, ["batch_size", "sequence_length"]))
        self.assertEqual(c["unet"]["inputs"]["timestep"], (1, []))
        self.assertEqual(c["unet"]["inputs"]["sample"][1], ["batch_size", 4, "height", "width"])
        self.assertEqual(c["vae_decoder"]["outputs"]["sample"][1][:2], ["batch_size", 3])
        for comp in exp.COMPONENTS:
            for name, (elem, dims) in {**c[comp]["inputs"], **c[comp]["outputs"]}.items():
                if name != "input_ids":
                    self.assertEqual(elem, 1, f"{comp}.{name} must be fp32 in the export")

    def test_opset_and_components(self):
        self.assertEqual(exp.DEFAULT_OPSET, 18)
        self.assertEqual(exp.COMPONENTS, ("text_encoder", "unet", "vae_decoder"))
        self.assertEqual(set(exp.CONTRACT), set(exp.COMPONENTS))

    def test_parity_tolerances_cover_contract_outputs(self):
        cfg = json.loads((DIFFUSION / "export_parity.json").read_text(encoding="utf-8"))
        fp32 = cfg["tolerances"]["fp32_vs_fp32"]
        for comp in exp.COMPONENTS:
            for name in exp.CONTRACT[comp]["outputs"]:
                self.assertIn(f"{comp}.{name}", fp32)
        self.assertIn("image_uint8", fp32)
        for t in fp32.values():
            self.assertGreater(t["min_cosine"], 0.999)
        self.assertEqual(cfg["inputs"]["latent_hw"], 64)
        self.assertEqual(cfg["inputs"]["seq_len"], 77)


class RequirementsTest(unittest.TestCase):
    def _pins(self):
        pins = {}
        for line in (DIFFUSION / "requirements.txt").read_text(encoding="utf-8").splitlines():
            line = line.split("#", 1)[0].strip()
            if line:
                m = re.match(r"([A-Za-z0-9_.\-\[\]]+)\s*(==|>=)\s*([0-9][^\s]*)", line)
                self.assertIsNotNone(m, line)
                pins[m.group(1).lower()] = (m.group(2), m.group(3))
        return pins

    def test_optimum_removed(self):
        pins = self._pins()
        self.assertFalse([p for p in pins if p.startswith("optimum")])
        for f in ("export_sd_turbo.sh", "generate_onnx.py", "validate_pipeline.py",
                  "convert_fp16.py", "export_sd_turbo.py", "compare_onnx_exports.py"):
            code = "\n".join(l for l in (DIFFUSION / f).read_text(encoding="utf-8").splitlines()
                             if not l.lstrip().startswith("#"))
            self.assertNotRegex(code, r"optimum[-_.]?(cli|onnxruntime|\.exporters)|from optimum|import optimum", f)

    def test_patched_versions(self):
        pins = self._pins()
        ver = lambda v: tuple(int(x) for x in re.findall(r"\d+", v)[:3])
        self.assertEqual(pins["torch"][0], "==")
        self.assertGreaterEqual(ver(pins["torch"][1]), (2, 13, 0))  # GHSA-rrmf-rvhw-rf47
        self.assertGreaterEqual(ver(pins["transformers"][1]), (5, 10, 1))  # GHSA-xrqw-3rrv-vx5w; 5.10.0 withdrawn
        self.assertGreaterEqual(ver(pins["diffusers"][1]), (0, 40, 0))
        self.assertGreaterEqual(ver(pins["onnx"][1]), (1, 23, 0))
        for p in ("onnxscript", "onnxruntime"):
            self.assertIn(p, pins)

    def test_dependabot_no_longer_ignores_export_stack(self):
        text = (ROOT / ".github/dependabot.yml").read_text(encoding="utf-8")
        self.assertNotIn("dependency-name: optimum", text)
        block = text.split("directory: /diffusion", 1)[1].split("- package-ecosystem", 1)[0]
        self.assertNotIn("dependency-name: torch", block)
        self.assertNotIn("dependency-name: transformers", block)


@unittest.skipUnless(_have("onnx"), "onnx not installed (diffusion/requirements.txt)")
class CheckContractTest(unittest.TestCase):
    def _graph(self, path, ids_type=7, ts_dims=(), hidden_dynamic=False):
        import onnx
        from onnx import TensorProto, helper

        hid = "hidden" if hidden_dynamic else 32
        ids = helper.make_tensor_value_info("input_ids", ids_type, ["batch_size", "sequence_length"])
        lhs = helper.make_tensor_value_info("last_hidden_state", TensorProto.FLOAT,
                                            ["batch_size", "sequence_length", hid])
        pool = helper.make_tensor_value_info("pooler_output", TensorProto.FLOAT, ["batch_size", hid])
        g = helper.make_graph([], "te", [ids], [lhs, pool])
        onnx.save(helper.make_model(g, opset_imports=[helper.make_opsetid("", 18)]), path)

    def test_accepts_contract_graph(self):
        with tempfile.TemporaryDirectory() as d:
            p = f"{d}/m.onnx"
            self._graph(p)
            self.assertEqual(exp.check_contract(p, "text_encoder"), [])
            self.assertEqual(exp.opset_of(p), 18)

    def test_rejects_wrong_dtype_and_dynamic_hidden(self):
        with tempfile.TemporaryDirectory() as d:
            p = f"{d}/m.onnx"
            self._graph(p, ids_type=6, hidden_dynamic=True)
            errs = exp.check_contract(p, "text_encoder")
            self.assertTrue(any("elem_type 6" in e for e in errs), errs)
            self.assertTrue(any("want static" in e for e in errs), errs)


@unittest.skipUnless(_have("onnx", "onnxruntime"), "onnx/onnxruntime not installed")
class Fp16DuplicateCastTest(unittest.TestCase):
    def test_drop_duplicate_nodes(self):
        from onnx import helper

        import convert_fp16

        cast = lambda: helper.make_node("Cast", ["s"], ["s_cast_to_fp32"], name="s_cast_to_fp32_node", to=1)
        other = helper.make_node("Identity", ["s_cast_to_fp32"], ["y"], name="id")
        g = helper.make_graph([cast(), cast(), other, cast()], "g", [], [])
        self.assertEqual(convert_fp16.drop_duplicate_nodes(g), 2)
        self.assertEqual([n.name for n in g.node], ["s_cast_to_fp32_node", "id"])
        self.assertEqual(convert_fp16.drop_duplicate_nodes(g), 0)


@unittest.skipUnless(_have("torch", "diffusers", "transformers", "onnx", "onnxscript", "onnxruntime"),
                     "export toolchain not installed (diffusion/requirements.txt)")
class TinyDynamoExportTest(unittest.TestCase):
    """Random-weight tiny SD2-like components through export_component."""

    @classmethod
    def setUpClass(cls):
        import torch
        from diffusers import AutoencoderKL, UNet2DConditionModel
        from transformers import CLIPTextConfig, CLIPTextModel

        torch.manual_seed(0)
        TextEncoder, UNet, VaeDecoder = exp._wrappers()
        te = CLIPTextModel(CLIPTextConfig(
            vocab_size=49408, hidden_size=32, intermediate_size=37, num_hidden_layers=2,
            num_attention_heads=4, max_position_embeddings=77, projection_dim=32)).eval()
        unet = UNet2DConditionModel(
            sample_size=64, in_channels=4, out_channels=4, layers_per_block=1,
            block_out_channels=(32, 64), down_block_types=("DownBlock2D", "CrossAttnDownBlock2D"),
            up_block_types=("CrossAttnUpBlock2D", "UpBlock2D"), cross_attention_dim=32,
            attention_head_dim=(2, 4), use_linear_projection=True, norm_num_groups=8).eval()
        vae = AutoencoderKL(
            in_channels=3, out_channels=3, block_out_channels=(32, 32, 32, 32), latent_channels=4,
            down_block_types=("DownEncoderBlock2D",) * 4, up_block_types=("UpDecoderBlock2D",) * 4,
            norm_num_groups=8, layers_per_block=1).eval()
        cls.mods = {"text_encoder": TextEncoder(te), "unet": UNet(unet), "vae_decoder": VaeDecoder(vae)}
        cls.tmp = tempfile.TemporaryDirectory()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def _export_and_check(self, comp):
        mod = self.mods[comp]
        path = f"{self.tmp.name}/{comp}/model.onnx"
        exp.export_component(mod, comp, exp.dummy_inputs(comp, hidden=32, batch=2), path)
        self.assertEqual(exp.check_contract(path, comp), [])
        self.assertEqual(exp.opset_of(path), exp.DEFAULT_OPSET)
        for kw in ({}, {"batch": 2, "seq": 16, "latent_hw": 32}):
            res = exp.ort_vs_torch(mod, comp, exp.dummy_inputs(comp, hidden=32, **kw), path)
            for name, st in res.items():
                self.assertLess(st["max_abs"], 1e-3, (comp, name, kw, st))
                self.assertGreater(st["cosine"], 0.99999, (comp, name, kw, st))
        return path

    def test_text_encoder(self):
        self._export_and_check("text_encoder")

    def test_unet(self):
        import onnx

        path = self._export_and_check("unet")
        ts = next(i for i in onnx.load(path).graph.input if i.name == "timestep")
        self.assertEqual(len(ts.type.tensor_type.shape.dim), 0)

    def test_vae_decoder_upsamples_8x(self):
        import onnxruntime as ort

        path = self._export_and_check("vae_decoder")
        s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        out = s.run(None, {"latent_sample": exp.dummy_inputs("vae_decoder", hidden=32)[0].numpy()})[0]
        self.assertEqual(out.shape, (1, 3, 512, 512))


if __name__ == "__main__":
    unittest.main()
