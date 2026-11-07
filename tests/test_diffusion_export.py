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

import importlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DIFFUSION = ROOT / "diffusion"
sys.path.insert(0, str(DIFFUSION))

# Imported after the sys.path tweak; stdlib-only at import time.
exp = importlib.import_module("export_sd_turbo")
compare_onnx_exports = importlib.import_module("compare_onnx_exports")


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


class OptimumReferenceTest(unittest.TestCase):
    """CONTRACT against the committed optimum-onnx 0.1.0 signature dump."""

    def test_contract_matches_optimum_reference(self):
        ref = json.loads((DIFFUSION / "optimum_reference_signature.json").read_text(encoding="utf-8"))
        self.assertEqual(ref["reference_env"]["optimum-onnx"], "0.1.0")
        codes = {"FLOAT": 1, "INT64": 7, "INT32": 6, "FLOAT16": 10}
        for comp in exp.COMPONENTS:
            r = ref["components"][comp]
            self.assertEqual(r["opset"], exp.DEFAULT_OPSET, comp)
            for kind in ("inputs", "outputs"):
                want = exp.CONTRACT[comp][kind]
                self.assertEqual([v["name"] for v in r[kind]], list(want), (comp, kind))
                for v in r[kind]:
                    elem, dims = want[v["name"]]
                    self.assertEqual(codes[v["elem_type"]], elem, (comp, v["name"]))
                    self.assertEqual(len(v["dims"]), len(dims), (comp, v["name"]))
                    for got, d in zip(v["dims"], dims):
                        if isinstance(d, int) or (isinstance(d, str) and d != "*"):
                            self.assertEqual(got, d, (comp, v["name"]))
                        elif d is None:
                            self.assertIsInstance(got, int, (comp, v["name"]))


class CompareGateTest(unittest.TestCase):
    """compare_onnx_exports.py must not pass when it compared nothing."""

    def test_empty_dirs_fail(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            rep = compare_onnx_exports.compare(a, b)
            self.assertFalse(rep["ok"])
            self.assertEqual(rep["checked"], 0)
            self.assertEqual(sum("missing" in f for f in rep["failures"]), 3)
            self.assertIn("nothing was compared against a recorded tolerance", rep["failures"])
            rc = subprocess.run([sys.executable, str(DIFFUSION / "compare_onnx_exports.py"), a, b],
                                capture_output=True, check=False).returncode
            self.assertEqual(rc, 1)

    def test_component_missing_on_one_side_fails(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            os.makedirs(f"{a}/unet")
            Path(f"{a}/unet/model.onnx").write_bytes(b"")
            rep = compare_onnx_exports.compare(a, b, components=("unet",))
            self.assertFalse(rep["ok"])
            self.assertIn("unet: model.onnx missing in cand", rep["failures"])

    def test_pipeline_needs_all_components(self):
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            rep = compare_onnx_exports.compare(a, b, pipeline=True, components=("unet",))
            self.assertIn("--pipeline needs all three components in both directories",
                          rep["failures"])


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
class Fp16HygieneTest(unittest.TestCase):
    """convert_fp16.py: metadata stripping, external-length check, CLI."""

    @staticmethod
    def _leaky_model():
        from onnx import TensorProto, helper

        def tag(obj):
            obj.doc_string = "/home/someone/xllama/diffusion/export_sd_turbo.py"
            helper.set_metadata_props(obj, {
                "pkg.torch.onnx.stack_trace": 'File "/Users/someone/x.py", line 1',
                "namespace": "C:\\Users\\someone"})
            return obj

        body = helper.make_graph([tag(helper.make_node("Identity", ["a"], ["b"]))], "body",
                                 [helper.make_tensor_value_info("a", TensorProto.FLOAT, [1])],
                                 [helper.make_tensor_value_info("b", TensorProto.FLOAT, [1])])
        tag(body)
        node = tag(helper.make_node("If", ["c"], ["y"], then_branch=body, else_branch=body))
        fn_node = tag(helper.make_node("Identity", ["p"], ["q"]))
        fn = tag(helper.make_function("local", "F", ["p"], ["q"], [fn_node],
                                      [helper.make_opsetid("", 18)]))
        x = tag(helper.make_tensor_value_info("c", TensorProto.BOOL, []))
        g = tag(helper.make_graph([node], "g", [x],
                                  [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])]))
        m = tag(helper.make_model(g, functions=[fn], opset_imports=[
            helper.make_opsetid("", 18), helper.make_opsetid("local", 1)]))
        return m

    def test_strip_metadata_everywhere(self):
        import convert_fp16

        m = self._leaky_model()
        self.assertGreater(len(_metadata_leaks(m)), 10)
        self.assertGreater(convert_fp16.strip_metadata(m), 0)
        self.assertEqual(_metadata_leaks(m), [])
        self.assertEqual(convert_fp16.strip_metadata(m), 0)

    @unittest.skipUnless(_have("onnx_ir"), "onnx_ir not installed")
    def test_export_strip_metadata_everywhere(self):
        import onnx_ir as ir

        model = ir.serde.deserialize_model(self._leaky_model())
        self.assertGreater(exp.strip_metadata(model), 0)
        self.assertEqual(_metadata_leaks(ir.serde.serialize_model(model)), [])

    def test_attach_weights_requires_length(self):
        import convert_fp16
        import numpy as np
        from onnx import TensorProto, helper, numpy_helper

        def model(length):
            t = TensorProto(name="w", data_type=TensorProto.FLOAT, dims=[4],
                            data_location=TensorProto.EXTERNAL)
            entries = {"location": "w.data", "offset": "0"}
            if length is not None:
                entries["length"] = str(length)
            for k, v in entries.items():
                t.external_data.add(key=k, value=v)
            return helper.make_model(helper.make_graph([], "g", [], [], initializer=[t]))

        with tempfile.TemporaryDirectory() as d:
            np.arange(4, dtype=np.float32).tofile(f"{d}/w.data")
            for length in (None, 0, 32):
                with self.assertRaisesRegex(ValueError, "recorded length"):
                    convert_fp16._attach_weights(model(length), d, {"w": TensorProto.FLOAT})
            m = model(16)
            convert_fp16._attach_weights(m, d, {"w": TensorProto.FLOAT})
            self.assertEqual(numpy_helper.to_array(m.graph.initializer[0]).tolist(), [0, 1, 2, 3])
            np.arange(2, dtype=np.float32).tofile(f"{d}/w.data")
            with self.assertRaisesRegex(ValueError, "truncated"):
                convert_fp16._attach_weights(model(16), d, {"w": TensorProto.FLOAT})

    def test_cli_help_and_missing_dir(self):
        cli = [sys.executable, str(DIFFUSION / "convert_fp16.py")]
        r = subprocess.run([*cli, "--help"], capture_output=True, text=True, check=False)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("usage:", r.stdout)
        with tempfile.TemporaryDirectory() as d:
            r = subprocess.run([*cli, f"{d}/missing", f"{d}/out"], capture_output=True,
                               text=True, check=False)
        self.assertEqual(r.returncode, 2)
        self.assertIn("input directory not found", r.stderr)


@unittest.skipUnless(_have("onnx", "onnxruntime"), "onnx/onnxruntime not installed")
class Fp16DuplicateCastTest(unittest.TestCase):
    def test_drop_duplicate_nodes(self):
        import convert_fp16
        from onnx import helper

        cast = lambda: helper.make_node("Cast", ["s"], ["s_cast_to_fp32"], name="s_cast_to_fp32_node", to=1)
        other = helper.make_node("Identity", ["s_cast_to_fp32"], ["y"], name="id")
        g = helper.make_graph([cast(), cast(), other, cast()], "g", [], [])
        self.assertEqual(convert_fp16.drop_duplicate_nodes(g), 2)
        self.assertEqual([n.name for n in g.node], ["s_cast_to_fp32_node", "id"])
        self.assertEqual(convert_fp16.drop_duplicate_nodes(g), 0)


HOME_PATTERNS = re.compile(r"/home/|/Users/|[A-Za-z]:\\|/workspace|~")


def _metadata_leaks(model):
    """Every metadata/doc string of a ModelProto that looks like a path, plus
    names that contain a home-path pattern. Empty list means clean."""
    from onnx import AttributeProto

    bad = []

    def text(where, value, strict=True):
        if value and ((strict and ("/" in value or "\\" in value)) or HOME_PATTERNS.search(value)):
            bad.append(f"{where}: {value[:120]}")

    def meta(where, obj):
        for p in getattr(obj, "metadata_props", ()):
            text(f"{where} metadata key", p.key)
            text(f"{where} metadata[{p.key}]", p.value)
            if p.key.startswith("pkg.torch.onnx"):
                bad.append(f"{where}: {p.key}")
        text(f"{where} doc_string", obj.doc_string)

    def nodes(where, ns):
        for n in ns:
            meta(f"{where}/node {n.name}", n)
            text(f"{where}/node name", n.name, strict=False)
            for a in n.attribute:
                meta(f"{where}/node {n.name} attr {a.name}", a)
                if a.type == AttributeProto.GRAPH:
                    graph(f"{where}/{n.name}.{a.name}", a.g)
                elif a.type == AttributeProto.GRAPHS:
                    for g in a.graphs:
                        graph(f"{where}/{n.name}.{a.name}", g)

    def graph(where, g):
        meta(where, g)
        for v in [*g.input, *g.output, *g.value_info, *g.initializer]:
            meta(f"{where}/value {v.name}", v)
            text(f"{where}/value name", v.name, strict=False)
        nodes(where, g.node)

    meta("model", model)
    for f in ("producer_name", "producer_version", "domain"):
        text(f"model {f}", getattr(model, f), strict=False)
    graph("graph", model.graph)
    for fn in model.functions:
        meta(f"function {fn.name}", fn)
        for v in fn.value_info:
            meta(f"function {fn.name}/value {v.name}", v)
        nodes(f"function {fn.name}", fn.node)
    return bad


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
        import onnx

        self.assertEqual(_metadata_leaks(onnx.load(path, load_external_data=False)), [], comp)
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

    def test_external_data_layout_and_fp16(self):
        """Force the >2 GB layout on the tiny UNet, then run convert_fp16.py on it."""
        import onnxruntime as ort

        root = f"{self.tmp.name}/ext"
        path = f"{root}/unet/model.onnx"
        old = os.environ.get("XLLAMA_EXPORT_EXTERNAL_DATA_BYTES")
        os.environ["XLLAMA_EXPORT_EXTERNAL_DATA_BYTES"] = "1"
        try:
            ext = exp.export_component(self.mods["unet"], "unet",
                                       exp.dummy_inputs("unet", hidden=32, batch=2), path)
        finally:
            if old is None:
                del os.environ["XLLAMA_EXPORT_EXTERNAL_DATA_BYTES"]
            else:
                os.environ["XLLAMA_EXPORT_EXTERNAL_DATA_BYTES"] = old
        self.assertTrue(ext)
        self.assertTrue(os.path.exists(path + ".data"))
        self.assertGreater(os.path.getsize(path + ".data"), os.path.getsize(path))
        self.assertEqual(exp.check_contract(path, "unet"), [])
        import onnx

        self.assertEqual(_metadata_leaks(onnx.load(path, load_external_data=False)), [])
        res = exp.ort_vs_torch(self.mods["unet"], "unet", exp.dummy_inputs("unet", hidden=32), path)
        self.assertLess(res["out_sample"]["max_abs"], 1e-3)

        import convert_fp16

        # The low-memory fp16 path (used for external data) must match ORT's default one.
        models = [convert_fp16.to_fp16(path, low_memory=lm).model for lm in (False, True)]
        for m in models:
            convert_fp16.drop_duplicate_nodes(m.graph)
        self.assertEqual(models[0].SerializeToString(), models[1].SerializeToString())
        del models

        out = f"{self.tmp.name}/ext-fp16"
        r = subprocess.run([sys.executable, str(DIFFUSION / "convert_fp16.py"), root, out],
                           capture_output=True, text=True, check=False)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        self.assertIn("unet: ", r.stdout)
        self.assertIn("LOAD OK (EXTENDED)", r.stdout)
        fp16 = f"{out}/unet/model.onnx"
        self.assertFalse(os.path.exists(fp16 + ".data"), "fp16 graph must be self-contained")
        self.assertEqual(_metadata_leaks(onnx.load(fp16)), [])
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        s = ort.InferenceSession(fp16, sess_options=so, providers=["CPUExecutionProvider"])
        self.assertEqual([i.name for i in s.get_inputs()], list(exp.CONTRACT["unet"]["inputs"]))
        self.assertEqual(s.get_inputs()[0].type, "tensor(float16)")

    def test_fp16_default_path_is_clean(self):
        """Inline (self-contained) exports through convert_fp16.py's default path."""
        import onnx

        root = f"{self.tmp.name}/inline"
        for comp in exp.COMPONENTS:
            ext = exp.export_component(self.mods[comp], comp,
                                       exp.dummy_inputs(comp, hidden=32, batch=2),
                                       f"{root}/{comp}/model.onnx")
            self.assertFalse(ext)
        out = f"{self.tmp.name}/inline-fp16"
        r = subprocess.run([sys.executable, str(DIFFUSION / "convert_fp16.py"), root, out],
                           capture_output=True, text=True, check=False)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
        for comp in exp.COMPONENTS:
            self.assertEqual(_metadata_leaks(onnx.load(f"{out}/{comp}/model.onnx")), [], comp)

    def test_default_threshold_is_protobuf_limit(self):
        old = os.environ.pop("XLLAMA_EXPORT_EXTERNAL_DATA_BYTES", None)
        try:
            self.assertEqual(exp.external_data_threshold(), exp.PROTOBUF_LIMIT)
        finally:
            if old is not None:
                os.environ["XLLAMA_EXPORT_EXTERNAL_DATA_BYTES"] = old

    def test_vae_decoder_upsamples_8x(self):
        import onnxruntime as ort

        path = self._export_and_check("vae_decoder")
        s = ort.InferenceSession(path, providers=["CPUExecutionProvider"])
        out = s.run(None, {"latent_sample": exp.dummy_inputs("vae_decoder", hidden=32)[0].numpy()})[0]
        self.assertEqual(out.shape, (1, 3, 512, 512))


if __name__ == "__main__":
    unittest.main()
