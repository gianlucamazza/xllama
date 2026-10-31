#!/usr/bin/env python3
# Copyright (c) 2026 Gianluca Mazza
# SPDX-License-Identifier: MIT
#
# First-party SD-Turbo -> ONNX (fp32) exporter (ADR 0003). Replaces
# `optimum-cli export onnx --task text-to-image`: each txt2img component is
# exported with `torch.onnx.export(..., dynamo=True)` and checked against the
# I/O contract the console pipeline reads (uwp/diffuse.cpp) and the host
# validator (validate_pipeline.py) runs.
#
#   python diffusion/export_sd_turbo.py [--model stabilityai/sd-turbo] \
#       [--revision <sha>] [--out sd-turbo-onnx] [--components unet,...] [--verify]
#
# Output layout matches the optimum export that convert_fp16.py consumes:
#   <out>/{text_encoder,unet,vae_decoder}/model.onnx (+ model.onnx.data when a
#   component exceeds the 2 GB protobuf limit), <out>/tokenizer/,
#   <out>/scheduler/, <out>/model_index.json.
#
# Host-only, offline toolchain fed first-party checkpoints. Not shipped in the
# MSIX. The module imports torch lazily so the contract below can be read (and
# unit-tested) without the export toolchain installed.
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time

DEFAULT_MODEL = "stabilityai/sd-turbo"
# Same opset as the optimum-onnx 0.1.0 export it replaces. DirectML
# loadability of the new graphs is proven only by the console gate (ADR 0003
# step 5).
DEFAULT_OPSET = 18
COMPONENTS = ("text_encoder", "unet", "vae_decoder")

# The console contract (ADR 0003): names, order, element types and axes per
# component, identical to what the optimum-onnx 0.1.0 export it replaces emits
# (read back from a reference export on 2026-10-31, see the #263 PR).
# Element types are ONNX TensorProto values: 1 = FLOAT, 7 = INT64.
# A dim is a str (dynamic, symbolic in the graph), an int (static, exact) or
# None (static, model-dependent: the hidden size is 1024 for SD-Turbo).
CONTRACT = {
    "text_encoder": {
        "inputs": {"input_ids": (7, ["batch_size", "sequence_length"])},
        "outputs": {
            "last_hidden_state": (1, ["batch_size", "sequence_length", None]),
            "pooler_output": (1, ["batch_size", None]),
        },
    },
    "unet": {
        "inputs": {
            "sample": (1, ["batch_size", 4, "height", "width"]),
            # Rank-0 float timestep, as optimum-onnx 0.1.0 emits; diffuse.cpp and
            # validate_pipeline.py read the declared rank and dtype and feed it.
            "timestep": (1, []),
            "encoder_hidden_states": (1, ["batch_size", "sequence_length", None]),
        },
        "outputs": {"out_sample": (1, ["batch_size", 4, "height", "width"])},
    },
    "vae_decoder": {
        "inputs": {"latent_sample": (1, ["batch_size", 4, "latent_height", "latent_width"])},
        # Output spatial dims are exporter expressions (8x the latent dims).
        "outputs": {"sample": (1, ["batch_size", 3, "*", "*"])},
    },
}

PROTOBUF_LIMIT = 2 * 1024**3


def check_contract(path: str, component: str) -> list[str]:
    """Return contract violations of an exported component (empty = OK).

    Checks I/O names, order, element types and rank. A dynamic axis in the
    contract must be symbolic in the graph; a static one must match.
    """
    import onnx

    m = onnx.load(path, load_external_data=False)
    want = CONTRACT[component]
    errs: list[str] = []
    for kind, vis in (("inputs", m.graph.input), ("outputs", m.graph.output)):
        got = {v.name: v for v in vis}
        names = [v.name for v in vis]
        if names != list(want[kind]):
            errs.append(f"{component} {kind}: {names} != {list(want[kind])}")
        for name, (elem, dims) in want[kind].items():
            if name not in got:
                continue
            tt = got[name].type.tensor_type
            if tt.elem_type != elem:
                errs.append(f"{component}.{name}: elem_type {tt.elem_type} != {elem}")
            gdims = list(tt.shape.dim)
            if len(gdims) != len(dims):
                errs.append(f"{component}.{name}: rank {len(gdims)} != {len(dims)}")
                continue
            for i, (g, w) in enumerate(zip(gdims, dims)):
                static = g.HasField("dim_value")
                if w is None:
                    if not static:
                        errs.append(f"{component}.{name}[{i}]: dynamic, want static")
                elif isinstance(w, int):
                    if not (static and g.dim_value == w):
                        errs.append(f"{component}.{name}[{i}]: {g} != {w}")
                elif static:
                    errs.append(f"{component}.{name}[{i}]: static {g.dim_value}, want dynamic")
                elif w != "*" and g.dim_param != w:
                    errs.append(f"{component}.{name}[{i}]: '{g.dim_param}' != '{w}'")
    return errs


def opset_of(path: str) -> int:
    import onnx

    m = onnx.load(path, load_external_data=False)
    return next(o.version for o in m.opset_import if o.domain in ("", "ai.onnx"))


# --------------------------------------------------------------------------
# Wrappers: plain tensor in, plain tensor(s) out, with the contract names.


def _wrappers():
    import torch

    class TextEncoder(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, input_ids):
            out = self.m(input_ids=input_ids, return_dict=True)
            return out.last_hidden_state, out.pooler_output

    class UNet(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, sample, timestep, encoder_hidden_states):
            return self.m(sample, timestep, encoder_hidden_states, return_dict=False)[0]

    class VaeDecoder(torch.nn.Module):
        def __init__(self, vae):
            super().__init__()
            self.vae = vae

        def forward(self, latent_sample):
            return self.vae.decode(latent_sample, return_dict=False)[0]

    return TextEncoder, UNet, VaeDecoder


def dummy_inputs(component: str, *, hidden: int, batch: int = 1, seq: int = 77,
                 latent_hw: int = 64):
    """Fixed, seeded example inputs (default: the console shapes) for one component.

    Export traces with batch 2: with a batch-1 example torch.export specialises
    the batch axis to 1 inside the conv shape rules (0/1 specialisation).
    """
    import torch

    g = torch.Generator().manual_seed(0)
    if component == "text_encoder":
        return (torch.randint(0, 49408, (batch, seq), generator=g, dtype=torch.int64),)
    if component == "unet":
        return (
            torch.randn(batch, 4, latent_hw, latent_hw, generator=g),
            torch.tensor(999.0, dtype=torch.float32),
            torch.randn(batch, seq, hidden, generator=g),
        )
    if component == "vae_decoder":
        return (torch.randn(batch, 4, latent_hw, latent_hw, generator=g),)
    raise KeyError(component)


def _dynamic_shapes(component: str):
    from torch.export import Dim

    # Bounds cover the console shapes (batch 1, 64x64 latents) with headroom.
    batch = Dim("batch_size", min=1, max=8)
    if component == "text_encoder":
        return {"input_ids": {0: batch, 1: Dim("sequence_length", min=2, max=77)}}
    if component == "unet":
        # Spatial dims must stay divisible by the UNet's 2**(levels-1)
        # down-sampling, so they are declared as multiples of 8.
        h, w = Dim("h8", min=2, max=32), Dim("w8", min=2, max=32)
        return {
            "sample": {0: batch, 2: 8 * h, 3: 8 * w},
            "timestep": {},
            "encoder_hidden_states": {0: batch, 1: Dim("sequence_length", min=2, max=77)},
        }
    if component == "vae_decoder":
        return {
            "latent_sample": {
                0: batch,
                2: Dim("latent_height", min=8, max=128),
                3: Dim("latent_width", min=8, max=128),
            }
        }
    raise KeyError(component)


_SYMBOL_NAMES = {
    "text_encoder": {
        "inputs": {"input_ids": ["batch_size", "sequence_length"]},
        "outputs": {
            "last_hidden_state": ["batch_size", "sequence_length", None],
            "pooler_output": ["batch_size", None],
        },
    },
    "unet": {
        "inputs": {
            "sample": ["batch_size", None, "height", "width"],
            "encoder_hidden_states": ["batch_size", "sequence_length", None],
        },
        "outputs": {"out_sample": ["batch_size", None, "height", "width"]},
    },
    "vae_decoder": {
        "inputs": {"latent_sample": ["batch_size", None, "latent_height", "latent_width"]},
        "outputs": {"sample": ["batch_size", None, None, None]},
    },
}


def _rename_symbols(model, component: str) -> None:
    """Give the graph's dynamic axes the optimum-style symbol names.

    Cosmetic for ORT (which binds by name and rank), but keeps the exported I/O
    readable and diff-able against the old export.
    """
    names = _SYMBOL_NAMES[component]
    for kind, vis in (("inputs", model.graph.input), ("outputs", model.graph.output)):
        for v in vis:
            want = names[kind].get(v.name)
            if not want:
                continue
            for d, n in zip(v.type.tensor_type.shape.dim, want):
                if n and not d.HasField("dim_value"):
                    d.dim_param = n


def export_component(module, component: str, args, path: str, *, opset: int = DEFAULT_OPSET):
    """Export one wrapped component with the dynamo exporter and save it.

    Components above the protobuf limit are written with one external data
    file next to the graph (`model.onnx.data`); convert_fp16.py and
    scripts/merge_onnx_external_data.py both read that layout.
    """
    import onnx
    import torch

    want = CONTRACT[component]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with torch.no_grad():
        prog = torch.onnx.export(
            module,
            args,
            dynamo=True,
            input_names=list(want["inputs"]),
            output_names=list(want["outputs"]),
            dynamic_shapes=_dynamic_shapes(component),
            opset_version=opset,
            optimize=True,
            verify=False,
            external_data=True,
        )
    model = prog.model_proto
    _rename_symbols(model, component)
    nbytes = model.ByteSize()
    data = os.path.basename(path) + ".data"
    for f in (path, path + ".data"):
        if os.path.exists(f):
            os.remove(f)
    if nbytes >= PROTOBUF_LIMIT:
        onnx.save_model(model, path, save_as_external_data=True, all_tensors_to_one_file=True,
                        location=data, size_threshold=1024)
    else:
        onnx.save_model(model, path)
    errs = check_contract(path, component)
    if errs:
        raise RuntimeError("contract violation:\n  " + "\n  ".join(errs))
    return path


def ort_vs_torch(module, component: str, args, path: str) -> dict:
    """Run the exported graph on ORT CPU (EXTENDED cap) against eager torch."""
    import numpy as np
    import onnxruntime as ort
    import torch

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    s = ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])
    feeds = {n: a.numpy() for n, a in zip(CONTRACT[component]["inputs"], args)}
    got = s.run(None, feeds)
    with torch.no_grad():
        ref = module(*args)
    ref = ref if isinstance(ref, tuple) else (ref,)
    out = {}
    for name, g, r in zip(CONTRACT[component]["outputs"], got, ref):
        r = r.numpy().astype(np.float64).ravel()
        g = g.astype(np.float64).ravel()
        cos = float(np.dot(r, g) / (np.linalg.norm(r) * np.linalg.norm(g) + 1e-30))
        out[name] = {"max_abs": float(np.abs(r - g).max()), "cosine": cos}
    return out


def load_components(model: str, revision: str | None, which):
    """Load the requested fp32 torch components from a diffusers checkpoint."""
    import torch
    from diffusers import AutoencoderKL, UNet2DConditionModel
    from transformers import CLIPTextModel

    TextEncoder, UNet, VaeDecoder = _wrappers()
    kw = {"revision": revision, "torch_dtype": torch.float32}
    if "text_encoder" in which:
        te = CLIPTextModel.from_pretrained(model, subfolder="text_encoder", **kw).eval()
        yield "text_encoder", TextEncoder(te), te.config.hidden_size
        del te
    if "unet" in which:
        unet = UNet2DConditionModel.from_pretrained(model, subfolder="unet", **kw).eval()
        yield "unet", UNet(unet), unet.config.cross_attention_dim
        del unet
    if "vae_decoder" in which:
        vae = AutoencoderKL.from_pretrained(model, subfolder="vae", **kw).eval()
        yield "vae_decoder", VaeDecoder(vae), None
        del vae


def copy_pipeline_files(model: str, revision: str | None, out: str) -> None:
    """Write tokenizer/, scheduler/ and model_index.json (validate_pipeline.py input)."""
    from diffusers import EulerDiscreteScheduler
    from huggingface_hub import hf_hub_download
    from transformers import CLIPTokenizer

    CLIPTokenizer.from_pretrained(model, subfolder="tokenizer", revision=revision).save_pretrained(
        os.path.join(out, "tokenizer")
    )
    EulerDiscreteScheduler.from_pretrained(
        model, subfolder="scheduler", revision=revision
    ).save_pretrained(os.path.join(out, "scheduler"))
    if os.path.isdir(model):
        src = os.path.join(model, "model_index.json")
    else:
        src = hf_hub_download(model, "model_index.json", revision=revision)
    shutil.copy2(src, os.path.join(out, "model_index.json"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0] if __doc__ else None)
    ap.add_argument("--model", default=DEFAULT_MODEL, help="hub id or local diffusers dir")
    ap.add_argument("--revision", default=None, help="hub commit to pin")
    ap.add_argument("--out", default="sd-turbo-onnx")
    ap.add_argument("--components", default=",".join(COMPONENTS))
    ap.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    ap.add_argument("--verify", action="store_true",
                    help="compare each graph on ORT CPU against eager torch")
    a = ap.parse_args(argv)
    which = [c.strip() for c in a.components.split(",") if c.strip()]
    bad = [c for c in which if c not in COMPONENTS]
    if bad:
        ap.error(f"unknown components {bad}; choose from {COMPONENTS}")

    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.makedirs(a.out, exist_ok=True)
    copy_pipeline_files(a.model, a.revision, a.out)
    report = {"model": a.model, "revision": a.revision, "opset": a.opset, "components": {}}
    hidden = None
    for comp, module, h in load_components(a.model, a.revision, which):
        if comp == "text_encoder" or comp == "unet":
            hidden = h
        if hidden is None:
            hidden = 1024  # SD2-class default; only used for the UNet dummy
        trace = dummy_inputs(comp, hidden=hidden, batch=2)
        path = os.path.join(a.out, comp, "model.onnx")
        t0 = time.time()
        print(f"[export] {comp} (dynamo, opset {a.opset}) ...", flush=True)
        export_component(module, comp, trace, path, opset=a.opset)
        del trace
        size = os.path.getsize(path)
        if os.path.exists(path + ".data"):
            size += os.path.getsize(path + ".data")
        rec = {"seconds": round(time.time() - t0, 1), "bytes": size, "opset": opset_of(path)}
        print(f"[export] {comp}: {size / 1e6:.1f} MB in {rec['seconds']} s, contract OK", flush=True)
        if a.verify:
            # Console shapes (batch 1, 77 tokens, 64x64 latents) plus a second,
            # smaller shape to prove the dynamic axes are really dynamic.
            rec["ort_vs_torch"] = ort_vs_torch(
                module, comp, dummy_inputs(comp, hidden=hidden), path)
            rec["ort_vs_torch_dynamic"] = ort_vs_torch(
                module, comp, dummy_inputs(comp, hidden=hidden, batch=2, seq=16, latent_hw=32),
                path)
            print(f"[verify] {comp}: {rec['ort_vs_torch']}", flush=True)
            print(f"[verify] {comp} (batch 2, smaller): {rec['ort_vs_torch_dynamic']}",
                  flush=True)
        report["components"][comp] = rec
        del module
    with open(os.path.join(a.out, "export_report.json"), "w") as f:
        json.dump(report, f, indent=2)
    print(f"[export] done -> {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
