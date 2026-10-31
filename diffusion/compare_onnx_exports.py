#!/usr/bin/env python3
# Copyright (c) 2026 Gianluca Mazza
# SPDX-License-Identifier: MIT
"""Compare two SD-Turbo ONNX export directories (ADR 0003 parity gate).

Checks that every component present in both directories declares the same I/O
signature (names, order, element types, rank, static dims) and opset, then runs
both through ONNX Runtime CPU (ORT_ENABLE_EXTENDED, the console cap) on the fixed
inputs in export_parity.json and checks max_abs / cosine against its recorded
tolerances. With --pipeline it also runs the 1-step text -> UNet -> VAE recipe of
validate_pipeline.py for each prompt and compares the uint8 images.

    python diffusion/compare_onnx_exports.py REF_DIR CAND_DIR \
        [--mode fp32_vs_fp32|fp16_vs_fp32] [--pipeline] [--json report.json]

Exit status 0 when every check passes, 1 otherwise.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PARITY = os.path.join(HERE, "export_parity.json")
COMPONENTS = ("text_encoder", "unet", "vae_decoder")
VAE_SCALE = 0.18215


def signature(path):
    """[(kind, name, elem_type, dims)] with dynamic dims as None."""
    import onnx

    m = onnx.load(path, load_external_data=False)
    sig = []
    for kind, vis in (("in", m.graph.input), ("out", m.graph.output)):
        for v in vis:
            t = v.type.tensor_type
            dims = [d.dim_value if d.HasField("dim_value") else None for d in t.shape.dim]
            sig.append((kind, v.name, t.elem_type, dims))
    opset = next((o.version for o in m.opset_import if o.domain in ("", "ai.onnx")), None)
    return sig, opset


def session(path):
    import onnxruntime as ort

    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
    return ort.InferenceSession(path, sess_options=so, providers=["CPUExecutionProvider"])


_NP = {
    "tensor(float)": "float32",
    "tensor(float16)": "float16",
    "tensor(int64)": "int64",
    "tensor(int32)": "int32",
}


def run(sess, feeds):
    """Feed float32/int64 arrays adapted to the session's declared types and ranks."""
    import numpy as np

    cast = {}
    for i in sess.get_inputs():
        a = np.asarray(feeds[i.name]).astype(_NP[i.type])
        if len(i.shape) == 0:
            a = a.reshape(())
        elif len(i.shape) == 1 and a.ndim == 0:
            a = a.reshape(1)
        cast[i.name] = a
    return {o.name: v.astype(np.float32) for o, v in zip(sess.get_outputs(), sess.run(None, cast))}


def stats(a, b):
    import numpy as np

    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
    return {"max_abs": float(np.abs(a - b).max()), "cosine": cos}


def fixed_inputs(comp, inp, hidden):
    import numpy as np

    rng = np.random.RandomState(inp["seed"])
    b, s, hw = inp["batch"], inp["seq_len"], inp["latent_hw"]
    if comp == "text_encoder":
        return {"input_ids": rng.randint(0, 49408, (b, s)).astype(np.int64)}
    if comp == "unet":
        return {
            "sample": rng.randn(b, 4, hw, hw).astype(np.float32),
            "timestep": np.array(float(inp["timestep"]), dtype=np.float32),
            "encoder_hidden_states": rng.randn(b, s, hidden).astype(np.float32),
        }
    return {"latent_sample": rng.randn(b, 4, hw, hw).astype(np.float32)}


def pipeline_image(root, prompt, inp):
    """validate_pipeline.py's recipe, in-process: returns the uint8 HWC image."""
    import numpy as np
    import torch
    from diffusers import EulerDiscreteScheduler
    from transformers import CLIPTokenizer

    tok = CLIPTokenizer.from_pretrained(os.path.join(root, "tokenizer"))
    ids = tok(prompt, padding="max_length", max_length=77, truncation=True,
              return_tensors="np")["input_ids"]
    hidden = run(session(f"{root}/text_encoder/model.onnx"), {"input_ids": ids})[
        "last_hidden_state"]
    sched = EulerDiscreteScheduler.from_pretrained(os.path.join(root, "scheduler"))
    sched.set_timesteps(inp["steps"])
    rng = np.random.RandomState(inp["seed"])
    hw = inp["latent_hw"]
    latent = rng.randn(1, 4, hw, hw).astype(np.float32) * float(sched.init_noise_sigma)
    unet = session(f"{root}/unet/model.onnx")
    for t in sched.timesteps:
        scaled = sched.scale_model_input(torch.from_numpy(latent), t).numpy()
        eps = run(unet, {"sample": scaled, "timestep": np.array(float(t), np.float32),
                         "encoder_hidden_states": hidden})["out_sample"]
        latent = sched.step(torch.from_numpy(eps), t, torch.from_numpy(latent)).prev_sample.numpy()
    img = run(session(f"{root}/vae_decoder/model.onnx"),
              {"latent_sample": latent / VAE_SCALE})["sample"]
    img = np.clip(img[0] * 0.5 + 0.5, 0, 1)
    return (img.transpose(1, 2, 0) * 255 + 0.5).astype(np.uint8)


def compare(ref, cand, mode="fp32_vs_fp32", pipeline=False, parity=PARITY):
    with open(parity) as f:
        cfg = json.load(f)
    inp, tol = cfg["inputs"], cfg["tolerances"][mode]
    report = {"ref": ref, "cand": cand, "mode": mode, "components": {}, "failures": []}
    fail = report["failures"]

    def check(key, st):
        t = tol.get(key)
        if t is None:
            return
        st["max_abs_tol"], st["min_cosine"] = t["max_abs"], t["min_cosine"]
        if not (st["max_abs"] <= t["max_abs"] and st["cosine"] >= t["min_cosine"]):
            fail.append(f"{key}: {st}")

    hidden = None
    for comp in COMPONENTS:
        rp, cp = f"{ref}/{comp}/model.onnx", f"{cand}/{comp}/model.onnx"
        if not (os.path.exists(rp) and os.path.exists(cp)):
            continue
        rs, ro = signature(rp)
        cs, co = signature(cp)
        rec = {"opset": {"ref": ro, "cand": co}}
        if mode == "fp32_vs_fp32":
            if rs != cs:
                fail.append(f"{comp}: signature differs\n    ref  {rs}\n    cand {cs}")
            if ro != co:
                fail.append(f"{comp}: opset {ro} != {co}")
        else:  # fp16 vs fp32: same names, order and rank; element types may differ
            strip = lambda s: [(k, n, len(d)) for k, n, _, d in s]
            if strip(rs) != strip(cs):
                fail.append(f"{comp}: names/ranks differ\n    ref  {rs}\n    cand {cs}")
        if comp == "text_encoder":
            hidden = next(d[-1] for k, n, _, d in rs if n == "last_hidden_state")
        if comp == "unet" and hidden is None:
            hidden = next(d[-1] for k, n, _, d in rs if n == "encoder_hidden_states")
        feeds = fixed_inputs(comp, inp, hidden)
        r_out, c_out = run(session(rp), feeds), run(session(cp), feeds)
        for name in r_out:
            if name not in c_out:
                continue
            st = stats(r_out[name], c_out[name])
            check(f"{comp}.{name}", st)
            rec[name] = st
        report["components"][comp] = rec

    if pipeline:
        report["images"] = {}
        for p in inp["prompts"]:
            a, b = pipeline_image(ref, p, inp), pipeline_image(cand, p, inp)
            st = stats(a, b)
            st["max_abs"] = int(abs(a.astype(int) - b.astype(int)).max())
            st["ref_std"] = float(a.std())
            check("image_uint8", st)
            report["images"][p or "<empty>"] = st
    report["ok"] = not fail
    return report


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("ref")
    ap.add_argument("cand")
    ap.add_argument("--mode", default="fp32_vs_fp32", choices=("fp32_vs_fp32", "fp16_vs_fp32"))
    ap.add_argument("--pipeline", action="store_true")
    ap.add_argument("--json")
    a = ap.parse_args(argv)
    rep = compare(a.ref, a.cand, a.mode, a.pipeline)
    text = json.dumps(rep, indent=2)
    print(text)
    if a.json:
        with open(a.json, "w") as f:
            f.write(text + "\n")
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
