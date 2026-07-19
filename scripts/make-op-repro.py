#!/usr/bin/env python3
"""Generate single-op ONNX repro models for the #111 DML kernel investigation.

The #91 root cause is the DML lowering of (Skip)SimplifiedLayerNormalization:
both funnel into DML_OPERATOR_MEAN_VARIANCE_NORMALIZATION2 with UseMean=false
(onnxruntime/core/providers/dml/.../DmlOperatorLayerNormalization.cpp), the
"RMSNorm" MVN2 configuration. This script emits minimal models that exercise
exactly that path — plus a non-simplified LayerNormalization control that uses
the same MVN2 op with UseMean=true (expected correct):

    simplified  -> one SimplifiedLayerNormalization node (default domain)
    skip        -> one com.microsoft.SkipSimplifiedLayerNormalization node
    layernorm   -> one LayerNormalization node (control)

Each variant writes <out>/<name>/repro.onnx plus repro-input.bin (fp32 payload,
inputs concatenated in model input order — the on-device runner contract, see
uwp/op-repro.cpp) and repro-expected.bin (fp32 reference computed here in
float64 then rounded, for host-side sanity only; the on-device verdict compares
the device's own CPU vs DML runs).

Usage:
    make-op-repro.py [-o build/op-repro] [--hidden 960] [--tokens 8] [--seed 91]
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

EPS = 9.999999747378752e-06  # the decoder's epsilon, bit-exact


def rms_ref(x, gamma):
    x64 = x.astype(np.float64)
    ms = (x64 * x64).mean(axis=-1, keepdims=True)
    return (x64 / np.sqrt(ms + EPS) * gamma.astype(np.float64)).astype(np.float32)


def ln_ref(x, gamma, beta):
    x64 = x.astype(np.float64)
    mu = x64.mean(axis=-1, keepdims=True)
    var = ((x64 - mu) ** 2).mean(axis=-1, keepdims=True)
    return (
        (x64 - mu) / np.sqrt(var + EPS) * gamma.astype(np.float64)
        + beta.astype(np.float64)
    ).astype(np.float32)


def build(variant, hidden, tokens, rng, out_dir):
    shape = [1, tokens, hidden]
    gamma = (
        rng.standard_normal(hidden).astype(np.float16).astype(np.float32) * 0.5 + 1.0
    )
    x = (rng.standard_normal(shape).astype(np.float16).astype(np.float32)) * 2.0

    inputs = [helper.make_tensor_value_info("x", TensorProto.FLOAT16, shape)]
    inits = [numpy_helper.from_array(gamma.astype(np.float16), "gamma")]
    payload = [x]

    if variant == "simplified":
        node = helper.make_node(
            "SimplifiedLayerNormalization",
            ["x", "gamma"],
            ["y"],
            axis=-1,
            epsilon=EPS,
            stash_type=1,
        )
        expected = rms_ref(x, gamma)
        outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT16, shape)]
    elif variant == "skip":
        skip = (rng.standard_normal(shape).astype(np.float16).astype(np.float32)) * 2.0
        payload.append(skip)
        inputs.append(helper.make_tensor_value_info("skip", TensorProto.FLOAT16, shape))
        node = helper.make_node(
            "SkipSimplifiedLayerNormalization",
            ["x", "skip", "gamma"],
            ["y", "", "", "sum"],
            domain="com.microsoft",
            epsilon=EPS,
        )
        expected = rms_ref(x + skip, gamma)
        outputs = [
            helper.make_tensor_value_info("y", TensorProto.FLOAT16, shape),
            helper.make_tensor_value_info("sum", TensorProto.FLOAT16, shape),
        ]
    elif variant == "layernorm":
        beta = rng.standard_normal(hidden).astype(np.float16).astype(np.float32) * 0.1
        inits.append(numpy_helper.from_array(beta.astype(np.float16), "beta"))
        node = helper.make_node(
            "LayerNormalization", ["x", "gamma", "beta"], ["y"], axis=-1, epsilon=EPS
        )
        expected = ln_ref(x, gamma, beta)
        outputs = [helper.make_tensor_value_info("y", TensorProto.FLOAT16, shape)]
    else:
        raise ValueError(variant)

    graph = helper.make_graph([node], f"oprepro_{variant}", inputs, outputs, inits)
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )
    model.ir_version = 8

    d = out_dir / variant
    d.mkdir(parents=True, exist_ok=True)
    onnx.save(model, d / "repro.onnx")
    np.concatenate([p.reshape(-1) for p in payload]).astype(np.float32).tofile(
        d / "repro-input.bin"
    )
    expected.reshape(-1).astype(np.float32).tofile(d / "repro-expected.bin")
    print(
        f"{variant}: repro.onnx + input ({sum(p.size for p in payload)} f32) "
        f"+ expected ({expected.size} f32) -> {d}"
    )


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--out", type=Path, default=Path("build/op-repro"))
    ap.add_argument("--hidden", type=int, default=960)
    ap.add_argument("--tokens", type=int, default=8)
    ap.add_argument("--seed", type=int, default=91)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    for variant in ("simplified", "skip", "layernorm"):
        build(variant, args.hidden, args.tokens, rng, args.out)


if __name__ == "__main__":
    main()
