#!/usr/bin/env python3
"""Chain repro for #111: decoder-shaped RMSNorm/MatMul stacks, CPU vs DML.

The single-op probes (make-op-repro.py) came back MATCH on the Series S
driver — the (Skip)SimplifiedLayerNormalization kernel is correct in
isolation. This generator raises the fidelity one notch: N transformer-ish
blocks of

    h_norm = SimplifiedLayerNormalization(h)        # pre-norm
    proj   = MatMul(h_norm, W_i)                    # 960x960 fp16
    h, _   = SkipSimplifiedLayerNormalization(h, proj)  # residual + post-norm
              (output 3 = h + proj feeds the next block, like the decoder)

ending with a final SimplifiedLayerNormalization "head". The float64
reference is computed here; the on-device verdict is CPU-vs-DML via
uwp/op-repro.cpp (same repro.onnx/repro-input.bin contract).

Usage:
    make-chain-repro.py [-o build/op-repro-chain] [--layers 8] [--tokens 6]
                        [--hidden 960] [--scale 2.0] [--seed 91]
"""

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

EPS = 9.999999747378752e-06


def rms64(x, gamma):
    ms = (x * x).mean(axis=-1, keepdims=True)
    return x / np.sqrt(ms + EPS) * gamma


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-o", "--out", type=Path, default=Path("build/op-repro-chain"))
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--tokens", type=int, default=6)
    ap.add_argument("--hidden", type=int, default=960)
    ap.add_argument("--scale", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=91)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    H, T = args.hidden, args.tokens
    shape = [1, T, H]

    x = rng.standard_normal(shape).astype(np.float16).astype(np.float64) * args.scale

    nodes, inits = [], []
    h_name = "x"
    h = x.copy()
    fp16 = lambda a: a.astype(np.float16)  # noqa: E731

    for i in range(args.layers):
        gamma_pre = fp16(rng.standard_normal(H) * 0.5 + 1.0)
        gamma_post = fp16(rng.standard_normal(H) * 0.5 + 1.0)
        # Orthogonal-ish small weights keep magnitudes decoder-like across depth.
        w = fp16(rng.standard_normal((H, H)) * (1.0 / np.sqrt(H)))
        inits += [
            numpy_helper.from_array(gamma_pre, f"g_pre_{i}"),
            numpy_helper.from_array(w, f"w_{i}"),
            numpy_helper.from_array(gamma_post, f"g_post_{i}"),
        ]
        nodes += [
            helper.make_node(
                "SimplifiedLayerNormalization",
                [h_name, f"g_pre_{i}"],
                [f"norm_{i}"],
                axis=-1,
                epsilon=EPS,
                stash_type=1,
            ),
            helper.make_node("MatMul", [f"norm_{i}", f"w_{i}"], [f"proj_{i}"]),
            helper.make_node(
                "SkipSimplifiedLayerNormalization",
                [h_name, f"proj_{i}", f"g_post_{i}"],
                [f"post_{i}", "", "", f"h_{i}"],
                domain="com.microsoft",
                epsilon=EPS,
            ),
        ]
        # Reference in float64 with per-step fp16 rounding to track the device.
        g_pre64 = gamma_pre.astype(np.float64)
        w64 = w.astype(np.float64)
        g_post64 = gamma_post.astype(np.float64)
        norm = fp16(rms64(h, g_pre64)).astype(np.float64)
        proj = fp16(norm @ w64).astype(np.float64)
        h = fp16(h + proj).astype(np.float64)
        _post = fp16(rms64(h, g_post64)).astype(
            np.float64
        )  # graph output post_i unused
        h_name = f"h_{i}"

    gamma_final = fp16(rng.standard_normal(H) * 0.5 + 1.0)
    inits.append(numpy_helper.from_array(gamma_final, "g_final"))
    nodes.append(
        helper.make_node(
            "SimplifiedLayerNormalization",
            [h_name, "g_final"],
            ["y"],
            axis=-1,
            epsilon=EPS,
            stash_type=1,
        )
    )
    expected = fp16(rms64(h, gamma_final.astype(np.float64))).astype(np.float32)

    graph = helper.make_graph(
        nodes,
        "oprepro_chain",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT16, shape)],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT16, shape)],
        inits,
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )
    model.ir_version = 8

    d = args.out / "chain"
    d.mkdir(parents=True, exist_ok=True)
    onnx.save(model, d / "repro.onnx")
    x.reshape(-1).astype(np.float32).tofile(d / "repro-input.bin")
    expected.reshape(-1).tofile(d / "repro-expected.bin")
    print(
        f"chain: {args.layers} blocks (2x SimplifiedLN + MatMul + skip residual each), "
        f"h final max|h|={np.abs(h).max():.1f} -> {d}"
    )


if __name__ == "__main__":
    main()
