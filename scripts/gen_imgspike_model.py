#!/usr/bin/env python3
# Copyright (c) 2024 Venere Labs
# SPDX-License-Identifier: MIT
#
# Generates bench/models/imgspike.onnx: a compute-bound conv stack
# (1x3x512x512 fp16, 17 Conv(3x3,64ch)+Relu layers, ~309 GFLOP/forward) used by
# the image DirectML spike (uwp/image-spike.cpp) as a faithful proxy for one
# diffusion UNet step. Deterministic (fixed seed); no torch/onnxscript needed.
#
# Usage:  python scripts/gen_imgspike_model.py [out.onnx]
import sys
import onnx
import numpy as np
from onnx import helper, TensorProto, numpy_helper

H = W = 512
CH = 64
N = 16
DT = TensorProto.FLOAT16
out_path = sys.argv[1] if len(sys.argv) > 1 else "bench/models/imgspike.onnx"

np.random.seed(0)
nodes, inits = [], []


def conv(nin, cin, cout, name, relu=True):
    w = (np.random.randn(cout, cin, 3, 3).astype(np.float16)) * 0.05
    b = np.zeros((cout,), np.float16)
    inits.append(numpy_helper.from_array(w, name + "_w"))
    inits.append(numpy_helper.from_array(b, name + "_b"))
    out = name + "_o"
    nodes.append(
        helper.make_node(
            "Conv",
            [nin, name + "_w", name + "_b"],
            [out],
            kernel_shape=[3, 3],
            pads=[1, 1, 1, 1],
            name=name,
        )
    )
    if not relu:
        return out
    r = name + "_r"
    nodes.append(helper.make_node("Relu", [out], [r], name=name + "_relu"))
    return r


c = conv("input", 3, CH, "c0")
for i in range(1, N):
    c = conv(c, CH, CH, f"c{i}")
# final conv to 3 channels, no relu; output name is "output"
w = (np.random.randn(3, CH, 3, 3).astype(np.float16)) * 0.05
b = np.zeros((3,), np.float16)
inits += [numpy_helper.from_array(w, "cf_w"), numpy_helper.from_array(b, "cf_b")]
nodes.append(
    helper.make_node(
        "Conv",
        [c, "cf_w", "cf_b"],
        ["output"],
        kernel_shape=[3, 3],
        pads=[1, 1, 1, 1],
        name="cf",
    )
)

g = helper.make_graph(
    nodes,
    "convspike",
    [helper.make_tensor_value_info("input", DT, [1, 3, H, W])],
    [helper.make_tensor_value_info("output", DT, [1, 3, H, W])],
    inits,
)
m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
m.ir_version = 10
onnx.checker.check_model(m)
onnx.save(m, out_path)
gflop = (N * CH * CH * 9 * H * W * 2) / 1e9
print(f"saved {out_path}: {len(nodes)} nodes, ~{gflop:.0f} GFLOP/forward")
