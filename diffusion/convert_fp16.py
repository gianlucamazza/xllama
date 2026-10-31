#!/usr/bin/env python3
# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
#
# Convert the fp32 SD-Turbo ONNX components to fp16 for the Xbox Series S GPU
# budget (3801 MB), producing the deployable console artifacts: each component
# self-contained < 2 GB (ONNX protobuf single-file limit + the merged-model
# requirement that avoids the AppContainer `weakly_canonical` crash,
# docs/uwp-constraints.md §8).
#
# Converter choice (verified 2026-07-08): onnxruntime.transformers'
# OnnxModel.convert_float_to_float16 — the ORT team's corrected fp16 pass.
# onnxconverter_common leaves mixed-type nodes in ALL three SD components
# (regardless of keep_io_types / shape-infer / node_block_list) and its output is
# rejected by ORT at load. See diffusion/README.md "Getting a runnable fp16 model".
#
# Load caveat baked into the pipeline: sessions must cap graph optimization at
# ORT_ENABLE_EXTENDED (ORT_ENABLE_ALL crashes on these graphs) — both
# uwp/diffuse.cpp and validate_pipeline.py do. This script load-tests each
# converted component at EXTENDED before declaring success.
#
# Usage:  python diffusion/convert_fp16.py <onnx_dir_in> <onnx_dir_out>
import os
import shutil
import sys

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, ValueInfoProto
from onnx.external_data_helper import ExternalDataInfo, uses_external_data
from onnxruntime.transformers import float16
from onnxruntime.transformers.onnx_model import OnnxModel

COMPONENTS = ["text_encoder", "unet", "vae_decoder"]
PROTOBUF_LIMIT = 2 * 1024**3


def drop_duplicate_nodes(graph):
    """Remove byte-identical duplicate nodes; returns how many were dropped.

    The dynamo exporter shares one initializer between several consumers (e.g.
    the [1, 1, 2, 2] scales of the three VAE-decoder Resize nodes). The fp16
    pass keeps Resize scales fp32 and inserts one `<init>_cast_to_fp32` Cast
    per consumer, all with the same name and output, which ORT rejects ("two
    nodes with same node name"). Identical nodes compute the same value, so
    keeping the first is exact.
    """
    seen, keep = set(), []
    for node in graph.node:
        key = node.SerializeToString()
        if key in seen:
            continue
        seen.add(key)
        keep.append(node)
    dropped = len(graph.node) - len(keep)
    if dropped:
        del graph.node[:]
        graph.node.extend(keep)
    return dropped


def _merge_value_types(model, inferred):
    """Copy element types (not shapes) from `inferred` into `model`.

    Same merge OnnxModel.convert_float_to_float16 does after symbolic shape
    inference: the fp16 pass needs the dtype of every intermediate value, and
    auto_merge shapes can be wrong (onnxruntime issue 15521), so shapes are dropped.
    """
    have = {vi.name for vi in model.graph.value_info}
    for vi in inferred.graph.value_info:
        tt = vi.type.tensor_type
        if not vi.name or vi.name in have or tt.elem_type == TensorProto.UNDEFINED:
            continue
        vi_copy = ValueInfoProto()
        vi_copy.CopyFrom(vi)
        vi_copy.type.tensor_type.ClearField("shape")
        model.graph.value_info.append(vi_copy)


def _value_info_from_dims(tensor):
    """float16.make_value_info_from_tensor without reading the tensor's data.

    The original takes the shape from numpy_helper.to_array(tensor), which is
    always tensor.dims; reading it would pull external weights back in.
    """
    return onnx.helper.make_tensor_value_info(tensor.name, tensor.data_type, list(tensor.dims))


def _attach_weights(model, base, dtypes):
    """Inline each external initializer, converting it to fp16 if the pass did.

    `dtypes` holds the element types before the fp16 pass; the pass only flips
    data_type on weight-free tensors, so the conversion runs here, one tensor at
    a time, with the same clamp as float16.convert_tensor_float_to_float16.
    """
    for t in model.graph.initializer:
        if not uses_external_data(t):
            continue
        info = ExternalDataInfo(t)
        was = dtypes[t.name]
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(was)
        count = (info.length // np.dtype(np_dtype).itemsize) if info.length else -1
        arr = np.fromfile(
            os.path.join(base, info.location), dtype=np_dtype, count=count, offset=info.offset or 0
        )
        if was == TensorProto.FLOAT and t.data_type == TensorProto.FLOAT16:
            arr = float16.convert_np_to_float16(arr)
        del t.external_data[:]
        t.data_location = TensorProto.DEFAULT
        t.raw_data = arr.tobytes()
        del arr


def to_fp16(src, low_memory=None):
    """Return the fp16 OnnxModel for the fp32 model at `src` (either layout).

    The default ORT path (OnnxModel.convert_float_to_float16) runs symbolic
    shape inference on a copy of the whole model and converts the weights in
    place; for the external-data SD-Turbo UNet (~3.46 GB fp32) that peaks near
    10 GB and is killed on a 16 GB host. The low-memory path (automatic when
    the model has external data) runs the same steps on the weight-free graph,
    whose initializers keep their dims: symbolic type inference, the
    float16.convert_float_to_float16 pass (which then only flips the dtype of
    external tensors) and the same Cast cleanup; it then reads each weight from
    the .data file and converts it to fp16 if the pass did. On the tiny model
    both paths give byte-identical graphs (tests/test_diffusion_export.py).
    """
    graph = onnx.load(src, load_external_data=False)
    if low_memory is None:
        low_memory = any(uses_external_data(t) for t in graph.graph.initializer)
    if not low_memory:
        del graph
        m = OnnxModel(onnx.load(src))
        m.convert_float_to_float16(keep_io_types=False)
        return m
    # Imported after onnx_model, which puts the helper's directory on sys.path.
    from onnxruntime.transformers.shape_infer_helper import SymbolicShapeInferenceHelper

    base = os.path.dirname(os.path.abspath(src))
    # Inference reads a few constant initializers (biases, scales) through
    # numpy_helper.to_array, which resolves external-data locations against the
    # working directory: run it from the model directory so those reads work.
    cwd = os.getcwd()
    os.chdir(base)
    try:
        inferred = SymbolicShapeInferenceHelper(graph).infer_shapes(
            graph, auto_merge=True, guess_output_rank=False
        )
    finally:
        os.chdir(cwd)
    if inferred is None:
        raise RuntimeError("symbolic shape inference failed on the weight-free graph")
    _merge_value_types(graph, inferred)
    del inferred
    dtypes = {t.name: t.data_type for t in graph.graph.initializer}
    orig = float16.make_value_info_from_tensor
    float16.make_value_info_from_tensor = _value_info_from_dims
    try:
        fp16 = float16.convert_float_to_float16(graph, keep_io_types=False, disable_shape_infer=True)
    finally:
        float16.make_value_info_from_tensor = orig
    m = OnnxModel(fp16)
    os.chdir(base)
    try:
        m.remove_cascaded_cast_nodes()
        # Its dtype lookup falls back to symbolic inference only for values the
        # pass left without value_info; if that inference cannot run here, ORT
        # logs it and uses value_info, exactly as on the default path.
        m.remove_useless_cast_nodes()
    finally:
        os.chdir(cwd)
    if not m.enable_shape_infer:
        print("[fp16] note: Cast cleanup ran on value_info dtypes only (weight-free graph)")
    _attach_weights(m.model, base, dtypes)
    return m


def main(argv):
    IN = argv[1] if len(argv) > 1 else "sd-turbo-onnx"
    OUT = argv[2] if len(argv) > 2 else "sd-turbo-onnx-fp16"
    os.makedirs(OUT, exist_ok=True)
    # Copy the non-weight pipeline files (scheduler, tokenizer, configs) verbatim.
    for name in os.listdir(IN):
        src = os.path.join(IN, name)
        if os.path.isdir(src) and name not in [*COMPONENTS, "vae_encoder"]:
            shutil.copytree(src, os.path.join(OUT, name), dirs_exist_ok=True)
        elif os.path.isfile(src):
            shutil.copy2(src, os.path.join(OUT, name))

    failed = 0
    for comp in COMPONENTS:
        src = os.path.join(IN, comp, "model.onnx")
        if not os.path.exists(src):
            print(f"[skip] {comp}: no model.onnx")
            continue
        print(f"[fp16] converting {comp} ...", flush=True)
        m = to_fp16(src)
        dup = drop_duplicate_nodes(m.model.graph)
        if dup:
            print(f"[fp16] {comp}: dropped {dup} duplicate Cast node(s)")
        dst_dir = os.path.join(OUT, comp)
        os.makedirs(dst_dir, exist_ok=True)
        dst = os.path.join(dst_dir, "model.onnx")
        m.save_model_to_file(dst, use_external_data_format=False)  # self-contained
        del m  # free the in-memory graph before the ORT load test
        disk = os.path.getsize(dst)
        if disk >= PROTOBUF_LIMIT:
            print(f"[fp16] {comp}: {disk // (1024 * 1024)} MB EXCEEDS 2GB — not deployable")
            failed += 1
            continue
        # Load-test at the same optimization level the console pipeline uses.
        so = ort.SessionOptions()
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        try:
            s = ort.InferenceSession(
                dst, sess_options=so, providers=["CPUExecutionProvider"]
            )
            ins = [(i.name, i.type) for i in s.get_inputs()]
            del s
            print(
                f"[fp16] {comp}: {disk // (1024 * 1024)} MB, LOAD OK (EXTENDED), in: {ins}"
            )
        except Exception as e:
            print(f"[fp16] {comp}: LOAD FAIL: {str(e)[:200]}")
            failed += 1

    print(f"[fp16] done -> {OUT}" + (f"  ({failed} FAILED)" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
