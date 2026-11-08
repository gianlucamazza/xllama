#!/usr/bin/env python3
# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
"""Specialize a first-party fp16 export to the native runner's fixed inputs.

Keep the dynamic export as the host reference. This separate profile proves
nonzero Reshape targets at the native fixed inputs before normalization. By
default, retain dynamic annotations to limit DML fusion. The static diagnostic
variant also replaces target expressions. Literal-zero targets fail closed.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import tempfile

SHAPES = {
    "text_encoder": {"input_ids": [1, 77]},
    "unet": {"sample": [1, 4, 64, 64], "timestep": [],
             "encoder_hidden_states": [1, 77, 1024]},
    "vae_decoder": {"latent_sample": [1, 4, 64, 64]},
}


def specialize(source, destination, shapes, *, dynamic_annotations=False):
    import numpy as np
    import onnx
    import onnxruntime as ort
    from onnx import helper, numpy_helper

    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError("The console profile must not overwrite the dynamic reference")
    model = onnx.load(source)
    original_inputs = [copy.deepcopy(v) for v in model.graph.input]
    if {v.name for v in model.graph.input} != set(shapes):
        raise ValueError("Input names differ from the fixed console profile")
    for value in model.graph.input:
        shape = shapes[value.name]
        declared = value.type.tensor_type.shape.dim
        if len(shape) != len(declared) or any(d <= 0 for d in shape):
            raise ValueError(f"Invalid fixed shape for {value.name}")
        for dim, actual in zip(declared, shape):
            if dim.HasField("dim_value") and dim.dim_value != actual:
                raise ValueError(f"Static dimension mismatch for {value.name}")
            dim.ClearField("dim_param")
            dim.dim_value = actual
    targets = [n for n in model.graph.node if n.op_type == "Reshape" and
               n.domain in ("", "ai.onnx") and
               any(a.name == "allowzero" and a.i == 1 for a in n.attribute)]
    names = list(dict.fromkeys(n.input[1] for n in targets))
    outputs = [copy.deepcopy(v) for v in model.graph.output]
    values = {}
    if names:
        # Fetch only shape expressions, not image tensors. With all input axes
        # fixed these are independent of activation contents. Reject data-
        # dependent shape expressions rather than sampling them with zero data.
        producers = {o: n for n in model.graph.node for o in n.output}
        initializers = {v.name for v in model.graph.initializer}
        def shape_only(name, seen):
            if name in initializers:
                return True
            if name in seen:
                return False
            node = producers.get(name)
            if node is None:
                return False
            if node.op_type in ("Constant", "Shape") and node.domain in ("", "ai.onnx"):
                return True
            if node.domain not in ("", "ai.onnx") or node.op_type not in {
                "Concat", "Squeeze", "Unsqueeze", "Reshape", "Gather", "Slice",
                "Cast", "Mul", "Div", "Add", "Sub", "Identity",
            }:
                return False
            return all(shape_only(x, seen | {name}) for x in node.input if x)
        if not all(shape_only(name, set()) for name in names):
            raise ValueError("A Reshape target depends on activation data or an unsupported expression")
        del model.graph.output[:]
        model.graph.output.extend(helper.make_tensor_value_info(n, onnx.TensorProto.INT64, [None])
                                  for n in names)
        with tempfile.TemporaryDirectory(prefix="xllama-shapes-") as tmp:
            probe = Path(tmp) / "probe.onnx"
            onnx.save(model, probe)
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
            so.intra_op_num_threads = 2
            optimized = Path(tmp) / "constant-shapes.onnx"
            so.optimized_model_filepath = str(optimized)
            session = ort.InferenceSession(str(probe), so, providers=["CPUExecutionProvider"])
            del session
            folded = onnx.load(optimized)
            constants = {v.name: numpy_helper.to_array(v) for v in folded.graph.initializer}
            aliases = {n.output[0]: n.input[0] for n in folded.graph.node
                       if n.op_type == "Identity" and n.domain in ("", "ai.onnx")}
            def constant(name, seen):
                if name in constants:
                    return constants[name]
                if name in aliases and name not in seen:
                    return constant(aliases[name], seen | {name})
                raise ValueError(f"Reshape target is not proven constant: {name}")
            values = {name: constant(name, set()) for name in names}
        del model.graph.output[:]
        model.graph.output.extend(outputs)
    existing = {v.name for v in model.graph.initializer} | {
        o for n in model.graph.node for o in n.output}
    for index, node in enumerate(targets):
        shape = np.asarray(values[node.input[1]])
        if shape.ndim != 1 or not np.issubdtype(shape.dtype, np.integer):
            raise ValueError("Reshape target must be a one-dimensional integer tensor")
        if np.any(shape == 0) or np.any(shape < -1) or np.count_nonzero(shape == -1) > 1:
            raise ValueError("Unsafe Reshape target: zero or invalid inferred dimensions")
        name = f"xllama_console_reshape_{index}"
        if name in existing:
            raise ValueError("Console shape initializer name collision")
        if not dynamic_annotations:
            model.graph.initializer.append(numpy_helper.from_array(shape.astype(np.int64), name))
            node.input[1] = name
        for attr in node.attribute:
            if attr.name == "allowzero":
                attr.i = 0
    if dynamic_annotations:
        # DML cannot fuse the entire dynamic network into one scratch-heavy
        # partition. Supported input shapes remain restricted by this profile;
        # retaining annotations does not restore general dynamic equivalence.
        del model.graph.input[:]
        model.graph.input.extend(original_inputs)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Publish only after validation; partial or rejected profiles never replace
    # an existing deployment artifact.
    with tempfile.TemporaryDirectory(dir=destination.parent, prefix=".console-profile-") as tmp:
        final = Path(tmp) / "model.onnx"
        onnx.save(model, final)
        onnx.checker.check_model(str(final))
        final.replace(destination)
    def digest(path):
        with path.open("rb") as stream:
            return hashlib.file_digest(stream, "sha256").hexdigest()
    return {"reshapes": len(targets), "constant_shape_targets": len(names), "fixed_inputs": shapes,
            "dynamic_annotations": dynamic_annotations,
            "source_sha256": digest(source), "profile_sha256": digest(destination)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path)
    ap.add_argument("destination", type=Path)
    modes = ap.add_mutually_exclusive_group()
    modes.add_argument("--dynamic-annotations", dest="dynamic_annotations", action="store_true",
                       help="retain input/shape annotations to limit DML fusion (default)")
    modes.add_argument("--static-inputs", dest="dynamic_annotations", action="store_false",
                       help="diagnostic fully static variant; failed Series S UNet resource gate")
    ap.set_defaults(dynamic_annotations=True)
    args = ap.parse_args()
    if args.source.resolve() == args.destination.resolve():
        ap.error("Use a separate destination for the fixed console profile")
    if args.destination.exists():
        ap.error("Destination must not exist; retain previous profiles as separate evidence")
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.destination.parent, prefix=".console-bundle-") as tmp:
        bundle = Path(tmp) / "bundle"
        report = {"profile": "sd-turbo-512-batch1-seq77",
                  "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  "dynamic_annotations": args.dynamic_annotations, "components": {}}
        for component, shapes in SHAPES.items():
            report["components"][component] = specialize(
                args.source / component / "model.onnx", bundle / component / "model.onnx", shapes,
                dynamic_annotations=args.dynamic_annotations)
            print(component, report["components"][component], flush=True)
        for name in ("tokenizer", "scheduler"):
            shutil.copytree(args.source / name, bundle / name)
        if (args.source / "model_index.json").exists():
            shutil.copy2(args.source / "model_index.json", bundle / "model_index.json")
        (bundle / "console-profile.json").write_text(json.dumps(report, indent=2) + "\n")
        bundle.rename(args.destination)



if __name__ == "__main__":
    main()
