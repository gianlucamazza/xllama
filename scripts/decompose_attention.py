#!/usr/bin/env python3
"""Decompose com.microsoft attention contrib ops into ONNX primitives (#91 plan B).

The Xbox Series S DML driver produces garbage logits from the fused attention
contrib kernels (GroupQueryAttention and MultiHeadAttention both fail with
NMSE ~0.98), while decomposed attention — SD-Turbo's MatMul/Softmax graph —
is correct on the same GPU. This script rewrites each MultiHeadAttention node
of an MHA-exported decoder (probe #94 style) into

    out = Reshape(Transpose(MatMul(Softmax(MatMul(q4, k4T)*scale + mask), v4)))

using only primitive ops. Graph I/O is untouched, so the result stays
ORT-GenAI compatible (past/present KV remain ordinary graph inputs/outputs;
KV concat and the GQA 5->15 head repeat already live in the graph as
primitive nodes in the MHA export).

Expected MHA node interface (validated on smollm2-360m-dml-fp16-mha):
    inputs  = [query(3D [B,S,H*d]), key(3D [B,T,H*d]), value(3D [B,T,H*d]),
               '', '', attention_bias([B,H,S,T] additive fp16), '', '']
    attrs   = num_heads, scale, unidirectional=0 (causality lives in the mask)

Usage:
    decompose_attention.py -i model.onnx -o out.onnx [--fp32-qk]

--fp32-qk casts the score/softmax path to fp32 (precision mitigation; the
fp16 default mirrors SD-Turbo, the proven-correct reference on this driver).
"""

import argparse
import sys

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

MSFT_DOMAIN = "com.microsoft"


def _attr_map(node):
    return {a.name: helper.get_attribute_value(a) for a in node.attribute}


def _tensor_dim(model, name, axis):
    """Static dim of value_info `name` at `axis`, or None."""
    for vi in (
        list(model.graph.value_info)
        + list(model.graph.input)
        + list(model.graph.output)
    ):
        if vi.name == name:
            dims = vi.type.tensor_type.shape.dim
            if axis < len(dims) and dims[axis].HasField("dim_value"):
                return dims[axis].dim_value
    return None


def decompose_mha(model, fp32_qk=False):
    g = model.graph
    mha_nodes = [
        n
        for n in g.node
        if n.op_type == "MultiHeadAttention" and n.domain == MSFT_DOMAIN
    ]
    if not mha_nodes:
        sys.exit("no com.microsoft.MultiHeadAttention nodes found — nothing to do")

    # Validate every node against the interface this surgery supports before
    # touching the graph, so a partial rewrite can never be emitted.
    for n in mha_nodes:
        attrs = _attr_map(n)
        ins = list(n.input) + [""] * (8 - len(n.input))
        if attrs.get("unidirectional", 0) != 0:
            sys.exit(
                f"{n.name}: unidirectional=1 unsupported (mask must carry causality)"
            )
        if ins[3] or ins[4] or ins[6] or ins[7]:
            sys.exit(
                f"{n.name}: bias/key_padding_mask/past inputs are wired — unsupported layout"
            )
        if not ins[5]:
            sys.exit(f"{n.name}: attention_bias input missing — unsupported layout")
        if "num_heads" not in attrs:
            sys.exit(f"{n.name}: num_heads attribute missing")

    first = mha_nodes[0]
    num_heads = _attr_map(first)["num_heads"]
    hidden = _tensor_dim(model, first.input[0], 2)
    if hidden is None:
        sys.exit(f"cannot infer hidden size from value_info of {first.input[0]}")
    head_size = hidden // num_heads
    scale = _attr_map(first).get("scale", 1.0 / float(np.sqrt(head_size)))

    # Shared initializers (Reshape 0/-1 semantics: 0 copies the input dim).
    shape_split = numpy_helper.from_array(
        np.array([0, 0, num_heads, head_size], dtype=np.int64), "decomp/shape_split"
    )
    shape_merge = numpy_helper.from_array(
        np.array([0, 0, -1], dtype=np.int64), "decomp/shape_merge"
    )
    scale_dtype = np.float32 if fp32_qk else np.float16
    scale_init = numpy_helper.from_array(
        np.array(scale, dtype=scale_dtype), "decomp/scale"
    )
    g.initializer.extend([shape_split, shape_merge, scale_init])

    new_graph_nodes = []
    for n in g.node:
        if not (n.op_type == "MultiHeadAttention" and n.domain == MSFT_DOMAIN):
            new_graph_nodes.append(n)
            continue

        attrs = _attr_map(n)
        if attrs["num_heads"] != num_heads or attrs.get("scale", scale) != scale:
            sys.exit(
                f"{n.name}: heterogeneous num_heads/scale across layers unsupported"
            )
        q, k, v = n.input[0], n.input[1], n.input[2]
        mask = n.input[5]
        base = n.name or n.output[0]

        def N(op, inputs, outputs, **kw):
            new_graph_nodes.append(
                helper.make_node(
                    op,
                    inputs,
                    outputs,
                    name=f"{base}/decomp/{outputs[0].rsplit('/', 1)[-1]}",
                    **kw,
                )
            )

        # [B,S,H*d] -> [B,H,S,d]  (k transposed straight to [B,H,d,T] for QK^T)
        N("Reshape", [q, "decomp/shape_split"], [f"{base}/q4"])
        N("Transpose", [f"{base}/q4"], [f"{base}/q4t"], perm=[0, 2, 1, 3])
        N("Reshape", [k, "decomp/shape_split"], [f"{base}/k4"])
        N("Transpose", [f"{base}/k4"], [f"{base}/k4t"], perm=[0, 2, 3, 1])
        N("Reshape", [v, "decomp/shape_split"], [f"{base}/v4"])
        N("Transpose", [f"{base}/v4"], [f"{base}/v4t"], perm=[0, 2, 1, 3])

        qt, kt, vt, mk = f"{base}/q4t", f"{base}/k4t", f"{base}/v4t", mask
        if fp32_qk:
            for src, dst in ((qt, "q32"), (kt, "k32"), (vt, "v32"), (mk, "mask32")):
                N("Cast", [src], [f"{base}/{dst}"], to=TensorProto.FLOAT)
            qt, kt, vt, mk = (
                f"{base}/q32",
                f"{base}/k32",
                f"{base}/v32",
                f"{base}/mask32",
            )

        N("MatMul", [qt, kt], [f"{base}/qk"])
        N("Mul", [f"{base}/qk", "decomp/scale"], [f"{base}/qk_scaled"])
        N("Add", [f"{base}/qk_scaled", mk], [f"{base}/scores"])
        N("Softmax", [f"{base}/scores"], [f"{base}/probs"], axis=-1)
        N("MatMul", [f"{base}/probs", vt], [f"{base}/ctx"])

        ctx = f"{base}/ctx"
        if fp32_qk:
            N("Cast", [ctx], [f"{base}/ctx16"], to=TensorProto.FLOAT16)
            ctx = f"{base}/ctx16"

        N("Transpose", [ctx], [f"{base}/ctx_t"], perm=[0, 2, 1, 3])
        new_graph_nodes.append(
            helper.make_node(
                "Reshape",
                [f"{base}/ctx_t", "decomp/shape_merge"],
                [n.output[0]],
                name=f"{base}/decomp/out",
            )
        )

    del g.node[:]
    g.node.extend(new_graph_nodes)
    return len(mha_nodes)


def decompose_rotary(model):
    """Rewrite com.microsoft.RotaryEmbedding into primitives (NeoX rotate-half).

    Interface (genai builder export): x [B,S,H*d] fp16, position_ids [B,S]
    int64, cos_cache/sin_cache [max_pos, d/2] initializers; interleaved=0.
    out[..., :d/2] = x1*cos - x2*sin ; out[..., d/2:] = x2*cos + x1*sin.
    """
    g = model.graph
    rot_nodes = [
        n for n in g.node if n.op_type == "RotaryEmbedding" and n.domain == MSFT_DOMAIN
    ]
    if not rot_nodes:
        return 0

    for n in rot_nodes:
        a = _attr_map(n)
        if a.get("interleaved", 0) != 0 or a.get("rotary_embedding_dim", 0) != 0:
            sys.exit(f"{n.name}: only interleaved=0, full-dim rotary supported")

    cos_cache, sin_cache, pos = (
        rot_nodes[0].input[2],
        rot_nodes[0].input[3],
        rot_nodes[0].input[1],
    )
    half = next(int(i.dims[1]) for i in g.initializer if i.name == cos_cache)
    head = 2 * half

    inits = [
        numpy_helper.from_array(np.array([0], dtype=np.int64), "decomp_rot/zero"),
        numpy_helper.from_array(np.array([half], dtype=np.int64), "decomp_rot/half"),
        numpy_helper.from_array(np.array([head], dtype=np.int64), "decomp_rot/head"),
        numpy_helper.from_array(np.array([3], dtype=np.int64), "decomp_rot/axis3"),
        numpy_helper.from_array(np.array([2], dtype=np.int64), "decomp_rot/axes2"),
        numpy_helper.from_array(
            np.array([0, 0, -1], dtype=np.int64), "decomp_rot/shape_merge"
        ),
    ]
    g.initializer.extend(inits)

    # One shared cos/sin gather + head-broadcast unsqueeze: every rotary node
    # reads the same position_ids.
    shared = [
        helper.make_node(
            "Gather",
            [cos_cache, pos],
            ["decomp_rot/cos_g"],
            name="decomp_rot/cos_gather",
            axis=0,
        ),
        helper.make_node(
            "Gather",
            [sin_cache, pos],
            ["decomp_rot/sin_g"],
            name="decomp_rot/sin_gather",
            axis=0,
        ),
        helper.make_node(
            "Unsqueeze",
            ["decomp_rot/cos_g", "decomp_rot/axes2"],
            ["decomp_rot/cos_u"],
            name="decomp_rot/cos_unsq",
        ),
        helper.make_node(
            "Unsqueeze",
            ["decomp_rot/sin_g", "decomp_rot/axes2"],
            ["decomp_rot/sin_u"],
            name="decomp_rot/sin_unsq",
        ),
    ]

    new_nodes, shared_placed = [], False
    for n in g.node:
        if not (n.op_type == "RotaryEmbedding" and n.domain == MSFT_DOMAIN):
            new_nodes.append(n)
            continue
        if not shared_placed:
            new_nodes.extend(shared)
            shared_placed = True

        x, base = n.input[0], (n.name or n.output[0])
        width = _tensor_dim(model, x, 2)
        if width is None or width % head:
            sys.exit(f"{n.name}: cannot infer head count from value_info of {x}")
        heads = width // head
        shape_split = f"decomp_rot/shape_split_{heads}"
        if not any(i.name == shape_split for i in g.initializer):
            g.initializer.append(
                numpy_helper.from_array(
                    np.array([0, 0, heads, head], dtype=np.int64), shape_split
                )
            )

        def N(op, inputs, outputs, **kw):
            new_nodes.append(
                helper.make_node(
                    op,
                    inputs,
                    outputs,
                    name=f"{base}/decomp/{outputs[0].rsplit('/', 1)[-1]}",
                    **kw,
                )
            )

        N("Reshape", [x, shape_split], [f"{base}/x4"])
        N(
            "Slice",
            [f"{base}/x4", "decomp_rot/zero", "decomp_rot/half", "decomp_rot/axis3"],
            [f"{base}/x1"],
        )
        N(
            "Slice",
            [f"{base}/x4", "decomp_rot/half", "decomp_rot/head", "decomp_rot/axis3"],
            [f"{base}/x2"],
        )
        N("Mul", [f"{base}/x1", "decomp_rot/cos_u"], [f"{base}/x1c"])
        N("Mul", [f"{base}/x2", "decomp_rot/sin_u"], [f"{base}/x2s"])
        N("Sub", [f"{base}/x1c", f"{base}/x2s"], [f"{base}/o1"])
        N("Mul", [f"{base}/x2", "decomp_rot/cos_u"], [f"{base}/x2c"])
        N("Mul", [f"{base}/x1", "decomp_rot/sin_u"], [f"{base}/x1s"])
        N("Add", [f"{base}/x2c", f"{base}/x1s"], [f"{base}/o2"])
        N("Concat", [f"{base}/o1", f"{base}/o2"], [f"{base}/o4"], axis=3)
        new_nodes.append(
            helper.make_node(
                "Reshape",
                [f"{base}/o4", "decomp_rot/shape_merge"],
                [n.output[0]],
                name=f"{base}/decomp/out",
            )
        )

    del g.node[:]
    g.node.extend(new_nodes)
    return len(rot_nodes)


def decompose_skipln(model):
    """Rewrite (Skip)SimplifiedLayerNormalization into primitive RMSNorm.

    Contrib semantics (stash_type=1): accumulate in fp32. Mirrored here with
    Cast fp16->fp32 around the mean/sqrt math. SkipSimplifiedLayerNormalization
    also emits output[3] = x + skip (residual chain input downstream).
    """
    g = model.graph
    targets = [
        n
        for n in g.node
        if n.op_type
        in ("SimplifiedLayerNormalization", "SkipSimplifiedLayerNormalization")
    ]
    if not targets:
        return 0

    g.initializer.append(
        numpy_helper.from_array(np.array([-1], dtype=np.int64), "decomp_ln/axes_last")
    )

    new_nodes = []
    for n in g.node:
        if n not in targets:
            new_nodes.append(n)
            continue
        a = _attr_map(n)
        eps = a["epsilon"]
        base = n.name or n.output[0]
        has_skip = n.op_type == "SkipSimplifiedLayerNormalization"
        gamma = n.input[2] if has_skip else n.input[1]

        def N(op, inputs, outputs, **kw):
            new_nodes.append(
                helper.make_node(
                    op,
                    inputs,
                    outputs,
                    name=f"{base}/decomp/{outputs[0].rsplit('/', 1)[-1]}",
                    **kw,
                )
            )

        if has_skip:
            sum_out = (
                n.output[3] if len(n.output) > 3 and n.output[3] else f"{base}/sum"
            )
            N("Add", [n.input[0], n.input[1]], [sum_out])
            x = sum_out
        else:
            x = n.input[0]

        eps_init = f"decomp_ln/eps_{base.replace('/', '_')}"
        g.initializer.append(
            numpy_helper.from_array(np.array(eps, dtype=np.float32), eps_init)
        )

        N("Cast", [x], [f"{base}/x32"], to=TensorProto.FLOAT)
        N("Mul", [f"{base}/x32", f"{base}/x32"], [f"{base}/sq"])
        N(
            "ReduceMean",
            [f"{base}/sq", "decomp_ln/axes_last"],
            [f"{base}/ms"],
            keepdims=1,
        )
        N("Add", [f"{base}/ms", eps_init], [f"{base}/ms_eps"])
        N("Sqrt", [f"{base}/ms_eps"], [f"{base}/rms"])
        N("Div", [f"{base}/x32", f"{base}/rms"], [f"{base}/normed32"])
        N("Cast", [f"{base}/normed32"], [f"{base}/normed16"], to=TensorProto.FLOAT16)
        new_nodes.append(
            helper.make_node(
                "Mul",
                [f"{base}/normed16", gamma],
                [n.output[0]],
                name=f"{base}/decomp/out",
            )
        )

    del g.node[:]
    g.node.extend(new_nodes)
    return len(targets)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-i", "--input", required=True, help="MHA-exported model.onnx")
    ap.add_argument(
        "-o", "--output", required=True, help="output path for the surgered model"
    )
    ap.add_argument(
        "--fp32-qk",
        action="store_true",
        help="compute scores/softmax in fp32 (precision mitigation)",
    )
    ap.add_argument(
        "--also-rotary",
        action="store_true",
        help="also decompose com.microsoft.RotaryEmbedding (escalation level 2)",
    )
    ap.add_argument(
        "--also-skipln",
        action="store_true",
        help="also decompose (Skip)SimplifiedLayerNormalization (escalation level 3)",
    )
    ap.add_argument(
        "--skip-attention",
        action="store_true",
        help="leave the fused attention nodes in place (isolation runs: decompose "
        "only rotary/skipln; works on GQA models too since attention is untouched)",
    )
    args = ap.parse_args()

    if args.skip_attention and not (args.also_rotary or args.also_skipln):
        ap.error("--skip-attention with no --also-* flag leaves nothing to do")

    model = onnx.load(args.input)
    replaced = 0 if args.skip_attention else decompose_mha(model, fp32_qk=args.fp32_qk)
    rot = decompose_rotary(model) if args.also_rotary else 0
    lns = decompose_skipln(model) if args.also_skipln else 0

    # Structural gate: no fused attention left (unless deliberately kept).
    # onnx.checker is advisory only — the genai builder emits ORT-internal ops
    # (SimplifiedLayerNormalization) in the default domain that the checker
    # rejects even on the *input* model; the authoritative validation is
    # loading the output in onnxruntime.
    assert args.skip_attention or not any(
        n.op_type in ("MultiHeadAttention", "GroupQueryAttention", "Attention")
        for n in model.graph.node
    ), "fused attention nodes survived the rewrite"
    try:
        onnx.checker.check_model(model)
    except onnx.checker.ValidationError as e:
        print(
            f"warning: onnx.checker rejected the model (pre-existing ORT-internal ops): {e}",
            file=sys.stderr,
        )

    onnx.save(model, args.output)
    print(
        f"decomposed {replaced} MultiHeadAttention, {rot} RotaryEmbedding, "
        f"{lns} (Skip)SimplifiedLayerNorm nodes "
        f"({'fp32' if args.fp32_qk else 'fp16'} score path) -> {args.output}"
    )


if __name__ == "__main__":
    main()
