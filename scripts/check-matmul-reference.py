#!/usr/bin/env python3
# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
#
# Diagnostic for #420: check every captured weight MUL_MAT of one arm against
# a float64 reference computed from the model file and that arm's own operand.
#
#   scripts/check-matmul-reference.py <model.gguf> <capture_dir>...
#                                     [--nmse-limit F] [--json-out PATH]
#
# The captures come from `--capture-spec '<names>;matmul_inputs=<stop>'`
# (tensor_capture.h). For each record:
#
#   reference = dequantize(W) @ dequantize(q8_K(operand))
#
# where q8_K is ggml's activation quantizer for K-quant weights. The reference
# is what an exact kernel returns for the int8 values the arm itself fed in,
# so the check needs no second arm and covers every token column, unlike
# `control-parity.py kernel`, which compares two arms and can only judge the
# columns whose operands are equal.
#
# It answers one question: does this arm's kernel compute the product it was
# given? It does not compare arms, and it is not the ADR 0010 criterion.
#
# Needs numpy and the gguf-py package in the llama.cpp submodule (which needs
# PyYAML). Only K-quant weights (q8_K operands) are supported.
#
# Exit 0 = every record within --nmse-limit (or no limit given), 1 = a record
# over the limit, 2 = invalid input.

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
QK_K = 256
Q8_K_WEIGHTS = {"q2_K", "q3_K", "q4_K", "q5_K", "q6_K"}


def dequantized_q8_k(x: np.ndarray) -> np.ndarray:
    """float64 values of x [n, tokens] after ggml's q8_K round trip, per column."""
    n, tokens = x.shape
    blocks = x.T.reshape(tokens, n // QK_K, QK_K)
    idx = np.argmax(np.abs(blocks), axis=2)[..., None]
    mx = np.take_along_axis(blocks, idx, axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        iscale = np.where(mx != 0, np.float32(-127.0) / mx, np.float32(0.0)).astype(np.float32)
        q = np.minimum(np.rint((iscale * blocks).astype(np.float32)), 127)
        d = np.where(iscale != 0, np.float32(1.0) / iscale, np.float32(0.0)).astype(np.float32)
    return (q.astype(np.float64) * d.astype(np.float64)).reshape(tokens, n).T


def record_error(weight: np.ndarray, operand: np.ndarray, result: np.ndarray):
    """weight [n_out, n_in] float64; operand [n_in, tokens]; result [n_out, tokens]."""
    ref = weight @ dequantized_q8_k(operand)
    diff = result.astype(np.float64) - ref
    num, den = (diff * diff).sum(axis=0), (ref * ref).sum(axis=0)
    total = float(den.sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        per = np.where(den > 0, num / den, np.where(num > 0, math.inf, 0.0))
    return {
        "nmse": float(num.sum() / total) if total > 0 else (0.0 if num.sum() == 0 else math.inf),
        "worst_token_nmse": float(per.max()),
        "max_abs": float(np.abs(diff).max()),
    }


def load_control_parity():
    spec = importlib.util.spec_from_file_location("control_parity", ROOT / "scripts/control-parity.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description="Weight MUL_MAT captures against a float64 reference.")
    ap.add_argument("model", type=Path, help="the GGUF both arms loaded")
    ap.add_argument("captures", type=Path, nargs="+", help="capture directories, one per arm")
    ap.add_argument("--nmse-limit", type=float, help="fail when a record's NMSE exceeds this")
    ap.add_argument("--json-out", type=Path, help="write the per-record table as JSON")
    args = ap.parse_args()
    if args.nmse_limit is not None and not (math.isfinite(args.nmse_limit) and args.nmse_limit > 0):
        ap.error("--nmse-limit must be finite and positive")

    cp = load_control_parity()
    try:
        sys.path.insert(0, str(ROOT / "llama.cpp/gguf-py"))
        import gguf
        from gguf.quants import dequantize
    except ImportError as e:
        print(f"❌ invalid input: gguf-py is not importable ({e})")
        return 2
    try:
        tensors = {t.name: t for t in gguf.GGUFReader(args.model).tensors}
        arms = {str(c): cp.load_matmuls(c) for c in args.captures}
    except (OSError, ValueError) as e:
        print(f"❌ invalid input: {e}")
        return 2

    rows, worst = [], 0.0
    first = next(iter(arms.values()))
    for arm, records in arms.items():
        if set(records) != set(first):
            print(f"❌ invalid input: {arm} records different products")
            return 2
    for key in first:
        wtype = first[key]["weight_type"]
        if key[1] not in tensors or wtype not in Q8_K_WEIGHTS or first[key]["in"].shape[0] % QK_K:
            print(f"❌ invalid input: {key[1]} ({wtype}) is not a K-quant weight of this model")
            return 2
        t = tensors[key[1]]
        weight = dequantize(t.data, t.tensor_type).astype(np.float64)
        row = {"eval": key[0], "weight": key[1], "occ": key[2], "weight_type": wtype, "arms": {}}
        for arm, records in arms.items():
            r = records[key]
            if weight.shape != (r["out"].shape[0], r["in"].shape[0]):
                print(f"❌ invalid input: {key[1]} shape does not match the capture")
                return 2
            row["arms"][arm] = record_error(weight, r["in"], r["out"])
            worst = max(worst, row["arms"][arm]["nmse"])
        rows.append(row)
        print(f"{key[0]:>3} {key[1]:<28} {wtype:<5} "
              + "  ".join(f"{v['nmse']:.2e}/{v['worst_token_nmse']:.2e}" for v in row["arms"].values()),
              flush=True)

    passed = args.nmse_limit is None or worst <= args.nmse_limit
    if args.json_out:
        args.json_out.write_text(json.dumps({
            "scope": "diagnostic; each arm against a float64 reference from its own operand",
            "model": args.model.name, "arms": list(arms), "records": len(rows),
            "worst_nmse": worst, "nmse_limit": args.nmse_limit, "pass": passed, "rows": rows,
        }, indent=1) + "\n")
    print(f"{len(rows)} records per arm; worst NMSE {worst:.3e}")
    print("✅ within limit" if passed else "❌ over limit")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
