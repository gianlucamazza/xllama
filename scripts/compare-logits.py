#!/usr/bin/env python3
# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
#
# Logit-parity comparator for the xllama harness. Diffs two float32 logit dumps
# produced by `xllama-cli --dump-logits` (or the on-device ORT path) and decides
# PASS/FAIL against tolerance thresholds. NumPy is the only external dependency.
#
# Usage:
#   scripts/compare-logits.py <a.bin> <b.bin> [--max-abs-diff F] [--nmse F]
#                             [--top-k N] [--top-k-min F]
#
# Each <x.bin> may have a sibling <x.bin.json> sidecar (written by the dumper)
# carrying vocab_size / backend / top1_piece. When present it is used to catch
# tokenizer/vocab misalignment, which makes an index-wise comparison meaningless.
#
# Metrics:
#   max_abs_diff  max |a_i - b_i|                       (kernel/quant drift)
#   nmse          ||a-b||^2 / ||a||^2                    (overall distribution)
#   top1_match    argmax(a) == argmax(b)                (does generation agree?)
#   topk_overlap  |top-k(a) ∩ top-k(b)| / k             (near-tie robustness)
#
# Exit 0 = PASS (within tolerance), 1 = FAIL, 2 = usage/IO error.

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from logit_comparison import compare_logits


def load_sidecar(bin_path: Path):
    side = Path(str(bin_path) + ".json")
    if not side.exists():
        return {}
    data = json.loads(side.read_text())
    if not isinstance(data, dict):
        raise ValueError(f"sidecar must be an object: {side}")
    return data


def load_dump(path: Path):
    raw = path.read_bytes()
    if not raw or len(raw) % 4:
        raise ValueError(f"dump must contain complete, nonempty float32 values: {path}")
    values = np.frombuffer(raw, dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise ValueError(f"non-finite logits: {path}")
    sidecar = load_sidecar(path)
    if "vocab_size" in sidecar:
        size = sidecar["vocab_size"]
        if type(size) is not int or size != values.size:
            raise ValueError(f"sidecar vocab_size does not match dump: {path}")
    return values, sidecar


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare two float32 logit dumps.")
    ap.add_argument("a", type=Path, help="reference dump (e.g. llama.cpp golden)")
    ap.add_argument("b", type=Path, help="candidate dump (e.g. ORT on-device)")
    ap.add_argument(
        "--max-abs-diff",
        type=float,
        default=0.5,
        help="max allowed |a-b| per logit (default 0.5)",
    )
    ap.add_argument(
        "--nmse",
        type=float,
        default=1e-3,
        help="max allowed normalized MSE (default 1e-3)",
    )
    ap.add_argument(
        "--top-k", type=int, default=10, help="k for top-k overlap (default 10)"
    )
    ap.add_argument(
        "--top-k-min",
        type=float,
        default=0.8,
        help="min top-k overlap fraction (default 0.8)",
    )
    args = ap.parse_args()

    if (
        not math.isfinite(args.max_abs_diff) or args.max_abs_diff < 0
        or not math.isfinite(args.nmse) or args.nmse < 0
        or args.top_k <= 0
        or not math.isfinite(args.top_k_min) or not 0 <= args.top_k_min <= 1
    ):
        ap.error("thresholds must be finite and nonnegative; top-k > 0 and overlap in [0, 1]")

    try:
        a, sa = load_dump(args.a)
        b, sb = load_dump(args.b)
    except (OSError, ValueError) as e:
        print(f"❌ NOK: {e}")
        return 2

    print(f"A: {args.a}  ({a.size} logits, backend={sa.get('backend', '?')})")
    print(f"B: {args.b}  ({b.size} logits, backend={sb.get('backend', '?')})")

    # A matching token string is diagnostic only, never numerical parity.
    if a.size != b.size:
        print(f"❌ NOK: shape mismatch — A={a.size}, B={b.size}; parity not computable")
        return 1
    for key in ("prompt", "tokenizer_sha256", "vocab_sha256"):
        if key in sa and key in sb and sa[key] != sb[key]:
            print(f"❌ NOK: sidecar {key} mismatch; parity not computable")
            return 1

    metrics = compare_logits(a, b, args.top_k)
    k = min(args.top_k, a.size)
    print(f"  max_abs_diff : {metrics.max_abs_diff:.6f}   (thr ≤ {args.max_abs_diff})")
    print(f"  nmse         : {metrics.nmse:.3e}   (thr ≤ {args.nmse})")
    print(f"  top1_match   : {metrics.top1_match}")
    print(f"  top{k}_overlap : {metrics.top_k_overlap:.2f}   (thr ≥ {args.top_k_min})")

    # top1_piece cross-check (detects silent index misalignment at equal size).
    pa, pb = sa.get("top1_piece"), sb.get("top1_piece")
    if pa is not None and pb is not None and pa != pb:
        print(f"  ⚠ top1_piece differ: A={pa!r} B={pb!r}")

    passed = metrics.passes(args.max_abs_diff, args.nmse, args.top_k_min) and not (
        pa is not None and pb is not None and pa != pb
    )
    print("✅ PASS" if passed else "❌ FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
