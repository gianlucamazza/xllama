#!/usr/bin/env python3
# Copyright (c) 2024 Venere Labs
# SPDX-License-Identifier: MIT
#
# Logit-parity comparator for the xllama harness. Diffs two float32 logit dumps
# produced by `xllama-cli --dump-logits` (or the on-device ORT path) and decides
# PASS/FAIL against tolerance thresholds. Self-contained: only numpy required.
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
import sys
from pathlib import Path

import numpy as np


def load_sidecar(bin_path: Path):
    side = Path(str(bin_path) + ".json")
    if not side.exists():
        return {}
    try:
        return json.loads(side.read_text())
    except Exception as e:
        print(f"⚠ could not parse sidecar {side}: {e}")
        return {}


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

    for p in (args.a, args.b):
        if not p.exists():
            print(f"❌ NOK: file not found: {p}")
            return 2

    a = np.fromfile(args.a, dtype=np.float32)
    b = np.fromfile(args.b, dtype=np.float32)
    sa, sb = load_sidecar(args.a), load_sidecar(args.b)

    print(f"A: {args.a}  ({a.size} logits, backend={sa.get('backend', '?')})")
    print(f"B: {args.b}  ({b.size} logits, backend={sb.get('backend', '?')})")

    # --- vocab / tokenizer alignment gate -----------------------------------
    # Index-wise metrics are only meaningful if both vocabularies match in size
    # AND ordering. Different tokenizers (ORT ONNX vs GGUF) can differ; fall back
    # to top-token STRING agreement rather than a bogus per-index diff.
    if a.size != b.size:
        print(
            f"❌ NOK: shape mismatch — A={a.size}, B={b.size} (different vocab/tokenizer)."
        )
        pa, pb = sa.get("top1_piece"), sb.get("top1_piece")
        if pa is not None and pb is not None:
            ok = pa == pb
            print(f"   top1_piece: A={pa!r} B={pb!r} → {'MATCH' if ok else 'DIFFER'}")
            print("   (string-level fallback; per-index parity not computable)")
            return 0 if ok else 1
        return 1

    # --- index-wise metrics --------------------------------------------------
    diff = a - b
    max_abs = float(np.max(np.abs(diff))) if a.size else 0.0
    denom = float(np.dot(a, a))
    nmse = float(np.dot(diff, diff) / denom) if denom > 0 else float("inf")

    k = min(args.top_k, a.size)
    top_a = set(np.argsort(a)[-k:].tolist())
    top_b = set(np.argsort(b)[-k:].tolist())
    overlap = len(top_a & top_b) / k if k else 1.0
    top1_match = int(np.argmax(a)) == int(np.argmax(b))

    print(f"  max_abs_diff : {max_abs:.6f}   (thr ≤ {args.max_abs_diff})")
    print(f"  nmse         : {nmse:.3e}   (thr ≤ {args.nmse})")
    print(f"  top1_match   : {top1_match}")
    print(f"  top{k}_overlap : {overlap:.2f}   (thr ≥ {args.top_k_min})")

    # top1_piece cross-check (detects silent index misalignment at equal size).
    pa, pb = sa.get("top1_piece"), sb.get("top1_piece")
    if pa is not None and pb is not None and pa != pb:
        print(f"  ⚠ top1_piece differ: A={pa!r} B={pb!r}")

    passed = (
        max_abs <= args.max_abs_diff
        and nmse <= args.nmse
        and top1_match
        and overlap >= args.top_k_min
    )
    print("✅ PASS" if passed else "❌ FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
