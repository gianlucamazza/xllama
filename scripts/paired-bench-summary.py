#!/usr/bin/env python3
# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
#
# Verdict for one cell of the preregistered #313 paired campaign
# (docs/gguf-gpu-decode.md, "Preregistered #313 experiment"): matched A/B bench
# rows written by scripts/bench-paired-knob.sh, A = baseline, B = experiment.
#
#   scripts/paired-bench-summary.py <cell_dir> --gates prefill,decode,ttft,peak
#                                   [--json-out PATH]
#
# Pair i is row i of A.csv and row i of B.csv; only pairs whose two arms are
# both recorded as ok in ledger.jsonl count. Per pair:
#   prefill ratio  = B.prompt_tok_s / A.prompt_tok_s   (higher is better)
#   decode ratio   = B.decode_tok_s / A.decode_tok_s   (higher is better)
#   TTFT ratio     = B.ttft_ms / A.ttft_ms             (lower is better)
#   peak delta     = B.peak_ws_mb - A.peak_ws_mb       (MiB, lower is better)
# Each statistic is the median over pairs, with a paired nonparametric
# percentile bootstrap 95 % interval (10,000 resamples of pairs, seed
# 20261107). Margins, frozen in the preregistration:
#   prefill >= 1.10 (lower bound), decode >= 0.97 (lower bound),
#   TTFT <= 1.05 (upper bound), peak delta <= 32 MiB (upper bound).
# A margin passes when the interval bound passes it, fails when the whole
# interval is on the wrong side, and is inconclusive otherwise. The cell
# passes only when every gate it is asked for passes.
#
# Exit 0 = cell PASS, 1 = FAIL or INCONCLUSIVE, 2 = invalid input.

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

SEED = 20261107
RESAMPLES = 10_000
GATES = {
    # name: (column, kind, margin, direction)
    "prefill": ("prompt_tok_s", "ratio", 1.10, "higher"),
    "decode": ("decode_tok_s", "ratio", 0.97, "higher"),
    "ttft": ("ttft_ms", "ratio", 1.05, "lower"),
    "peak": ("peak_ws_mb", "delta", 32.0, "lower"),
}


class Invalid(ValueError):
    pass


def load_rows(path: Path):
    if not path.is_file():
        raise Invalid(f"missing {path}")
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def ok_pairs(cell: Path):
    ledger = cell / "ledger.jsonl"
    if not ledger.is_file():
        raise Invalid(f"missing {ledger}")
    status = {}
    for line in ledger.read_text().splitlines():
        if line.strip():
            e = json.loads(line)
            status.setdefault(e["pair"], {})[e["arm"]] = e["status"]
    return sorted(p for p, arms in status.items() if arms.get("A") == "ok" and arms.get("B") == "ok")


def pair_values(a, b, column, kind):
    out = []
    for ra, rb in zip(a, b):
        try:
            va, vb = float(ra[column]), float(rb[column])
        except (KeyError, ValueError):
            raise Invalid(f"column {column} missing or not a number") from None
        if not (math.isfinite(va) and math.isfinite(vb)) or (kind == "ratio" and va <= 0):
            raise Invalid(f"column {column} has a non-positive or non-finite value")
        out.append(vb / va if kind == "ratio" else vb - va)
    return np.array(out, dtype=np.float64)


def median_interval(values, rng):
    idx = rng.integers(0, values.size, size=(RESAMPLES, values.size))
    meds = np.median(values[idx], axis=1)
    return float(np.median(values)), float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))


def judge(direction, margin, lo, hi):
    if direction == "higher":
        return "pass" if lo >= margin else ("fail" if hi < margin else "inconclusive")
    return "pass" if hi <= margin else ("fail" if lo > margin else "inconclusive")


def main() -> int:
    ap = argparse.ArgumentParser(description="Paired A/B bench verdict for one campaign cell.")
    ap.add_argument("cell", type=Path, help="directory with A.csv, B.csv and ledger.jsonl")
    ap.add_argument("--gates", required=True, help="comma-separated: prefill,decode,ttft,peak")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()
    try:
        gates = [g.strip() for g in args.gates.split(",") if g.strip()]
        if not gates or any(g not in GATES for g in gates):
            raise Invalid(f"gates must be among {sorted(GATES)}")
        a, b = load_rows(args.cell / "A.csv"), load_rows(args.cell / "B.csv")
        pairs = ok_pairs(args.cell)
        if not pairs or len(a) < len(pairs) or len(b) < len(pairs):
            raise Invalid("no complete pair, or fewer rows than complete pairs")
        a, b = [a[p - 1] for p in pairs], [b[p - 1] for p in pairs]
        for ra, rb in zip(a, b):
            if ra.get("model") != rb.get("model") or ra.get("n_prompt_tok") != rb.get("n_prompt_tok"):
                raise Invalid("a pair mixes models or prompt lengths")
        values = {g: pair_values(a, b, GATES[g][0], GATES[g][1]) for g in gates}
    except (Invalid, OSError, json.JSONDecodeError) as e:
        print(f"❌ invalid input: {e}")
        return 2

    rng = np.random.default_rng(SEED)
    rows, verdicts = {}, []
    for g in gates:
        column, kind, margin, direction = GATES[g]
        vals = values[g]
        med, lo, hi = median_interval(vals, rng)
        v = judge(direction, margin, lo, hi)
        verdicts.append(v)
        rows[g] = {"column": column, "kind": kind, "median": med, "ci95": [lo, hi],
                   "margin": margin, "direction": direction, "verdict": v,
                   "values": vals.tolist()}
        unit = " MiB" if kind == "delta" else ""
        print(f"  {g:<8} median {med:.4f}{unit}  95% CI [{lo:.4f}, {hi:.4f}]  "
              f"margin {'>=' if direction == 'higher' else '<='} {margin:g}{unit}  {v}")
    cell = "pass" if all(v == "pass" for v in verdicts) else (
        "fail" if "fail" in verdicts else "inconclusive")
    summary = {"cell": str(args.cell), "model": a[0]["model"], "n_prompt_tok": a[0]["n_prompt_tok"],
               "pairs": len(pairs), "seed": SEED, "resamples": RESAMPLES, "gates": rows,
               "verdict": cell}
    if args.json_out:
        args.json_out.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"{a[0]['model']} P={a[0]['n_prompt_tok']} pairs={len(pairs)}: {cell.upper()}")
    return 0 if cell == "pass" else 1


if __name__ == "__main__":
    sys.exit(main())
