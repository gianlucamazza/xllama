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
#   scripts/paired-bench-summary.py <cell_dir> --profile bias-add|bias-add-control|rope
#                                   [--json-out PATH]
#   scripts/paired-bench-summary.py <cell_dir> --report prefill,decode,ttft,peak
#                                   [--json-out PATH]
# --report is for an estimate with no preregistered margin (#582): the same
# medians and intervals, no verdict, exit 0 on valid input.
# Profiles require 20 complete alternating pairs of one preregistered model and
# host tag. bias-add-control also requires identical measured prefill and decode
# submission counts in both arms of every pair. rope (A = bias ADD on, B = bias
# ADD and ROPE on) shares the bias ADD gates and models, and requires B to
# submit fewer decode and no more prefill batches than A in every pair, so a
# speed-up cannot come from a knob that never took effect.
#
# Pair i is row i of A.csv and row i of B.csv; only pairs whose two arms are
# both recorded as ok in ledger.jsonl count. A "retry" ledger line is an attempt
# that failed before writing a row (bench-paired-knob.sh --retries); it is not
# a pair and is ignored. Per pair:
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
import re
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

BIAS_ADD_GATES = {
    "prefill": ("prompt_tok_s", "ratio", 0.97, "higher"),
    "decode": ("decode_tok_s", "ratio", 1.03, "higher"),
    "ttft": ("ttft_ms", "ratio", 1.03, "lower"),
    "peak": GATES["peak"],
}
BIAS_ADD_CONTROL_GATES = {
    name: (column, kind, (0.98, 1.02), "within")
    for name, (column, kind, _, _) in BIAS_ADD_GATES.items()
    if name != "peak"
}
BIAS_ADD_PAIRS = 20
PHASE_SUBMISSIONS = re.compile(
    r"(?m)^\[xllama\] d3d12-phase phase=(prefill|decode) calls=\d+ submissions=(\d+)\b"
)

BIAS_ADD_MODELS = {
    "qwen25-coder-1.5b": "Q4_K_M",
    "qwen25-coder-3b": "Q4_K_M",
}
BIAS_ADD_CONTROL_MODELS = {"lfm25-1.2b-instruct": "Q4_0"}
# profile: (gates, models, submission check). The check compares the measured
# phase counters of the two arms of a pair: "equal" or "fewer" (B < A in decode,
# B <= A in prefill); None reads no log.
PROFILES = {
    "bias-add": (BIAS_ADD_GATES, BIAS_ADD_MODELS, None),
    "bias-add-control": (BIAS_ADD_CONTROL_GATES, BIAS_ADD_CONTROL_MODELS, "equal"),
    "rope": (BIAS_ADD_GATES, BIAS_ADD_MODELS, "fewer"),
}
BIAS_ADD_ROW_FIELDS = {
    "backend": "d3d12",
    "n_ctx": "2048",
    "n_gen_tok": "128",
    "run_index": "2",
    "host": "xbox-series-s-declared-u512-g99-noeog-greedy",
}


class Invalid(ValueError):
    pass


def load_rows(path: Path):
    if not path.is_file():
        raise Invalid(f"missing {path}")
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def ok_pairs(cell: Path, expected_count=None):
    ledger = cell / "ledger.jsonl"
    if not ledger.is_file():
        raise Invalid(f"missing {ledger}")
    status = {}
    entries = []
    for line in ledger.read_text().splitlines():
        if line.strip():
            e = json.loads(line)
            if e["status"] == "retry":
                continue
            status.setdefault(e["pair"], {})[e["arm"]] = e["status"]
            entries.append((e["pair"], e.get("order"), e["arm"], e["status"]))
    if expected_count is not None:
        expected = []
        for pair in range(1, expected_count + 1):
            order = "AB" if pair % 2 else "BA"
            expected.extend((pair, order, arm, "ok") for arm in order)
        if entries != expected:
            raise Invalid(f"expected {expected_count} complete alternating AB/BA pairs")
    return sorted(p for p, arms in status.items() if arms.get("A") == "ok" and arms.get("B") == "ok")


def submission_counts(path: Path):
    # One warm-up and one measured run, each with both phases in order.
    phases = PHASE_SUBMISSIONS.findall(path.read_text(errors="replace"))
    if [phase for phase, _ in phases] != ["prefill", "decode", "prefill", "decode"]:
        raise Invalid(f"expected warm-up and measured D3D12 phase counters in {path.name}")
    return {phase: int(count) for phase, count in phases[2:]}


def submissions_agree(check, a, b):
    if check == "equal":
        return a == b
    if check == "fewer":
        return b["decode"] < a["decode"] and b["prefill"] <= a["prefill"]
    raise Invalid(f"unknown submission check {check!r}")


def validate_profile_rows(rows, models):
    model = rows[0].get("model")
    if model not in models:
        raise Invalid(f"model {model!r} does not match the profile")
    prompt_length = rows[0].get("n_prompt_tok")
    if not prompt_length or not re.fullmatch(r"[0-9]+", prompt_length) or int(prompt_length) <= 0:
        raise Invalid("missing or invalid prompt length")
    fields = dict(BIAS_ADD_ROW_FIELDS, model=model, quant=models[model], n_prompt_tok=prompt_length)
    for row in rows:
        for field, expected in fields.items():
            if row.get(field) != expected:
                raise Invalid(f"expected {field}={expected!r}, got {row.get(field)!r}")


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
    if direction == "within":
        lower, upper = margin
        if lower <= lo and hi <= upper:
            return "pass"
        return "fail" if hi < lower or lo > upper else "inconclusive"
    if direction == "higher":
        return "pass" if lo >= margin else ("fail" if hi < margin else "inconclusive")
    return "pass" if hi <= margin else ("fail" if lo > margin else "inconclusive")


def main() -> int:
    ap = argparse.ArgumentParser(description="Paired A/B bench verdict for one campaign cell.")
    ap.add_argument("cell", type=Path, help="directory with A.csv, B.csv and ledger.jsonl")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--gates", help="comma-separated: prefill,decode,ttft,peak for #313")
    mode.add_argument("--profile", choices=tuple(PROFILES))
    mode.add_argument("--report", help="comma-separated gates, reported without a margin")
    ap.add_argument("--json-out", type=Path)
    args = ap.parse_args()
    try:
        profile_specs, profile_models, check = PROFILES[args.profile] if args.profile else (
            None, None, None)
        specs = profile_specs or GATES
        names = args.report if args.report is not None else args.gates
        gates = list(specs) if args.profile else [g.strip() for g in names.split(",") if g.strip()]
        if not gates or any(g not in specs for g in gates):
            raise Invalid(f"gates must be among {sorted(GATES)}")
        a, b = load_rows(args.cell / "A.csv"), load_rows(args.cell / "B.csv")
        pairs = ok_pairs(args.cell, BIAS_ADD_PAIRS if args.profile else None)
        if args.profile and (len(a) != BIAS_ADD_PAIRS or len(b) != BIAS_ADD_PAIRS):
            raise Invalid(f"expected {BIAS_ADD_PAIRS} measured rows per arm")
        if not pairs or len(a) < len(pairs) or len(b) < len(pairs):
            raise Invalid("no complete pair, or fewer rows than complete pairs")
        a, b = [a[p - 1] for p in pairs], [b[p - 1] for p in pairs]
        if args.profile:
            validate_profile_rows(a + b, profile_models)
        for ra, rb in zip(a, b):
            if ra.get("model") != rb.get("model") or ra.get("n_prompt_tok") != rb.get("n_prompt_tok"):
                raise Invalid("a pair mixes models or prompt lengths")
        values = {g: pair_values(a, b, specs[g][0], specs[g][1]) for g in gates}
        mismatched_pairs = []
        if check:
            for pair in pairs:
                counts = [submission_counts(args.cell / f"pair{pair}-{arm}-app.log")
                          for arm in "AB"]
                if not submissions_agree(check, *counts):
                    mismatched_pairs.append(pair)
    except (Invalid, OSError, KeyError, TypeError, json.JSONDecodeError) as e:
        print(f"❌ invalid input: {e}")
        return 2

    rng = np.random.default_rng(SEED)
    rows, verdicts = {}, []
    for g in gates:
        column, kind, margin, direction = specs[g]
        vals = values[g]
        med, lo, hi = median_interval(vals, rng)
        unit = " MiB" if kind == "delta" else ""
        if args.report is not None:
            rows[g] = {"column": column, "kind": kind, "median": med, "ci95": [lo, hi],
                       "values": vals.tolist()}
            print(f"  {g:<8} median {med:.4f}{unit}  95% CI [{lo:.4f}, {hi:.4f}]")
            continue
        v = judge(direction, margin, lo, hi)
        verdicts.append(v)
        rows[g] = {"column": column, "kind": kind, "median": med, "ci95": [lo, hi],
                   "margin": margin, "direction": direction, "verdict": v,
                   "values": vals.tolist()}
        if direction == "within":
            limit = f"[{margin[0]:g}, {margin[1]:g}]"
        else:
            limit = f"{'>=' if direction == 'higher' else '<='} {margin:g}{unit}"
        print(f"  {g:<8} median {med:.4f}{unit}  95% CI [{lo:.4f}, {hi:.4f}]  margin {limit}  {v}")
    if check:
        verdict = "fail" if mismatched_pairs else "pass"
        verdicts.append(verdict)
        rows["submissions"] = {"mismatched_pairs": mismatched_pairs, "verdict": verdict}
        print(f"  submissions {verdict}, mismatched pairs {mismatched_pairs}")
    if args.report is not None:
        cell = "report"
    else:
        cell = "pass" if all(v == "pass" for v in verdicts) else (
            "fail" if "fail" in verdicts else "inconclusive")
    summary = {"cell": str(args.cell), "model": a[0]["model"], "n_prompt_tok": a[0]["n_prompt_tok"],
               "pairs": len(pairs), "seed": SEED, "resamples": RESAMPLES, "gates": rows,
               "verdict": cell}
    if args.profile:  # keep the --gates output as it was; a profile cell names its profile
        summary["profile"] = args.profile
    if args.json_out:
        args.json_out.write_text(json.dumps(summary, indent=1) + "\n")
    print(f"{a[0]['model']} P={a[0]['n_prompt_tok']} pairs={len(pairs)}: {cell.upper()}")
    return 0 if cell in ("pass", "report") else 1


if __name__ == "__main__":
    sys.exit(main())
