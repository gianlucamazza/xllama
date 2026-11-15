#!/usr/bin/env python3
# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
#
# Graph-order intermediate tensor comparison for the #420 control diagnosis.
#
# Compares two capture directories written by `xllama-cli --capture-tensors`
# (or the console `tensor-capture.txt` knob) with the same prompt, model and
# capture spec. Records are matched by (eval, name, occurrence of that name in
# the eval) and visited in the reference's evaluation order, which is graph
# dependency order.
#
# Usage:
#   scripts/compare-tensors.py <ref_dir> <cand_dir> [--floor <alt_dir>]
#                              [--nmse F] [--floor-ratio F] [--json-out PATH]
#
# Per record: max_abs, nmse = ||a-b||^2/||a||^2 and cosine similarity.
# --floor names a second reference-class capture (for example the same CPU
# arm built with another compiler or kernel layout). Its per-record nmse
# against <ref_dir> is the drift that two accepted CPU paths already show;
# a candidate record "diverges" when its nmse exceeds --floor-ratio times
# that floor (and an absolute --nmse floor, so near-zero floors do not flag
# noise). Without --floor only the absolute --nmse threshold is used.
#
# These thresholds are diagnostic only: they locate where error grows. They
# are not the frozen logit gate (scripts/compare-logits.py), do not change
# its verdict and do not authorize any performance trial.
#
# Exit 0 = no divergent record, 1 = divergent record found,
#      2 = invalid input (missing/malformed manifest, mismatched record
#          sets or shapes, non-finite values, bad arguments).

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np


class CaptureError(ValueError):
    pass


def load_capture(path: Path):
    """Return (ordered list of keys, {key: (meta, values)}, skipped list)."""
    manifest = path / "manifest.jsonl"
    if not manifest.is_file():
        raise CaptureError(f"missing manifest: {manifest}")
    order, records, skipped, seen = [], {}, [], {}
    for lineno, line in enumerate(manifest.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            meta = json.loads(line)
        except json.JSONDecodeError as e:
            raise CaptureError(f"{manifest}:{lineno}: bad JSON: {e}") from None
        if not isinstance(meta, dict) or "name" not in meta or "eval" not in meta:
            raise CaptureError(f"{manifest}:{lineno}: record needs name and eval")
        # Graph names are not unique (Qcur-N names several nodes), so a
        # record is identified by its occurrence within the evaluation.
        base = (meta["eval"], meta["name"])
        occ = seen.get(base, 0)
        seen[base] = occ + 1
        key = (meta["eval"], meta["name"], occ)
        if "skipped" in meta:
            skipped.append(key)
            continue
        for field in ("file", "values", "ne"):
            if field not in meta:
                raise CaptureError(f"{manifest}:{lineno}: missing {field}")
        f = path / meta["file"]
        if f.resolve().parent != path.resolve():
            raise CaptureError(f"{manifest}:{lineno}: file escapes capture dir")
        raw = f.read_bytes() if f.is_file() else b""
        if not raw or len(raw) % 4:
            raise CaptureError(f"incomplete or missing float32 data: {f}")
        values = np.frombuffer(raw, dtype=np.float32)
        if values.size != meta["values"]:
            raise CaptureError(f"{f}: {values.size} values, manifest says {meta['values']}")
        if not np.all(np.isfinite(values)):
            raise CaptureError(f"non-finite values: {f}")
        order.append(key)
        records[key] = (meta, values)
    if not records:
        raise CaptureError(f"no records in {manifest}")
    return order, records, skipped


def metrics(a: np.ndarray, b: np.ndarray):
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    diff = a64 - b64
    denom = float(np.dot(a64, a64))
    nb = float(np.dot(b64, b64))
    nmse = float(np.dot(diff, diff) / denom) if denom > 0 else (0.0 if nb == 0 else math.inf)
    cos = float(np.dot(a64, b64) / math.sqrt(denom * nb)) if denom > 0 and nb > 0 else 1.0
    return {"max_abs": float(np.max(np.abs(diff))), "nmse": nmse, "cosine": cos}


def check_pair(ref, cand, label):
    r_order, r_rec, r_skip = ref
    c_order, c_rec, c_skip = cand
    if set(r_rec) != set(c_rec):
        only_r = sorted(set(r_rec) - set(c_rec))[:5]
        only_c = sorted(set(c_rec) - set(r_rec))[:5]
        raise CaptureError(f"{label}: record sets differ (ref-only {only_r}, other-only {only_c})")
    if sorted(r_skip) != sorted(c_skip):
        raise CaptureError(f"{label}: skipped records differ")
    for key in r_order:
        ma, mb = r_rec[key][0], c_rec[key][0]
        if list(ma["ne"]) != list(mb["ne"]) or ma.get("mode") != mb.get("mode"):
            raise CaptureError(f"{label}: shape/mode mismatch at {key}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Compare two intermediate tensor captures.")
    ap.add_argument("ref", type=Path, help="reference capture directory (e.g. console CPU)")
    ap.add_argument("cand", type=Path, help="candidate capture directory (e.g. console GPU)")
    ap.add_argument("--floor", type=Path, help="second reference-class capture for the drift floor")
    ap.add_argument("--nmse", type=float, default=1e-4,
                    help="absolute diagnostic nmse threshold per record (default 1e-4)")
    ap.add_argument("--floor-ratio", type=float, default=4.0,
                    help="with --floor: flag nmse above ratio x floor nmse (default 4)")
    ap.add_argument("--json-out", type=Path, help="write the per-record table as JSON")
    args = ap.parse_args()
    if (not math.isfinite(args.nmse) or args.nmse < 0
            or not math.isfinite(args.floor_ratio) or args.floor_ratio < 1):
        ap.error("--nmse must be finite and >= 0; --floor-ratio finite and >= 1")

    try:
        ref = load_capture(args.ref)
        cand = load_capture(args.cand)
        check_pair(ref, cand, "candidate")
        floor = None
        if args.floor:
            floor = load_capture(args.floor)
            check_pair(ref, floor, "floor")
    except (OSError, CaptureError) as e:
        print(f"❌ invalid input: {e}")
        return 2

    rows, first = [], None
    for key in ref[0]:
        meta, a = ref[1][key]
        m = metrics(a, cand[1][key][1])
        row = {"eval": key[0], "name": key[1], "occ": key[2], "op": meta.get("op"), **m}
        limit = args.nmse
        if floor is not None:
            f = metrics(a, floor[1][key][1])
            row["floor_nmse"] = f["nmse"]
            limit = max(args.nmse, args.floor_ratio * f["nmse"])
        row["limit"] = limit
        row["divergent"] = m["nmse"] > limit
        if row["divergent"] and first is None:
            first = row
        rows.append(row)

    print(f"{'eval':>4} {'name':<22} {'op':<10} {'nmse':>10} {'floor':>10} {'max_abs':>10} {'cos':>9}")
    for r in rows:
        mark = "  <-- first divergence" if r is first else ("  !" if r["divergent"] else "")
        fl = f"{r['floor_nmse']:.2e}" if "floor_nmse" in r else "-"
        name = r["name"] + (f"#{r['occ']}" if r["occ"] else "")
        print(f"{r['eval']:>4} {name:<22} {str(r['op']):<10} {r['nmse']:>10.2e} {fl:>10} "
              f"{r['max_abs']:>10.4f} {r['cosine']:>9.6f}{mark}")

    summary = {
        "ref": str(args.ref), "cand": str(args.cand),
        "floor": str(args.floor) if args.floor else None,
        "nmse_threshold": args.nmse, "floor_ratio": args.floor_ratio if args.floor else None,
        "records": len(rows), "divergent": sum(r["divergent"] for r in rows),
        "first_divergence": first, "scope": "diagnostic; not the frozen logit gate",
        "rows": rows,
    }
    if args.json_out:
        args.json_out.write_text(json.dumps(summary, indent=1) + "\n")
    if first is None:
        print("✅ no divergent record")
        return 0
    print(f"❌ first divergence: eval {first['eval']} {first['name']}#{first['occ']} ({first['op']}) "
          f"nmse {first['nmse']:.3e} > {first['limit']:.3e}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
