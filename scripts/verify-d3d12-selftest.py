#!/usr/bin/env python3
# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
"""Verify a d3d12be-result.csv against the requested console selftest profile."""

import argparse
import csv
import math
import re
import sys
from datetime import datetime
from pathlib import Path

FIELDS = "type n k ncols rel_err gpu_ms packed_gbs ok d3d12_ran host date error".split()
# run_d3d12_selftest's producer contract. The CSV type includes the operation
# (bare quant names are MUL_MAT) and, for RoPE, the mode and staged variant.
DECODE = {
    "q4_0": [(8192, 2048), (2048, 8192)],
    "q4_k": [(11008, 2048), (2048, 11008)],
    "q6_k": [(2048, 11008), (65536, 1024)],
    "q5_k": [(11008, 2048), (2048, 11008)],
    "iq4_xs": [(2048, 2048)],
    "iq3_xxs": [(2048, 2048), (3, 256)],
}
BIAS = ("add_f32", 129, 0, 2)
ROPE = [(mode + suffix, 128, heads, cols) for mode in ("rope_neox", "rope_norm")
        for suffix, heads, cols in (("", 16, 1), ("", 2, 1), ("", 2, 512),
                                    ("_staged", 16, 7), ("", 4, 3))]
UNAVAILABLE = re.compile(r"^\[xllama\] d3d12: optional (q5_k|iq4_xs|iq3_xxs) "
                         r"t(?:64|128) pipeline unavailable: .+ — type stays on CPU$", re.M)


def inventory(unavailable, bias, rope):
    expected = {("get_rows_q6_k", 151936, 2048, 6), ("get_rows_q6_k", 65536, 1024, 6)}
    for quant, shapes in DECODE.items():
        if quant in unavailable:
            continue
        expected.update((quant, n, k, 1) for n, k in shapes)
        for cols in ((7,) if quant == "iq3_xxs" else (7, 512)):
            expected.add((quant, 1024, 1024, cols))
    if bias:
        expected.add(BIAS)
    if rope:
        expected.update(ROPE)
    return expected


def verify(stream, log="", bias=False, rope=False, allow_unavailable=False):
    unavailable = set(UNAVAILABLE.findall(log)) if allow_unavailable else set()
    expected = inventory(unavailable, bias, rope)
    reader = csv.DictReader(stream, strict=True)
    if sorted(reader.fieldnames or []) != sorted(FIELDS):
        raise ValueError("missing, duplicate or unexpected CSV column")
    seen = set()
    ordered = []
    for row in reader:
        if None in row or any(not (row[k] or "").strip() for k in FIELDS):
            raise ValueError(f"line {reader.line_num}: incomplete or overlong row")
        # The producer (uwp/inference-bridge.cpp) labels every run xbox-series-s;
        # a Series X selftest needs the label from the device before this can
        # accept it (bench/README.md, hardware declaration).
        if row["host"] != "xbox-series-s":
            raise ValueError(f"line {reader.line_num}: expected xbox-series-s host")
        try:
            timestamp = datetime.strptime(row["date"], "%Y-%m-%dT%H:%M:%SZ")
        except ValueError as exc:
            raise ValueError(f"line {reader.line_num}: invalid UTC date") from exc
        if timestamp.isoformat(timespec="seconds") + "Z" != row["date"]:
            raise ValueError(f"line {reader.line_num}: noncanonical UTC date")
        typ = row["type"]
        key = (typ, *(int(row[k]) for k in ("n", "k", "ncols")))
        if key not in expected:
            raise ValueError(f"unexpected case {key}")
        if key in seen:
            raise ValueError(f"duplicate case {key}")
        seen.add(key)
        ordered.append(key)
        if row["ok"] != "1" or row["d3d12_ran"] != "1":
            raise ValueError(f"{key}: requires ok=1 and d3d12_ran=1")
        err, ms, gbs = (float(row[k]) for k in ("rel_err", "gpu_ms", "packed_gbs"))
        if not all(math.isfinite(x) and x >= 0 for x in (err, ms, gbs)):
            raise ValueError(f"{key}: invalid metric")
        # RoPE reports NMSE. ADD and GET_ROWS compare exactly.
        if key in ROPE:
            note = re.fullmatch(r"nmse=(\S+) max_abs=(\S+) bit_identical=[01]", row["error"])
            if not note:
                raise ValueError(f"{key}: missing RoPE comparison")
            nmse, max_abs = map(float, note.groups())
            if nmse != err or not math.isfinite(max_abs) or max_abs < 0:
                raise ValueError(f"{key}: invalid RoPE comparison")
            tolerance = 1e-7
        else:
            if row["error"] != "-":
                raise ValueError(f"{key}: {row['error']}")
            tolerance = 1e-5 if typ in DECODE else 0
        if err > tolerance:
            raise ValueError(f"{key}: rel_err {err} exceeds {tolerance}")
        if typ in DECODE and key[3] == 1 and key != ("iq3_xxs", 3, 256, 1):
            floor = 55 if typ == "iq3_xxs" else 100
            if gbs < floor:
                raise ValueError(f"{key}: packed_gbs {gbs} below {floor}")
    if expected - seen:
        raise ValueError(f"missing cases {sorted(expected - seen)}")
    if bias and (ordered[0] != BIAS or "[xllama] d3d12be: bias ADD cold_scratch=1" not in log.splitlines()
                 or not re.search(r"^\[xllama\] d3d12be: add_f32 .* ok=1 err=-$", log, re.M)):
        raise ValueError("first cold bias ADD success missing")
    if rope and ([key for key in ordered if key in ROPE] != ROPE
                 or "[xllama] d3d12be: rope cold_scratch=1" not in log.splitlines()):
        raise ValueError("ordered RoPE cases or cold scratch proof missing")
    return len(seen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--run-log", type=Path, help="console log slice for this run")
    parser.add_argument("--require-bias-add", action="store_true")
    parser.add_argument("--require-rope", action="store_true")
    parser.add_argument("--allow-unavailable-mmv", action="store_true",
                        help="omit optional types whose pipeline initialization failed in this run")
    args = parser.parse_args()
    try:
        if (args.allow_unavailable_mmv or args.require_bias_add or args.require_rope) and not args.run_log:
            raise ValueError("profile requires --run-log")
        log = args.run_log.read_text(errors="replace") if args.run_log else ""
        with args.csv.open(newline="") as stream:
            count = verify(stream, log, args.require_bias_add, args.require_rope, args.allow_unavailable_mmv)
    except (ValueError, OSError, csv.Error) as exc:
        print(f"GATE=FAIL ({exc})", file=sys.stderr)
        return 1
    print(f"Verified {count} distinct selftest cases\nGATE=PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
