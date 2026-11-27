#!/usr/bin/env python3
# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
"""Run the mandatory native CPU smoke against a frozen independent reference."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = ROOT / "tests/golden/cpu-smoke.json"
GOLDEN_SHA256 = "716df46e1214a027cc3989cf72baecca62c3f20cf2a16d287f2619d2f893ca5e"
MAX_ABS = 0.006
MAX_NMSE = 1e-5


def judge(actual, golden):
    for key in ("prompt_ids", "generated_ids", "positions"):
        if actual[key] != golden[key]:
            raise ValueError(f"{key}: expected {golden[key]}, got {actual[key]}")
    ref = np.asarray(golden["rows"], dtype=np.float64)
    rows = np.asarray(actual["rows"], dtype=np.float64)
    if rows.shape != ref.shape or ref.shape != (5, 19):
        raise ValueError(f"incomplete capture: got {rows.shape}, reference {ref.shape}")
    if not (np.isfinite(rows).all() and np.isfinite(ref).all()):
        raise ValueError("nonfinite logit")
    for pos, row, expected in zip(actual["positions"], rows, ref):
        diff = row - expected
        maximum = np.abs(diff).max()
        nmse = (diff @ diff) / (expected @ expected)
        print(f"position={pos} max_abs={maximum:.6g} nmse={nmse:.6g} top1={row.argmax()}")
        if maximum > MAX_ABS or nmse > MAX_NMSE or row.argmax() != expected.argmax():
            raise ValueError(f"logit mismatch at position {pos}")
    keys = ("generated_ids", "continuation_ids", "reset_ids")
    if len(actual["turns"]) != len(keys):
        raise ValueError("missing session turns")
    for turn, key, prompt_count in zip(actual["turns"], keys, (5, 2, 3)):
        expected = "".join(chr(ord("a") + i - 3) for i in golden[key])
        if turn["text"] != expected or turn["n_prompt"] != prompt_count:
            raise ValueError(f"{key}: expected {expected!r} for {prompt_count} tokens, got {turn}")
    if actual["turns"][0]["n_p_eval"] != 5 or actual["turns"][1]["n_p_eval"] != 2:
        raise ValueError("incorrect prefill count on first or continuation turn")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    subprocess.run([sys.executable, ROOT / "scripts/cpu-smoke-fixture.py", args.model], check=True)
    data = GOLDEN.read_bytes()
    if hashlib.sha256(data).hexdigest() != GOLDEN_SHA256:
        raise ValueError("unexpected CPU smoke golden digest")
    print(f"CPU reference {GOLDEN.name} sha256={GOLDEN_SHA256}", flush=True)
    with tempfile.TemporaryDirectory() as tmp:
        result = Path(tmp) / "result.json"
        subprocess.run([args.build_dir / "tests/xllama-cpu-smoke", args.model, result], check=True)
        judge(json.loads(result.read_bytes()), json.loads(data))
    print("PASS native CPU prefill, four incremental decode rows and three session turns")


if __name__ == "__main__":
    main()
