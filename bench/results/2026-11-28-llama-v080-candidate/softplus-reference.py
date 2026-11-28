#!/usr/bin/env python3
# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
"""Compare captured Qwen3.5 layer-1 softplus values with a decimal reference."""

import argparse
import base64
from decimal import Decimal, localcontext
import json
from pathlib import Path

import numpy as np


def read_node(directory, name):
    records = [json.loads(line) for line in (directory / "manifest.jsonl").read_text().splitlines()]
    matches = [r for r in records if r["eval"] == 0 and r["name"] == name]
    if (len(matches) != 1 or matches[0].get("ne") != [16, 7, 1, 1]
            or matches[0].get("type") != "f32"):
        raise ValueError(f"{directory}: expected one 16 x 7 float32 {name} capture")
    values = np.fromfile(directory / matches[0]["file"], dtype="<f4").reshape(7, 16)
    if not np.isfinite(values).all():
        raise ValueError(f"{directory}: nonfinite values in {name}")
    return values


def nearest_float32(value):
    center = np.float32(float(value))
    neighbors = (np.nextafter(center, np.float32(-np.inf)), center,
                 np.nextafter(center, np.float32(np.inf)))
    # Compare in decimal precision and resolve an exact midpoint by ties-to-even.
    def rounding_key(candidate):
        distance = abs(value - Decimal.from_float(float(candidate)))
        return distance, int(candidate.view(np.uint32)) & 1

    return min(neighbors, key=rounding_key)


def compare(actual, reference, rounded):
    values = actual.ravel()
    if np.any(values < 0):
        raise ValueError("softplus values must be nonnegative")
    ulps = np.abs(values.view("<i4").astype(np.int64) - rounded.view("<i4").astype(np.int64))
    errors = values.astype(np.float64) - reference
    return {
        "exact_rounding_count": int(np.count_nonzero(ulps == 0)),
        "max_ulp": int(ulps.max()),
        "max_abs": float(np.abs(errors).max()),
        "rms": float(np.sqrt(np.mean(errors**2))),
    }


def from_captures(model, baseline_capture, candidate_capture):
    from gguf import GGUFReader

    baseline = read_node(baseline_capture, "alpha-1")
    candidate = read_node(candidate_capture, "alpha-1")
    if not np.array_equal(baseline, candidate):
        raise ValueError("the two captures have different layer-1 inputs")
    weights = GGUFReader(model)
    bias = next(t for t in weights.tensors if t.name == "blk.1.ssm_dt.bias")
    if bias.tensor_type.name != "F32" or bias.data.shape != (16,):
        raise ValueError("expected 16 float32 bias values from the Qwen3.5 checkpoint")
    return baseline, bias.data, {
        "baseline": read_node(baseline_capture, "a_softplus-1"),
        "candidate": read_node(candidate_capture, "a_softplus-1"),
    }


def from_receipt():
    receipt = json.loads(Path(__file__).with_name("catalogue.json").read_text())
    witness = receipt["qwen35_controls"]["softplus"]["layer1_witness"]

    def unpack(name, shape):
        raw = base64.b64decode(witness[name], validate=True)
        values = np.frombuffer(raw, dtype="<f4").reshape(shape)
        if not np.isfinite(values).all():
            raise ValueError(f"nonfinite {name} in the receipt")
        return values

    return unpack("alpha", (7, 16)), unpack("checkpoint_bias", (16,)), {
        "baseline": unpack("baseline_output", (7, 16)),
        "candidate": unpack("candidate_output", (7, 16)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="*", type=Path,
        help="MODEL BASELINE_CAPTURE CANDIDATE_CAPTURE, or omit to replay the archived values",
    )
    args = parser.parse_args()
    if len(args.paths) not in (0, 3):
        parser.error("provide the model and both capture directories, or omit all three")
    alpha, bias, outputs = from_captures(*args.paths) if args.paths else from_receipt()
    biased = np.add(alpha, bias, dtype=np.float32)
    with localcontext() as context:
        context.prec = 65
        values = [(Decimal(1) + Decimal.from_float(float(x)).exp()).ln() for x in biased.ravel()]
        rounded = np.array([nearest_float32(x) for x in values], dtype="<f4")
    reference = np.array(values, dtype=np.float64)
    for name, output in outputs.items():
        print(name, json.dumps(compare(output, reference, rounded)))


if __name__ == "__main__":
    main()
