"""Byte parity and paired statistics for verdicts (ADR 0005 §1, §7)."""

from __future__ import annotations

import statistics

TOLERANCE = 0.01


def admissible(
    byte_counts: dict[str, int], target_bytes: float, tol: float = TOLERANCE
) -> tuple[bool, list[str]]:
    """Serialized sizes must each be within ±tol of the target AND within tol of each other."""
    reasons = []
    for name, b in byte_counts.items():
        if abs(b / target_bytes - 1) > tol:
            reasons.append(f"{name}: {b} B is {100 * (b / target_bytes - 1):+.2f}% from target")
    if byte_counts:
        hi, lo = max(byte_counts.values()), min(byte_counts.values())
        if hi / lo - 1 > tol:
            reasons.append(f"spread max/min - 1 = {100 * (hi / lo - 1):.2f}% > {100 * tol:.0f}%")
    return not reasons, reasons


def paired_sigma(a: dict[int, float], b: dict[int, float]) -> dict:
    """Mean and sample sd of per-seed differences a - b over the seeds both conditions share."""
    seeds = sorted(set(a) & set(b))
    if len(seeds) < 2 or set(a) != set(b):
        raise ValueError("paired sigma needs the same >= 2 seeds in both conditions")
    diffs = [a[s] - b[s] for s in seeds]
    return {
        "seeds": seeds,
        "diffs": diffs,
        "mean": statistics.fmean(diffs),
        "sd": statistics.stdev(diffs),
        "n": len(diffs),
    }
