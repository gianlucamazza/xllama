# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
"""Numerical comparison for validated float32 vocabulary rows."""

from dataclasses import dataclass
import math

import numpy as np


@dataclass(frozen=True)
class LogitComparison:
    max_abs_diff: float
    nmse: float
    top1_match: bool
    top_k_overlap: float

    def passes(self, max_abs_diff: float, nmse: float, top_k_min: float) -> bool:
        return (
            math.isfinite(self.max_abs_diff)
            and self.max_abs_diff <= max_abs_diff
            and math.isfinite(self.nmse)
            and self.nmse <= nmse
            and self.top1_match
            and self.top_k_overlap >= top_k_min
        )


def compare_logits(reference: np.ndarray, candidate: np.ndarray, top_k: int) -> LogitComparison:
    """Compare two equally sized, nonempty, finite rows with top_k > 0."""
    wide_reference = reference.astype(np.float64)
    diff = wide_reference - candidate
    max_abs = float(np.max(np.abs(diff)))
    reference_energy = float(np.dot(wide_reference, wide_reference))
    error_energy = float(np.dot(diff, diff))
    if reference_energy == 0:
        nmse = 0.0 if error_energy == 0 else math.inf
    else:
        nmse = error_energy / reference_energy

    k = min(top_k, reference.size)
    top_reference = set(np.argsort(reference)[-k:].tolist())
    top_candidate = set(np.argsort(candidate)[-k:].tolist())
    return LogitComparison(
        max_abs_diff=max_abs,
        nmse=nmse,
        top1_match=bool(np.argmax(reference) == np.argmax(candidate)),
        top_k_overlap=len(top_reference & top_candidate) / k,
    )
