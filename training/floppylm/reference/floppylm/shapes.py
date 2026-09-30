"""Shape solver: fill a bit budget to within tolerance by choosing d, n_layers and d_ff."""

from __future__ import annotations

from dataclasses import replace

from .model import GPTConfig

HEAD_DIM = 16


def nominal_bits(cfg: GPTConfig) -> float:
    return cfg.nominal_bits()


def fill_d_ff(
    cfg: GPTConfig, budget_bits: float, ratio: float = 1.0, step: int = 1
) -> GPTConfig | None:
    """Largest d_ff (multiple of `step`) with nominal_bits * ratio <= budget.

    `ratio` is the expected coded/nominal ratio (1.0 = no entropy-coding gain assumed).
    """
    lo, hi = step, 16 * cfg.d
    if nominal_bits(replace(cfg, d_ff=lo)) * ratio > budget_bits:
        return None
    while hi - lo > step:
        mid = (lo + hi) // 2 // step * step
        if nominal_bits(replace(cfg, d_ff=mid)) * ratio <= budget_bits:
            lo = mid
        else:
            hi = mid
    return replace(cfg, d_ff=lo)


def grid(
    base: GPTConfig,
    budget_bits: float,
    widths=range(64, 257, 16),
    layers=range(1, 17),
    ff_range=(2.0, 6.0),
    emb_share=(0.08, 0.20),
    ratio: float = 1.0,
    fill_min: float = 0.995,
) -> list[GPTConfig]:
    """All (d, n_layers) whose budget-filling d_ff lands in ff_range * d (or * 2/3 for SwiGLU)."""
    out = []
    k = 2 / 3 if base.mlp == "swiglu" else 1.0
    for d in widths:
        for n in layers:
            cfg = fill_d_ff(
                replace(base, d=d, n_layers=n, n_heads=d // HEAD_DIM), budget_bits, ratio
            )
            if cfg is None:
                break
            if not ff_range[0] * k * d <= cfg.d_ff <= ff_range[1] * k * d:
                continue
            bits = nominal_bits(cfg)
            share = cfg.vocab * d * 4 / bits
            if bits * ratio >= fill_min * budget_bits and emb_share[0] <= share <= emb_share[1]:
                out.append(cfg)
    return out
