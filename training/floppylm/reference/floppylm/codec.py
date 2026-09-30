"""Weight codecs: how a weight tensor becomes symbols + scales on the floppy, and back.

A codec is the single source of truth for a quantized tensor: the forward pass uses `weight()`,
`pack` stores `encode()`, `unpack` restores `decode()`, and `nominal_bits()` drives the shape
solver. Vector codecs (E1) implement the same interface and may carry a shared section.

Scale semantics (roadmap S9):
- every stored scale is finite, >= 0 and <= FP16_MAX; a non-finite weight raises CodecError;
- a row whose weights are all zero gets scale 0 and reconstructs to exactly zero in every
  format (even-level grids have no zero level, so the scale carries it);
- a positive scale below FP16_MIN is stored as FP16_MIN;
- `row8log` stores an fp16 base plus one byte per row: code 0 = scale 0, codes 1..255 =
  base * 2**((code - 128) / 16), clamped to that range.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

FP16_MIN = 6.103515625e-05  # smallest normal fp16
FP16_MAX = 65504.0
SCALE_POLICIES = ("row16", "row8log", "tensor16")
LOG_STEPS_PER_OCTAVE = 16


class CodecError(ValueError):
    """Invalid weights or invalid stored scales."""


def _to_fp16(s: torch.Tensor) -> torch.Tensor:
    """Round non-negative scales to stored fp16 values under the documented rules."""
    if not torch.isfinite(s).all() or (s < 0).any():
        raise CodecError("scale must be finite and non-negative")
    if (s > FP16_MAX).any():
        raise CodecError(f"scale exceeds fp16 max ({float(s.max()):.3g})")
    s = torch.where(s > 0, s.clamp_min(FP16_MIN), s)
    return s.half().float()


@dataclass(frozen=True)
class ScalarCodec:
    """Uniform scalar quantizer with stored scales.

    ternary: q = sign(w) * [|w| > delta * mean|w|], scale = mean|w| over kept entries.
    even levels: midrise grid (±0.5, ±1.5, ...) * s, s = mult * mean|w|.
    """

    name: str
    levels: int
    mult: float = 1.0
    delta: float = 0.5
    policy: str = "row16"

    def __post_init__(self) -> None:
        if self.policy not in SCALE_POLICIES or self.levels < 2:
            raise CodecError(f"invalid codec {self}")

    @property
    def half(self) -> float:
        return (self.levels - 1) / 2

    @property
    def symbol_bits(self) -> float:
        return math.log2(self.levels)

    # -- scales ---------------------------------------------------------------------------

    def _row8log_codes(self, s: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        base = _to_fp16(s.max().reshape(1))[0]
        if base == 0:
            return base, torch.zeros_like(s, dtype=torch.uint8)
        code = torch.round(torch.log2(s.clamp_min(1e-30) / base) * LOG_STEPS_PER_OCTAVE) + 128
        code = torch.where(s > 0, code.clamp(1, 255), torch.zeros_like(code))
        return base, code.to(torch.uint8)

    @staticmethod
    def _row8log_values(base: torch.Tensor, code: torch.Tensor) -> torch.Tensor:
        v = base * torch.exp2((code.float() - 128) / LOG_STEPS_PER_OCTAVE)
        return torch.where(code > 0, v, torch.zeros_like(v))

    def store_scale(self, s: torch.Tensor) -> torch.Tensor:
        """Map raw per-row scales (rows, 1) to exactly the values the policy stores."""
        if self.policy == "row16":
            return _to_fp16(s)
        if self.policy == "tensor16":
            return _to_fp16(s[:1]).expand_as(s).clone()
        base, code = self._row8log_codes(s)
        return self._row8log_values(base, code)

    def scale_bits(self, rows: int) -> int:
        return {"row16": 16 * rows, "tensor16": 16, "row8log": 16 + 8 * rows}[self.policy]

    def scale_nbytes(self, rows: int) -> int:
        return self.scale_bits(rows) // 8

    def scale_to_bytes(self, s: torch.Tensor) -> bytes:
        s = s.reshape(-1)
        if self.policy == "row16":
            return s.half().numpy().astype("<f2").tobytes()
        if self.policy == "tensor16":
            return s[:1].half().numpy().astype("<f2").tobytes()
        base, code = self._row8log_codes(s)
        return base.half().numpy().astype("<f2").tobytes() + code.numpy().tobytes()

    def scale_from_bytes(self, b: bytes, rows: int) -> torch.Tensor:
        if len(b) != self.scale_nbytes(rows):
            raise CodecError("scale section has the wrong size")
        if self.policy == "row16":
            v = torch.from_numpy(np.frombuffer(b, "<f2", rows).astype(np.float32))
        elif self.policy == "tensor16":
            v = torch.full((rows,), float(np.frombuffer(b, "<f2", 1)[0]))
        else:
            base = torch.tensor(float(np.frombuffer(b, "<f2", 1)[0]))
            code = torch.from_numpy(np.frombuffer(b, np.uint8, rows, 2).copy())
            v = self._row8log_values(base, code)
        if not torch.isfinite(v).all() or (v < 0).any():
            raise CodecError("stored scale is not finite and non-negative")
        return v.view(rows, 1)

    # -- quantization ---------------------------------------------------------------------

    def _mean_abs(self, a: torch.Tensor) -> torch.Tensor:
        m = a.mean(1, keepdim=True)
        return a.mean().expand_as(m) if self.policy == "tensor16" else m

    def _raw_scale(self, w: torch.Tensor) -> torch.Tensor:
        a = w.abs()
        m = self._mean_abs(a)
        if self.levels == 3:
            keep = a > self.delta * m
            kept, cnt = (a * keep).sum(1, keepdim=True), keep.sum(1, keepdim=True)
            if self.policy == "tensor16":
                kept, cnt = kept.sum().expand_as(kept), cnt.sum().expand_as(cnt)
            return torch.where(cnt > 0, kept / cnt.clamp_min(1), torch.zeros_like(kept))
        return self.mult * m

    def quantize(self, w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Symbols in [0, levels) and the stored per-row scale (rows, 1), float32."""
        w = w.detach().float()
        if not torch.isfinite(w).all():
            raise CodecError("weights contain NaN or Inf")
        s = self.store_scale(self._raw_scale(w))
        if self.levels == 3:
            q = torch.sign(w) * (w.abs() > self.delta * self._mean_abs(w.abs()))
        else:
            x = torch.where(
                s > 0, w / torch.where(s > 0, s, torch.ones_like(s)), torch.zeros_like(w)
            )
            if self.levels % 2:
                q = torch.round(x).clamp(-self.half, self.half)
            else:
                q = (torch.floor(x) + 0.5).clamp(-self.half, self.half)
        return (q + self.half).round().to(torch.int64), s

    def dequant(self, sym: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        return (sym.float() - self.half) * s

    def weight(self, w: torch.Tensor) -> torch.Tensor:
        """Straight-through fake quantization used in the forward pass."""
        sym, s = self.quantize(w)
        return self.dequant(sym, s) + (w - w.detach())  # exact value, identity gradient

    # -- storage --------------------------------------------------------------------------

    def encode(self, w: torch.Tensor) -> tuple[np.ndarray, bytes]:
        sym, s = self.quantize(w)
        return sym.flatten().numpy(), self.scale_to_bytes(s)

    def decode(self, sym: np.ndarray, scale: bytes, shape: tuple[int, int]) -> torch.Tensor:
        n = shape[0] * shape[1]
        if sym.size != n or (n and (sym.min() < 0 or sym.max() >= self.levels)):
            raise CodecError("symbol stream does not match the tensor")
        s = self.scale_from_bytes(scale, shape[0])
        return self.dequant(torch.from_numpy(np.ascontiguousarray(sym)).view(shape), s)

    def nominal_bits(self, shape: tuple[int, int]) -> float:
        return shape[0] * shape[1] * self.symbol_bits + self.scale_bits(shape[0])

    def shared_state(self) -> bytes:
        return b""


def scalar(name: str, policy: str = "row16", delta: float = 0.5) -> ScalarCodec:
    if name == "ternary":
        return ScalarCodec("ternary", 3, delta=delta, policy=policy)
    if name == "2bit":
        return ScalarCodec("2bit", 4, mult=1.135, policy=policy)  # Lloyd-Max step, Gaussian
    if name == "4bit":
        return ScalarCodec("4bit", 16, mult=0.42, policy=policy)
    raise CodecError(f"unknown format {name!r}")
