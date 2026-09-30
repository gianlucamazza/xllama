"""Tiny GPT whose every stored number goes through a codec or an fp16 round trip."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .codec import SCALE_POLICIES, ScalarCodec, scalar

MLPS = ("gelu", "relu2", "swiglu")
FORMATS = ("ternary", "2bit", "4bit")


@dataclass(frozen=True)
class GPTConfig:
    vocab: int = 256
    d: int = 96
    n_layers: int = 3
    n_heads: int = 6
    d_ff: int = 384
    mlp: str = "gelu"  # gelu | relu2 | swiglu
    core_fmt: str = "ternary"
    emb_fmt: str = "4bit"
    scale_policy: str = "row16"
    delta: float = 0.5
    qk_norm: bool = False
    ctx: int = 256

    def __post_init__(self) -> None:
        ints = ("vocab", "d", "n_layers", "n_heads", "d_ff", "ctx")
        if not all(isinstance(getattr(self, k), int) and getattr(self, k) > 0 for k in ints):
            raise ValueError(f"config sizes must be positive ints: {self}")
        if self.d % self.n_heads or (self.d // self.n_heads) % 2:
            raise ValueError("d must split into heads of even size")
        if self.mlp not in MLPS or self.core_fmt not in FORMATS or self.emb_fmt not in FORMATS:
            raise ValueError(f"unknown mlp or format: {self}")
        if self.scale_policy not in SCALE_POLICIES:
            raise ValueError(f"unknown scale policy: {self}")
        if not (0 <= self.delta < 4) or not isinstance(self.qk_norm, bool):
            raise ValueError(f"invalid delta or qk_norm: {self}")
        if self.vocab > 65535 or self.ctx > 1 << 16 or self.n_layers > 255 or self.d > 65535:
            raise ValueError(f"config out of serializable range: {self}")

    def to_dict(self) -> dict:
        return asdict(self)

    def tensor_shapes(self) -> list[tuple[str, tuple[int, ...]]]:
        """Stored tensors in serialization order: ("q", (rows, cols)) or ("n", (d,))."""
        d, f = self.d, self.d_ff
        up = 2 * f if self.mlp == "swiglu" else f
        block = [
            ("n", (d,)),
            ("q", (3 * d, d)),
            ("q", (d, d)),
            ("n", (d,)),
            ("q", (up, d)),
            ("q", (d, f)),
        ]
        return [("q", (self.vocab, d))] + block * self.n_layers + [("n", (d,))]

    def codecs(self) -> tuple[ScalarCodec, ScalarCodec]:
        """(embedding codec, core codec)."""
        emb = scalar(self.emb_fmt, self.scale_policy)
        return emb, scalar(self.core_fmt, self.scale_policy, self.delta)

    def nominal_bits(self) -> float:
        emb, core = self.codecs()
        bits = 0.0
        for i, (kind, shape) in enumerate(self.tensor_shapes()):
            if kind == "n":
                bits += 16 * shape[0]
            else:
                bits += (emb if i == 0 else core).nominal_bits(shape)
        return bits


def _fp16(w: torch.Tensor) -> torch.Tensor:
    return w.detach().half().float() + (w - w.detach())


class QLinear(nn.Module):
    """Linear layer whose weight is always the codec's reconstruction (frozen after unpack)."""

    def __init__(self, d_in: int, d_out: int, codec: ScalarCodec, std: float) -> None:
        super().__init__()
        self.codec = codec
        self.frozen = False
        self.canonical: bytes | None = None  # exact stored record after unpack; re-saved verbatim
        self.weight = nn.Parameter(torch.randn(d_out, d_in) * std)

    def qweight(self) -> torch.Tensor:
        return self.weight if self.frozen else self.codec.weight(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.qweight())


class RMSNorm(nn.Module):
    def __init__(self, d: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(x, (x.size(-1),)) * _fp16(self.weight)


class Rope(nn.Module):
    def __init__(self, head_dim: int, ctx: int) -> None:
        super().__init__()
        inv = 1.0 / (10000 ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        ang = torch.arange(ctx, dtype=torch.float32)[:, None] * inv[None]
        self.register_buffer("cos", ang.cos(), persistent=False)
        self.register_buffer("sin", ang.sin(), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.size(-2)
        cos, sin = self.cos[:t], self.sin[:t]
        x1, x2 = x[..., ::2], x[..., 1::2]
        return torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), -1).flatten(-2)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig, codec: ScalarCodec, rope: Rope) -> None:
        super().__init__()
        d, f = cfg.d, cfg.d_ff
        out_std = 1 / math.sqrt(2 * cfg.n_layers)
        self.cfg, self.rope = cfg, rope
        self.norm1, self.norm2 = RMSNorm(d), RMSNorm(d)
        self.qkv = QLinear(d, 3 * d, codec, 1 / math.sqrt(d))
        self.proj = QLinear(d, d, codec, out_std / math.sqrt(d))
        up = 2 * f if cfg.mlp == "swiglu" else f
        self.fc = QLinear(d, up, codec, 1 / math.sqrt(d))
        self.fc2 = QLinear(f, d, codec, out_std / math.sqrt(f))

    def _mlp(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x)
        if self.cfg.mlp == "swiglu":
            a, b = h.chunk(2, -1)
            h = F.silu(a) * b
        elif self.cfg.mlp == "relu2":
            h = F.relu(h).square()
        else:
            h = F.gelu(h)
        return self.fc2(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.norm1(x)).split(d, -1)
        q, k, v = (z.view(b, t, self.cfg.n_heads, -1).transpose(1, 2) for z in (q, k, v))
        if self.cfg.qk_norm:
            q, k = F.rms_norm(q, (q.size(-1),)), F.rms_norm(k, (k.size(-1),))
        a = F.scaled_dot_product_attention(self.rope(q), self.rope(k), v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(b, t, d))
        return x + self._mlp(self.norm2(x))


class TinyGPT(nn.Module):
    def __init__(self, cfg: GPTConfig) -> None:
        super().__init__()
        assert cfg.d % cfg.n_heads == 0
        self.cfg = cfg
        emb, core = cfg.codecs()
        rope = Rope(cfg.d // cfg.n_heads, cfg.ctx)
        self.emb = QLinear(cfg.d, cfg.vocab, emb, 1 / math.sqrt(cfg.d))  # (vocab, d), tied
        self.blocks = nn.ModuleList(Block(cfg, core, rope) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.d)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        w = self.emb.qweight()
        x = F.embedding(idx, w)
        for blk in self.blocks:
            x = blk(x)
        return F.linear(self.norm(x), w)

    def train(self, mode: bool = True):
        if mode and hasattr(self, "_artifact_state"):
            raise RuntimeError("Loaded FLP2 models are inference-only; resume a trunk checkpoint")
        return super().train(mode)

    def stored_tensors(self) -> list[QLinear | RMSNorm]:
        """Fixed serialization order."""
        out: list[QLinear | RMSNorm] = [self.emb]
        for b in self.blocks:
            out += [b.norm1, b.qkv, b.proj, b.norm2, b.fc, b.fc2]
        return out + [self.norm]

    def nominal_bits(self) -> float:
        return self.cfg.nominal_bits()

    def stored_params(self) -> int:
        return sum(t.weight.numel() for t in self.stored_tensors() if isinstance(t, QLinear))

    def core_params(self) -> int:
        return self.stored_params() - self.emb.weight.numel()

    def flops_per_token(self) -> float:
        """Forward FLOPs per token: linear layers + attention scores/values (causal ~ctx/2)."""
        c = self.cfg
        attn = 2 * 2 * c.n_layers * (c.ctx / 2) * c.d
        return 2 * self.stored_params() + attn
