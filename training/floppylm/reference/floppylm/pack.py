"""Serialize a TinyGPT to the bytes that sit on the floppy, and back.

Layout (FLP2):
  "FLP2" | u16 header_len | header JSON (GPTConfig, sorted keys, no spaces)
  | u32 shared_len | shared section (codebooks / seeds, counted once; empty for scalar codecs)
  | per tensor in `GPTConfig.tensor_shapes()` order:
      quantized -> scale bytes | u32 len | tagged symbol stream (rans.encode_best)
      norm      -> fp16 values

`len(pack(model))` is the model's byte count (ADR 0003 §1). A model returned by `unpack` keeps
its stored records verbatim, so `pack(unpack(blob)) == blob`.
"""

from __future__ import annotations

import json
import struct

import numpy as np
import torch

from . import rans
from .codec import CodecError
from .model import GPTConfig, QLinear, TinyGPT

MAGIC = b"FLP2"
LEGACY_MAGIC = b"FLP1"
LEGACY_REVISION = "f9e0732"  # last git revision that reads FLP1 (ADR 0006)


class FormatError(ValueError):
    """The blob is not a valid FloppyLM artifact."""


class _Reader:
    def __init__(self, blob: bytes) -> None:
        self.blob, self.off = blob, 0

    def take(self, n: int, what: str) -> bytes:
        if self.off + n > len(self.blob):
            raise FormatError(f"truncated while reading {what}")
        out = self.blob[self.off : self.off + n]
        self.off += n
        return out

    def u32(self, what: str) -> int:
        return struct.unpack("<I", self.take(4, what))[0]


def _header(cfg: GPTConfig) -> bytes:
    return json.dumps(cfg.to_dict(), separators=(",", ":"), sort_keys=True).encode()


def _record(t: QLinear) -> bytes:
    if t.frozen and t.canonical is not None:
        return t.canonical
    sym, scale = t.codec.encode(t.weight.detach())
    stream = rans.encode_best(sym, t.codec.levels)
    return scale + struct.pack("<I", len(stream)) + stream


def pack_sections(m: TinyGPT) -> tuple[bytes, dict[str, int]]:
    """Return the blob and its exact byte breakdown by section (sums to len(blob))."""
    if hasattr(m, "_artifact_state"):
        current = dict(m.named_parameters())
        if m.cfg != m._artifact_config or current.keys() != m._artifact_state.keys():
            raise FormatError("loaded model configuration or parameters were modified")
        for name, original in m._artifact_state.items():
            if not torch.equal(current[name].detach().cpu(), original):
                raise FormatError(f"loaded inference model was modified: {name}")
    header = _header(m.cfg)
    shared = b"".join(t.codec.shared_state() for t in m.stored_tensors() if isinstance(t, QLinear))
    out = bytearray(MAGIC + struct.pack("<H", len(header)) + header)
    out += struct.pack("<I", len(shared)) + shared
    parts = {
        "header": len(MAGIC) + 2 + len(header) + 4,
        "shared": len(shared),
        "emb": 0,
        "core": 0,
        "scales": 0,
        "norms": 0,
    }
    for t in m.stored_tensors():
        if not isinstance(t, QLinear):
            b = t.weight.detach().half().numpy().astype("<f2").tobytes()
            parts["norms"] += len(b)
            out += b
            continue
        rec = _record(t)
        ns = t.codec.scale_nbytes(t.weight.shape[0])
        parts["scales"] += ns
        parts["emb" if t is m.emb else "core"] += len(rec) - ns
        out += rec
    blob = bytes(out)
    if sum(parts.values()) != len(blob):
        raise AssertionError("section accounting does not match the blob")
    return blob, parts


def pack(m: TinyGPT) -> bytes:
    return pack_sections(m)[0]


def _read_config(r: _Reader) -> GPTConfig:
    magic = r.take(4, "magic")
    if magic == LEGACY_MAGIC:
        raise FormatError(
            f"FLP1 (pre-v2) is no longer supported; read it with git revision {LEGACY_REVISION}"
        )
    if magic != MAGIC:
        raise FormatError(f"unknown magic {magic!r}")
    (hlen,) = struct.unpack("<H", r.take(2, "header length"))
    raw_bytes = r.take(hlen, "header")
    try:
        raw = json.loads(raw_bytes)
        if not isinstance(raw, dict) or set(raw) != set(GPTConfig.__dataclass_fields__):
            raise FormatError("header keys do not match GPTConfig")
        cfg = GPTConfig(**raw)
    except FormatError:
        raise
    except (ValueError, TypeError) as e:
        raise FormatError(f"invalid header: {e}") from e
    if _header(cfg) != raw_bytes:
        raise FormatError("header is not in canonical form")
    return cfg


def unpack(blob: bytes) -> TinyGPT:
    """Rebuild the model exactly as stored. Raises FormatError on any malformed input."""
    r = _Reader(bytes(blob))
    cfg = _read_config(r)
    if r.take(r.u32("shared length"), "shared section"):
        raise FormatError("scalar codecs carry no shared section")
    m = TinyGPT(cfg)
    with torch.no_grad():
        for i, t in enumerate(m.stored_tensors()):
            w = t.weight
            if not isinstance(t, QLinear):
                v = np.frombuffer(r.take(2 * w.numel(), f"norm {i}"), "<f2").astype(np.float32)
                if not np.isfinite(v).all():
                    raise FormatError(f"norm {i} is not finite")
                w.copy_(torch.from_numpy(v))
                continue
            start = r.off
            scale = r.take(t.codec.scale_nbytes(w.shape[0]), f"scales {i}")
            stream = r.take(r.u32(f"stream length {i}"), f"stream {i}")
            try:
                w.copy_(t.codec.decode(rans.decode_best(stream), scale, tuple(w.shape)))
            except (rans.StreamError, CodecError) as e:
                raise FormatError(f"tensor {i}: {e}") from e
            t.frozen = True
            t.canonical = r.blob[start : r.off]
    if r.off != len(r.blob):
        raise FormatError(f"{len(r.blob) - r.off} trailing bytes")
    m.requires_grad_(False)
    m._artifact_config = m.cfg
    m._artifact_state = {name: p.detach().cpu().clone() for name, p in m.named_parameters()}
    m.eval()
    return m
