"""Entropy coders for integer symbol streams.

rANS: static byte-wise rANS (ryg_rans style), 15-bit probabilities.
  layout: n (u32) | alphabet (u16) | freqs (u16 * alphabet) | state + renorm bytes
bitpack: fixed-rate packing, exact for power-of-two alphabets, 64-bit chunks otherwise.
  layout: n (u32) | alphabet (u16) | payload
`best` picks the shorter of the two and prefixes a one-byte tag. Every table is inside the
stream, so its bytes are always counted.
"""

import math
import struct

import numpy as np

PROB_BITS = 15
L = 1 << 23
MAX_ALPHABET = 1 << 16
TAG_RANS, TAG_BITPACK = 0, 1


class StreamError(ValueError):
    """Truncated or corrupt symbol stream."""


def _check(sym: np.ndarray, alphabet: int) -> np.ndarray:
    sym = np.asarray(sym, dtype=np.int64).ravel()
    assert 2 <= alphabet <= MAX_ALPHABET
    assert sym.size == 0 or (sym.min() >= 0 and sym.max() < alphabet)
    return sym


def _freqs(sym: np.ndarray, alphabet: int, M: int) -> np.ndarray:
    counts = np.bincount(sym, minlength=alphabet).astype(np.int64)
    if counts.sum() == 0:
        return np.zeros(alphabet, dtype=np.int64)
    f = np.floor(counts * M / counts.sum()).astype(np.int64)
    f[(counts > 0) & (f == 0)] = 1
    while f.sum() > M:  # many rare symbols bumped to 1: take back from the largest
        f[np.argmax(f)] -= f.sum() - M
        f[f < 0] = 0
    f[np.argmax(f)] += M - f.sum()
    assert f.sum() == M and f.max() < (1 << 16)
    return f


def rans_encode(sym: np.ndarray, alphabet: int, prob_bits: int = PROB_BITS) -> bytes:
    sym = _check(sym, alphabet)
    M = 1 << prob_bits
    if alphabet > M:
        raise ValueError("alphabet larger than the probability range")
    freq = _freqs(sym, alphabet, M)
    header = struct.pack("<IH", sym.size, alphabet % MAX_ALPHABET) + freq.astype("<u2").tobytes()
    if sym.size == 0:
        return header
    cum = np.concatenate([[0], np.cumsum(freq)[:-1]]).tolist()
    freq_l = freq.tolist()
    out = bytearray()
    x = L
    for s in reversed(sym.tolist()):
        f = freq_l[s]
        x_max = ((L >> prob_bits) << 8) * f
        while x >= x_max:
            out.append(x & 0xFF)
            x >>= 8
        x = ((x // f) << prob_bits) + (x % f) + cum[s]
    out += x.to_bytes(4, "little")
    out.reverse()
    return header + bytes(out)


def rans_decode(blob: bytes, prob_bits: int = PROB_BITS) -> np.ndarray:
    M = 1 << prob_bits
    if len(blob) < 6:
        raise StreamError("rANS header truncated")
    n, alphabet = struct.unpack_from("<IH", blob, 0)
    alphabet = alphabet or MAX_ALPHABET
    off = 6 + 2 * alphabet
    if len(blob) < off:
        raise StreamError("rANS frequency table truncated")
    freq = np.frombuffer(blob, dtype="<u2", count=alphabet, offset=6).astype(np.int64)
    if n == 0:
        if len(blob) != off:
            raise StreamError("trailing bytes after empty rANS stream")
        return np.zeros(0, dtype=np.int64)
    if freq.sum() != M:
        raise StreamError("rANS frequencies do not sum to the probability range")
    cum = np.concatenate([[0], np.cumsum(freq)[:-1]])
    lookup = np.repeat(np.arange(alphabet), freq).tolist()
    freq_l, cum_l = freq.tolist(), cum.tolist()
    data = blob[off:]
    if len(data) < 4:
        raise StreamError("rANS state truncated")
    x = int.from_bytes(data[:4], "big")
    pos = 4
    out = [0] * n
    try:
        for i in range(n):
            slot = x & (M - 1)
            s = lookup[slot]
            out[i] = s
            x = freq_l[s] * (x >> prob_bits) + slot - cum_l[s]
            while x < L:
                x = (x << 8) | data[pos]
                pos += 1
    except IndexError as e:
        raise StreamError("rANS payload truncated") from e
    if pos != len(data) or x != L:
        raise StreamError("rANS payload corrupt (state or length mismatch)")
    return np.array(out, dtype=np.int64)


def _chunk(alphabet: int) -> int:
    """Symbols per 64-bit word for non power-of-two alphabets."""
    k = int(64 / math.log2(alphabet))
    while alphabet**k >= 2**64:
        k -= 1
    return max(1, k)


def bitpack_encode(sym: np.ndarray, alphabet: int) -> bytes:
    sym = _check(sym, alphabet)
    header = struct.pack("<IH", sym.size, alphabet % MAX_ALPHABET)
    bits = (alphabet - 1).bit_length()
    if alphabet == 1 << bits:
        planes = ((sym[:, None] >> np.arange(bits - 1, -1, -1)) & 1).astype(np.uint8)
        return header + np.packbits(planes.ravel()).tobytes()
    k = _chunk(alphabet)
    pad = (-sym.size) % k
    s = np.concatenate([sym, np.zeros(pad, dtype=np.int64)]).reshape(-1, k).tolist()
    words = []
    for row in s:
        v = 0
        for d in row:
            v = v * alphabet + d
        words.append(v)
    return header + np.array(words, dtype="<u8").tobytes()


def bitpack_decode(blob: bytes) -> np.ndarray:
    if len(blob) < 6:
        raise StreamError("bitpack header truncated")
    n, alphabet = struct.unpack_from("<IH", blob, 0)
    alphabet = alphabet or MAX_ALPHABET
    if alphabet < 2:
        raise StreamError("bitpack alphabet invalid")
    data = blob[6:]
    bits = (alphabet - 1).bit_length()
    if alphabet == 1 << bits:
        if len(data) != (n * bits + 7) // 8:
            raise StreamError("bitpack payload has the wrong length")
        planes = np.unpackbits(np.frombuffer(data, dtype=np.uint8))[: n * bits].reshape(n, bits)
        return (planes.astype(np.int64) << np.arange(bits - 1, -1, -1)).sum(1)
    k = _chunk(alphabet)
    if len(data) != 8 * ((n + k - 1) // k):
        raise StreamError("bitpack payload has the wrong length")
    out = []
    for v in np.frombuffer(data, dtype="<u8").tolist():
        row = []
        for _ in range(k):
            v, d = divmod(v, alphabet)
            row.append(d)
        if v:
            raise StreamError("bitpack word out of range")
        out.extend(reversed(row))
    return np.array(out[:n], dtype=np.int64)


def encode_best(sym: np.ndarray, alphabet: int) -> bytes:
    cands = [bytes([TAG_BITPACK]) + bitpack_encode(sym, alphabet)]
    if alphabet <= 1 << PROB_BITS:
        cands.append(bytes([TAG_RANS]) + rans_encode(sym, alphabet))
    return min(cands, key=len)


def decode_best(blob: bytes) -> np.ndarray:
    if not blob or blob[0] not in (TAG_RANS, TAG_BITPACK):
        raise StreamError("unknown coder tag")
    return (rans_decode if blob[0] == TAG_RANS else bitpack_decode)(blob[1:])
