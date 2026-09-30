#!/usr/bin/env python3
# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
"""Build random-weight `lfm2` GGUFs of arbitrary shape for the shape speed gate.

Decode and prefill speed depend on the tensor shapes and quant types, not on
the weight values. So a randomly initialised GGUF of a candidate shape can be
timed on Series S before any training is paid for. The output is NOT a model:
it produces noise. The filename and `general.tags` say so, so it can never be
mistaken for a catalogue entry.

The tokenizer and chat template are copied from a real LFM2 template GGUF.
With `vocab < template vocab`, the tokenizer is truncated to the first N ids
and merges that reference a dropped token are removed. LFM2 ids are
merge-ordered, so this approximates a smaller trained BPE vocab.

Usage:
  scripts/make-shape-gguf.py --spec bench/configs/shape-sweep.json --dry-run
  scripts/make-shape-gguf.py --spec bench/configs/shape-sweep.json \
      --template ~/.cache/xllama-gguf/LFM2.5-350M-Q4_K_M.gguf \
      --out-dir build/shape-gguf [--only ref-350m,d24]

Output per shape: <out-dir>/shape-<id>/lfm2-shape-<id>-rand-<QUANT>.gguf
(the directory name is what goes under LocalState/models/ for the bench).
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "llama.cpp" / "gguf-py"))

HEAD_DIM = 64  # LFM2 family: attn_q_norm / attn_k_norm are [64]
L_CACHE = 3  # shortconv kernel length of every released LFM2
QK_K = 256  # Q4_K/Q6_K super-block: every matmul input dim must be a multiple
# Bits per weight for the desk estimate only. Q4_0 and Q4_K are both 4.5 bpw;
# the tied embedding/output goes to Q6_K (6.5625) under both llama-quantize
# mixes. The shipped LFM2.5 QAD Q4_0 files use exactly that layout: every
# matmul Q4_0, token_embd Q6_K. Measured file size is what the plan compares.
BPW_Q4 = 4.5
BPW_Q6K = 6.5625

TOKENIZER_REBUILT = {
    "tokenizer.ggml.tokens",
    "tokenizer.ggml.token_type",
    "tokenizer.ggml.merges",
    "tokenizer.ggml.scores",
}


@dataclass
class Shape:
    id: str
    n_layer: int
    n_embd: int
    n_ff: int
    n_vocab: int = 65536
    n_head_kv: int = 8
    attn_layers: list = field(default_factory=list)
    note: str = ""

    @property
    def n_head(self):
        return self.n_embd // HEAD_DIM

    @property
    def n_embd_kv(self):
        return self.n_head_kv * HEAD_DIM


def default_attn_layers(n_layer):
    """Spread attention blocks like LFM2.5 (6 of 16, first at index 2)."""
    n_attn = max(1, round(n_layer * 6 / 16))
    if n_attn == 1:
        return [n_layer // 2]
    lo, hi = 2, n_layer - 2
    return sorted({round(lo + i * (hi - lo) / (n_attn - 1)) for i in range(n_attn)})


def load_spec(path):
    data = json.loads(Path(path).read_text())
    shapes = []
    for s in data["shapes"]:
        shape = Shape(
            id=s["id"],
            n_layer=s["n_layer"],
            n_embd=s["n_embd"],
            n_ff=s["n_ff"],
            n_vocab=s.get("n_vocab", 65536),
            n_head_kv=s.get("n_head_kv", 8),
            attn_layers=s.get("attn_layers") or default_attn_layers(s["n_layer"]),
            note=s.get("note", ""),
        )
        validate_shape(shape)
        shapes.append(shape)
    ids = [s.id for s in shapes]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate shape ids in {path}")
    return shapes


def validate_shape(s):
    errs = []
    for name in ("n_embd", "n_ff"):
        if getattr(s, name) % QK_K:
            errs.append(f"{name}={getattr(s, name)} is not a multiple of {QK_K}")
    if s.n_embd % HEAD_DIM:
        errs.append(f"n_embd={s.n_embd} is not a multiple of head_dim {HEAD_DIM}")
    if s.n_head % s.n_head_kv:
        errs.append(f"n_head={s.n_head} is not a multiple of n_head_kv={s.n_head_kv}")
    if not s.attn_layers or any(i < 0 or i >= s.n_layer for i in s.attn_layers):
        errs.append(f"attn_layers {s.attn_layers} out of range for n_layer={s.n_layer}")
    if errs:
        raise ValueError(f"shape {s.id}: " + "; ".join(errs))


def tensor_plan(s):
    """(name, ggml ne) for every tensor, in the order llama.cpp's lfm2 loader expects."""
    D, F = s.n_embd, s.n_ff
    plan = [("token_embd.weight", (D, s.n_vocab)), ("token_embd_norm.weight", (D,))]
    attn = set(s.attn_layers)
    for i in range(s.n_layer):
        p = f"blk.{i}."
        plan.append((p + "attn_norm.weight", (D,)))
        if i in attn:
            plan += [
                (p + "attn_q.weight", (D, D)),
                (p + "attn_k.weight", (D, s.n_embd_kv)),
                (p + "attn_v.weight", (D, s.n_embd_kv)),
                (p + "attn_output.weight", (D, D)),
                (p + "attn_q_norm.weight", (HEAD_DIM,)),
                (p + "attn_k_norm.weight", (HEAD_DIM,)),
            ]
        else:
            plan += [
                (p + "shortconv.conv.weight", (L_CACHE, D)),
                (p + "shortconv.in_proj.weight", (D, 3 * D)),
                (p + "shortconv.out_proj.weight", (D, D)),
            ]
        plan += [
            (p + "ffn_norm.weight", (D,)),
            (p + "ffn_gate.weight", (D, F)),
            (p + "ffn_up.weight", (D, F)),
            (p + "ffn_down.weight", (F, D)),
        ]
    return plan


def is_matmul(name, ne):
    return len(ne) == 2 and "shortconv.conv" not in name


def estimate(s):
    """Desk numbers: parameters and weight bytes read per decoded token."""
    total = embd = 0
    for name, ne in tensor_plan(s):
        n = 1
        for d in ne:
            n *= d
        total += n
        if name == "token_embd.weight":
            embd = n
    body = total - embd
    # Per decoded token: every block matmul plus the tied output head (the full
    # embedding matrix is read once as the LM head; the input lookup is one row).
    read_bytes = body * BPW_Q4 / 8 + embd * BPW_Q6K / 8
    kv_per_tok = len(s.attn_layers) * 2 * s.n_embd_kv * 2  # f16 K+V
    return {
        "params_M": total / 1e6,
        "embd_M": embd / 1e6,
        "body_M": body / 1e6,
        "read_MB_per_tok": read_bytes / 1e6,
        "kv_KiB_per_tok": kv_per_tok / 1024,
    }


def truncate_tokenizer(tokens, token_types, merges, n_vocab):
    """Keep ids [0, n_vocab) and the merges whose parts and result all survive."""
    if n_vocab > len(tokens):
        raise ValueError(f"n_vocab={n_vocab} exceeds template vocab {len(tokens)}")
    kept = tokens[:n_vocab]
    keep = set(kept)
    out_merges = []
    for m in merges:
        a, sep, b = m.partition(" ")
        if sep and a in keep and b in keep and (a + b) in keep:
            out_merges.append(m)
    return kept, token_types[:n_vocab], out_merges


def read_template(path):
    from gguf import GGUFReader, GGUFValueType

    r = GGUFReader(str(path))
    arch = r.fields["general.architecture"].contents()
    if arch != "lfm2":
        raise ValueError(f"template {path} is {arch!r}, need an lfm2 GGUF")
    copied = []
    for key, f in r.fields.items():
        if not key.startswith("tokenizer.") or key in TOKENIZER_REBUILT:
            continue
        vtype = f.types[0]
        sub = f.types[-1] if vtype == GGUFValueType.ARRAY else None
        copied.append((key, f.contents(), vtype, sub))
    tokens = r.fields["tokenizer.ggml.tokens"].contents()
    types = [int(t) for t in r.fields["tokenizer.ggml.token_type"].contents()]
    merges = r.fields["tokenizer.ggml.merges"].contents()
    rope = r.fields["lfm2.rope.freq_base"].contents()
    eps = r.fields["lfm2.attention.layer_norm_rms_epsilon"].contents()
    return copied, tokens, types, merges, float(rope), float(eps)


def write_f16_gguf(s, template, path, seed):
    import numpy as np
    from gguf import GGUFValueType, GGUFWriter

    copied, tokens, types, merges, rope, eps = template
    tokens, types, merges = truncate_tokenizer(tokens, types, merges, s.n_vocab)

    w = GGUFWriter(str(path), "lfm2")
    w.add_name(f"lfm2-shape-{s.id} (RANDOM WEIGHTS, speed gate only)")
    w.add_array("general.tags", ["xllama-shape-gate", "random-weights", "not-a-model"])
    w.add_file_type(1)  # MOSTLY_F16
    w.add_uint32("lfm2.block_count", s.n_layer)
    w.add_uint32("lfm2.context_length", 128000)
    w.add_uint32("lfm2.embedding_length", s.n_embd)
    w.add_uint32("lfm2.feed_forward_length", s.n_ff)
    w.add_uint32("lfm2.attention.head_count", s.n_head)
    attn = set(s.attn_layers)
    w.add_array(
        "lfm2.attention.head_count_kv",
        [s.n_head_kv if i in attn else 0 for i in range(s.n_layer)],
    )
    w.add_float32("lfm2.rope.freq_base", rope)
    w.add_float32("lfm2.attention.layer_norm_rms_epsilon", eps)
    w.add_uint32("lfm2.vocab_size", s.n_vocab)
    w.add_uint32("lfm2.shortconv.l_cache", L_CACHE)
    for key, val, vtype, sub in copied:
        w.add_key_value(key, val, vtype, sub_type=sub)
    w.add_array("tokenizer.ggml.tokens", tokens)
    w.add_key_value(
        "tokenizer.ggml.token_type",
        types,
        GGUFValueType.ARRAY,
        sub_type=GGUFValueType.INT32,
    )
    w.add_array("tokenizer.ggml.merges", merges)

    plan = tensor_plan(s)

    def dtype_for(name, ne):
        return np.float16 if is_matmul(name, ne) else np.float32

    for name, ne in plan:
        dt = np.dtype(dtype_for(name, ne))
        n = 1
        for d in ne:
            n *= d
        w.add_tensor_info(name, tuple(reversed(ne)), dt, n * dt.itemsize)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()

    rng = np.random.default_rng(seed)
    for name, ne in plan:
        shape = tuple(reversed(ne))
        if name.endswith("norm.weight"):
            arr = np.ones(shape, dtype=np.float32)
        else:
            # std 0.02 keeps activations bounded through RMSNorm; values are irrelevant
            arr = (rng.standard_normal(shape, dtype=np.float32) * 0.02).astype(
                dtype_for(name, ne)
            )
        w.write_tensor_data(arr)
        del arr
    w.close()


def find_quantize():
    # linux-test is the day-to-day preset; a stale tree can ship a binary whose
    # shared libs no longer match, so only return one that actually starts.
    for preset in ("linux-test", "linux-release", "linux-debug"):
        p = ROOT / "build" / preset / "bin" / "llama-quantize"
        if p.exists():
            r = subprocess.run([str(p), "--help"], capture_output=True, check=False)
            if r.returncode != 127:
                return p
    return None


def build(s, template, out_dir, quant, quantize_bin, seed):
    model_dir = out_dir / f"shape-{s.id}"
    model_dir.mkdir(parents=True, exist_ok=True)
    final = model_dir / f"lfm2-shape-{s.id}-rand-{quant}.gguf"
    with tempfile.TemporaryDirectory(dir=out_dir) as tmp:
        f16 = Path(tmp) / "f16.gguf"
        write_f16_gguf(s, template, f16, seed)
        subprocess.run([str(quantize_bin), str(f16), str(final), quant], check=True)
    return final


def print_table(shapes):
    hdr = f"{'id':<12}{'L':>4}{'D':>6}{'F':>7}{'V':>7}{'attn':>5}{'params_M':>10}{'embd_M':>8}{'read_MB/tok':>12}{'KV_KiB/tok':>11}"
    print(hdr)
    for s in shapes:
        e = estimate(s)
        print(
            f"{s.id:<12}{s.n_layer:>4}{s.n_embd:>6}{s.n_ff:>7}{s.n_vocab:>7}{len(s.attn_layers):>5}"
            f"{e['params_M']:>10.1f}{e['embd_M']:>8.1f}{e['read_MB_per_tok']:>12.1f}{e['kv_KiB_per_tok']:>11.1f}"
        )


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--spec", required=True, help="shape sweep JSON")
    ap.add_argument(
        "--template", help="real lfm2 GGUF (tokenizer + chat template source)"
    )
    ap.add_argument("--out-dir", default=str(ROOT / "build" / "shape-gguf"))
    ap.add_argument("--only", help="comma-separated shape ids")
    ap.add_argument(
        "--quant",
        default="Q4_0",
        help="llama-quantize type (default: shipped QAD layout)",
    )
    ap.add_argument(
        "--quantize-bin", help="llama-quantize (default: build/<preset>/bin)"
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--dry-run", action="store_true", help="print the desk estimate and exit"
    )
    args = ap.parse_args(argv)

    shapes = load_spec(args.spec)
    if args.only:
        wanted = set(args.only.split(","))
        unknown = wanted - {s.id for s in shapes}
        if unknown:
            ap.error(f"unknown shape ids: {', '.join(sorted(unknown))}")
        shapes = [s for s in shapes if s.id in wanted]

    print_table(shapes)
    if args.dry_run:
        return 0

    if not args.template:
        ap.error("--template is required unless --dry-run")
    quantize_bin = Path(args.quantize_bin) if args.quantize_bin else find_quantize()
    if not quantize_bin or not quantize_bin.exists():
        ap.error(
            "llama-quantize not found; build a linux preset or pass --quantize-bin"
        )

    try:
        import gguf  # noqa: F401
    except ModuleNotFoundError:
        ap.error(
            "gguf-py not found: run `git submodule update --init llama.cpp` (e.g. in a fresh worktree)"
        )

    template = read_template(Path(os.path.expanduser(args.template)))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for s in shapes:
        path = build(s, template, out_dir, args.quant, quantize_bin, args.seed)
        print(f"{s.id}: {path} ({path.stat().st_size / 2**20:.0f} MiB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
