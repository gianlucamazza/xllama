#!/usr/bin/env python3
# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
"""Build a small random-weight MoE GGUF from a real checkpoint's header.

Expert streaming must produce the same logits as a resident load on every
architecture it admits (plan: docs/superpowers/plans/2026-11-16-moe-streaming-
all-models.md, Task 4/5). A full checkpoint is too large for a host reference,
so this writes a fixture with the real architecture's metadata, tokenizer and
tensor layout, but few layers, few experts and small experts:

- layers: the first ones, until every layer kind (tensor-name signature) and
  at least --moe-layers MoE layers are present;
- experts: --experts (default 8), --used active (default 2);
- expert width: --ff-exp (default 256, a multiple of the Q4_K super-block).

Everything else (embedding width, attention, shared experts, vocabulary) keeps
the template's shape. The output is NOT a model: weights are random, the
filename and general.tags say so.

The template can be a local GGUF or an https URL; for a URL only the header is
downloaded (HTTP Range) into a sparse file of the full size.

Usage:
  scripts/make-moe-fixture.py --template <file|url> --out fixture.gguf \
      [--experts 8] [--used 2] [--ff-exp 256] [--moe-layers 1] [--seed 0] \
      [--quant Q4_K_M|none]
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "llama.cpp" / "gguf-py"))

TABLE_SOURCE = ROOT / "src" / "bridge" / "moe_streaming.cpp"
BLK = re.compile(r"^blk\.(\d+)\.(.+)$")


def supported_archs(source=TABLE_SOURCE):
    """Architectures of the C++ table (one home: src/bridge/moe_streaming.cpp)."""
    text = Path(source).read_text()
    block = text[text.index("const MoeArch kArchs[]") :]
    block = block[: block.index("};")]
    return re.findall(r'\{"([a-z0-9_.-]+)",', block)


def fetch_header(url, dest, first=8 << 20, limit=256 << 20):
    """Download only the GGUF header of `url` into a sparse file of full size."""
    from gguf import GGUFReader

    head = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(head) as r:
        total = int(r.headers.get("x-linked-size") or r.headers["content-length"])
    want = first
    while True:
        req = urllib.request.Request(url, headers={"Range": f"bytes=0-{want - 1}"})
        with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
            f.write(r.read())
            f.truncate(total)
        try:
            GGUFReader(str(dest))
            return dest
        except Exception:
            if want >= limit:
                raise
            want *= 2


def arch_key(arch, name):
    return f"{arch}.{name}"


def plan_layers(tensor_names, n_layer, moe_layers):
    """Indices of the template layers to keep, in order."""
    per_layer = {}
    for name in tensor_names:
        m = BLK.match(name)
        if m:
            per_layer.setdefault(int(m.group(1)), set()).add(m.group(2))
    kinds_all = {frozenset(per_layer.get(i, ())) for i in range(n_layer)}
    kept, kinds, moe = [], set(), 0
    for i in range(n_layer):
        sig = frozenset(per_layer.get(i, ()))
        is_moe = any(t.startswith("ffn_gate_inp") for t in sig)
        if sig not in kinds or (is_moe and moe < moe_layers):
            kept.append(i)
            kinds.add(sig)
            moe += int(is_moe)
        if kinds == kinds_all and moe >= moe_layers:
            break
    if moe < moe_layers:
        raise ValueError("template has fewer MoE layers than --moe-layers")
    return kept


def shrink_shape(base, suffix, shape, n_expert, n_ff_exp, n_shared, experts, ff_exp):
    """Template tensor shape (gguf order, ne0 first) -> fixture shape, by position.

    Positions follow llama.cpp's create_tensor calls: gate/up/gate_up experts
    are [n_embd, ff, n_expert], down experts [ff, n_embd, n_expert]; their
    biases drop ne0 accordingly. Matching by value is wrong when n_embd equals
    the expert width (gpt-oss: both 2880).
    """
    out = list(shape)
    if base in ("ffn_gate_exps", "ffn_up_exps", "ffn_gate_up_exps"):
        width = 2 * ff_exp if base == "ffn_gate_up_exps" else ff_exp
        if suffix == "weight":
            out[1], out[2] = width, experts
        elif suffix == "bias":
            out[0], out[1] = width, experts
        else:  # scale
            out[0] = experts
    elif base == "ffn_down_exps":
        if suffix == "weight":
            out[0], out[2] = ff_exp, experts
        elif suffix == "bias":
            out[1] = experts
        else:
            out[0] = experts
    elif base == "ffn_gate_inp":
        out[-1] = experts
    elif base == "exp_probs_b":
        out[0] = experts
    elif base.endswith("_shexp") and n_shared and n_ff_exp > 0:
        # Shared experts sized n_ff_exp * n_shared (afmoe) shrink with the experts.
        for i, d in enumerate(out):
            if d == n_ff_exp * n_shared:
                out[i] = ff_exp * n_shared
    return out


def read_template(path):
    from gguf import GGUFReader, GGUFValueType

    r = GGUFReader(str(path))
    arch = r.fields["general.architecture"].contents()
    kvs = []
    for key, f in r.fields.items():
        if key.startswith("GGUF.") or key == "general.architecture":
            continue
        vtype = f.types[0]
        sub = f.types[-1] if vtype == GGUFValueType.ARRAY else None
        kvs.append((key, f.contents(), vtype, sub))
    tensors = [(t.name, [int(x) for x in t.shape]) for t in r.tensors]
    return arch, kvs, tensors


def build_plan(arch, kvs, tensors, experts, used, ff_exp, moe_layers):
    kv = {k: v for k, v, _, _ in kvs}
    n_layer = int(kv[arch_key(arch, "block_count")])
    n_expert = int(kv[arch_key(arch, "expert_count")])
    ff = kv.get(arch_key(arch, "expert_feed_forward_length"))
    n_ff_exp = int(max(ff) if isinstance(ff, list) else ff) if ff is not None else -1
    if experts > n_expert:
        raise ValueError(f"--experts {experts} exceeds template n_expert {n_expert}")
    n_shared = int(kv.get(arch_key(arch, "expert_shared_count"), 0) or 0)
    if kv.get(arch_key(arch, "expert_shared_feed_forward_length")) is not None:
        n_shared = 0  # the shared width has its own key; keep it as is
    kept = plan_layers([n for n, _ in tensors], n_layer, moe_layers)
    # A scalar sliding window without a per-layer pattern means llama.cpp applies
    # its default period (4 for smallthinker): keep one full period of layers.
    has_swa = kv.get(arch_key(arch, "attention.sliding_window")) is not None
    has_pattern = kv.get(arch_key(arch, "attention.sliding_window_pattern")) is not None
    if has_swa and not has_pattern and len(kept) < min(4, n_layer):
        kept = sorted(set(kept) | set(range(min(4, n_layer))))
    remap = {old: new for new, old in enumerate(kept)}

    out_kvs = []
    for key, val, vtype, sub in kvs:
        short = key[len(arch) + 1 :] if key.startswith(arch + ".") else None
        if short == "block_count":
            val = len(kept)
        elif short == "expert_count":
            val = experts
        elif short == "expert_used_count":
            val = min(int(val), used, experts)
        elif short == "expert_feed_forward_length":
            val = [ff_exp] * len(kept) if isinstance(val, list) else ff_exp
        elif short in ("expert_group_count", "expert_group_used_count"):
            val = 1
        elif short == "nextn_predict_layers":
            val = 0
        elif short == "leading_dense_block_count":
            val = sum(1 for i in kept if i < int(val))
        elif short is not None and isinstance(val, list) and len(val) == n_layer:
            val = [val[i] for i in kept]
        elif key == "general.file_type":
            val = 1  # MOSTLY_F16 before quantization
        out_kvs.append((key, val, vtype, sub))

    out_tensors = []
    for name, shape in tensors:
        m = BLK.match(name)
        if m:
            old = int(m.group(1))
            if old not in remap:
                continue
            rest = m.group(2)
            base, _, suffix = rest.partition(".")
            name = f"blk.{remap[old]}.{rest}"
            shape = shrink_shape(base, suffix, shape, n_expert, n_ff_exp, n_shared, experts, ff_exp)
        out_tensors.append((name, shape))
    return out_kvs, out_tensors, kept


F32_ONLY = ("conv", "ssm_", "norm", "sinks")


def is_matmul_weight(name, shape):
    """f16 only for matmul inputs; ggml's conv/SSM ops and norms require f32."""
    return (
        len(shape) >= 2
        and name.endswith(".weight")
        and not any(part in name for part in F32_ONLY)
    )


def write_f16(arch, kvs, tensors, path, seed):
    import numpy as np
    from gguf import GGUFWriter

    w = GGUFWriter(str(path), arch)
    for key, val, vtype, sub in kvs:
        if key in ("general.name", "general.tags"):
            continue
        w.add_key_value(key, val, vtype, sub_type=sub)
    w.add_name(f"{arch}-moe-fixture (RANDOM WEIGHTS, streaming parity only)")
    w.add_array("general.tags", ["xllama-moe-fixture", "random-weights", "not-a-model"])

    def dtype_for(name, shape):
        return np.float16 if is_matmul_weight(name, shape) else np.float32

    for name, shape in tensors:
        dt = np.dtype(dtype_for(name, shape))
        n = 1
        for d in shape:
            n *= d
        w.add_tensor_info(name, tuple(reversed(shape)), dt, n * dt.itemsize)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    rng = np.random.default_rng(seed)
    for name, shape in tensors:
        np_shape = tuple(reversed(shape))
        if "norm" in name and name.endswith(".weight"):
            arr = np.ones(np_shape, dtype=np.float32)
        elif name.endswith(".bias"):
            arr = np.zeros(np_shape, dtype=np.float32)
        else:
            arr = (rng.standard_normal(np_shape, dtype=np.float32) * 0.02).astype(
                dtype_for(name, shape)
            )
        w.write_tensor_data(arr)
        del arr
    w.close()


def find_quantize():
    for preset in ("linux-test", "linux-release", "linux-debug"):
        p = ROOT / "build" / preset / "bin" / "llama-quantize"
        if p.exists():
            return p
    return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--template", required=True, help="GGUF file or https URL")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--used", type=int, default=2)
    ap.add_argument("--ff-exp", type=int, default=256)
    ap.add_argument("--moe-layers", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quant", default="Q4_K_M", help="llama-quantize type, or 'none'")
    ap.add_argument("--cache-dir", type=Path, default=Path.home() / ".cache" / "xllama-gguf")
    a = ap.parse_args(argv)

    template = a.template
    if template.startswith("https://"):
        a.cache_dir.mkdir(parents=True, exist_ok=True)
        local = a.cache_dir / ("header-" + template.rstrip("/").split("/")[-1])
        if not local.exists():
            fetch_header(template, local)
        template = local
    arch, kvs, tensors = read_template(template)
    if arch not in supported_archs():
        raise SystemExit(f"architecture {arch!r} is not in the streaming table")
    kvs, tensors, kept = build_plan(
        arch, kvs, tensors, a.experts, a.used, a.ff_exp, a.moe_layers
    )
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.quant == "none":
        write_f16(arch, kvs, tensors, a.out, a.seed)
    else:
        q = find_quantize()
        if not q:
            raise SystemExit("llama-quantize not found; build linux-test or pass --quant none")
        with tempfile.TemporaryDirectory(dir=a.out.parent) as tmp:
            f16 = Path(tmp) / "f16.gguf"
            write_f16(arch, kvs, tensors, f16, a.seed)
            subprocess.run([str(q), str(f16), str(a.out), a.quant], check=True,
                           stdout=subprocess.DEVNULL)
    print(f"{a.out}: arch={arch} layers={kept} experts={a.experts} ff_exp={a.ff_exp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
