#!/usr/bin/env python3
# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
#
# GPU control comparison of ADR 0010. Two modes, both driven by a declaration
# file that is written and committed before the GPU arm is measured:
#
#   scripts/control-parity.py kernel --declaration D <cpu_capture> <gpu_capture>
#   scripts/control-parity.py spread --declaration D --cpu-console DIR
#                                    --cpu-host DIR --gpu DIR
#
# kernel (part 1): both captures come from `--capture-spec
#   '<names>;matmul_inputs=l_out-0'`, which records every weight MUL_MAT of the
#   first layer with its activation operand. A token column is comparable when
#   the two arms fed the product the same operand: bit-identical floats, or,
#   for K-quant weights, equal int8 values under ggml's q8_K activation
#   quantizer. On the comparable columns each product must reach the declared
#   per-record NMSE limit, and at least the declared fraction of columns must
#   be comparable. Columns whose operand differs are counted, not judged: a
#   different operand is not a kernel finding. Equal int8 values do not make
#   the operands equal: each 256-value block keeps its own float scale, so the
#   result inherits the operand's remaining difference. `in_nmse` reports that
#   difference on the comparable columns; the declared limit has to sit above
#   it and far below what a wrong kernel produces.
#
# spread (part 2): each directory holds `<prompt id>.bin` (+ `.bin.json`) for
#   every declared prompt. The reference spread is the largest max_abs_diff and
#   the largest NMSE between the two CPU references over the prompt set. The
#   GPU arm, measured against the console CPU arm, passes when its largest
#   max_abs_diff and NMSE stay within margin x spread, with the same top token
#   and the declared top-k overlap on every prompt.
#
# Declaration (JSON):
#   { "schema": "xllama.control-declaration/v1", "model": "<catalogue id>",
#     "kernel": { "nmse_limit": 1e-9, "min_comparable_fraction": 0.9 },
#     "spread": { "margin": 1.2, "top_k": 10, "top_k_min": 0.8 },
#     "prompts": [ { "id": "p1", "sha256": "<sha256 of the UTF-8 prompt>" } ] }
# No value has a default: the criterion is whatever was declared. The file's
# sha256 is printed and written to --json-out so a verdict names its criterion.
#
# This is not scripts/compare-logits.py, whose frozen gate and defaults are
# unchanged. Exit 0 = PASS, 1 = FAIL, 2 = invalid input or declaration.

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np

SCHEMA = "xllama.control-declaration/v1"
QK_K = 256
# Weight types whose CPU kernels quantize the activation operand with q8_K.
Q8_K_WEIGHTS = {"q2_K", "q3_K", "q4_K", "q5_K", "q6_K"}


class Invalid(ValueError):
    pass


def _number(obj, key, where, lo, hi, lo_open=False):
    v = obj.get(key) if isinstance(obj, dict) else None
    if type(v) not in (int, float) or not math.isfinite(v):
        raise Invalid(f"declaration {where}.{key} must be a finite number")
    if v < lo or v > hi or (lo_open and v == lo):
        raise Invalid(f"declaration {where}.{key} out of range")
    return float(v)


def load_declaration(path: Path, need: str):
    try:
        raw = path.read_bytes()
        decl = json.loads(raw)
    except (OSError, ValueError) as e:
        raise Invalid(f"cannot read declaration {path}: {e}") from None
    if not isinstance(decl, dict) or decl.get("schema") != SCHEMA:
        raise Invalid(f"declaration schema must be {SCHEMA}")
    if not isinstance(decl.get("model"), str) or not decl["model"]:
        raise Invalid("declaration model must be a non-empty string")
    out = {"model": decl["model"], "sha256": hashlib.sha256(raw).hexdigest()}
    if need == "kernel":
        k = decl.get("kernel")
        out["nmse_limit"] = _number(k, "nmse_limit", "kernel", 0.0, 1.0, lo_open=True)
        out["min_comparable_fraction"] = _number(
            k, "min_comparable_fraction", "kernel", 0.0, 1.0, lo_open=True)
    else:
        s = decl.get("spread")
        out["margin"] = _number(s, "margin", "spread", 1.0, 1e6)
        out["top_k_min"] = _number(s, "top_k_min", "spread", 0.0, 1.0)
        top_k = s.get("top_k")
        if type(top_k) is not int or top_k <= 0:
            raise Invalid("declaration spread.top_k must be a positive integer")
        out["top_k"] = top_k
        prompts = decl.get("prompts")
        if not isinstance(prompts, list) or not prompts:
            raise Invalid("declaration prompts must be a non-empty list")
        seen = {}
        for p in prompts:
            pid, sha = (p.get("id"), p.get("sha256")) if isinstance(p, dict) else (None, None)
            if (not isinstance(pid, str) or not pid or "/" in pid or "\\" in pid
                    or pid.startswith(".") or pid in seen):
                raise Invalid(f"declaration prompt id invalid or repeated: {pid!r}")
            if (not isinstance(sha, str) or len(sha) != 64
                    or any(c not in "0123456789abcdef" for c in sha)):
                raise Invalid(f"declaration prompt {pid}: sha256 must be 64 lowercase hex digits")
            seen[pid] = sha
        out["prompts"] = seen
    return out


# --- kernel mode -------------------------------------------------------------


def quantize_q8_k(x: np.ndarray) -> np.ndarray:
    """int8 values ggml's quantize_row_q8_K gives each column of x [n, tokens].

    Per 256-value block: iscale = -127 / (first value of largest magnitude),
    q = min(127, round-half-even(iscale * x)), all in float32.
    """
    n, tokens = x.shape
    blocks = x.T.reshape(tokens, n // QK_K, QK_K)
    idx = np.argmax(np.abs(blocks), axis=2)[..., None]
    mx = np.take_along_axis(blocks, idx, axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        iscale = np.where(mx != 0, np.float32(-127.0) / mx, np.float32(0.0)).astype(np.float32)
        q = np.rint((iscale * blocks).astype(np.float32))
    return np.minimum(q, 127).astype(np.int16).reshape(tokens, n)


def load_matmuls(path: Path):
    manifest = path / "manifest.jsonl"
    if not manifest.is_file():
        raise Invalid(f"missing manifest: {manifest}")
    records, seen = {}, {}

    def read(name, shape):
        f = path / name
        if f.resolve().parent != path.resolve():
            raise Invalid(f"{manifest}: file escapes capture dir: {name}")
        raw = f.read_bytes() if f.is_file() else b""
        values = np.frombuffer(raw, dtype=np.float32) if raw and len(raw) % 4 == 0 else None
        if values is None or values.size != shape[0] * shape[1]:
            raise Invalid(f"{f}: missing, incomplete or not {shape[0]}x{shape[1]} float32")
        if not np.all(np.isfinite(values)):
            raise Invalid(f"non-finite values: {f}")
        return values.reshape(shape[1], shape[0]).T  # [n, tokens], column = token

    for lineno, line in enumerate(manifest.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            meta = json.loads(line)
        except json.JSONDecodeError as e:
            raise Invalid(f"{manifest}:{lineno}: bad JSON: {e}") from None
        if not isinstance(meta, dict):
            raise Invalid(f"{manifest}:{lineno}: record must be an object")
        if meta.get("op") == "MUL_MAT" and "skipped" in meta:
            raise Invalid(f"{manifest}:{lineno}: product {meta.get('name')} was not captured "
                          f"({meta['skipped']})")
        if meta.get("matmul") is not True:
            continue
        for field in ("eval", "name", "ne", "file", "weight_type", "input_file", "input_ne"):
            if field not in meta:
                raise Invalid(f"{manifest}:{lineno}: matmul record missing {field}")
        shapes = []
        for ne in (meta["ne"], meta["input_ne"]):
            if (not isinstance(ne, list) or len(ne) != 4 or any(type(v) is not int for v in ne)
                    or ne[0] <= 0 or ne[1] <= 0 or ne[2] != 1 or ne[3] != 1):
                raise Invalid(f"{manifest}:{lineno}: product must be [n, tokens, 1, 1]: {ne}")
            shapes.append((ne[0], ne[1]))
        if shapes[0][1] != shapes[1][1]:
            raise Invalid(f"{manifest}:{lineno}: operand and result token counts differ")
        base = (meta["eval"], meta["name"])
        occ = seen.get(base, 0)
        seen[base] = occ + 1
        records[(*base, occ)] = {
            "weight_type": meta["weight_type"],
            "out": read(meta["file"], shapes[0]),
            "in": read(meta["input_file"], shapes[1]),
        }
    if not records:
        raise Invalid(f"no matmul records in {manifest}; capture with ';matmul_inputs=<stop>'")
    return records


def comparable_columns(a_in, b_in, weight_type):
    """(mask of comparable token columns, columns that are bit-identical)."""
    identical = np.all(a_in.view(np.uint32) == b_in.view(np.uint32), axis=0)
    if weight_type not in Q8_K_WEIGHTS or a_in.shape[0] % QK_K:
        return identical, identical
    same_q = np.all(quantize_q8_k(a_in) == quantize_q8_k(b_in), axis=1)
    return identical | same_q, identical


def _sci(v):
    return "-" if v is None else f"{v:.2e}"


def run_kernel(args) -> int:
    decl = load_declaration(args.declaration, "kernel")
    cpu, gpu = load_matmuls(args.cpu), load_matmuls(args.gpu)
    if set(cpu) != set(gpu):
        only = sorted(set(cpu) ^ set(gpu))[:5]
        raise Invalid(f"the two captures record different products, e.g. {only}")
    rows, passed = [], True
    print(f"declaration {decl['sha256']}  model={decl['model']}  "
          f"nmse_limit={decl['nmse_limit']:g}  "
          f"min_comparable_fraction={decl['min_comparable_fraction']:g}")
    print(f"{'eval':>4} {'weight':<28} {'type':<5} {'tokens':>6} {'same_in':>7} "
          f"{'in_nmse':>10} {'nmse':>10} {'worst_tok':>10}  verdict")
    for key in cpu:  # manifest order, which is graph order
        a, b = cpu[key], gpu[key]
        if (a["out"].shape != b["out"].shape or a["in"].shape != b["in"].shape
                or a["weight_type"] != b["weight_type"]):
            raise Invalid(f"shape or weight type differs at {key}")
        mask, identical = comparable_columns(a["in"], b["in"], a["weight_type"])
        tokens = int(mask.size)
        n_cmp = int(mask.sum())
        fraction = n_cmp / tokens
        nmse = worst = operand_nmse = None
        if n_cmp:
            ia = a["in"][:, mask].astype(np.float64)
            idiff = ia - b["in"][:, mask].astype(np.float64)
            iden = float((ia * ia).sum())
            operand_nmse = float((idiff * idiff).sum() / iden) if iden > 0 else 0.0
            ra = a["out"][:, mask].astype(np.float64)
            diff = ra - b["out"][:, mask].astype(np.float64)
            num, den = (diff * diff).sum(axis=0), (ra * ra).sum(axis=0)
            total = float(den.sum())
            nmse = float(num.sum() / total) if total > 0 else (0.0 if num.sum() == 0 else math.inf)
            with np.errstate(divide="ignore", invalid="ignore"):
                per = np.where(den > 0, num / den, np.where(num > 0, math.inf, 0.0))
            worst = float(per.max())
        ok = (nmse is not None and nmse <= decl["nmse_limit"]
              and fraction >= decl["min_comparable_fraction"])
        if nmse is None or fraction < decl["min_comparable_fraction"]:
            verdict = "FAIL (too few comparable columns)"
        else:
            verdict = "ok" if ok else "FAIL (kernel)"
        passed = passed and ok
        rows.append({
            "eval": key[0], "weight": key[1], "occ": key[2], "weight_type": a["weight_type"],
            "tokens": tokens, "comparable_tokens": n_cmp,
            "bit_identical_tokens": int(identical.sum()), "comparable_fraction": fraction,
            "nmse": nmse, "worst_token_nmse": worst, "operand_nmse": operand_nmse, "pass": ok,
        })
        print(f"{key[0]:>4} {key[1]:<28} {a['weight_type']:<5} {tokens:>6} {n_cmp:>7} "
              f"{_sci(operand_nmse):>10} {_sci(nmse):>10} {_sci(worst):>10}  {verdict}")
    summary = {
        "mode": "kernel", "adr": "0010 part 1", "declaration_sha256": decl["sha256"],
        "model": decl["model"], "nmse_limit": decl["nmse_limit"],
        "min_comparable_fraction": decl["min_comparable_fraction"],
        "cpu": str(args.cpu), "gpu": str(args.gpu), "products": len(rows),
        "pass": passed, "rows": rows,
    }
    return finish(args, summary)


# --- spread mode -------------------------------------------------------------


def load_logits(directory: Path, pid: str, sha: str):
    path = directory / f"{pid}.bin"
    raw = path.read_bytes() if path.is_file() else b""
    if not raw or len(raw) % 4:
        raise Invalid(f"missing or incomplete float32 dump: {path}")
    values = np.frombuffer(raw, dtype=np.float32)
    if not np.all(np.isfinite(values)):
        raise Invalid(f"non-finite logits: {path}")
    side_path = Path(str(path) + ".json")
    try:
        side = json.loads(side_path.read_text())
    except (OSError, ValueError) as e:
        raise Invalid(f"missing or unreadable sidecar {side_path}: {e}") from None
    if not isinstance(side, dict) or not isinstance(side.get("prompt"), str):
        raise Invalid(f"sidecar must record the prompt: {side_path}")
    if hashlib.sha256(side["prompt"].encode("utf-8")).hexdigest() != sha:
        raise Invalid(f"{side_path}: prompt is not the declared prompt {pid}")
    if "vocab_size" in side and side["vocab_size"] != values.size:
        raise Invalid(f"{side_path}: vocab_size does not match the dump")
    return values.astype(np.float64)


def pair_metrics(ref, other, top_k):
    diff = ref - other
    denom = float(np.dot(ref, ref))
    k = min(top_k, ref.size)
    overlap = len(set(np.argsort(ref)[-k:].tolist()) & set(np.argsort(other)[-k:].tolist())) / k
    return {
        "max_abs_diff": float(np.max(np.abs(diff))),
        "nmse": float(np.dot(diff, diff) / denom) if denom > 0 else math.inf,
        "top1_match": int(np.argmax(ref)) == int(np.argmax(other)),
        "topk_overlap": overlap,
    }


def run_spread(args) -> int:
    decl = load_declaration(args.declaration, "spread")
    rows = []
    for pid, sha in decl["prompts"].items():
        console = load_logits(args.cpu_console, pid, sha)
        host = load_logits(args.cpu_host, pid, sha)
        gpu = load_logits(args.gpu, pid, sha)
        if not console.size == host.size == gpu.size:
            raise Invalid(f"prompt {pid}: the three dumps have different sizes")
        rows.append({
            "prompt": pid,
            "reference": pair_metrics(console, host, decl["top_k"]),
            "gpu": pair_metrics(console, gpu, decl["top_k"]),
        })
    spread = {m: max(r["reference"][m] for r in rows) for m in ("max_abs_diff", "nmse")}
    worst = {m: max(r["gpu"][m] for r in rows) for m in ("max_abs_diff", "nmse")}
    limit = {m: decl["margin"] * spread[m] for m in spread}
    ranking_ok = all(r["gpu"]["top1_match"] and r["gpu"]["topk_overlap"] >= decl["top_k_min"]
                     for r in rows)
    passed = (ranking_ok and all(math.isfinite(worst[m]) and worst[m] <= limit[m] for m in worst))

    print(f"declaration {decl['sha256']}  model={decl['model']}  margin={decl['margin']:g}  "
          f"prompts={len(rows)}")
    print(f"{'prompt':<16} {'ref max_abs':>11} {'ref nmse':>10} {'gpu max_abs':>11} "
          f"{'gpu nmse':>10} {'top1':>5} {'top' + str(decl['top_k']):>6}")
    for r in rows:
        g, ref = r["gpu"], r["reference"]
        print(f"{r['prompt']:<16} {ref['max_abs_diff']:>11.6f} {ref['nmse']:>10.3e} "
              f"{g['max_abs_diff']:>11.6f} {g['nmse']:>10.3e} {str(g['top1_match']):>5} "
              f"{g['topk_overlap']:>6.2f}")
        if not ref["top1_match"]:
            print(f"  note: the two CPU references disagree on the top token of {r['prompt']}")
    for m in ("max_abs_diff", "nmse"):
        ratio = worst[m] / spread[m] if spread[m] > 0 else (1.0 if worst[m] == 0 else math.inf)
        print(f"  {m:<13}: gpu {worst[m]:.6g} vs spread {spread[m]:.6g} "
              f"(x{ratio:.3f}, limit x{decl['margin']:g})")
    summary = {
        "mode": "spread", "adr": "0010 part 2", "declaration_sha256": decl["sha256"],
        "model": decl["model"], "margin": decl["margin"], "top_k": decl["top_k"],
        "top_k_min": decl["top_k_min"], "reference_spread": spread, "gpu_worst": worst,
        "limit": limit, "ranking_ok": ranking_ok, "pass": passed, "rows": rows,
    }
    return finish(args, summary)


def finish(args, summary) -> int:
    if args.json_out:
        args.json_out.write_text(json.dumps(summary, indent=1) + "\n")
    print("✅ PASS" if summary["pass"] else "❌ FAIL")
    return 0 if summary["pass"] else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="ADR 0010 GPU control comparison.")
    sub = ap.add_subparsers(dest="mode", required=True)
    k = sub.add_parser("kernel", help="part 1: first-layer MUL_MAT results on equal operands")
    k.add_argument("cpu", type=Path, help="CPU arm capture directory")
    k.add_argument("gpu", type=Path, help="GPU arm capture directory")
    s = sub.add_parser("spread", help="part 2: GPU logits against the CPU reference spread")
    s.add_argument("--cpu-console", type=Path, required=True, help="console CPU arm dumps")
    s.add_argument("--cpu-host", type=Path, required=True, help="host CPU reference dumps")
    s.add_argument("--gpu", type=Path, required=True, help="console GPU arm dumps")
    for p in (k, s):
        p.add_argument("--declaration", type=Path, required=True,
                       help="criterion declared before the GPU arm was measured")
        p.add_argument("--json-out", type=Path, help="write the verdict and its rows as JSON")
    args = ap.parse_args()
    try:
        return run_kernel(args) if args.mode == "kernel" else run_spread(args)
    except (Invalid, OSError) as e:
        print(f"❌ invalid input: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
