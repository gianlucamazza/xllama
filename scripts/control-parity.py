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
# Two more modes implement the 2026-11-15 amendment of ADR 0010, which is the
# criterion for new controls (declaration schema v2):
#
#   scripts/control-parity.py reference --declaration D --model M.gguf
#                                       <capture_dir>...
#   scripts/control-parity.py distribution --declaration D --cpu-console DIR
#                                          --cpu-host DIR --gpu DIR
#
# reference (part 1): every weight MUL_MAT of each capture must be within the
#   declared NMSE limit of a float64 product of the dequantized weight and the
#   arm's own quantized operand (scripts/check-matmul-reference.py; q8_K for
#   K-quant weights, q8_0 for Q4_0, Q5_0 and Q8_0). One arm is
#   enough, so every token column and every captured layer is judged. Each
#   declared capture must reach its declared last layer.
#
# distribution (part 2): each directory holds `<prompt id>.bin.positions.bin`
#   (+ `.json`), the logits at many prompt positions. Per position it takes
#   the KL divergence from the console CPU arm to the GPU arm and to the host
#   CPU arm, and whether the top token agrees. The GPU arm passes when, at the
#   declared confidence (bootstrap over prompts):
#     mean KLD(cpu, gpu) <= kld_ratio_limit x mean KLD(cpu, host), and
#     top-token agreement(cpu, gpu) >= agreement(cpu, host) - top1_margin,
#   over at least the declared number of positions. `distribution.prompts`
#   names which declared prompts this part uses.
#
#   Schema v2 declaration:
#   { "schema": "xllama.control-declaration/v2", "model": "<catalogue id>",
#     "reference": { "nmse_limit": 1e-7,
#                    "captures": [ { "prompt": "k1", "last_layer": 35 } ] },
#     "distribution": { "kld_ratio_limit": 1.5, "top1_margin": 0.01,
#                       "confidence": 0.95, "min_positions": 4000,
#                       "bootstrap": 2000, "seed": 1,
#                       "prompts": [ "p1" ] },
#     "prompts": [ { "id": "p1", "sha256": "<sha256 of the UTF-8 prompt>" } ] }
#
# This is not scripts/compare-logits.py, whose frozen gate and defaults are
# unchanged. Exit 0 = PASS, 1 = FAIL, 2 = invalid input or declaration.

import argparse
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

import numpy as np

SCHEMA = "xllama.control-declaration/v1"
SCHEMA_V2 = "xllama.control-declaration/v2"
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
    want = SCHEMA_V2 if need in ("reference", "distribution") else SCHEMA
    if not isinstance(decl, dict) or decl.get("schema") != want:
        raise Invalid(f"declaration schema must be {want} for the {need} mode")
    if not isinstance(decl.get("model"), str) or not decl["model"]:
        raise Invalid("declaration model must be a non-empty string")
    out = {"model": decl["model"], "sha256": hashlib.sha256(raw).hexdigest()}
    if need == "kernel":
        k = decl.get("kernel")
        out["nmse_limit"] = _number(k, "nmse_limit", "kernel", 0.0, 1.0, lo_open=True)
        out["min_comparable_fraction"] = _number(
            k, "min_comparable_fraction", "kernel", 0.0, 1.0, lo_open=True)
    elif need == "reference":
        r = decl.get("reference")
        out["nmse_limit"] = _number(r, "nmse_limit", "reference", 0.0, 1.0, lo_open=True)
        caps = r.get("captures")
        if not isinstance(caps, list) or not caps:
            raise Invalid("declaration reference.captures must be a non-empty list")
        out["captures"] = {}
        for c in caps:
            pid, last = (c.get("prompt"), c.get("last_layer")) if isinstance(c, dict) else (None, None)
            if not isinstance(pid, str) or not pid or type(last) is not int or last < 0:
                raise Invalid("declaration reference.captures entries need prompt and last_layer")
            out["captures"][pid] = last
        out["prompts"] = _prompts(decl)
        if any(pid not in out["prompts"] for pid in out["captures"]):
            raise Invalid("declaration reference.captures must name declared prompt ids")
    elif need == "distribution":
        d = decl.get("distribution")
        out["kld_ratio_limit"] = _number(d, "kld_ratio_limit", "distribution", 1.0, 1e6)
        out["top1_margin"] = _number(d, "top1_margin", "distribution", 0.0, 1.0)
        out["confidence"] = _number(d, "confidence", "distribution", 0.5, 1.0, lo_open=True)
        if out["confidence"] >= 1.0:
            raise Invalid("declaration distribution.confidence must be below 1")
        for key, lo in (("min_positions", 1), ("bootstrap", 100), ("seed", 0)):
            v = d.get(key)
            if type(v) is not int or v < lo:
                raise Invalid(f"declaration distribution.{key} must be an integer >= {lo}")
            out[key] = v
        declared = _prompts(decl)
        ids = d.get("prompts")
        if (not isinstance(ids, list) or not ids or len(set(ids)) != len(ids)
                or any(i not in declared for i in ids)):
            raise Invalid("declaration distribution.prompts must list distinct declared prompt ids")
        out["prompts"] = {i: declared[i] for i in ids}
    else:
        s = decl.get("spread")
        out["margin"] = _number(s, "margin", "spread", 1.0, 1e6)
        out["top_k_min"] = _number(s, "top_k_min", "spread", 0.0, 1.0)
        top_k = s.get("top_k")
        if type(top_k) is not int or top_k <= 0:
            raise Invalid("declaration spread.top_k must be a positive integer")
        out["top_k"] = top_k
        out["prompts"] = _prompts(decl)
    return out


def _prompts(decl):
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
    return seen


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


# --- reference mode (amendment, part 1) ---------------------------------------


def _load_reference_tool():
    import importlib.util
    path = Path(__file__).resolve().parent / "check-matmul-reference.py"
    spec = importlib.util.spec_from_file_location("check_matmul_reference", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_rows(records, weights, tool):
    """Per-record error of one capture; `weights(name)` returns [n_out, n_in] float64."""
    rows = []
    for key, r in records.items():
        quantizer = tool.operand_quantizer(r["weight_type"], r["in"].shape[0])
        if quantizer is None:
            raise Invalid(f"{key[1]} ({r['weight_type']}): no reference for this weight type")
        w = weights(key[1])
        if w.shape != (r["out"].shape[0], r["in"].shape[0]):
            raise Invalid(f"{key[1]}: weight shape does not match the capture")
        rows.append({"eval": key[0], "weight": key[1], "weight_type": r["weight_type"],
                     **tool.record_error(w, r["in"], r["out"], quantizer)})
    return rows


def run_reference(args) -> int:
    decl = load_declaration(args.declaration, "reference")
    tool = _load_reference_tool()
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "llama.cpp/gguf-py"))
        import gguf
        from gguf.quants import dequantize
    except ImportError as e:
        raise Invalid(f"gguf-py is not importable ({e})") from None
    tensors = {t.name: t for t in gguf.GGUFReader(args.model).tensors}
    blocks = [int(m.group(1)) for m in (re.match(r"blk\.(\d+)\.", n) for n in tensors) if m]
    if not blocks:
        raise Invalid(f"{args.model.name} has no blk.N tensors")
    last_block = max(blocks)
    if max(decl["captures"].values()) != last_block or any(
            v > last_block for v in decl["captures"].values()):
        raise Invalid(f"declared captures must stay within layer {last_block} and one must "
                      f"reach it")

    def weights(name):
        if name not in tensors:
            raise Invalid(f"{name} is not a tensor of {args.model.name}")
        t = tensors[name]
        return dequantize(t.data, t.tensor_type).astype(np.float64)

    captures, seen, passed, worst = [], set(), True, 0.0
    print(f"declaration {decl['sha256']}  model={decl['model']}  "
          f"nmse_limit={decl['nmse_limit']:g}")
    for cap in args.captures:
        pid = cap.name
        if pid not in decl["captures"]:
            raise Invalid(f"capture directory {pid} is not a declared capture prompt")
        records = load_matmuls(cap)
        need = f"blk.{decl['captures'][pid]}."
        if not any(k[1].startswith(need) for k in records):
            raise Invalid(f"{cap}: capture does not reach layer {decl['captures'][pid]}")
        rows = reference_rows(records, weights, tool)
        cap_worst = max(r["nmse"] for r in rows)
        ok = math.isfinite(cap_worst) and cap_worst <= decl["nmse_limit"]
        passed, worst = passed and ok, max(worst, cap_worst)
        seen.add(pid)
        captures.append({"arm": args.arm, "prompt": pid, "records": len(rows),
                         "worst_nmse": cap_worst,
                         "worst_token_nmse": max(r["worst_token_nmse"] for r in rows),
                         "pass": ok, "rows": rows})
        print(f"  {pid:<20} records {len(rows):>4}  worst NMSE {cap_worst:.3e}  "
              f"{'ok' if ok else 'FAIL'}")
    missing = sorted(set(decl["captures"]) - seen)
    if missing:
        raise Invalid(f"declared captures not provided: {missing}")
    return finish(args, {
        "mode": "reference", "adr": "0010 amendment part 1",
        "declaration_sha256": decl["sha256"], "model": decl["model"], "arm": args.arm,
        "nmse_limit": decl["nmse_limit"], "worst_nmse": worst, "pass": passed,
        "captures": captures,
    })


# --- distribution mode (amendment, part 2) -------------------------------------


def load_positions(directory: Path, pid: str, sha: str):
    base = directory / f"{pid}.bin.positions"
    try:
        side = json.loads(Path(str(base) + ".json").read_text())
    except (OSError, ValueError) as e:
        raise Invalid(f"missing or unreadable sidecar {base}.json: {e}") from None
    if not isinstance(side, dict) or not isinstance(side.get("prompt"), str):
        raise Invalid(f"sidecar must record the prompt: {base}.json")
    if hashlib.sha256(side["prompt"].encode("utf-8")).hexdigest() != sha:
        raise Invalid(f"{base}.json: prompt is not the declared prompt {pid}")
    vocab, positions = side.get("vocab_size"), side.get("positions")
    if (type(vocab) is not int or vocab <= 0 or not isinstance(positions, list) or not positions
            or any(type(p) is not int for p in positions)):
        raise Invalid(f"{base}.json: needs vocab_size and a non-empty positions list")
    if positions[0] < 0 or any(b <= a for a, b in zip(positions, positions[1:])):
        raise Invalid(f"{base}.json: positions must be distinct and increasing")
    path = Path(str(base) + ".bin")
    count = len(positions) * vocab
    try:
        with path.open("rb") as dump:
            if os.fstat(dump.fileno()).st_size != count * np.dtype(np.float32).itemsize:
                raise Invalid(f"{path}: size does not match {len(positions)} x {vocab} float32")
            data = np.fromfile(dump, dtype=np.float32, count=count)
            if data.size != count or dump.read(1):
                raise Invalid(f"{path}: size changed while reading the dump")
    except OSError as e:
        raise Invalid(f"unable to read dump {path}: {e}") from None
    if not np.all(np.isfinite(data)):
        raise Invalid(f"non-finite logits: {path}")
    return data.reshape(len(positions), vocab), positions


def log_softmax(x):
    x = x.astype(np.float64)
    x = x - x.max(axis=1, keepdims=True)
    return x - np.log(np.exp(x).sum(axis=1, keepdims=True))


def position_stats(ref, other):
    """Per position: KL(ref || other) in nats and top-token agreement."""
    lr, lo = log_softmax(ref), log_softmax(other)
    kld = (np.exp(lr) * (lr - lo)).sum(axis=1)
    return np.maximum(kld, 0.0), ref.argmax(axis=1) == other.argmax(axis=1)


def run_distribution(args) -> int:
    decl = load_declaration(args.declaration, "distribution")
    per = []  # per prompt: n, sums of kld and agreement for gpu and host
    for pid, sha in decl["prompts"].items():
        c, pc = load_positions(args.cpu_console, pid, sha)
        h, ph = load_positions(args.cpu_host, pid, sha)
        g, pg = load_positions(args.gpu, pid, sha)
        if not (pc == ph == pg) or not (c.shape == h.shape == g.shape):
            raise Invalid(f"prompt {pid}: the three dumps cover different positions")
        kg, ag = position_stats(c, g)
        kh, ah = position_stats(c, h)
        per.append({"prompt": pid, "positions": len(pc), "kld_gpu": float(kg.sum()),
                    "kld_host": float(kh.sum()), "top1_gpu": int(ag.sum()),
                    "top1_host": int(ah.sum()), "kld_gpu_max": float(kg.max()),
                    "kld_host_max": float(kh.max())})
    n = np.array([p["positions"] for p in per], dtype=np.float64)
    cols = {k: np.array([p[k] for p in per], dtype=np.float64)
            for k in ("kld_gpu", "kld_host", "top1_gpu", "top1_host")}
    total = int(n.sum())

    def stats(idx):
        m = n[idx].sum()
        kg, kh = cols["kld_gpu"][idx].sum() / m, cols["kld_host"][idx].sum() / m
        ratio = kg / kh if kh > 0 else (1.0 if kg == 0 else math.inf)
        deficit = (cols["top1_host"][idx].sum() - cols["top1_gpu"][idx].sum()) / m
        return ratio, deficit, kg, kh

    everything = np.arange(len(per))
    ratio, deficit, kld_gpu, kld_host = stats(everything)
    rng = np.random.default_rng(decl["seed"])
    boot = np.array([stats(rng.integers(0, len(per), len(per)))[:2]
                     for _ in range(decl["bootstrap"])])
    ratio_ub = float(np.quantile(boot[:, 0], decl["confidence"]))
    deficit_ub = float(np.quantile(boot[:, 1], decl["confidence"]))
    enough = total >= decl["min_positions"]
    passed = (enough and math.isfinite(ratio_ub) and ratio_ub <= decl["kld_ratio_limit"]
              and deficit_ub <= decl["top1_margin"])

    print(f"declaration {decl['sha256']}  model={decl['model']}  prompts={len(per)}  "
          f"positions={total} (declared minimum {decl['min_positions']})")
    print(f"{'prompt':<16} {'pos':>5} {'KLD gpu':>10} {'KLD host':>10} {'top1 gpu':>9} "
          f"{'top1 host':>9}")
    for p in per:
        print(f"{p['prompt']:<16} {p['positions']:>5} {p['kld_gpu'] / p['positions']:>10.3e} "
              f"{p['kld_host'] / p['positions']:>10.3e} {p['top1_gpu'] / p['positions']:>9.4f} "
              f"{p['top1_host'] / p['positions']:>9.4f}")
    print(f"  mean KLD: gpu {kld_gpu:.4e}, host {kld_host:.4e}; ratio {ratio:.3f}, "
          f"upper bound {ratio_ub:.3f} (limit {decl['kld_ratio_limit']:g})")
    print(f"  top-token agreement deficit: {deficit:+.4f}, upper bound {deficit_ub:+.4f} "
          f"(margin {decl['top1_margin']:g})")
    if not enough:
        print("  too few positions for the declared minimum")
    return finish(args, {
        "mode": "distribution", "adr": "0010 amendment part 2",
        "declaration_sha256": decl["sha256"], "model": decl["model"],
        "positions": total, "min_positions": decl["min_positions"],
        "confidence": decl["confidence"], "bootstrap": decl["bootstrap"], "seed": decl["seed"],
        "mean_kld_gpu": kld_gpu, "mean_kld_host": kld_host, "kld_ratio": ratio,
        "kld_ratio_upper_bound": ratio_ub, "kld_ratio_limit": decl["kld_ratio_limit"],
        "top1_agreement_gpu": float(cols["top1_gpu"].sum() / total),
        "top1_agreement_host": float(cols["top1_host"].sum() / total),
        "top1_deficit": deficit, "top1_deficit_upper_bound": deficit_ub,
        "top1_margin": decl["top1_margin"], "pass": passed, "rows": per,
    })


def _finite(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: _finite(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_finite(v) for v in value]
    return value


def finish(args, summary) -> int:
    if args.json_out:
        # A non-finite figure (an undefined ratio, for instance) becomes null,
        # so the verdict file is always standard JSON.
        args.json_out.write_text(json.dumps(_finite(summary), indent=1, allow_nan=False) + "\n")
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
    r = sub.add_parser("reference", help="amendment part 1: products against an exact reference")
    r.add_argument("--model", type=Path, required=True, help="the GGUF the arm loaded")
    r.add_argument("--arm", default="gpu", help="label written into the verdict (default gpu)")
    r.add_argument("captures", type=Path, nargs="+",
                   help="capture directories named after their declared prompt id")
    d = sub.add_parser("distribution", help="amendment part 2: per-position KLD and top token")
    d.add_argument("--cpu-console", type=Path, required=True, help="console CPU arm dumps")
    d.add_argument("--cpu-host", type=Path, required=True, help="host CPU reference dumps")
    d.add_argument("--gpu", type=Path, required=True, help="console GPU arm dumps")
    for p in (k, s, r, d):
        p.add_argument("--declaration", type=Path, required=True,
                       help="criterion declared before the GPU arm was measured")
        p.add_argument("--json-out", type=Path, help="write the verdict and its rows as JSON")
    args = ap.parse_args()
    try:
        return {"kernel": run_kernel, "spread": run_spread, "reference": run_reference,
                "distribution": run_distribution}[args.mode](args)
    except (Invalid, OSError) as e:
        print(f"❌ invalid input: {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
