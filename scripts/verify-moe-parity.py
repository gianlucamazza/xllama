#!/usr/bin/env python3
# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
"""Validate frozen SSD MoE reference rows and gate the complete streamed campaign."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np


from logit_comparison import compare_logits

STEPS = [0, 1, 2, 3, 5, 9]
CACHES = [2, 512]
STRICT_COMPARISON = {
    "max_abs_diff": 0, "nmse": 0, "top_k": 10, "top_k_min": 1,
    "same_top_token": True, "reject_nonfinite": True,
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def require_value(actual, expected, location):
    # JSON comparison keeps bools distinct from token IDs and cache sizes.
    require(json.dumps(actual, sort_keys=True) == json.dumps(expected, sort_keys=True),
            f"{location} does not match the campaign")


def unique_object(pairs):
    obj = {}
    for key, value in pairs:
        require(key not in obj, f"duplicate JSON key {key!r}")
        obj[key] = value
    return obj


def read_object(path):
    raw = path.read_bytes()
    try:
        obj = json.loads(raw.decode("utf-8"), object_pairs_hook=unique_object)
        require(type(obj) is dict, "expected a JSON object")
    except ValueError as error:
        raise ValueError(f"{path}: {error}") from error
    return raw, obj


def digest(data):
    return hashlib.sha256(data).hexdigest()


def check_hex(value, length, location):
    require(isinstance(value, str) and re.fullmatch(rf"[0-9a-f]{{{length}}}", value),
            f"{location} must be a lowercase {length}-digit hex value")


def check_tokens(ids, count, vocab, location):
    require(type(ids) is list and len(ids) == count and
            all(type(token) is int and 0 <= token < vocab for token in ids),
            f"{location} must contain {count} valid token IDs")


def validate_declaration(declaration):
    require_value(declaration["schema"], "xllama-moe-parity-preparation-v1", "declaration schema")
    execution = declaration["execution"]
    require_value(execution["reference_rows_after_feeding"], STEPS, "reference steps")
    require_value(execution["cache_slots"], CACHES, "cache profiles")
    for key, expected in STRICT_COMPARISON.items():
        require_value(declaration["comparison"][key], expected, f"comparison.{key}")
    identity = {
        "checkpoint_sha256": declaration["checkpoint"]["sha256"],
        **{key: declaration[key] for key in ("template_sha256", "tokenizer_sha256", "vocab_sha256")},
    }
    for key, value in identity.items():
        check_hex(value, 64, key)

    cases = declaration["cases"]
    require(type(cases) is list and len(cases) == 4, "expected four declared cases")
    case_ids = set()
    for case in cases:
        name = case["id"]
        require(isinstance(name, str) and re.fullmatch(r"[a-z0-9]+(?:-[a-z0-9]+)*", name),
                "invalid declared case ID")
        require(name not in case_ids, f"duplicate case {name}")
        case_ids.add(name)
        vocab = case["vocab_size"]
        require(type(vocab) is int and vocab >= STRICT_COMPARISON["top_k"], f"{name}: invalid vocabulary size")
        prompt = case["prompt_token_ids"]
        require(0 < len(prompt) <= execution["prompt_tokens_max"], f"{name}: invalid prompt length")
        check_tokens(prompt, len(prompt), vocab, f"{name}: prompt")
        bos = case["bos_token_id"]
        require(type(bos) is int and prompt[0] == bos and prompt.count(bos) == 1,
                f"{name}: expected exactly one leading BOS")
        check_tokens(case["teacher_forced_token_ids"], STEPS[-1], vocab, f"{name}: continuation")
        check_hex(case["rendered_prompt_sha256"], 64, f"{name}: rendered prompt hash")
        hashes = case["resident_row_sha256_by_fed_tokens"]
        require(type(hashes) is dict and set(hashes) == set(map(str, STEPS)),
                f"{name}: incomplete resident row hashes")
        for value in hashes.values():
            check_hex(value, 64, f"{name}: resident row hash")
    return identity


def validate_receipt(receipt, declaration, declaration_hash):
    require_value(receipt["schema"], "xllama-moe-parity-receipt-v1", "receipt schema")
    require_value(receipt["declaration_sha256"], declaration_hash, "receipt declaration hash")
    candidate = receipt["candidate_commit"]
    check_hex(candidate, 40, "candidate commit")
    require(candidate != "0" * 40, "candidate commit cannot be the null object ID")
    pinned_candidate = declaration["source"]["upstream_candidate_commit"]
    if pinned_candidate is not None:
        require_value(candidate, pinned_candidate, "receipt candidate commit")

    review = receipt["cpu_kernel_review"]
    require(isinstance(review, str) and review.strip(), "CPU kernel equivalence needs a review reference")
    reference = declaration["resident_reference"]
    execution = declaration["execution"]
    settings = {
        **{key: reference[key] for key in ("platform", "cpu", "compiler", "cmake", "environment")},
        **{key: execution[key] for key in ("threads", "context", "n_batch", "n_ubatch")},
        "repacking": False,
        "flash_attention": False,
    }
    require_value(receipt["cpu_settings"], settings, "receipt CPU settings")

    builds = receipt["streamed_build_sha256"]
    require(type(builds) is dict and set(builds) == set(map(str, CACHES)),
            "receipt must bind both streamed builds")
    for cache, value in builds.items():
        check_hex(value, 64, f"cache {cache} build hash")
    return candidate, builds


def load_row(path, vocab_size, expected_hash):
    raw = path.read_bytes()
    require(len(raw) == vocab_size * 4, f"{path}: expected one full float32 vocabulary row")
    values = np.frombuffer(raw, dtype="<f4")
    require(np.isfinite(values).all(), f"{path}: nonfinite logits")
    require_value(digest(raw), expected_hash, f"{path}: row SHA-256")
    return values


def load_reference(directory, cases, template_hash):
    template = directory / "chat-template.jinja"
    require_value(digest(template.read_bytes()), template_hash, f"{template}: template SHA-256")
    rows = {}
    for case in cases:
        name = case["id"]
        rendered = directory / name / "rendered.txt"
        require_value(digest(rendered.read_bytes()), case["rendered_prompt_sha256"],
                      f"{rendered}: rendered prompt SHA-256")
        metadata = directory / name / "tokens.json"
        _, tokens = read_object(metadata)
        for key in ("vocab_size", "bos_token_id", "prompt_token_ids", "teacher_forced_token_ids"):
            require_value(tokens.get(key), case[key], f"{metadata}: {key}")
        for step in STEPS:
            path = directory / name / f"row-{step}.bin"
            rows[name, step] = load_row(
                path, case["vocab_size"], case["resident_row_sha256_by_fed_tokens"][str(step)],
            )
    return rows


def load_streamed(directory, cases, identity):
    rows = {}
    for case in cases:
        name = case["id"]
        for step in STEPS:
            path = directory / name / f"row-{step}.bin"
            _, sidecar = read_object(path.with_suffix(".bin.json"))
            for key, expected in {
                **identity,
                "vocab_size": case["vocab_size"],
                "prompt_token_ids": case["prompt_token_ids"],
                "fed_continuation_token_ids": case["teacher_forced_token_ids"][:step],
            }.items():
                require_value(sidecar.get(key), expected, f"{path}: sidecar {key}")
            prompt = sidecar.get("prompt")
            require(isinstance(prompt, str) and digest(prompt.encode("utf-8")) == case["rendered_prompt_sha256"],
                    f"{path}: rendered prompt hash mismatch")
            rows[name, step] = load_row(path, case["vocab_size"], sidecar.get("row_sha256"))
    return rows


def compare_campaign(reference, streamed, thresholds):
    results = []
    for cache, rows in streamed.items():
        for (case, step), candidate in rows.items():
            metrics = compare_logits(reference[case, step], candidate, thresholds["top_k"])
            passed = metrics.passes(thresholds["max_abs_diff"], thresholds["nmse"], thresholds["top_k_min"])
            results.append({
                "case": case, "step": step, "cache_slots": cache,
                "status": "PASS" if passed else "FAIL",
                "max_abs_diff": metrics.max_abs_diff,
                "nmse": metrics.nmse if math.isfinite(metrics.nmse) else None,
                "top1_match": metrics.top1_match,
                "top_k_overlap": metrics.top_k_overlap,
            })
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("declaration", type=Path)
    parser.add_argument("resident", type=Path)
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--streamed-2", type=Path)
    parser.add_argument("--streamed-512", type=Path)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args()

    report = {"schema": "xllama-moe-parity-report-v1", "pairs": []}
    try:
        streamed = {2: args.streamed_2, 512: args.streamed_512}
        complete_inputs = [*streamed.values(), args.receipt]
        if args.reference_only:
            require(not any(complete_inputs), "--reference-only takes only the declaration and resident directory")
        else:
            require(all(complete_inputs), "full parity needs --streamed-2, --streamed-512 and --receipt")
            require(len({path.resolve() for path in (args.resident, *streamed.values())}) == 3,
                    "resident and streamed directories must be distinct")

        raw, declaration = read_object(args.declaration)
        declaration_hash = digest(raw)
        identity = validate_declaration(declaration)
        report["declaration_sha256"] = declaration_hash
        cases = declaration["cases"]
        reference = load_reference(args.resident, cases, identity["template_sha256"])
        report["resident_rows"] = len(reference)
        if args.reference_only:
            report["status"], code = "REFERENCE_VERIFIED", 0
        else:
            receipt_raw, receipt = read_object(args.receipt)
            candidate, builds = validate_receipt(receipt, declaration, declaration_hash)
            report.update(
                candidate_commit=candidate, receipt_sha256=digest(receipt_raw), streamed_build_sha256=builds,
            )
            # The row arrays retain the immutable bytes checked by the loaders.
            # Complete preflight before comparing any pair.
            captures = {
                cache: load_streamed(directory, cases, {
                    **identity, "candidate_commit": candidate, "cache_slots": cache,
                    "build_sha256": builds[str(cache)],
                })
                for cache, directory in streamed.items()
            }
            report["pairs"] = compare_campaign(reference, captures, declaration["comparison"])
            passed = all(result["status"] == "PASS" for result in report["pairs"])
            report["status"], code = ("PASS", 0) if passed else ("FAIL", 1)
    except (OSError, ValueError, TypeError, KeyError) as error:
        report["status"], code = "INVALID_INPUT", 2
        report["error"] = str(error)
    print(json.dumps(report, indent=2, allow_nan=False))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
