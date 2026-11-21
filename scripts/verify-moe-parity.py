#!/usr/bin/env python3
# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
"""Validate frozen SSD MoE reference rows and gate the complete streamed campaign."""

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
import tempfile

import numpy as np


COMPARATOR = Path(__file__).with_name("compare-logits.py")
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


def validate_row(path, case, step, identity, streamed):
    _, sidecar = read_object(path.with_suffix(".bin.json"))
    for key, expected in {
        **identity,
        "vocab_size": case["vocab_size"],
        "prompt_token_ids": case["prompt_token_ids"],
        "fed_continuation_token_ids": case["teacher_forced_token_ids"][:step],
        **streamed,
    }.items():
        require_value(sidecar.get(key), expected, f"{path}: sidecar {key}")
    prompt = sidecar.get("prompt")
    require(isinstance(prompt, str) and digest(prompt.encode("utf-8")) == case["rendered_prompt_sha256"],
            f"{path}: rendered prompt hash mismatch")
    raw = path.read_bytes()
    require(len(raw) == case["vocab_size"] * 4, f"{path}: expected one full float32 vocabulary row")
    require(np.isfinite(np.frombuffer(raw, dtype="<f4")).all(), f"{path}: nonfinite logits")
    expected = sidecar.get("row_sha256") if streamed else case["resident_row_sha256_by_fed_tokens"][str(step)]
    require_value(digest(raw), expected, f"{path}: row SHA-256")
    return raw


def validate_arm(directory, declaration, identity, streamed, snapshot=None):
    for case in declaration["cases"]:
        name = case["id"]
        if snapshot is not None:
            (snapshot / name).mkdir(parents=True)
        for step in STEPS:
            file = f"row-{step}.bin"
            raw = validate_row(directory / name / file, case, step, identity, streamed)
            if snapshot is not None:
                (snapshot / name / file).write_bytes(raw)


def compare(declaration, reference, streamed):
    flags = [
        value for key in ("max_abs_diff", "nmse", "top_k", "top_k_min")
        for value in ("--" + key.replace("_", "-"), str(declaration["comparison"][key]))
    ]
    results = []
    for cache, directory in streamed.items():
        for case in declaration["cases"]:
            name = case["id"]
            for step in STEPS:
                row = Path(name) / f"row-{step}.bin"
                proc = subprocess.run(
                    [sys.executable, str(COMPARATOR), str(reference / row), str(directory / row), *flags],
                    capture_output=True, text=True, check=False,
                )
                require(proc.returncode in (0, 1),
                        f"comparator error for cache {cache}, {row}: {(proc.stderr or proc.stdout).strip()}")
                results.append({
                    "case": name, "step": step, "cache_slots": cache,
                    "status": "FAIL" if proc.returncode else "PASS",
                    "comparator_output": "\n".join(proc.stdout.splitlines()[2:]),
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
        resident_rows = len(declaration["cases"]) * len(STEPS)
        if args.reference_only:
            validate_arm(args.resident, declaration, identity, {})
            report["resident_rows"] = resident_rows
            report["status"], code = "REFERENCE_VERIFIED", 0
        else:
            receipt_raw, receipt = read_object(args.receipt)
            candidate, builds = validate_receipt(receipt, declaration, declaration_hash)
            report.update(
                candidate_commit=candidate, receipt_sha256=digest(receipt_raw), streamed_build_sha256=builds,
            )
            # Compare the bytes just verified. Optional sidecar annotations and
            # later changes to the source files cannot affect the verdict.
            with tempfile.TemporaryDirectory(prefix="xllama-moe-parity-") as snapshot:
                root = Path(snapshot)
                reference = root / "resident"
                validate_arm(args.resident, declaration, identity, {}, reference)
                report["resident_rows"] = resident_rows
                copies = {cache: root / str(cache) for cache in streamed}
                for cache, directory in streamed.items():
                    validate_arm(directory, declaration, identity, {
                        "candidate_commit": candidate, "cache_slots": cache,
                        "build_sha256": builds[str(cache)],
                    }, copies[cache])
                report["pairs"] = compare(declaration, reference, copies)
            passed = all(result["status"] == "PASS" for result in report["pairs"])
            report["status"], code = ("PASS", 0) if passed else ("FAIL", 1)
    except (OSError, ValueError, TypeError, KeyError) as error:
        report["status"], code = "INVALID_INPUT", 2
        report["error"] = str(error)
    print(json.dumps(report, indent=2))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
