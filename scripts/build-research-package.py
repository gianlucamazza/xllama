#!/usr/bin/env python3
"""Validate the citable xLlama research package against benchmark evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
FIGURES = PAPER / "figures"
GENERATED = PAPER / "generated"


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def claims_digest(claims: dict) -> str:
    encoded = json.dumps(claims, indent=2, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2


def resolve_claim(claim: dict) -> dict:
    source = ROOT / "bench" / "results" / claim["source"]
    if not source.is_file():
        raise ValueError(f"missing claim source: {claim['source']}")
    matches = [
        row for row in load_rows(source)
        if all(row.get(key) == value for key, value in claim["selector"].items())
    ]
    if not matches:
        raise ValueError(f"claim selector has no match: {claim['id']}")
    selection = claim.get("selection", "only")
    repeats = [row for row in matches if row.get("run_index", "0") not in ("", "0")]
    if selection == "median":
        selected = repeats or matches
        if len(selected) < 2:
            raise ValueError(f"claim {claim['id']} requests median with fewer than two rows")
        values = {metric: median([float(row[metric]) for row in selected]) for metric in claim["metrics"]}
        return {"rows": selected, "values": values, "run_count": len(selected)}
    if selection == "max_decode":
        selected = max(matches, key=lambda row: float(row["decode_tok_s"]))
        return {"rows": [selected], "values": {metric: float(selected[metric]) for metric in claim["metrics"]}, "run_count": 1}
    if selection != "only" or len(matches) != 1:
        raise ValueError(f"claim {claim['id']} selector matched {len(matches)} rows; declare selection")
    selected = matches[0]
    return {"rows": [selected], "values": {metric: float(selected[metric]) for metric in claim["metrics"]}, "run_count": 1}


def validate_claims(claims: dict) -> dict[str, dict]:
    seen: set[str] = set()
    resolved = {}
    for claim in claims["claims"]:
        claim_id = claim["id"]
        if claim_id in seen:
            raise ValueError(f"duplicate claim id: {claim_id}")
        seen.add(claim_id)
        resolved[claim_id] = resolve_claim(claim)
        for metric in claim["metrics"]:
            if metric not in resolved[claim_id]["rows"][0]:
                raise ValueError(f"claim {claim_id} metric is absent: {metric}")
    return resolved


def current_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def decode_figure(claims: dict, resolved: dict[str, dict]) -> str:
    points = []
    for claim in claims["claims"]:
        if claim["id"] not in {"lfm25-230m-console-baseline", "lfm25-350m-console-baseline"}:
            continue
        points.append((claim["id"], resolved[claim["id"]]["values"]["decode_tok_s"]))
    bar_width, base, scale = 150, 180, 1.1
    bars = []
    labels = []
    for index, (claim_id, value) in enumerate(points):
        x = 90 + index * 190
        bar_height = round(value * scale)
        bars.append(f'<rect x="{x}" y="{base - bar_height}" width="{bar_width}" height="{bar_height}" fill="#3776ab"/>')
        label = "230M" if "230m" in claim_id else "350M"
        labels.append(f'<text x="{x + bar_width // 2}" y="205" text-anchor="middle">{label}</text>')
        labels.append(f'<text x="{x + bar_width // 2}" y="{base - bar_height - 8}" text-anchor="middle">{value:.1f}</text>')
    return "\n".join([
        '<svg xmlns="http://www.w3.org/2000/svg" width="520" height="220" viewBox="0 0 520 220">',
        '<title>Xbox Series S LFM decode throughput</title>',
        '<text x="260" y="20" text-anchor="middle">Decode tok/s (selected raw rows)</text>',
        '<line x1="60" y1="180" x2="480" y2="180" stroke="#333"/>',
        *bars,
        *labels,
        '</svg>',
        '',
    ])


def generated_table(claims: dict, resolved: dict[str, dict]) -> str:
    lines = [
        "# Generated research evidence",
        "",
        "| Claim | Status | Runs | Metrics | Source |",
        "| --- | --- | ---: | --- | --- |",
    ]
    for claim in claims["claims"]:
        result = resolved[claim["id"]]
        metrics = "; ".join(f"{key}={value:.2f}" for key, value in result["values"].items())
        lines.append(
            f"| `{claim['id']}` | {claim.get('status', 'historical')} | "
            f"{result['run_count']} | {metrics} | `{claim['source']}` |"
        )
    lines.extend(["", "Generated by `scripts/build-research-package.py`; do not edit by hand.", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="validate only")
    args = parser.parse_args()
    claims_path = PAPER / "claims.json"
    manifest_path = PAPER / "research-manifest.json"
    claims = json.loads(claims_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if claims.get("schema_version") != 1 or manifest.get("schema_version") != 1:
        raise SystemExit("unsupported research package schema")
    resolved = validate_claims(claims)
    digest = claims_digest(claims)
    if args.check and manifest.get("claims_sha256") != digest:
        raise SystemExit("research-manifest.json claims_sha256 is stale")
    if not args.check:
        manifest["claims_sha256"] = digest
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    FIGURES.mkdir(exist_ok=True)
    GENERATED.mkdir(exist_ok=True)
    table = GENERATED / "benchmarks.md"
    expected_table = generated_table(claims, resolved)
    if args.check:
        if not table.is_file() or table.read_text(encoding="utf-8") != expected_table:
            raise SystemExit("paper/generated/benchmarks.md is stale")
    else:
        table.write_text(expected_table, encoding="utf-8")
    figure = FIGURES / "model-decode.svg"
    expected_figure = decode_figure(claims, resolved)
    if args.check:
        if not figure.is_file() or figure.read_text(encoding="utf-8") != expected_figure:
            raise SystemExit("paper/figures/model-decode.svg is stale")
    else:
        figure.write_text(expected_figure, encoding="utf-8")
    print(
        f"OK research package claims={len(claims['claims'])} "
        f"source_commit={current_commit()} claims_sha256={digest}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
