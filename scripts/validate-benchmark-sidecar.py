#!/usr/bin/env python3
"""Validate provenance metadata for a publishable benchmark claim."""

import argparse
import hashlib
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("sidecar", type=Path)
    parser.add_argument("--prompt", type=Path)
    args = parser.parse_args()
    try:
        data = json.loads(args.sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        parser.error(f"cannot read sidecar: {exc}")
    required = {
        "schema_version", "model", "model_bytes", "model_sha256", "quant", "build",
        "hardware", "n_ctx", "prompt_sha256", "sampling", "warmup_runs",
        "recorded_runs", "thermal_rule", "ambient_c", "power",
    }
    missing = sorted(required - data.keys())
    if missing:
        parser.error(f"missing sidecar fields: {', '.join(missing)}")
    if data["schema_version"] != 1:
        parser.error("unsupported sidecar schema")
    if not isinstance(data["model_bytes"], int) or data["model_bytes"] <= 0:
        parser.error("model_bytes must be positive")
    digest = data["model_sha256"]
    if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        parser.error("model_sha256 must be lowercase SHA-256")
    if data["warmup_runs"] < 1 or data["recorded_runs"] < 3:
        parser.error("publishable claims require warmup_runs >= 1 and recorded_runs >= 3")
    power = data["power"]
    for field in ("wattmeter", "sample_interval_ms", "idle_w", "prefill_w", "decode_w"):
        if field not in power:
            parser.error(f"missing power field: {field}")
    if args.prompt:
        actual = hashlib.sha256(args.prompt.read_bytes()).hexdigest()
        if actual != data["prompt_sha256"]:
            parser.error("prompt_sha256 does not match --prompt")
    print(f"OK sidecar schema=1 file={args.sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
