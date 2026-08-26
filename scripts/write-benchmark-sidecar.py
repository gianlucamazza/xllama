#!/usr/bin/env python3
"""Write the provenance sidecar required for a publishable benchmark claim."""

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-bytes", type=int, required=True)
    parser.add_argument("--model-sha256", required=True)
    parser.add_argument("--quant", required=True)
    parser.add_argument("--build", required=True)
    parser.add_argument("--hardware", required=True)
    parser.add_argument("--ctx", type=int, required=True)
    parser.add_argument("--prompt", type=Path, required=True)
    parser.add_argument("--sampling", required=True, help="JSON object")
    parser.add_argument("--warmup-runs", type=int, required=True)
    parser.add_argument("--recorded-runs", type=int, required=True)
    parser.add_argument("--thermal-rule", required=True)
    parser.add_argument("--ambient-c", type=float, required=True)
    parser.add_argument("--wattmeter", required=True)
    parser.add_argument("--power-sample-ms", type=int, required=True)
    parser.add_argument("--idle-w", type=float, required=True)
    parser.add_argument("--prefill-w", type=float, required=True)
    parser.add_argument("--decode-w", type=float, required=True)
    args = parser.parse_args()

    if args.warmup_runs < 1 or args.recorded_runs < 3:
        parser.error("publishable claims require at least one warm-up and three recorded runs")
    try:
        sampling = json.loads(args.sampling)
    except json.JSONDecodeError as exc:
        parser.error(f"--sampling must be valid JSON: {exc}")
    if not isinstance(sampling, dict):
        parser.error("--sampling must be a JSON object")

    prompt_hash = hashlib.sha256(args.prompt.read_bytes()).hexdigest()
    payload = {
        "schema_version": 1,
        "model": args.model,
        "model_bytes": args.model_bytes,
        "model_sha256": args.model_sha256,
        "quant": args.quant,
        "build": args.build,
        "hardware": args.hardware,
        "n_ctx": args.ctx,
        "prompt_sha256": prompt_hash,
        "sampling": sampling,
        "warmup_runs": args.warmup_runs,
        "recorded_runs": args.recorded_runs,
        "thermal_rule": args.thermal_rule,
        "ambient_c": args.ambient_c,
        "power": {
            "wattmeter": args.wattmeter,
            "sample_interval_ms": args.power_sample_ms,
            "idle_w": args.idle_w,
            "prefill_w": args.prefill_w,
            "decode_w": args.decode_w,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{args.output.name}.", dir=args.output.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, args.output)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
