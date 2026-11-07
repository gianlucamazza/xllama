#!/usr/bin/env python3
"""Validate a benchmark CSV without rewriting historical evidence."""

import argparse
import csv
import math
from pathlib import Path

CORE = {
    "model", "quant", "backend", "n_ctx", "n_threads", "prompt_tok_s",
    "decode_tok_s", "peak_ws_mb", "load_ms", "gpu_mem_mb", "gpu_budget_mb",
}
V1 = CORE | {"n_prompt_tok", "n_gen_tok", "max_length", "host", "date", "run_index"}
V2 = V1 | {"prefill_ms", "ttft_ms"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_file", type=Path)
    args = parser.parse_args()
    with args.csv_file.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or ())
        if len(fields) != len(reader.fieldnames or ()):
            parser.error("duplicate benchmark fields")
        if not CORE <= fields:
            missing = sorted(CORE - fields)
            parser.error(f"missing benchmark fields: {', '.join(missing)}")
        version = 2 if V2 <= fields else 1
        if version == 2 and fields != V2:
            parser.error("schema v2 contains unexpected or missing fields")
        for number, row in enumerate(reader, start=2):
            # Historical campaigns include single-field comment lines, not runs.
            if (row.get("model") or "").startswith("#") and all(
                value is None for field, value in row.items() if field != "model"
            ):
                continue
            if None in row or any(value is None for value in row.values()):
                parser.error(f"row {number} has the wrong field count")
            for field in ("n_ctx", "n_threads", "n_prompt_tok", "n_gen_tok", "max_length", "run_index"):
                if row.get(field, ""):
                    try:
                        value = int(row[field])
                    except ValueError:
                        parser.error(f"row {number}: {field} must be a non-negative integer")
                    if value < 0:
                        parser.error(f"row {number}: {field} must be a non-negative integer")
            for field in ("prompt_tok_s", "decode_tok_s", "peak_ws_mb", "load_ms",
                          "gpu_mem_mb", "gpu_budget_mb", "prefill_ms", "ttft_ms"):
                if row.get(field, ""):
                    try:
                        value = float(row[field])
                    except ValueError:
                        parser.error(f"row {number}: {field} must be finite and non-negative")
                    if not math.isfinite(value) or value < 0:
                        parser.error(f"row {number}: {field} must be finite and non-negative")
    print(f"OK schema=v{version} file={args.csv_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
