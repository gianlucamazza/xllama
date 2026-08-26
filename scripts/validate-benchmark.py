#!/usr/bin/env python3
"""Validate a benchmark CSV without rewriting historical evidence."""

import argparse
import csv
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
        if not CORE <= fields:
            missing = sorted(CORE - fields)
            parser.error(f"missing benchmark fields: {', '.join(missing)}")
        version = 2 if V2 <= fields else 1
        if version == 2 and fields != V2:
            parser.error("schema v2 contains unexpected or missing fields")
        for number, row in enumerate(reader, start=2):
            if len(row) != len(reader.fieldnames or ()):
                parser.error(f"row {number} has the wrong field count")
            for field in ("n_ctx", "n_prompt_tok", "n_gen_tok", "run_index"):
                if row.get(field, "") and int(row[field]) < 0:
                    parser.error(f"row {number}: {field} must be non-negative")
            for field in ("prompt_tok_s", "decode_tok_s", "load_ms", "prefill_ms", "ttft_ms"):
                if row.get(field, "") and float(row[field]) < 0:
                    parser.error(f"row {number}: {field} must be non-negative")
    print(f"OK schema=v{version} file={args.csv_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
