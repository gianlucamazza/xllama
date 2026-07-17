#!/usr/bin/env python3
"""Emit a LocalState manifest override snippet for a merged GGUF (publish stub)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--name", required=True, help="catalogue model dir name")
    ap.add_argument("--display", default="", help="picker display string")
    ap.add_argument("--gguf", required=True, type=Path, help="merged .gguf path (for size)")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="write JSON here (default: stdout)",
    )
    args = ap.parse_args()
    size = args.gguf.stat().st_size if args.gguf.is_file() else 0
    doc = {
        "models": [
            {
                "name": args.name,
                "display": args.display or f"{args.name} (finetuned)",
                "kind": "gguf",
                "files": [
                    {
                        "filename": "model.gguf",
                        "approx_bytes": size,
                    }
                ],
            }
        ]
    }
    text = json.dumps(doc, indent=2) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
        print(f"wrote {args.out}", flush=True)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
