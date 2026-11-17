#!/usr/bin/env python3
"""Parse opt-in `[xllama] d3d12-layer` log lines (#421).

gpu_ms and wait_ms are recorded timestamps. The token `missing` means that
value was not observed; it is never filled in from wall time or from
exec_wait_ms alone. wait_ms is the submission→fence remainder
(exec_wait_ms − gpu_ms) attributed equally across dispatches in that
submission.

Usage:
  python3 scripts/parse_d3d12_layer_log.py [xllama.log]
  python3 scripts/parse_d3d12_layer_log.py --json [xllama.log]

Reads stdin when no file is given. Lines that are not d3d12-layer records
are ignored.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, Union

PREFIX = "[xllama] d3d12-layer "


def parse_ms(token: str) -> Optional[float]:
    """Return a float milliseconds value, or None for the explicit missing token."""
    if token == "missing":
        return None
    try:
        return float(token)
    except ValueError as exc:
        raise ValueError(f"not a milliseconds field: {token!r}") from exc


def parse_layer(token: str) -> Union[int, str]:
    if token == "unattributed":
        return "unattributed"
    try:
        return int(token)
    except ValueError as exc:
        raise ValueError(f"not a layer field: {token!r}") from exc


def _fields(body: str) -> dict[str, str]:
    out: dict[str, str] = {}
    for part in body.split():
        if "=" not in part:
            if part == "summary" and "summary" not in out:
                out["summary"] = "1"
                continue
            raise ValueError(f"not a key=value field: {part!r}")
        key, value = part.split("=", 1)
        if not key or key in out:
            raise ValueError(f"duplicate or empty field: {part!r}")
        out[key] = value
    return out


def parse_d3d12_layer_line(line: str) -> Optional[dict[str, Any]]:
    """Parse one d3d12-layer log line. None if the line is not one."""
    text = line.rstrip("\r\n")
    if not text.startswith(PREFIX):
        return None
    fields = _fields(text[len(PREFIX):])
    phase = fields.get("phase")
    if not phase:
        raise ValueError("d3d12-layer line missing phase")
    if fields.get("summary") == "1":
        required = ("rows", "timed", "missing", "gpu_ms", "wait_ms", "wait_missing")
        absent = [k for k in required if k not in fields]
        if absent:
            raise ValueError("d3d12-layer summary missing " + ",".join(absent))
        return {
            "kind": "summary",
            "phase": phase,
            "rows": int(fields["rows"]),
            "timed": int(fields["timed"]),
            "missing": int(fields["missing"]),
            "gpu_ms": parse_ms(fields["gpu_ms"]),
            "wait_ms": parse_ms(fields["wait_ms"]),
            "wait_missing": int(fields["wait_missing"]),
        }
    required = ("layer", "op", "count", "gpu_ms", "wait_ms", "gpu_missing", "wait_missing")
    absent = [k for k in required if k not in fields]
    if absent:
        raise ValueError("d3d12-layer row missing " + ",".join(absent))
    return {
        "kind": "row",
        "phase": phase,
        "layer": parse_layer(fields["layer"]),
        "op": fields["op"],
        "count": int(fields["count"]),
        "gpu_ms": parse_ms(fields["gpu_ms"]),
        "wait_ms": parse_ms(fields["wait_ms"]),
        "gpu_missing": int(fields["gpu_missing"]),
        "wait_missing": int(fields["wait_missing"]),
    }


def parse_d3d12_layer_log(text: str) -> list[dict[str, Any]]:
    records = []
    for raw in text.splitlines():
        rec = parse_d3d12_layer_line(raw)
        if rec is not None:
            records.append(rec)
    return records


def parse_d3d12_layer_lines(lines: Iterable[str]) -> list[dict[str, Any]]:
    records = []
    for raw in lines:
        rec = parse_d3d12_layer_line(raw)
        if rec is not None:
            records.append(rec)
    return records


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", nargs="?", help="xllama.log (stdin if omitted)")
    parser.add_argument("--json", action="store_true", help="write JSON to stdout")
    args = parser.parse_args(argv)
    if args.log is None:
        text = sys.stdin.read()
    else:
        text = Path(args.log).read_text(encoding="utf-8")
    records = parse_d3d12_layer_log(text)
    if args.json:
        json.dump(records, sys.stdout, indent=2)
        sys.stdout.write("\n")
        return 0
    for rec in records:
        gpu = "missing" if rec["gpu_ms"] is None else f"{rec['gpu_ms']:.3f}"
        wait = "missing" if rec["wait_ms"] is None else f"{rec['wait_ms']:.3f}"
        if rec["kind"] == "summary":
            print(f"{rec['phase']} summary rows={rec['rows']} timed={rec['timed']} "
                  f"missing={rec['missing']} gpu_ms={gpu} wait_ms={wait} "
                  f"wait_missing={rec['wait_missing']}")
        else:
            print(f"{rec['phase']} layer={rec['layer']} op={rec['op']} "
                  f"count={rec['count']} gpu_ms={gpu} wait_ms={wait} "
                  f"gpu_missing={rec['gpu_missing']} wait_missing={rec['wait_missing']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
