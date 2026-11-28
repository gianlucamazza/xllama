#!/usr/bin/env python3
"""Read D3D12 phase and opt-in layer timings from an xllama log (#421).

Layer gpu_ms sums measured dispatch intervals. Layer wait_ms divides the
execute-through-fence interval minus the submission GPU time equally among
that submission's dispatches. A missing measurement stays missing.

Usage:
  python3 scripts/parse_d3d12_layer_log.py [xllama.log]
  python3 scripts/parse_d3d12_layer_log.py --json [xllama.log]
  python3 scripts/parse_d3d12_layer_log.py --csv sidecar.csv [xllama.log]
  python3 scripts/parse_d3d12_layer_log.py --phase-summary [xllama.log]

`--phase-summary` writes JSON for each logged D3D12 backend phase in order.
It uses the phase and dispatch summary counters at their printed precision.
The CPU execute-through-fence interval includes GPU execution. The benchmark
records the inference phases. A client clock measures end-to-end request time.

`--csv` copies one row for each data line, in log order. It does not group
rows or add a run id, so the same phase, layer and op can appear twice when
the log holds two runs. A missing timestamp stays the word `missing`.
Summary lines are checked by the reader and are not copied into the CSV.
The bench CSV schema is unchanged. The sidecar path must be a different
file from the log: the script refuses to overwrite the file it just read.

Reads stdin when no file is given.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Optional, Union

CSV_HEADER = (
    "phase",
    "layer",
    "op",
    "count",
    "gpu_ms",
    "wait_ms",
    "gpu_missing",
    "wait_missing",
)

PREFIX = "[xllama] d3d12-layer "
PHASE_PREFIX = "[xllama] d3d12-phase "
DISPATCH_PREFIX = "[xllama] d3d12-dispatch "


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
    return parse_d3d12_layer_lines(text.splitlines())


def parse_d3d12_layer_lines(lines: Iterable[str]) -> list[dict[str, Any]]:
    records = []
    for raw in lines:
        rec = parse_d3d12_layer_line(raw)
        if rec is not None:
            records.append(rec)
    return records


def _count(fields: dict[str, str], key: str) -> int:
    value = int(fields[key])
    if value < 0:
        raise ValueError(f"negative {key}")
    return value


def _duration(token: str) -> float:
    value = float(token)
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid duration: {token!r}")
    return value


def _phase_record(fields: dict[str, str], line: int) -> dict[str, Any]:
    phase = fields["phase"]
    if not phase:
        raise ValueError("empty phase")
    submissions = _count(fields, "submissions")
    valid = _count(fields, "ts_valid")
    missing = _count(fields, "ts_missing")
    gpu = None if fields["gpu_ms"] == "unavailable" else _duration(fields["gpu_ms"])
    if valid + missing != submissions or (gpu is not None) != (valid > 0 and missing == 0):
        raise ValueError("submission timestamp coverage disagrees with phase counters")
    through_fence = _duration(fields["exec_wait_ms"])
    return {
        "log_line": line,
        "phase": phase,
        "graph_calls": _count(fields, "calls"),
        "submissions": submissions,
        "dispatches": _count(fields, "dispatches"),
        "graph_wall_ms": _duration(fields["wall_ms"]),
        "cpu_quant_ms": _duration(fields["quant_ms"]),
        "cpu_record_ms": _duration(fields["record_ms"]),
        "execute_through_fence_ms": through_fence,
        "submission_ts_valid": valid,
        "submission_ts_missing": missing,
        "submission_gpu_ms": gpu,
        "fence_residual_ms": round(through_fence - gpu, 3) if gpu is not None else None,
        "dispatch_profile": None,
    }


def _dispatch_summary(fields: dict[str, str], dispatches: int) -> dict[str, Any]:
    timed = _count(fields, "timed")
    missing = _count(fields, "missing")
    kinds = _count(fields, "kinds")
    gpu = None if fields["gpu_ms"] == "missing" else _duration(fields["gpu_ms"])
    if timed + missing != dispatches or (gpu is not None) != (timed > 0):
        raise ValueError("dispatch timestamp coverage disagrees with phase counters")
    if kinds > timed or (kinds == 0) != (timed == 0):
        raise ValueError("dispatch kinds disagree with timed dispatches")
    return {"kinds": kinds, "timed": timed, "missing": missing, "timed_dispatch_gpu_ms": gpu}


def phase_summary(lines: Iterable[str]) -> dict[str, Any]:
    phases = []
    current = None
    for number, line in enumerate(lines, 1):
        try:
            if line.startswith(PHASE_PREFIX):
                current = _phase_record(_fields(line[len(PHASE_PREFIX):]), number)
                phases.append(current)
            elif line.startswith(DISPATCH_PREFIX):
                fields = _fields(line[len(DISPATCH_PREFIX):])
                if current is None or fields["phase"] != current["phase"]:
                    raise ValueError("dispatch record has no matching preceding phase")
                if fields.get("summary") == "1":
                    current["dispatch_profile"] = _dispatch_summary(fields, current["dispatches"])
                    current = None
            elif line.startswith((PREFIX, "[xllama] done:", "[xllama] bench model:",
                                  "[xllama] headless bench:")):
                # The dispatch block precedes layer records and the next run.
                current = None
        except KeyError as exc:
            raise ValueError(f"line {number}: missing {exc.args[0]}") from exc
        except ValueError as exc:
            raise ValueError(f"line {number}: {exc}") from exc
    if not phases:
        raise ValueError("no d3d12-phase records")
    return {"scope": "D3D12 backend graph_compute calls", "phases": phases}


def ms_cell(value: Optional[float]) -> str:
    if value is None:
        return "missing"
    return f"{value:.3f}"


def same_file(left: Path, right: Path) -> bool:
    """True when both paths name one inode, including a hard link or a symlink."""
    try:
        if left.resolve() == right.resolve():
            return True
    except OSError:
        return False
    try:
        return left.samefile(right)
    except OSError:
        return False


def write_layer_csv(records: list[dict[str, Any]], path: Path) -> int:
    rows = [rec for rec in records if rec["kind"] == "row"]
    if not rows:
        raise ValueError("no d3d12-layer rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(CSV_HEADER)
        for rec in rows:
            writer.writerow(
                [
                    rec["phase"],
                    rec["layer"],
                    rec["op"],
                    rec["count"],
                    ms_cell(rec["gpu_ms"]),
                    ms_cell(rec["wait_ms"]),
                    rec["gpu_missing"],
                    rec["wait_missing"],
                ]
            )
    return len(rows)


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", nargs="?", help="xllama.log (stdin if omitted)")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--json", action="store_true", help="write layer records as JSON")
    mode.add_argument("--csv", type=Path, help="write data rows to this CSV sidecar")
    mode.add_argument("--phase-summary", action="store_true",
                      help="write one JSON observation per D3D12 backend phase")
    args = parser.parse_args(argv)
    try:
        source = nullcontext(sys.stdin) if args.log is None else Path(args.log).open(encoding="utf-8")
        with source as lines:
            if args.phase_summary:
                print(json.dumps(phase_summary(lines), indent=2))
                return 0
            records = parse_d3d12_layer_lines(lines)
    except (OSError, ValueError) as exc:
        print(f"parse_d3d12_layer_log: {exc}", file=sys.stderr)
        return 1
    if args.csv:
        if args.log is not None and same_file(Path(args.log), args.csv):
            print("parse_d3d12_layer_log: --csv would overwrite the log", file=sys.stderr)
            return 2
        try:
            count = write_layer_csv(records, args.csv)
        except ValueError as exc:
            print(f"parse_d3d12_layer_log: {exc}", file=sys.stderr)
            return 1
        print(f"wrote {count} rows to {args.csv}", file=sys.stderr)
        return 0
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
