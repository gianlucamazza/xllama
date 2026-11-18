#!/usr/bin/env python3
"""Write a CSV sidecar from ``[xllama] d3d12-dispatch`` log lines.

The bench CSV schema stays as it is. This file is the per-layer attribution
for the run whose log it was parsed from. A summary line that does not match
the rows of its phase is an error: the sidecar is not written.
"""

import argparse
import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

ROW = re.compile(
    r"\[xllama\] d3d12-dispatch phase=(\S+) op=(\S+) type=(\S+) cpg=(\d+) "
    r"threads=(\d+) cols=(\S+) layer=(-?\d+) count=(\d+) gpu_ms=([0-9.]+) "
    r"ms_per=([0-9.]+) weight_mib=([0-9.]+) columns=(\d+)\s*$"
)
SUMMARY = re.compile(
    r"\[xllama\] d3d12-dispatch phase=(\S+) summary kinds=(\d+) timed=(\d+) "
    r"missing=(\d+) gpu_ms=([0-9.]+)\s*$"
)
HEADER = [
    "phase",
    "op",
    "type",
    "cpg",
    "threads",
    "cols",
    "layer",
    "count",
    "gpu_ms",
    "ms_per",
    "weight_mib",
    "columns",
]


class ParseError(Exception):
    pass


def parse_log(text):
    rows = []
    summaries = []
    for lineno, line in enumerate(text.splitlines(), 1):
        if "[xllama] d3d12-dispatch " not in line:
            continue
        row = ROW.search(line)
        if row:
            rows.append(
                {
                    "phase": row.group(1),
                    "op": row.group(2),
                    "type": row.group(3),
                    "cpg": row.group(4),
                    "threads": row.group(5),
                    "cols": row.group(6),
                    "layer": row.group(7),
                    "count": row.group(8),
                    "gpu_ms": row.group(9),
                    "ms_per": row.group(10),
                    "weight_mib": row.group(11),
                    "columns": row.group(12),
                }
            )
            continue
        summary = SUMMARY.search(line)
        if summary:
            summaries.append(
                {
                    "phase": summary.group(1),
                    "kinds": int(summary.group(2)),
                    "timed": int(summary.group(3)),
                    "missing": int(summary.group(4)),
                    "gpu_ms": summary.group(5),
                    "line": lineno,
                }
            )
            continue
        raise ParseError(f"unparsed d3d12-dispatch line {lineno}: {line.strip()}")
    if not rows and not summaries:
        raise ParseError("no d3d12-dispatch lines")
    by_phase = defaultdict(list)
    for row in rows:
        by_phase[row["phase"]].append(row)
    seen = set()
    for summary in summaries:
        phase = summary["phase"]
        if phase in seen:
            raise ParseError(f"duplicate summary for phase {phase} at line {summary['line']}")
        seen.add(phase)
        phase_rows = by_phase.get(phase, [])
        if summary["kinds"] != len(phase_rows):
            raise ParseError(
                f"phase {phase}: summary kinds={summary['kinds']} but {len(phase_rows)} rows"
            )
        timed = sum(int(row["count"]) for row in phase_rows)
        if summary["timed"] != timed:
            raise ParseError(f"phase {phase}: summary timed={summary['timed']} but rows sum {timed}")
        gpu = sum(float(row["gpu_ms"]) for row in phase_rows)
        if f"{gpu:.3f}" != summary["gpu_ms"]:
            raise ParseError(
                f"phase {phase}: summary gpu_ms={summary['gpu_ms']} but rows sum {gpu:.3f}"
            )
    return rows


def write_csv(rows, path):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=HEADER, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="app log containing d3d12-dispatch lines")
    parser.add_argument("-o", "--output", type=Path, required=True, help="CSV sidecar to write")
    args = parser.parse_args(argv)
    try:
        text = args.log.read_text(encoding="utf-8", errors="replace")
        rows = parse_log(text)
    except (OSError, ParseError) as exc:
        print(f"d3d12-dispatch-sidecar: {exc}", file=sys.stderr)
        return 1
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.output)
    print(f"wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
