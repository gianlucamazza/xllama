#!/usr/bin/env python3
"""Attribute ORT kernel time per execution provider from a profiling JSON.

When session_options.enable_profiling is set, ONNX Runtime writes a
Chrome-trace event array; every <node>_kernel_time event carries
args.provider ("DmlExecutionProvider" or "CPUExecutionProvider").
This is the definitive answer to "did DML actually run on the GPU,
or silently fall back to CPU?" — it does not depend on log routing
or on ORT being a full (non-minimal) build.

Usage: python3 scripts/analyze_ort_profile.py <profile.json> [--top N] [--log xllama.log]

The last stdout line is greppable:
    VERDICT: GPU                      (DML kernel time >= 90%)
    VERDICT: MIXED (dml=X% cpu=Y%)    (both providers present)
    VERDICT: CPU-FALLBACK             (zero DML kernel events)
"""

import argparse
import json
import sys
from collections import defaultdict


def parse_events(text):
    """Parse a Chrome-trace event array, tolerating a truncated tail.

    ORT leaves the array unterminated if the process dies before session
    teardown; trim to the last complete object and close the array.
    """
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        cut = text.rfind("},")
        if cut == -1:
            raise
        data = json.loads(text[: cut + 1] + "]")
    if isinstance(data, dict):
        data = data.get("traceEvents", [])
    return data


def summarize(events):
    """Group kernel-time events by execution provider."""
    providers = defaultdict(lambda: {"count": 0, "dur_us": 0, "ops": defaultdict(int)})
    kernel_events = 0
    session_runs = 0
    for ev in events:
        name = str(ev.get("name", ""))
        if name == "SessionRun":
            session_runs += 1
        if ev.get("cat") != "Node" or not name.endswith("_kernel_time"):
            continue
        kernel_events += 1
        args = ev.get("args", {}) or {}
        prov = providers[args.get("provider", "unknown")]
        prov["count"] += 1
        dur = int(ev.get("dur", 0) or 0)
        prov["dur_us"] += dur
        prov["ops"][args.get("op_name", "?")] += dur
    return {
        "providers": {k: dict(v, ops=dict(v["ops"])) for k, v in providers.items()},
        "kernel_events": kernel_events,
        "session_runs": session_runs,
    }


def verdict(summary):
    providers = summary["providers"]
    total = sum(p["dur_us"] for p in providers.values())
    if total == 0 or summary["kernel_events"] == 0:
        return "VERDICT: NO-KERNEL-EVENTS"
    dml = sum(p["dur_us"] for name, p in providers.items() if "Dml" in name)
    if dml == 0:
        return "VERDICT: CPU-FALLBACK"
    if dml / total >= 0.9:
        return "VERDICT: GPU"
    return f"VERDICT: MIXED (dml={dml * 100 // total}% cpu={(total - dml) * 100 // total}%)"


def grep_node_placements(log_path):
    """Secondary probe: node-placement lines only appear at ORT severity VERBOSE
    and only in full (non-minimal) ORT builds — absence is not evidence."""
    hits = []
    with open(log_path, errors="replace") as fp:
        for line in fp:
            if "Node placements" in line or "nodes placed on" in line.lower():
                hits.append(line.rstrip())
    return hits


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("profile", help="ORT profiling JSON (ort_profile_<ts>.json)")
    ap.add_argument(
        "--top", type=int, default=5, help="top ops per provider (default 5)"
    )
    ap.add_argument(
        "--log", help="optional xllama.log to grep for node-placement lines"
    )
    args = ap.parse_args()

    with open(args.profile, errors="replace") as fp:
        events = parse_events(fp.read())
    summary = summarize(events)

    print(
        f"events: {len(events)} total, {summary['kernel_events']} kernel, "
        f"{summary['session_runs']} SessionRun"
    )
    for name, prov in sorted(
        summary["providers"].items(), key=lambda kv: -kv[1]["dur_us"]
    ):
        print(f"\n{name}: {prov['count']} kernels, {prov['dur_us'] / 1000.0:.1f} ms")
        top_ops = sorted(prov["ops"].items(), key=lambda kv: -kv[1])[: args.top]
        for op, dur in top_ops:
            print(f"  {op}: {dur / 1000.0:.1f} ms")

    if args.log:
        hits = grep_node_placements(args.log)
        print(f"\nnode-placement lines in {args.log}: {len(hits)}")
        for line in hits[:10]:
            print(f"  {line}")

    print()
    print(verdict(summary))


if __name__ == "__main__":
    main()
