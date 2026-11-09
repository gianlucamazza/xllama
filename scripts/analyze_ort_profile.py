#!/usr/bin/env python3
"""Attribute ORT kernel time per execution provider from a profiling JSON.

When session_options.enable_profiling is set, ONNX Runtime writes a
Chrome-trace event array; every <node>_kernel_time event carries
args.provider ("DmlExecutionProvider" or "CPUExecutionProvider").
Provider events identify ORT execution-provider placement, independently of
log routing. Their durations are host trace intervals, not GPU hardware
timestamps; fused nodes can hide internal operator and layer placement.

Usage: python3 scripts/analyze_ort_profile.py <profile.json> [--top N] [--log xllama.log]

The last stdout line is greppable:
    VERDICT: GPU                      (DML kernel time >= 90%)
    VERDICT: MIXED (dml=X% cpu=Y%)    (both providers present)
    VERDICT: CPU-FALLBACK             (zero DML kernel events)
"""

import argparse
import json
import hashlib
import math
from pathlib import Path
import sys
from collections import defaultdict


def parse_trace(text):
    """Parse a Chrome-trace event array, tolerating a truncated tail.

    ORT leaves the array unterminated if the process dies before session
    teardown; trim to the last complete object and close the array.
    """
    complete = True
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        complete = False
        cut = text.rfind("},")
        if cut == -1:
            raise
        data = json.loads(text[: cut + 1] + "]")
    if isinstance(data, dict):
        data = data["traceEvents"]
    if not isinstance(data, list) or not all(isinstance(ev, dict) for ev in data):
        raise ValueError("trace must be an event array or an object containing traceEvents")
    return data, complete


def parse_events(text):
    """Compatibility helper for forensic callers; use parse_trace for integrity."""
    return parse_trace(text)[0]


def summarize(events):
    """Group kernel-time events by execution provider."""
    providers = defaultdict(lambda: {"count": 0, "dur_us": 0, "ops": defaultdict(int)})
    kernel_events = 0
    session_runs = 0
    nodes = {}
    invalid_events = 0
    for ev in events:
        name = str(ev.get("name", ""))
        if name == "SessionRun":
            session_runs += 1
        if ev.get("cat") != "Node" or not name.endswith("_kernel_time"):
            continue
        kernel_events += 1
        args = ev.get("args", {}) or {}
        dur = ev.get("dur")
        if (not isinstance(args, dict) or type(dur) not in (int, float)
                or not math.isfinite(dur) or dur < 0):
            invalid_events += 1
            continue
        provider = args.get("provider", "unknown")
        if not isinstance(provider, str) or not provider:
            provider = "unknown"
        prov = providers[provider]
        prov["count"] += 1
        prov["dur_us"] += dur
        op = str(args.get("op_name", "?"))
        prov["ops"][op] += dur
        key = (provider, name)
        node = nodes.setdefault(key, {"provider": provider, "name": name,
                                      "op": op, "count": 0, "dur_us": 0})
        node["count"] += 1
        node["dur_us"] += dur
    return {
        "providers": {k: dict(v, ops=dict(v["ops"])) for k, v in providers.items()},
        "kernel_events": kernel_events,
        "session_runs": session_runs,
        "invalid_events": invalid_events,
        "nodes": sorted(nodes.values(), key=lambda n: (-n["dur_us"], n["name"])),
        "timing_scope": "ORT host trace event intervals; not GPU hardware time",
        "aggregation": "sum of events; may overlap; not end-to-end latency",
        "run_success": None,
        "layer_attribution": "node names only; fused nodes may hide layer boundaries",
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
    ap.add_argument("--strict", action="store_true",
                    help="reject truncated/invalid/empty or unattributed profiles")
    ap.add_argument("--json-out", type=Path, help="write attributable machine-readable report")
    args = ap.parse_args()
    if args.top <= 0:
        ap.error("--top must be positive")

    try:
        raw = Path(args.profile).read_bytes()
        events, complete = parse_trace(raw.decode("utf-8"))
        summary = summarize(events)
    except (OSError, ValueError, KeyError) as exc:
        print(f"INVALID PROFILE: {exc}", file=sys.stderr)
        return 2
    valid = (complete and summary["kernel_events"] > 0 and summary["session_runs"] > 0
             and summary["invalid_events"] == 0 and summary["providers"]
             and all(p in ("CPUExecutionProvider", "DmlExecutionProvider")
                     for p in summary["providers"])
             and sum(p["dur_us"] for p in summary["providers"].values()) > 0)
    summary.update(trace_complete=complete, profile_valid=bool(valid),
                   profile_sha256=hashlib.sha256(raw).hexdigest())
    if args.json_out:
        try:
            args.json_out.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
        except (OSError, ValueError) as exc:
            print(f"REPORT WRITE FAILED: {exc}", file=sys.stderr)
            return 2
    print(summary["timing_scope"])
    if not valid:
        print("DIAGNOSTIC ONLY: incomplete, invalid or unattributed profile")

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
    print(verdict(summary) if valid else "VERDICT: INVALID-PROFILE")
    return 1 if args.strict and not valid else 0


if __name__ == "__main__":
    sys.exit(main())
