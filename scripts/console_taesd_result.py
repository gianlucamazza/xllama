#!/usr/bin/env python3
# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
"""Freshness and optional peak-memory helpers for the taesd console gate.

The gate must never report a VAE time from a prior run's diffuse-result.csv.
A row is from this run when its `date` is at or after the recorded UTC start,
or when a valid pre-run snapshot exists and the file contents changed.

Peak memory is logged only when the Device Portal process list carries a
PeakWorkingSet / PrivateWorkingSet field (or *Size). WorkingSetSize is current
RSS and is not reported as peak.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

DATE_FMT = "%Y-%m-%dT%H:%M:%SZ"

# Field names the Xbox / Windows Device Portal process list has used. Only
# these count as peak; WorkingSetSize is current and is ignored.
_PEAK_KEYS = (
    "PeakWorkingSet",
    "PeakWorkingSetSize",
    "PrivateWorkingSet",
    "PrivateWorkingSetSize",
)

_PACKAGE_NEEDLES = ("xllama", "GianlucaMazza.xllama", "VenereLabs.xllama")


def last_result_row(text: str) -> Optional[Dict[str, str]]:
    """Return the last CSV row, or None if the text is not a result table."""
    if not text or not text.strip():
        return None
    try:
        rows = list(csv.DictReader(io.StringIO(text)))
    except csv.Error:
        return None
    if not rows:
        return None
    row = { (k or "").strip(): (v or "").strip() for k, v in rows[-1].items() }
    if not any(row.values()):
        return None
    return row


def parse_csv_date(value: str) -> Optional[datetime]:
    """Parse the app's UTC `date` column. None if missing or malformed."""
    if not value:
        return None
    try:
        return datetime.strptime(value, DATE_FMT).replace(tzinfo=timezone.utc)
    except ValueError:
        return None


def _vae_ms(row: Dict[str, str]) -> str:
    key = next((k for k in row if k and "vae" in k.lower() and "ms" in k.lower()), None)
    return row.get(key, "") if key else ""


def assess_result(
    after_text: str,
    run_start_utc: str,
    before_text: str = "",
) -> Tuple[bool, str, str]:
    """Decide whether `after_text` is a this-run diffuse-result.csv.

    Returns (ok, vae_ms, message). vae_ms is set only when ok is True.
    """
    after_row = last_result_row(after_text)
    if after_row is None:
        return False, "", "FAIL: no diffuse-result.csv row from this run"

    vae = _vae_ms(after_row)
    date_raw = after_row.get("date", "")
    date = parse_csv_date(date_raw)
    run_start = parse_csv_date(run_start_utc)
    before_row = last_result_row(before_text)
    before_valid = before_row is not None
    content_changed = (before_text or "").strip() != (after_text or "").strip()
    date_fresh = date is not None and run_start is not None and date >= run_start

    if before_valid and not content_changed:
        return (
            False,
            "",
            "FAIL: stale diffuse-result.csv row "
            f"(date={date_raw or 'missing'}; unchanged since before this run)",
        )
    if date is not None and run_start is not None and not date_fresh and not (
        before_valid and content_changed
    ):
        return (
            False,
            "",
            "FAIL: stale diffuse-result.csv row "
            f"(date={date_raw}; run started {run_start_utc})",
        )
    if not vae:
        return False, "", "FAIL: this-run diffuse-result.csv row has no vae_ms"
    return True, vae, ""


def _is_app_process(proc: Dict[str, Any]) -> bool:
    image = str(proc.get("ImageName", "")).lower()
    package = str(proc.get("PackageFullName", ""))
    if "xllama" in image:
        return True
    return any(n in package for n in _PACKAGE_NEEDLES)


def _first_peak_bytes(proc: Dict[str, Any]) -> Optional[int]:
    lower = {str(k).lower(): v for k, v in proc.items()}
    for name in _PEAK_KEYS:
        raw = lower.get(name.lower())
        if raw is None or raw == "":
            continue
        try:
            value = int(raw)
        except (TypeError, ValueError):
            continue
        if value < 0:
            continue
        return value
    return None


def peak_mem_mb(process_json: str) -> Optional[int]:
    """Return peak MB for the app process, or None if no peak field is present."""
    try:
        data = json.loads(process_json)
    except json.JSONDecodeError:
        return None
    processes: Iterable[Any]
    if isinstance(data, dict):
        processes = data.get("Processes") or []
    elif isinstance(data, list):
        processes = data
    else:
        return None
    peaks: List[int] = []
    for proc in processes:
        if not isinstance(proc, dict) or not _is_app_process(proc):
            continue
        nbytes = _first_peak_bytes(proc)
        if nbytes is not None:
            peaks.append(nbytes // (1024 * 1024))
    if not peaks:
        return None
    return max(peaks)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    assess = sub.add_parser("assess", help="accept a this-run diffuse-result.csv row")
    assess.add_argument("--after", type=argparse.FileType("r"), default=None)
    assess.add_argument("--before", type=argparse.FileType("r"), default=None)
    assess.add_argument("--run-start", required=True)

    peak = sub.add_parser("peak-mem", help="read peak MB from a WDP process list")
    peak.add_argument("input", nargs="?", type=argparse.FileType("r"), default=None)

    args = parser.parse_args(argv)
    if args.cmd == "assess":
        after_text = args.after.read() if args.after else sys.stdin.read()
        before_text = args.before.read() if args.before else ""
        ok, vae, message = assess_result(after_text, args.run_start, before_text)
        if not ok:
            print(message, file=sys.stderr)
            return 1
        print(vae)
        return 0

    text = args.input.read() if args.input else sys.stdin.read()
    mb = peak_mem_mb(text)
    if mb is not None:
        print(mb)
    return 0


if __name__ == "__main__":
    sys.exit(main())
