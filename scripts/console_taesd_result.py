#!/usr/bin/env python3
# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
"""Freshness and sampled-memory helpers for the taesd console gate.

The gate must never report a VAE time from a prior run's diffuse-result.csv.
A row is from this run only when:

  (valid CSV date >= valid run_start)
  OR (a valid pre-run snapshot exists AND the file contents changed
      AND the row is the taesd job: model=sd-turbo-fp16, steps=1, seed=42)

Anything else fails. A missing or invalid date, or an invalid run_start,
is not a pass unless the snapshot fallback applies.

Device Portal exposes PrivateWorkingSet (current), not a peak. The gate
keeps a running max of that field sampled on an interval and labels it
max_sampled_private_ws_mb.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

DATE_FMT = "%Y-%m-%dT%H:%M:%SZ"
EXPECTED_MODEL = "sd-turbo-fp16"
EXPECTED_STEPS = "1"
EXPECTED_SEED = "42"
PRIVATE_WS_KEY = "privateworkingset"


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
    row = {(k or "").strip(): (v or "").strip() for k, v in rows[-1].items()}
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


def _is_expected_job(row: Dict[str, str]) -> bool:
    return (
        row.get("model") == EXPECTED_MODEL
        and row.get("steps") == EXPECTED_STEPS
        and row.get("seed") == EXPECTED_SEED
    )


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
    snapshot_fresh = before_valid and content_changed and _is_expected_job(after_row)

    if date_fresh or snapshot_fresh:
        if not vae:
            return False, "", "FAIL: this-run diffuse-result.csv row has no vae_ms"
        return True, vae, ""

    if before_valid and not content_changed:
        return (
            False,
            "",
            "FAIL: stale diffuse-result.csv row "
            f"(date={date_raw or 'missing'}; unchanged since before this run)",
        )
    if before_valid and content_changed and not _is_expected_job(after_row):
        return (
            False,
            "",
            "FAIL: rewritten diffuse-result.csv row is not the taesd job "
            f"(model={after_row.get('model', '')!s} steps={after_row.get('steps', '')} "
            f"seed={after_row.get('seed', '')}; "
            f"want {EXPECTED_MODEL} steps={EXPECTED_STEPS} seed={EXPECTED_SEED})",
        )
    if date is None:
        return (
            False,
            "",
            "FAIL: diffuse-result.csv row has missing or invalid date "
            f"({date_raw or 'missing'}); no this-run proof",
        )
    if run_start is None:
        return (
            False,
            "",
            "FAIL: run_start is missing or invalid "
            f"({run_start_utc or 'missing'}); cannot compare CSV date",
        )
    return (
        False,
        "",
        "FAIL: stale diffuse-result.csv row "
        f"(date={date_raw}; run started {run_start_utc})",
    )


def _is_app_process(proc: Dict[str, Any], package_family: str) -> bool:
    if not package_family:
        return False
    return package_family in str(proc.get("PackageFullName", ""))


def _private_ws_bytes(proc: Dict[str, Any]) -> Optional[int]:
    for key, raw in proc.items():
        if str(key).lower() != PRIVATE_WS_KEY:
            continue
        if raw is None or raw == "":
            return None
        try:
            value = int(raw)
        except (TypeError, ValueError):
            return None
        if value < 0:
            return None
        return value
    return None


def private_ws_mb(process_json: str, package_family: str) -> Optional[int]:
    """Return current PrivateWorkingSet MB for the package family, or None."""
    if not package_family:
        return None
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
    samples = []
    for proc in processes:
        if not isinstance(proc, dict) or not _is_app_process(proc, package_family):
            continue
        nbytes = _private_ws_bytes(proc)
        if nbytes is not None:
            samples.append(nbytes // (1024 * 1024))
    if not samples:
        return None
    return max(samples)


def keep_running_max(prev: Optional[int], sample: Optional[int]) -> Optional[int]:
    """Running max of sampled current values. None means 'no sample yet'."""
    if sample is None:
        return prev
    if prev is None:
        return sample
    return max(prev, sample)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    assess = sub.add_parser("assess", help="accept a this-run diffuse-result.csv row")
    assess.add_argument("--after", type=argparse.FileType("r"), default=None)
    assess.add_argument("--before", type=argparse.FileType("r"), default=None)
    assess.add_argument("--run-start", required=True)

    ws = sub.add_parser(
        "private-ws",
        help="read PrivateWorkingSet MB for a package family from a WDP process list",
    )
    ws.add_argument("input", nargs="?", type=argparse.FileType("r"), default=None)
    ws.add_argument(
        "--package-family",
        required=True,
        help="Package family name (Name_PublisherId), from PFN or XLLAMA_PACKAGE_FAMILY",
    )

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
    mb = private_ws_mb(text, args.package_family)
    if mb is not None:
        print(mb)
    return 0


if __name__ == "__main__":
    sys.exit(main())
