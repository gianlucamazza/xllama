#!/usr/bin/env python3
"""Record hung/timeout bench runs and decide whether a campaign may exit 0.

The 2026-10-11 LFM2-2.6B re-measure timed out on run 4; bench-xbox-ort.sh
skipped it and exited 0 after writing 2 of 3 measurement rows. This module is
the one place the harness formats a failed row and decides campaign success,
so a missing or hung run cannot look like a clean OK.
"""

from __future__ import annotations

import argparse
import datetime as dt
import sys


CSV_HEADER = (
    "model,quant,backend,n_ctx,n_threads,prompt_tok_s,decode_tok_s,peak_ws_mb,"
    "load_ms,gpu_mem_mb,gpu_budget_mb,n_prompt_tok,n_gen_tok,max_length,host,"
    "date,run_index,prefill_ms,ttft_ms"
)

FAILED_REASONS = ("timeout", "stall", "hung")
FAILED_HOST_MARKERS = tuple(f"-{reason}" for reason in FAILED_REASONS)


def is_failed_host(host: str) -> bool:
    return any(marker in (host or "") for marker in FAILED_HOST_MARKERS)


def host_from_row(row: str) -> str:
    fields = row.split(",")
    if len(fields) < 15:
        return ""
    return fields[14]


def is_failed_row(row: str) -> bool:
    return is_failed_host(host_from_row(row.strip()))


def expected_recorded_runs(n_runs: int) -> int:
    """Warmup (run 1) is dropped when at least two runs were requested."""
    if n_runs < 1:
        return 0
    return n_runs - 1 if n_runs >= 2 else n_runs


def format_failed_row(
    *,
    model: str,
    host_base: str,
    run_index: int,
    reason: str,
    date: str | None = None,
    n_ctx: int = 0,
    n_threads: int = 0,
) -> str:
    if reason not in FAILED_REASONS:
        raise ValueError(f"unsupported bench failure reason: {reason}")
    if date is None:
        date = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    host = host_base if host_base.endswith(f"-{reason}") else f"{host_base}-{reason}"
    return (
        f"{model},unknown,unknown,{n_ctx},{n_threads},0.00,0.00,0,0,0,0,0,0,0,"
        f"{host},{date},{run_index},0.0,0.0"
    )


def measurement_row_count(rows: list[str]) -> int:
    return sum(1 for row in rows if row.strip() and not is_failed_row(row))


def campaign_should_fail(
    *,
    n_runs: int,
    rows: list[str],
    failed_runs: int,
) -> bool:
    if failed_runs > 0:
        return True
    return measurement_row_count(rows) < expected_recorded_runs(n_runs)


def campaign_exit_code(
    *,
    n_runs: int,
    rows: list[str],
    failed_runs: int,
) -> int:
    return 1 if campaign_should_fail(n_runs=n_runs, rows=rows, failed_runs=failed_runs) else 0


def _cmd_format_row(args: argparse.Namespace) -> int:
    print(
        format_failed_row(
            model=args.model,
            host_base=args.host,
            run_index=args.run_index,
            reason=args.reason,
            n_ctx=args.n_ctx,
            n_threads=args.n_threads,
        )
    )
    return 0


def _cmd_check(args: argparse.Namespace) -> int:
    rows = [line for line in (args.row or []) if line.strip()]
    code = campaign_exit_code(
        n_runs=args.n_runs, rows=rows, failed_runs=args.failed_runs
    )
    expected = expected_recorded_runs(args.n_runs)
    measured = measurement_row_count(rows)
    if code != 0:
        print(
            f"Error: bench campaign incomplete: measured={measured} "
            f"expected={expected} failed_runs={args.failed_runs}",
            file=sys.stderr,
        )
    else:
        print(f"ok measured={measured} expected={expected}")
    return code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    fmt = sub.add_parser("format-row", help="print one hung/timeout CSV row")
    fmt.add_argument("--model", required=True)
    fmt.add_argument("--host", default="xbox-series-s")
    fmt.add_argument("--run-index", type=int, required=True)
    fmt.add_argument("--reason", choices=FAILED_REASONS, required=True)
    fmt.add_argument("--n-ctx", type=int, default=0)
    fmt.add_argument("--n-threads", type=int, default=0)
    fmt.set_defaults(func=_cmd_format_row)

    chk = sub.add_parser("check", help="exit 1 if the campaign is incomplete")
    chk.add_argument("--n-runs", type=int, required=True)
    chk.add_argument("--failed-runs", type=int, default=0)
    chk.add_argument(
        "--row",
        action="append",
        default=[],
        help="CSV data row to count (repeatable; include failed rows)",
    )
    chk.set_defaults(func=_cmd_check)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
