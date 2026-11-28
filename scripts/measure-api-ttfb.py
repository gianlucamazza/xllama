#!/usr/bin/env python3
# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
#
# Client-side request timing for the LAN API (#421). Unlike the bench CSV's
# ttft_ms (measured inside the app, load excluded), this times what a LAN
# client sees, with the network and HTTP handling included.
#
# For each streamed POST /v1/chat/completions it records, in milliseconds
# from just before the connection is opened:
#   connect_ms        TCP connection established
#   headers_ms        complete HTTP response header received
#   first_byte_ms     first response body byte
#   first_content_ms  first SSE chunk with non-empty choices[0].delta.content
#   done_ms           `data: [DONE]` received
# A field that never happened is null and the attempt fails. Each attempt is
# retained, including failures: a timeout, non-200 status, malformed SSE,
# missing [DONE] or no content delta is recorded with its error, never
# dropped or replaced.
#
# Usage:
#   scripts/measure-api-ttfb.py --host <ip> [--port 11434] --model <id>
#       --prompt <text> [--repeats N] [--max-tokens N] [--timeout S]
#       [--hardware <label>] --json-out <path>
#
# Exit 0 = every attempt complete, 1 = at least one failed attempt,
#      2 = invalid arguments.

import argparse
import http.client
import io
import json
import math
import socket
import sys
import time
from datetime import datetime, timezone


STAGES = ("connect_ms", "headers_ms", "first_byte_ms", "first_content_ms", "done_ms")


def request_percentiles(completed):
    """Linear sample quantiles of client-observed request milestones (milliseconds)."""
    result = {}
    for stage in STAGES:
        values = sorted(attempt[stage] for attempt in completed)
        estimates = {}
        for percentile in (50, 95, 99):
            value = None
            if values:
                index, remainder = divmod((len(values) - 1) * percentile, 100)
                value = values[index]
                if remainder:
                    weight = remainder / 100
                    value = round(value * (1 - weight) + values[index + 1] * weight, 3)
            estimates[f"p{percentile}"] = value
        result[stage] = estimates
    return result


def ms_since(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000.0, 3)


def response_framing(header_block):
    """Return (chunked, content length); a missing length uses connection close."""
    headers = http.client.parse_headers(io.BytesIO(header_block))
    encodings = headers.get_all("Transfer-Encoding", [])
    lengths = headers.get_all("Content-Length", [])
    if encodings:
        if lengths:
            raise ValueError("response has both Transfer-Encoding and Content-Length")
        if ",".join(encodings).strip().lower() != "chunked":
            raise ValueError("unsupported response Transfer-Encoding")
        return True, None
    if not lengths:
        return False, None
    values = [value.strip() for line in lengths for value in line.split(",")]
    if any(not value.isascii() or not value.isdecimal() for value in values):
        raise ValueError("invalid response Content-Length")
    sizes = {int(value) for value in values}
    if len(sizes) != 1:
        raise ValueError("conflicting response Content-Length values")
    return False, sizes.pop()


def one_attempt(host, port, model, prompt, max_tokens, timeout):
    body = json.dumps({
        "model": model, "stream": True, "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }).encode()
    req = (f"POST /v1/chat/completions HTTP/1.1\r\nHost: {host}:{port}\r\n"
           f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n"
           f"Connection: close\r\n\r\n").encode() + body
    rec = {"connect_ms": None, "headers_ms": None, "first_byte_ms": None,
           "first_content_ms": None, "done_ms": None, "status": None,
           "content_chunks": 0, "error": None}
    t0 = time.perf_counter()
    sock = None
    try:
        sock = socket.create_connection((host, port), timeout=timeout)
        rec["connect_ms"] = ms_since(t0)
        sock.settimeout(timeout)
        sock.sendall(req)
        buf = b""
        while b"\r\n\r\n" not in buf:
            data = sock.recv(65536)
            if not data:
                raise ValueError("connection closed before response header")
            buf += data
        head, rest = buf.split(b"\r\n\r\n", 1)
        rec["headers_ms"] = ms_since(t0)
        status_line = head.split(b"\r\n", 1)[0].decode("latin-1")
        parts = status_line.split()
        if len(parts) < 2 or not parts[1].isdigit():
            raise ValueError(f"malformed status line: {status_line!r}")
        rec["status"] = int(parts[1])
        if rec["status"] != 200:
            raise ValueError(f"HTTP {rec['status']}")
        chunked, remaining = response_framing(head.partition(b"\r\n")[2] + b"\r\n\r\n")

        raw = rest
        if raw and remaining != 0:
            rec["first_byte_ms"] = ms_since(t0)
        decoded, text_chars = b"", 0
        previous_cr = False

        def dechunk(data):
            """Return (decoded payload, leftover undecoded bytes, finished)."""
            out = b""
            while True:
                line_end = data.find(b"\r\n")
                if line_end < 0:
                    return out, data, False
                size_txt = data[:line_end].split(b";", 1)[0].strip()
                if not size_txt or any(c not in b"0123456789abcdefABCDEF" for c in size_txt):
                    raise ValueError(f"malformed chunk size: {size_txt!r}")
                size = int(size_txt, 16)
                if size == 0:
                    # The terminal chunk includes a (possibly empty) trailer
                    # section followed by a blank line. It may span reads.
                    trailers = data[line_end + 2:]
                    if trailers.startswith(b"\r\n") or b"\r\n\r\n" in trailers:
                        return out, b"", True
                    return out, data, False
                if len(data) < line_end + 2 + size + 2:
                    return out, data, False
                if data[line_end + 2 + size:line_end + 4 + size] != b"\r\n":
                    raise ValueError("malformed chunk delimiter")
                out += data[line_end + 2:line_end + 2 + size]
                data = data[line_end + 2 + size + 2:]

        pending = raw
        finished = False
        while True:
            if chunked:
                payload, pending, finished = dechunk(pending)
            elif remaining is not None:
                payload, pending = pending[:remaining], b""
                remaining -= len(payload)
                finished = remaining == 0
            else:
                payload, pending = pending, b""
            if payload:
                # SSE permits LF, CR and CRLF. A CR and its optional LF can
                # arrive in separate HTTP chunks or socket reads.
                if previous_cr and payload.startswith(b"\n"):
                    payload = payload[1:]
                previous_cr = payload.endswith(b"\r")
                decoded += payload.replace(b"\r\n", b"\n").replace(b"\r", b"\n")
            while b"\n\n" in decoded:
                event, decoded = decoded.split(b"\n\n", 1)
                lines = event.decode("utf-8").split("\n")
                data_lines = [line[5:].removeprefix(" ") for line in lines if line.startswith("data:")]
                if not data_lines:
                    continue
                data = "\n".join(data_lines).strip()
                if rec["done_ms"] is not None:
                    raise ValueError("data after [DONE]")
                if data == "[DONE]":
                    rec["done_ms"] = ms_since(t0)
                    continue
                obj = json.loads(data)
                if not isinstance(obj, dict):
                    raise ValueError("stream event must be a JSON object")
                if "error" in obj:
                    raise ValueError(f"stream error record: {obj['error']}")
                choices = obj.get("choices", [])
                if not isinstance(choices, list) or (choices and not isinstance(choices[0], dict)):
                    raise ValueError("invalid stream choices")
                delta = choices[0].get("delta") if choices else None
                if delta is not None and not isinstance(delta, dict):
                    raise ValueError("invalid stream delta")
                piece = delta.get("content") if delta else None
                if piece is not None and not isinstance(piece, str):
                    raise ValueError("invalid stream content")
                if piece:
                    rec["content_chunks"] += 1
                    text_chars += len(piece)
                    if rec["first_content_ms"] is None:
                        rec["first_content_ms"] = ms_since(t0)
            if finished:
                break
            data = sock.recv(65536)
            if not data:
                if chunked:
                    raise ValueError("chunked response closed before terminal chunk completed")
                if remaining is not None:
                    raise ValueError("response closed before Content-Length bytes received")
                break
            if rec["first_byte_ms"] is None:
                rec["first_byte_ms"] = ms_since(t0)
            pending += data
        rec["text_chars"] = text_chars
        if rec["done_ms"] is None:
            raise ValueError("stream ended without data: [DONE]")
        if rec["first_content_ms"] is None:
            raise ValueError("no content delta before [DONE]")
    except (OSError, ValueError, http.client.HTTPException) as e:
        rec["error"] = f"{type(e).__name__}: {e}"
    finally:
        if sock is not None:
            sock.close()
    return rec


def main() -> int:
    ap = argparse.ArgumentParser(description="Client-side LAN API stream timing.")
    ap.add_argument("--host", required=True)
    ap.add_argument("--port", type=int, default=11434)
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=32)
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--hardware", default="undeclared",
                    help="operator-declared label, e.g. xbox-series-s-declared")
    ap.add_argument("--json-out", required=True)
    args = ap.parse_args()
    if (args.repeats < 1 or args.max_tokens < 1 or not 0 < args.port < 65536
            or not math.isfinite(args.timeout) or args.timeout <= 0 or not args.prompt):
        ap.error("repeats/max-tokens >= 1, port 1..65535, timeout > 0, non-empty prompt")

    attempts = []
    for i in range(args.repeats):
        rec = one_attempt(args.host, args.port, args.model, args.prompt,
                          args.max_tokens, args.timeout)
        rec["attempt"] = i
        attempts.append(rec)
        state = "ok" if rec["error"] is None else rec["error"]
        print(f"attempt {i}: headers={rec['headers_ms']} first_content={rec['first_content_ms']} "
              f"done={rec['done_ms']} ms  {state}")

    ok = [a for a in attempts if a["error"] is None]
    percentiles = request_percentiles(ok)
    failed = len(attempts) - len(ok)

    out = {
        "schema": "xllama-api-ttfb/1",
        "utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": f"{args.host}:{args.port}/v1/chat/completions",
        "model": args.model, "prompt_chars": len(args.prompt),
        "max_tokens": args.max_tokens, "hardware_declared": args.hardware,
        "scope": "client-observed LAN timing incl. network, HTTP and any model load; "
                 "not the in-app ttft_ms and not a throughput result",
        "attempts": attempts, "complete": len(ok), "failed": failed,
        "error_rate": failed / len(attempts),
        "median_ms": {stage: estimates["p50"] for stage, estimates in percentiles.items()},
        "latency_distribution": {
            "completed_requests": len(ok),
            "method": "linear (R-7)",
            "percentiles_ms": percentiles,
        },
    }
    with open(args.json_out, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    return 0 if len(ok) == len(attempts) else 1


if __name__ == "__main__":
    sys.exit(main())
