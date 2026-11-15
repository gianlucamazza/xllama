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
import json
import math
import socket
import sys
import time
from datetime import datetime, timezone


def ms_since(t0: float) -> float:
    return round((time.perf_counter() - t0) * 1000.0, 3)


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
        headers = {k.strip().lower(): v.strip() for k, _, v in
                   (line.decode("latin-1").partition(":") for line in head.split(b"\r\n")[1:])}
        chunked = "chunked" in headers.get("transfer-encoding", "").lower()
        if rec["status"] != 200:
            raise ValueError(f"HTTP {rec['status']}")

        raw = rest
        if raw:
            rec["first_byte_ms"] = ms_since(t0)
        decoded, text = b"", ""

        def dechunk(data):
            """Return (decoded payload, leftover undecoded bytes, finished)."""
            out = b""
            while True:
                line_end = data.find(b"\r\n")
                if line_end < 0:
                    return out, data, False
                size_txt = data[:line_end].split(b";", 1)[0].strip()
                try:
                    size = int(size_txt, 16)
                except ValueError:
                    raise ValueError(f"malformed chunk size: {size_txt!r}") from None
                if size == 0:
                    return out, b"", True
                if len(data) < line_end + 2 + size + 2:
                    return out, data, False
                out += data[line_end + 2:line_end + 2 + size]
                data = data[line_end + 2 + size + 2:]

        pending = raw
        finished = False
        while True:
            if chunked:
                payload, pending, finished = dechunk(pending)
            else:
                payload, pending = pending, b""
            decoded += payload
            while b"\n\n" in decoded:
                event, decoded = decoded.split(b"\n\n", 1)
                for line in event.decode("utf-8").splitlines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        rec["done_ms"] = ms_since(t0)
                        continue
                    if rec["done_ms"] is not None:
                        raise ValueError("data after [DONE]")
                    obj = json.loads(data)
                    if "error" in obj:
                        raise ValueError(f"stream error record: {obj['error']}")
                    choices = obj.get("choices") or []
                    delta = (choices[0].get("delta") or {}) if choices else {}
                    piece = delta.get("content") or ""
                    if piece:
                        rec["content_chunks"] += 1
                        text += piece
                        if rec["first_content_ms"] is None:
                            rec["first_content_ms"] = ms_since(t0)
            if finished:
                break
            data = sock.recv(65536)
            if not data:
                break
            if rec["first_byte_ms"] is None:
                rec["first_byte_ms"] = ms_since(t0)
            pending += data
        rec["text_chars"] = len(text)
        if rec["done_ms"] is None:
            raise ValueError("stream ended without data: [DONE]")
        if rec["first_content_ms"] is None:
            raise ValueError("no content delta before [DONE]")
    except (OSError, ValueError, json.JSONDecodeError, UnicodeDecodeError) as e:
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

    def median(key):
        vals = sorted(a[key] for a in ok)
        if not vals:
            return None
        mid = len(vals) // 2
        return vals[mid] if len(vals) % 2 else round((vals[mid - 1] + vals[mid]) / 2, 3)

    out = {
        "schema": "xllama-api-ttfb/1",
        "utc": datetime.now(timezone.utc).isoformat(),
        "endpoint": f"{args.host}:{args.port}/v1/chat/completions",
        "model": args.model, "prompt_chars": len(args.prompt),
        "max_tokens": args.max_tokens, "hardware_declared": args.hardware,
        "scope": "client-observed LAN timing incl. network, HTTP and any model load; "
                 "not the in-app ttft_ms and not a throughput result",
        "attempts": attempts, "complete": len(ok), "failed": len(attempts) - len(ok),
        "median_ms": {k: median(k) for k in
                      ("connect_ms", "headers_ms", "first_byte_ms", "first_content_ms", "done_ms")},
    }
    with open(args.json_out, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    return 0 if len(ok) == len(attempts) else 1


if __name__ == "__main__":
    sys.exit(main())
