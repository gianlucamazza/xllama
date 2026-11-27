# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
"""Exercise the client-side API timing probe against a local fake server."""
import itertools
import json
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
from contextlib import suppress
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/measure-api-ttfb.py"


def chunk(data: bytes) -> bytes:
    return f"{len(data):x}\r\n".encode() + data + b"\r\n"


def sse(obj) -> bytes:
    return b"data: " + (obj if isinstance(obj, bytes) else json.dumps(obj).encode()) + b"\n\n"


def delta(text):
    return {"object": "chat.completion.chunk", "choices": [{"delta": {"content": text}}]}


GOOD = [sse({"choices": [{"delta": {"role": "assistant"}}]}), sse(delta("Hel")),
        sse(delta("lo")), sse({"choices": [{"delta": {}, "finish_reason": "stop"}]}),
        sse(b"[DONE]")]


class FakeServer:
    def __init__(self, status=200, events=GOOD, split=False, terminator=b"0\r\n\r\n"):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.port = self.sock.getsockname()[1]
        self.statuses = itertools.cycle([status] if isinstance(status, int) else status)
        self.events, self.split = events, split
        self.terminator = terminator
        self.thread = threading.Thread(target=self.serve, daemon=True)
        self.thread.start()

    def serve(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            # A probe that rejects a malformed event may close before [DONE].
            with conn, suppress(BrokenPipeError, ConnectionResetError):
                buf = b""
                while b"\r\n\r\n" not in buf:
                    buf += conn.recv(65536)
                head, body = buf.split(b"\r\n\r\n", 1)
                length = int([l for l in head.split(b"\r\n") if l.lower().startswith(b"content-length")][0].split(b":")[1])
                while len(body) < length:
                    body += conn.recv(65536)
                json.loads(body)  # the probe must send a valid request body
                status = next(self.statuses)
                if status != 200:
                    payload = b'{"error":{"message":"busy"}}'
                    conn.sendall(f"HTTP/1.1 {status} Busy\r\nContent-Length: {len(payload)}\r\n\r\n".encode() + payload)
                    continue
                conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n")
                for ev in self.events:
                    data = chunk(ev)
                    if self.split:  # chunk boundaries inside the size line and the payload
                        for i in range(0, len(data), 3):
                            conn.sendall(data[i:i + 3])
                    else:
                        conn.sendall(data)
                conn.sendall(self.terminator)

    def close(self):
        self.sock.close()


class MeasureApiTtfbTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / "ttfb.json"

    def run_probe(self, port, *extra, repeats=2):
        r = subprocess.run([sys.executable, str(SCRIPT), "--host", "127.0.0.1", "--port", str(port),
                            "--model", "m", "--prompt", "hi", "--repeats", str(repeats),
                            "--timeout", "10", "--json-out", str(self.out), *extra],
                           capture_output=True, text=True, check=False)
        data = json.loads(self.out.read_text()) if self.out.exists() else None
        return r, data

    def serve(self, **kw):
        s = FakeServer(**kw)
        self.addCleanup(s.close)
        return s.port

    def test_complete_stream_records_ordered_timings(self):
        for split in (False, True):
            r, data = self.run_probe(self.serve(split=split))
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertEqual(data["complete"], 2)
            a = data["attempts"][0]
            self.assertEqual(a["content_chunks"], 2)
            self.assertEqual(a["text_chars"], 5)
            order = [a[k] for k in ("connect_ms", "headers_ms", "first_byte_ms",
                                    "first_content_ms", "done_ms")]
            self.assertEqual(order, sorted(order))
            self.assertIsNotNone(data["median_ms"]["first_content_ms"])

    def test_busy_status_is_a_retained_failure(self):
        r, data = self.run_probe(self.serve(status=503))
        self.assertEqual(r.returncode, 1)
        self.assertEqual(data["failed"], 2)
        self.assertIn("HTTP 503", data["attempts"][0]["error"])
        self.assertIsNone(data["median_ms"]["done_ms"])
        self.assertEqual(data["error_rate"], 1)
        self.assertEqual(data["latency_distribution"]["completed_requests"], 0)
        for estimates in data["latency_distribution"]["percentiles_ms"].values():
            self.assertEqual(estimates, dict.fromkeys(("p50", "p95", "p99")))

    def test_request_quantiles_include_only_completions_and_failures_count_towards_error_rate(self):
        r, rec = self.run_probe(self.serve(status=[200, 503, 200, 200, 200, 503]), repeats=6)
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertEqual([a["status"] for a in rec["attempts"]], [200, 503, 200, 200, 200, 503])
        self.assertEqual(rec["error_rate"], 2 / 6)
        dist = rec["latency_distribution"]
        self.assertEqual(dist["completed_requests"], 4)
        self.assertEqual(dist["method"], "linear (R-7)")
        self.assertEqual(set(dist["percentiles_ms"]), set(rec["median_ms"]))
        for field, quantiles in dist["percentiles_ms"].items():
            values = sorted(a[field] for a in rec["attempts"] if a["error"] is None)
            self.assertEqual(quantiles["p50"], round((values[1] + values[2]) / 2, 3))
            self.assertEqual(quantiles["p50"], rec["median_ms"][field])
            # For four observations R-7 uses zero-based ranks 2.85 and 2.97.
            self.assertAlmostEqual(quantiles["p95"], values[2] * 0.15 + values[3] * 0.85, delta=0.000501)
            self.assertAlmostEqual(quantiles["p99"], values[2] * 0.03 + values[3] * 0.97, delta=0.000501)

    def test_single_complete_request_has_one_empirical_observation(self):
        r, rec = self.run_probe(self.serve(), repeats=1)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(rec["error_rate"], 0)
        self.assertEqual(rec["latency_distribution"]["completed_requests"], 1)
        for field, quantiles in rec["latency_distribution"]["percentiles_ms"].items():
            self.assertEqual(quantiles, dict.fromkeys(("p50", "p95", "p99"), rec["attempts"][0][field]))

    def test_missing_done_fails(self):
        r, data = self.run_probe(self.serve(events=GOOD[:-1]), repeats=1)
        self.assertEqual(r.returncode, 1)
        self.assertIn("[DONE]", data["attempts"][0]["error"])

    def test_sse_line_endings_across_http_chunks(self):
        for ending in (b"\r\n", b"\r"):
            with self.subTest(ending=ending):
                payload = b"".join(GOOD).replace(b"\n", ending)
                # Each byte is its own HTTP chunk (including the two bytes of CRLF).
                port = self.serve(events=[bytes([b]) for b in payload], split=True,
                                  terminator=b"0\r\nX-Probe: ok\r\n\r\n")
                r, rec = self.run_probe(port, repeats=1)
                self.assertEqual(r.returncode, 0, r.stdout)
                self.assertEqual(rec["attempts"][0]["content_chunks"], 2)

    def test_multiline_sse_data_is_one_content_event(self):
        event = b'data: {"choices":\ndata: [{"delta": {"content": "hey"}}]}\n\n'
        r, rec = self.run_probe(self.serve(events=[event, GOOD[-1]], split=True), repeats=1)
        self.assertEqual(r.returncode, 0, r.stdout)
        self.assertEqual(rec["attempts"][0]["content_chunks"], 1)
        self.assertEqual(rec["attempts"][0]["text_chars"], 3)

    def test_duplicate_done_and_incomplete_http_termination_are_retained_failures(self):
        for events, terminator, error in ((GOOD + [GOOD[-1]], b"0\r\n\r\n", "data after"),
                                          (GOOD, b"", "terminal chunk"),
                                          (GOOD, b"0\r\n", "terminal chunk")):
            with self.subTest(terminator=terminator):
                port = self.serve(events=events, terminator=terminator, split=True)
                r, rec = self.run_probe(port, repeats=1)
                self.assertEqual(r.returncode, 1, r.stdout)
                self.assertIn(error, rec["attempts"][0]["error"])
                self.assertEqual(rec["error_rate"], 1)
                self.assertIsNone(rec["median_ms"]["done_ms"])

    def test_no_content_delta_fails(self):
        r, data = self.run_probe(self.serve(events=[GOOD[0], GOOD[-1]]), repeats=1)
        self.assertEqual(r.returncode, 1)
        self.assertIn("no content", data["attempts"][0]["error"])

    def test_stream_error_record_and_malformed_sse_fail(self):
        r, data = self.run_probe(self.serve(events=[sse({"error": "boom"}), GOOD[-1]]), repeats=1)
        self.assertEqual(r.returncode, 1)
        self.assertIn("boom", data["attempts"][0]["error"])
        self.out.unlink()
        r, data = self.run_probe(self.serve(events=[sse(b"{not json"), GOOD[-1]]), repeats=1)
        self.assertEqual(r.returncode, 1)
        self.assertIsNotNone(data["attempts"][0]["error"])

    def test_invalid_event_shapes_remain_in_the_error_rate_denominator(self):
        for event in (b"null", {"choices": {}}, {"choices": [0]},
                      {"choices": [{"delta": 2}]}, {"choices": [{"delta": {"content": [1]}}]}):
            with self.subTest(event=event):
                r, data = self.run_probe(self.serve(events=[sse(event), GOOD[-1]]), repeats=1)
                self.assertEqual(r.returncode, 1, r.stderr)
                self.assertEqual(data["error_rate"], 1)
                self.assertEqual(data["latency_distribution"]["completed_requests"], 0)

    def test_unreachable_endpoint_is_retained(self):
        s = socket.socket()
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
        s.close()
        r, data = self.run_probe(port, repeats=1)
        self.assertEqual(r.returncode, 1)
        self.assertIsNone(data["attempts"][0]["connect_ms"])

    def test_invalid_arguments(self):
        for extra in (["--repeats", "0"], ["--timeout", "nan"], ["--port", "0"]):
            r = subprocess.run([sys.executable, str(SCRIPT), "--host", "h", "--model", "m",
                                "--prompt", "p", "--json-out", str(self.out), *extra],
                               capture_output=True, text=True, check=False)
            self.assertEqual(r.returncode, 2, extra)


if __name__ == "__main__":
    unittest.main()
