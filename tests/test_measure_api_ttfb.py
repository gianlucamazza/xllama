# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
"""Exercise the client-side API timing probe against a local fake server."""
import json
import socket
import subprocess
import sys
import tempfile
import threading
import unittest
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
    def __init__(self, status=200, events=GOOD, split=False):
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(8)
        self.port = self.sock.getsockname()[1]
        self.status, self.events, self.split = status, events, split
        self.thread = threading.Thread(target=self.serve, daemon=True)
        self.thread.start()

    def serve(self):
        while True:
            try:
                conn, _ = self.sock.accept()
            except OSError:
                return
            with conn:
                buf = b""
                while b"\r\n\r\n" not in buf:
                    buf += conn.recv(65536)
                head, body = buf.split(b"\r\n\r\n", 1)
                length = int([l for l in head.split(b"\r\n") if l.lower().startswith(b"content-length")][0].split(b":")[1])
                while len(body) < length:
                    body += conn.recv(65536)
                json.loads(body)  # the probe must send a valid request body
                if self.status != 200:
                    payload = b'{"error":{"message":"busy"}}'
                    conn.sendall(f"HTTP/1.1 {self.status} Busy\r\nContent-Length: {len(payload)}\r\n\r\n".encode() + payload)
                    continue
                conn.sendall(b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nTransfer-Encoding: chunked\r\n\r\n")
                for ev in self.events:
                    data = chunk(ev)
                    if self.split:  # chunk boundaries inside the size line and the payload
                        for i in range(0, len(data), 3):
                            conn.sendall(data[i:i + 3])
                    else:
                        conn.sendall(data)
                conn.sendall(b"0\r\n\r\n")

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

    def test_missing_done_fails(self):
        r, data = self.run_probe(self.serve(events=GOOD[:-1]), repeats=1)
        self.assertEqual(r.returncode, 1)
        self.assertIn("[DONE]", data["attempts"][0]["error"])

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
