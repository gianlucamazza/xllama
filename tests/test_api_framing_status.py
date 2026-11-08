# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
"""Exercise the actual socket framing gate with complete training-status replies."""

import contextlib
import io
import json
import os
from pathlib import Path
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    (ROOT / "scripts/validate-api.sh")
    .read_text()
    .split("<<'PYFRAME'\n", 1)[1]
    .split("\nPYFRAME", 1)[0]
)


class ReplySocket:
    def __init__(self, status, body):
        header = f"HTTP/1.1 {status} Status\r\nContent-Length: {len(body)}\r\n\r\n"
        self.reply = header.encode() + body

    def sendall(self, data):
        pass

    def shutdown(self, mode):
        pass

    def recv(self, size):
        result, self.reply = self.reply[:size], self.reply[size:]
        return result

    def close(self):
        pass


def status_reply(count):
    # Real device-train progress makes the field lie beyond the display preview.
    return json.dumps(
        {"state": "ok", "progress": {"details": "x" * 512}, "usable_samples": count}
    ).encode()


class ApiFramingStatusTests(unittest.TestCase):
    def run_gate(self, before, after, before_status=200):
        replies = (
            [(before_status, before)]
            + [(code, b"{}") for code in (400, 400, 413, 411, 400, 408, 200)]
            + [(200, after)]
        )
        output = io.StringIO()
        with mock.patch.dict(os.environ, {"XBOX_IP": "127.0.0.1", "PORT": "12345"}), mock.patch(
            "socket.create_connection", side_effect=[ReplySocket(*reply) for reply in replies]
        ) as connect, contextlib.redirect_stdout(output):
            with self.assertRaises(SystemExit) as exit_result:
                exec(compile(SOURCE, "validate-api.sh:PYFRAME", "exec"), {})
        self.assertEqual(connect.call_count, 9)
        return exit_result.exception.code, output.getvalue()

    def test_long_status_body_is_parsed_before_display_truncation(self):
        code, output = self.run_gate(status_reply(17), status_reply(17))
        self.assertEqual(code, 0, output)
        self.assertIn("usable_samples unchanged (17)", output)

    def test_changed_counter_fails(self):
        code, output = self.run_gate(status_reply(17), status_reply(18))
        self.assertEqual(code, 1, output)
        self.assertIn("usable_samples 17 -> 18", output)

    def test_invalid_status_response_fails_closed(self):
        for body, status in ((b"{truncated", 200), (status_reply(17), 500)):
            with self.subTest(body=body, status=status):
                code, output = self.run_gate(body, status_reply(17), status)
                self.assertEqual(code, 1, output)

    def test_invalid_counter_fails_closed(self):
        for count in (True, -1, "17", None):
            with self.subTest(count=count):
                code, output = self.run_gate(status_reply(count), status_reply(count))
                self.assertEqual(code, 1, output)


if __name__ == "__main__":
    unittest.main()
