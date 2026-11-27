#!/usr/bin/env python3
# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
"""Optional SDK contract fixture; no network or console measurements."""

import json
import runpy
import unittest
from pathlib import Path

import httpx2
import openai

check_journey = runpy.run_path(str(Path(__file__).resolve().parents[1]
                                   / "scripts/check-sdk-chat-journey.py"))["check_journey"]


class SDKChatJourney(unittest.TestCase):
    def exercise(self, city="Paris", finish="stop", recovery_status=503, error_message="busy"):
        requests = []
        model = "fixture-gguf"
        call = {"id": "fixture-call-3", "type": "function", "function": {
            "name": "get_weather", "arguments": json.dumps({"city": city})}}

        def respond(request):
            requests.append(request.url.path)
            n = len(requests)
            if n == 1:
                self.assertEqual((request.method, request.url.path), ("GET", "/v1/models"))
                return httpx2.Response(200, json={"object": "list", "data": [
                    {"id": model, "object": "model", "created": 0, "owned_by": "fixture"}]})
            self.assertEqual((request.method, request.url.path), ("POST", "/v1/chat/completions"))
            body = json.loads(request.content)
            self.assertEqual(body["model"], model)
            base = {"id": "fixture", "created": 0, "model": model, "object": "chat.completion"}
            if n == 2:
                self.assertEqual(body["tool_choice"]["function"]["name"], call["function"]["name"])
                message = {"role": "assistant", "content": None, "tool_calls": [call]}
                return httpx2.Response(200, json={**base, "choices": [
                    {"index": 0, "message": message, "finish_reason": "tool_calls"}]})
            if n in (3, 4):
                self.assertIs(body["stream"], True)
                if n == 3:
                    self.assertEqual(body["messages"][-2]["tool_calls"], [call])
                    self.assertEqual(body["messages"][-1]["tool_call_id"], call["id"])
                chunks = [({"role": "assistant"}, None), ({"content": "Rain."}, None), ({}, finish)]
                frames = ["data: " + json.dumps({**base, "object": "chat.completion.chunk",
                          "choices": [{"index": 0, "delta": delta, "finish_reason": reason}]})
                          for delta, reason in chunks]
                return httpx2.Response(200, headers={"content-type": "text/event-stream"},
                                       content="\n\n".join([*frames, "data: [DONE]", ""]) + "\n")
            if n == 5:
                return httpx2.Response(recovery_status, json={"error": {"message": error_message}})
            self.assertEqual(n, 6)
            return httpx2.Response(200, json={**base, "choices": [{"index": 0, "finish_reason": "stop",
                                   "message": {"role": "assistant", "content": "OK"}}]})

        with openai.OpenAI(api_key="fixture", base_url="http://fixture.invalid/v1", max_retries=0,
                           http_client=httpx2.Client(transport=httpx2.MockTransport(respond))) as client:
            check_journey(client, model)
        self.assertEqual(len(requests), 6)

    def test_sdk_round_trip(self):
        self.exercise()

    def test_rejects_malformed_tool_semantics(self):
        with self.assertRaisesRegex(ValueError, "unexpected tool"):
            self.exercise(city="Rome")

    def test_requires_terminal_stream_choice(self):
        with self.assertRaisesRegex(ValueError, "incomplete streamed"):
            self.exercise(finish=None)

    def test_propagates_out_of_memory(self):
        with self.assertRaises(openai.APIStatusError):
            self.exercise(error_message="out of memory")

    def test_propagates_recovery_http_failure(self):
        with self.assertRaises(openai.APIStatusError):
            self.exercise(recovery_status=400)


if __name__ == "__main__":
    unittest.main()
