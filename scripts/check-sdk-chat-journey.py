#!/usr/bin/env python3
# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
"""Exercise the official OpenAI Python SDK against an explicitly selected xLlama LAN host."""

import argparse
import json
import time
from urllib.parse import urlsplit

import httpx2
import openai


def check_journey(client, model):
    if model not in {item.id for item in client.models.list().data}:
        raise ValueError(f"model missing from discovery: {model}")
    options = {"model": model, "temperature": 0, "seed": 42}
    tool = {"type": "function", "function": {
        "name": "get_weather", "description": "Current weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"]},
    }}
    messages = [{"role": "user", "content": "What is the weather in Paris?"}]
    chat = client.chat.completions
    proposal = chat.create(**options, messages=messages, tools=[tool], max_tokens=128,
                           tool_choice={"type": "function", "function": {"name": "get_weather"}})
    choice = proposal.choices[0]
    calls = choice.message.tool_calls or []
    if choice.finish_reason != "tool_calls" or len(calls) != 1:
        raise ValueError("expected one tool proposal")
    call = calls[0]
    if (call.type != "function" or not call.id or call.function.name != "get_weather"
            or json.loads(call.function.arguments) != {"city": "Paris"}):
        raise ValueError("unexpected tool proposal or arguments")

    # The host supplies a fixed fixture. Nothing executes the proposed function.
    messages += [{"role": "assistant", "content": None, "tool_calls": [call.to_dict()]},
                 {"role": "tool", "tool_call_id": call.id, "content": "18 C, light rain"}]
    text = []
    finish = None
    with chat.create(**options, messages=messages, tools=[tool],
                     max_tokens=96, stream=True) as stream:
        for chunk in stream:
            for part in chunk.choices:
                if part.delta.tool_calls:
                    raise ValueError("unexpected second tool proposal")
                if part.delta.content:
                    text.append(part.delta.content)
                if part.finish_reason:
                    finish = part.finish_reason
    if not "".join(text).strip() or finish not in {"stop", "length"}:
        raise ValueError("incomplete streamed tool-result answer")

    # Close a separate stream early, then require a real chat to run again.
    with chat.create(**options, messages=[{"role": "user", "content": "Count from 1 to 100."}],
                     max_tokens=256, stream=True) as stream:
        first = next((part for chunk in stream for part in chunk.choices), None)
        if first is None or first.finish_reason is not None:
            raise ValueError("no unfinished stream to close")
    start = time.monotonic()
    deadline = start + 30
    while time.monotonic() < deadline:
        try:
            probe = chat.create(**options, messages=[{"role": "user", "content": "Say OK."}],
                                max_tokens=8, timeout=max(0.001, deadline - time.monotonic()))
        except openai.APIStatusError as error:
            if (error.status_code != 503 or not isinstance(error.body, dict)
                    or error.body.get("message") != "busy"):
                raise
            time.sleep(0.25)
            continue
        if not any(choice.message.content for choice in probe.choices):
            raise ValueError("recovery chat returned no content")
        elapsed = time.monotonic() - start
        if elapsed <= 30:
            return round(elapsed, 3)
        break
    raise TimeoutError("chat did not recover within 30 seconds after SDK stream close")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", required=True, help="Console origin, e.g. http://192.168.1.2:11434")
    parser.add_argument("--model", required=True, help="Installed GGUF with named tool support")
    parser.add_argument("--package", required=True, help="Expected /health package version")
    args = parser.parse_args()
    origin = urlsplit(args.origin)
    if (origin.scheme not in {"http", "https"} or not origin.hostname or origin.username is not None
            or origin.path not in {"", "/"} or origin.query or origin.fragment):
        parser.error("--origin must be an HTTP(S) origin without credentials, path, query or fragment")
    with httpx2.Client(trust_env=False, timeout=120) as http:
        response = http.get(args.origin.rstrip("/") + "/health")
        response.raise_for_status()
        health = response.json()
        if (health.get("service") != "xllama" or health.get("version") != args.package
                or health.get("api") != 1 or health.get("capabilities", {}).get("tools") is not True):
            raise ValueError("health did not identify the expected package and tool-capable API")
        with openai.OpenAI(api_key="unused-local", base_url=args.origin.rstrip("/") + "/v1",
                           http_client=http, max_retries=0, timeout=120) as client:
            seconds = check_journey(client, args.model)
    print(json.dumps({"result": "pass", "mode": "live", "sdk": openai.__version__,
                      "model": args.model, "package": args.package, "runtime": health.get("runtime"),
                      "seconds_to_recovery_chat": seconds}))


if __name__ == "__main__":
    main()
