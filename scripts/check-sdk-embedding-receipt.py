#!/usr/bin/env python3
# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
"""Replay captured Xbox embedding requests through the official Python SDK."""

import argparse
import base64
import json
import struct
from pathlib import Path

import httpx2
import openai


def check_row(row):
    payload = dict(row["request"])
    encoding = payload["encoding_format"]
    if encoding == "base64":
        # Exercise the SDK default: base64 on the wire, floats for the caller.
        del payload["encoding_format"]

    def replay(request):
        if (
            request.method != "POST"
            or request.url.path != row["route"]
            or json.loads(request.content) != row["request"]
        ):
            raise ValueError("SDK request differs from console receipt")
        return httpx2.Response(row["status"], json=row["response"])

    with openai.OpenAI(
        api_key="offline",
        base_url="http://receipt.invalid/v1",
        http_client=httpx2.Client(transport=httpx2.MockTransport(replay)),
        max_retries=0,
    ) as client:
        response = client.embeddings.create(**payload)

    expected = row["response"]
    if (
        response.object != "list"
        or response.model != expected["model"]
        or response.usage.to_dict() != expected["usage"]
        or not response.data
        or len(response.data) != len(expected["data"])
    ):
        raise ValueError("SDK embedding metadata differs from console receipt")
    for item, raw in zip(response.data, expected["data"]):
        vector = raw["embedding"]
        if encoding == "base64":
            binary = base64.b64decode(vector, validate=True)
            vector = [value for (value,) in struct.iter_unpack("<f", binary)]
        if item.index != raw["index"] or not vector or item.embedding != vector:
            raise ValueError("SDK embedding values differ from console receipt")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path)
    receipt = json.loads(parser.parse_args().receipt.read_text(encoding="utf-8"))
    checked = set()
    for row in receipt["requests"]:
        if row["route"] == "/v1/embeddings" and row["status"] == 200:
            encoding = row["request"].get("encoding_format")
            if encoding in {"float", "base64"}:
                check_row(row)
                checked.add(encoding)
    if checked != {"float", "base64"}:
        raise ValueError("receipt requires successful float and base64 responses")
    print(f"PASS openai-python {openai.__version__}: float and default base64")


if __name__ == "__main__":
    main()
