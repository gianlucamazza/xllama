#!/usr/bin/env python3
"""Live embedding contract checks against the Xbox LAN API."""

import argparse
import base64
import json
import math
import struct
import time
import urllib.error
import urllib.request
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    parser.add_argument("model", choices=("embed-bge-m3", "embed-nomic-v2-moe"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    width = 1024 if args.model == "embed-bge-m3" else 768
    records = []

    def request(route, fields, expected=200):
        payload = {"model": args.model, **fields}
        req = urllib.request.Request(
            args.url + route,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        started = time.monotonic()
        try:
            response = urllib.request.urlopen(req, timeout=300)
        except urllib.error.HTTPError as error:
            response = error
        with response:
            status = response.status
            body = json.load(response)
        records.append(
            {
                "route": route,
                "request": payload,
                "status": status,
                "wall_ms": (time.monotonic() - started) * 1000,
                "response": body,
            }
        )
        assert status == expected, (route, status, body)
        if expected != 200:
            if route.startswith("/api/"):
                assert isinstance(body.get("error"), str) and body["error"], body
            else:
                assert isinstance(body.get("error"), dict), body
                assert isinstance(body["error"].get("message"), str), body
        return body

    def vector(values, expected_width=width):
        assert len(values) == expected_width, len(values)
        assert all(math.isfinite(v) for v in values)
        assert abs(math.sqrt(sum(v * v for v in values)) - 1) < 1e-4
        return values

    text = "search_query: find a red car"
    try:
        batch = request(
            "/api/embed",
            {"input": [text, "search_document: a red car"], "dimensions": 0},
        )
        assert len(batch["embeddings"]) == 2
        for values in batch["embeddings"]:
            vector(values)
        assert batch["prompt_eval_count"] > 0
        assert batch["total_duration"] >= batch["load_duration"] >= 0

        legacy = request("/api/embeddings", {"prompt": text})
        vector(legacy["embedding"])
        floats = request(
            "/v1/embeddings", {"input": [text], "encoding_format": "float"}
        )
        assert floats["object"] == "list" and len(floats["data"]) == 1
        assert floats["data"][0]["index"] == 0
        reference = vector(floats["data"][0]["embedding"])
        encoded = request(
            "/v1/embeddings", {"input": text, "encoding_format": "base64"}
        )
        raw = base64.b64decode(encoded["data"][0]["embedding"], validate=True)
        assert len(raw) == width * 4
        decoded = vector(struct.unpack("<" + "f" * width, raw))
        assert max(abs(a - b) for a, b in zip(reference, decoded)) < 1e-5
        assert max(abs(a - b) for a, b in zip(reference, legacy["embedding"])) < 1e-5
        assert max(abs(a - b) for a, b in zip(reference, batch["embeddings"][0])) < 1e-5
        assert encoded["usage"]["prompt_tokens"] == floats["usage"]["prompt_tokens"] > 0

        request("/api/embed", {"input": ""}, 400)
        request("/api/embed", {"input": []}, 400)
        request("/api/embeddings", {"prompt": ""}, 400)
        request("/v1/embeddings", {"input": []}, 400)
        request("/v1/embeddings", {"input": text, "encoding_format": "invalid"}, 400)
        request("/api/embed", {"input": text, "dimensions": width + 1}, 400)
        if args.model == "embed-bge-m3":
            request("/api/embed", {"input": text, "dimensions": 256}, 400)
        else:
            reduced = request("/api/embed", {"input": text, "dimensions": 256})
            vector(reduced["embeddings"][0], 256)

        # Use an explicit small context to isolate truncation from machine speed.
        long_text = "search_document: " + "embedding text " * 180
        request(
            "/api/embed",
            {"input": long_text, "options": {"num_ctx": 128}, "truncate": False},
            400,
        )
        truncated = request(
            "/api/embed",
            {"input": long_text, "options": {"num_ctx": 128}, "truncate": True},
        )
        vector(truncated["embeddings"][0])
        assert truncated["prompt_eval_count"] == 128
        # Omitting num_ctx must restore catalogue defaults, not reuse 128.
        restored = request("/api/embed", {"input": long_text, "truncate": True})
        vector(restored["embeddings"][0])
        assert restored["prompt_eval_count"] > 128
        request("/api/embed", {"input": text, "options": {"num_ctx": 31}}, 400)
        request("/api/embed", {"input": text, "options": {"num_ctx": 8193}}, 400)
        request(
            "/v1/chat/completions",
            {"messages": [{"role": "user", "content": "Hi"}]},
            400,
        )
        print(f"embed: PASS ({args.model}, {len(records)} live requests)", flush=True)
    finally:
        if args.out:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(
                json.dumps({"model": args.model, "requests": records}, indent=2) + "\n"
            )


if __name__ == "__main__":
    main()
