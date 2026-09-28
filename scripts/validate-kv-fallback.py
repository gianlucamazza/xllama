#!/usr/bin/env python3
"""Force a divergent restored snapshot and compare Xbox output with cold inference."""

import argparse
import json
import re

from console_test import Console, fixture, generations, settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    console = Console(args.out)
    cid = "ap-kv-divergent-probe"
    actions = [
        {"op": "load_chat", "id": cid},
        {
            "op": "send",
            "text": "Summarize this harbour log in one sentence. "
            + "The harbour master logged tide tables, cargo manifests and crane maintenance. "
            * 40,
            "timeout_s": 180,
        },
        # Loading an empty fixed-ID chat saves the first one without creating
        # an untracked random conversation.
        {"op": "load_chat", "id": cid + "-other"},
        {"op": "mark", "label": "snapshot-saved", "timeout_s": 120},
        {"op": "load_chat", "id": cid},
        {"op": "send", "text": "Say goodbye.", "timeout_s": 180},
        {"op": "mark", "label": "fallback-finished", "timeout_s": 120},
        {"op": "quit"},
    ]
    try:
        console.setup(actions, settings())
        fixture(console, cid, [])
        fixture(console, cid + "-other", [])
        console.command("start-app", console.pfn)
        console.wait(
            lambda: (
                "prompt budget:" in console.log() and "api: listening" in console.log()
            )
        )
        busy = []
        for route in ["/api/embed", "/v1/embeddings"]:
            code, response = console.api(
                route, {"model": "embed-bge-m3", "input": "busy probe"}
            )
            assert code == 503, (route, code, response)
            error = response.get("error")
            assert isinstance(error, dict if route.startswith("/v1/") else str)
            assert (
                error.get("message") if isinstance(error, dict) else error
            ) == "busy"
            busy.append({"route": route, "status": code, "response": response})
        (console.out / "busy-errors.json").write_text(json.dumps(busy, indent=2) + "\n")
        console.wait(
            lambda: (
                console.fetch("autopilot-mark.txt", optional=True) == b"snapshot-saved"
            )
        )
        console.wait(lambda: "KV state saved" in console.log())
        document = json.loads(console.fetch(cid + ".json", "chats"))
        assert document["messages"][-1]["role"] == "assistant"
        document["messages"][-1]["content"] = "Edited assistant reply."
        console.put(cid + ".json", json.dumps(document), "chats")
        console.delete("autopilot-mark.txt")
        console.wait(
            lambda: (
                console.fetch("autopilot-mark.txt", optional=True)
                == b"fallback-finished"
            )
        )
        log = console.log()
        (console.out / "fallback-device.log").write_text(log)
        assert "KV snapshot restored" in log and "KV rewind unsupported" in log
        assert re.search(r"full re-prefill.*common=\d+ resident=\d+ prompt=\d+", log)
        document = json.loads(console.fetch(cid + ".json", "chats"))
        actual = document["messages"][-1]["content"]
        messages = [{"role": "system", "content": settings()["system_prompt"]}]
        messages += [
            {"role": m["role"], "content": m["content"]}
            for m in document["messages"][:-1]
        ]
        # Swap to another backend/model, then create a genuinely cold LFM Session.
        code, _ = console.api(
            "/api/embed", {"model": "embed-bge-m3", "input": "cold reference"}
        )
        assert code == 200
        code, reference = console.api(
            "/v1/chat/completions",
            {
                "model": "lfm25-350m",
                "messages": messages,
                "max_tokens": 24,
                "temperature": 0,
                "top_p": 1,
            },
        )
        assert code == 200, reference
        expected = reference["choices"][0]["message"]["content"]
        assert actual == expected, (actual, expected)
        assert generations(log)[-1] == reference["usage"]["prompt_tokens"]
        (console.out / "cold-reference.json").write_text(
            json.dumps(reference, indent=2) + "\n"
        )
        console.delete("autopilot-mark.txt")
        console.wait(
            lambda: console.fetch("autopilot-done.txt", optional=True) == b"ok"
        )
        print(
            "KV divergent snapshot: PASS (cold-equivalent output and full token count)"
        )
    finally:
        (console.out / "final-device.log").write_text(console.log())
        console.restore()


if __name__ == "__main__":
    main()
