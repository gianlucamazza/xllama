#!/usr/bin/env python3
"""Validate xllama's native Ollama routes and model residency on a deployed build.

Run it from another host on the LAN (validate-api.sh compat does), for example:

    python3 scripts/validate-api-compat.py http://<console-ip>:11434 lfm25-350m \\
        --tool-model lfm25-1.2b-instruct

The model must be a GGUF chat model already on the console. Contract:
docs/api-endpoint.md (Native Ollama generation).
"""

import argparse
import importlib.metadata
import json
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen


def call(base, route, body=None):
    payload = None if body is None else json.dumps(body).encode()
    return urlopen(
        Request(base + route, data=payload, headers={"Content-Type": "application/json"}),
        timeout=120,
    )


def read_json(base, route, body=None):
    with call(base, route, body) as reply:
        return json.load(reply)


def expect_error(base, route, body, status, text="", native=True):
    try:
        read_json(base, route, body)
    except HTTPError as exc:
        with exc:
            error = json.load(exc)["error"]
            assert exc.code == status, (route, exc.code, error)
            message = error if native else error["message"]
            assert isinstance(message, str) and message and text in message, error
    else:
        raise AssertionError(f"{route} should return HTTP {status}")


def native_stream(base, route, request, text):
    with call(base, route, request) as reply:
        assert reply.headers.get_content_type() == "application/x-ndjson"
        chunks = [json.loads(line) for line in reply]
    assert len(chunks) >= 2 and all(chunk["done"] is False for chunk in chunks[:-1]), chunks
    final = chunks[-1]
    assert final["done"] is True and final["done_reason"] in {"stop", "length"}, final
    assert final["eval_count"] >= 1 and final["prompt_eval_count"] >= 1, final
    if text:
        key = (lambda c: c["message"]["content"]) if route == "/api/chat" else (
            lambda c: c["response"]
        )
        assert "".join(map(key, chunks)).strip(), chunks
    return chunks


def validate_native(base, model, tool_model, ort_model):
    health = read_json(base, "/health")
    assert health["capabilities"]["ollama_chat"] is True, health["capabilities"]
    print(f"MSIX {health['version']}")
    models = {m["id"]: m for m in read_json(base, "/v1/models")["data"]}
    assert models[model]["capabilities"]["ollama_chat"] is True, models[model]
    assert any(item["model"] == model for item in read_json(base, "/api/tags")["models"])

    messages = [{"role": "user", "content": "Count from one to five."}]
    chat = {"model": model, "messages": messages}
    prompt = {"model": model, "prompt": messages[0]["content"]}
    short = {"temperature": 0, "num_predict": 16}
    native_stream(base, "/api/chat", {**chat, "options": short}, text=True)
    native_stream(base, "/api/generate", {**prompt, "options": short}, text=True)
    for route, body in (("/api/chat", chat), ("/api/generate", prompt)):
        reply = read_json(base, route, {**body, "stream": False, "options": short})
        assert reply["done"] is True and reply["model"] == model, reply
        text = reply["message"]["content"] if route == "/api/chat" else reply["response"]
        assert text.strip(), reply
    capped = read_json(base, "/api/generate", {
        **prompt, "stream": False, "options": {"temperature": 0, "num_predict": 1},
    })
    assert capped["done_reason"] == "length" and capped["eval_count"] == 1, capped
    print("Ollama NDJSON and JSON framing passed")

    for options, text in (({"num_gpu": 1}, "num_gpu"), ({"temperature": 3}, "temperature"),
                          ({"stop": [1]}, "stop"), ({"num_predict": 0}, "num_predict")):
        expect_error(base, "/api/chat", {**chat, "options": options}, 400, text)
    for field, value in (("format", "json"), ("images", ["AA=="]), ("raw", True)):
        expect_error(base, "/api/generate", {**prompt, field: value}, 400, field)
    expect_error(base, "/api/chat", {**chat, "model": "xllama-nonexistent-model"}, 404)
    thinking = models[model]["capabilities"]["reasoning"]
    read_json(base, "/api/chat", {**chat, "stream": False, "options": short, "think": thinking})
    # #648: think maps onto the reasoning control the model reports.
    flipped = {**chat, "stream": False, "options": short, "think": not thinking}
    if models[model]["capabilities"].get("reasoning_control"):
        read_json(base, "/api/chat", flipped)
    else:
        expect_error(base, "/api/chat", flipped, 400, "reasoning")
    expect_error(base, "/api/chat", {**chat, "think": "max"}, 400, "think")
    if ort_model:
        expect_error(base, "/api/chat", {**chat, "model": ort_model}, 400, "/v1/chat/completions")
    print("Refused fields and think passed")

    for route, body in (("/api/chat", {"model": model, "messages": []}),
                        ("/api/generate", {"model": model})):
        loaded = read_json(base, route, body)
        assert loaded["done"] is True and loaded["done_reason"] == "load", loaded

    # The first NDJSON line is written once the request owns the permit; a
    # second compute request is busy until the stream ends.
    long_chat = {"model": model, "messages": [
        {"role": "user", "content": "Write 100 numbered sentences about planets."}
    ], "options": {"temperature": 0, "num_predict": 128}}
    with call(base, "/api/chat", long_chat) as reply:
        assert json.loads(reply.readline())["done"] is False
        expect_error(base, "/api/chat", {**chat, "options": short}, 503, "busy")
    read_json(base, "/api/chat", {**chat, "options": short, "stream": False})
    print(f"Ollama load and busy passed with {model}")

    assert models[tool_model]["capabilities"]["tools"] is True, tool_model
    tool = {"type": "function", "function": {
        "name": "get_weather", "description": "Read the current weather for a city.",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}},
                       "required": ["city"]},
    }}
    history = [{"role": "user",
                "content": ("Call get_weather for Rome. Quote the weather condition returned "
                            "by the tool in your answer.")}]
    tool_request = {"model": tool_model, "messages": history, "tools": [tool],
                    "options": {"temperature": 0, "num_predict": 128}}
    chunks = native_stream(base, "/api/chat", tool_request, text=False)
    tool_messages = [c["message"] for c in chunks if c["message"].get("tool_calls")]
    assert len(tool_messages) == 1, chunks
    calls = tool_messages[0]["tool_calls"]
    assert len(calls) == 1 and calls[0]["function"]["name"] == "get_weather", calls
    assert isinstance(calls[0]["function"]["arguments"], dict), calls
    history.extend([tool_messages[0], {"role": "tool", "tool_name": "get_weather",
                                       "content": "Weather condition: purple fog."}])
    # The follow-up omits tools: the replay still renders through the template.
    result = read_json(base, "/api/chat", {
        "model": tool_model, "messages": history, "stream": False,
        "options": {"temperature": 0, "num_predict": 48},
    })
    answer = result["message"]["content"].casefold()
    assert "purple" in answer and "fog" in answer, result
    print(f"Ollama tool proposal and replay passed with {tool_model}")


def wait_until(base, predicate, seconds=15):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        try:
            models = read_json(base, "/api/ps")["models"]
            assert len(models) <= 1, models
            if predicate(models):
                return models
        except HTTPError as exc:
            with exc:
                assert exc.code == 503, exc.code  # a turn holds the session
        time.sleep(0.2)
    raise AssertionError("resident state did not converge")


def validate_residency(base, model):
    """keep_alive, /api/ps and options.num_ctx on a session the API loads.

    A reduced context is a new load, so the session is the API's even when the
    console app had this model resident: only then do leases and unloads apply.
    """
    health = read_json(base, "/health")
    for capability in ("ollama_keep_alive", "chat_context"):
        assert health["capabilities"][capability] is True, capability
    catalogue = read_json(base, "/api/tags")["models"]
    assert any(item["model"] == model for item in catalogue), catalogue
    chat = {"model": model, "messages": [{"role": "user", "content": "Count to three."}]}
    for route, body in (("/api/chat", chat), ("/v1/chat/completions", chat)):
        native = route.startswith("/api/")
        expect_error(base, route, {**body, "options": {"num_ctx": 31}}, 400, "num_ctx", native)
        expect_error(base, route, {**body, "options": {"num_ctx": 10 ** 7}}, 400, "num_ctx",
                     native)
    expect_error(base, "/api/chat", {**chat, "keep_alive": "tomorrow"}, 400, "keep_alive")

    ctx = 384
    resident = lambda items, **fields: (len(items) == 1 and items[0]["model"] == model
                                        and all(items[0].get(k) == v for k, v in fields.items()))
    # Omitted keep_alive: resident with no expiry (unlike Ollama's five minutes).
    loaded = read_json(base, "/api/generate", {"model": model, "options": {"num_ctx": ctx}})
    assert loaded["done_reason"] == "load", loaded
    items = wait_until(base, lambda i: resident(i, context_length=ctx, owner="api"))
    assert "expires_at" not in items[0], items
    # The OpenAI extension reuses the same load, and the reply limit follows it.
    capped = read_json(base, "/v1/chat/completions", {
        **chat, "temperature": 0, "max_completion_tokens": 1, "options": {"num_ctx": ctx}})
    assert capped["choices"][0]["finish_reason"] == "length", capped
    assert capped["usage"]["completion_tokens"] == 1, capped
    wait_until(base, lambda i: resident(i, context_length=ctx, owner="api"))

    # A finite lease expires the API's session after the reply, not before.
    reply = read_json(base, "/api/generate", {
        "model": model, "prompt": "Say hi.", "stream": False, "keep_alive": "3s",
        "options": {"num_ctx": ctx, "num_predict": 1, "temperature": 0}})
    assert reply["done_reason"] == "length" and reply["eval_count"] == 1, reply
    observed = time.monotonic()
    wait_until(base, lambda i: resident(i) and "expires_at" in i[0])
    wait_until(base, lambda i: not i)
    assert time.monotonic() - observed >= 2.5, "the lease expired early"

    # A request without keep_alive ends a running lease.
    read_json(base, "/api/chat", {"model": model, "keep_alive": "30s",
                                  "options": {"num_ctx": ctx}})
    wait_until(base, lambda i: resident(i, owner="api") and "expires_at" in i[0])
    read_json(base, "/v1/chat/completions", {**chat, "max_completion_tokens": 4,
                                             "options": {"num_ctx": ctx}})
    wait_until(base, lambda i: resident(i) and "expires_at" not in i[0])

    # keep_alive 0 unloads the API's session, idempotently.
    for _ in range(2):
        unloaded = read_json(base, "/api/generate", {"model": model, "keep_alive": 0})
        assert unloaded["done_reason"] == "unload", unloaded
        wait_until(base, lambda i: not i)
    print(f"keep_alive, /api/ps and options.num_ctx passed with {model}")


def validate_sdks(base, model):
    import ollama
    import openai

    messages = [{"role": "user", "content": "Name a city in Italy."}]
    oai = openai.OpenAI(base_url=base + "/v1", api_key="local", max_retries=0)
    assert any(item.id == model for item in oai.models.list().data)
    assert oai.chat.completions.create(
        model=model, messages=messages, max_completion_tokens=32).choices[0].message.content
    native = ollama.Client(host=base)
    assert any(item.model == model for item in native.list().models)
    assert native.chat(model=model, messages=messages, options={"num_predict": 32}).message.content
    events = native.chat(model=model, messages=messages, options={"num_predict": 32}, stream=True)
    assert any(event.message.content for event in events)
    assert native.generate(model=model, prompt="Say hi.", options={"num_predict": 16}).response
    for package in ("openai", "ollama"):
        print(f"{package} {importlib.metadata.version(package)} passed")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("url", help="the LAN API base URL, http://<console-ip>:<port>")
    parser.add_argument("model", help="a GGUF chat model id from /v1/models")
    parser.add_argument("--tool-model", required=True, help="a tool-capable GGUF model id")
    parser.add_argument("--ort-model", default="", help="an ONNX model the native routes refuse")
    parser.add_argument("--sdks", action="store_true",
                        help="also run the installed openai and ollama Python packages")
    args = parser.parse_args()
    base = args.url.rstrip("/")
    validate_native(base, args.model, args.tool_model, args.ort_model)
    validate_residency(base, args.model)
    if args.sdks:
        validate_sdks(base, args.model)


if __name__ == "__main__":
    main()
