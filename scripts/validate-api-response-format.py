#!/usr/bin/env python3
# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
"""Check the OpenAI chat format and generation-field contracts against a running LAN endpoint."""

import argparse

import openai


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url", help="console URL including /v1")
    parser.add_argument("model", help="installed chat model ID")
    args = parser.parse_args()
    with openai.OpenAI(
        base_url=args.base_url, api_key="local", timeout=180, max_retries=0
    ) as client:
        request = {
            "model": args.model,
            "messages": [{"role": "user", "content": "Say hello."}],
            "temperature": 0,
            "max_completion_tokens": 32,
        }
        for options in (
            {},
            {"response_format": {"type": "text"}},
            {"extra_body": {"response_format": None}},
            {"n": 1, "logprobs": False, "top_logprobs": 0,
             "presence_penalty": 0, "frequency_penalty": 0},
            {"extra_body": dict.fromkeys(("n", "logprobs", "top_logprobs",
                                          "presence_penalty", "frequency_penalty"))},
        ):
            completion = client.chat.completions.create(**request, **options)
            if not completion.choices or completion.choices[0].message.role != "assistant":
                raise ValueError("expected an assistant completion")
        # The nonexistent model proves validation precedes model lookup.
        request["model"] = "xllama-missing-format-test-model"
        for invalid in (
            {"type": "json_object"},
            {"type": "json_schema", "json_schema": {"name": "result", "schema": {}}},
            {"type": "text", "schema": {}},
            {"type": "unknown"},
            {"type": 0},
            {},
            [],
            "text",
        ):
            # extra_body sends the unsupported cases verbatim.
            try:
                client.chat.completions.create(
                    **request, stream=True, extra_body={"response_format": invalid}
                )
            except openai.BadRequestError as exc:
                if exc.code != "unsupported_response_format" or exc.param != "response_format":
                    raise ValueError(f"unexpected error for {invalid!r}: {exc}") from exc
            else:
                raise ValueError(f"expected HTTP 400 for {invalid!r}")
        for field, value in (
            ("n", 2), ("n", True), ("logprobs", True), ("logprobs", 0),
            ("top_logprobs", 3), ("presence_penalty", -0.5),
            ("frequency_penalty", "0"),
        ):
            for stream in (False, True):
                try:
                    client.chat.completions.create(
                        **request, stream=stream, extra_body={field: value}
                    )
                except openai.BadRequestError as exc:
                    if exc.code != "unsupported_openai_generation_field" or exc.param != field:
                        raise ValueError(f"unexpected error for {field}: {exc}") from exc
                else:
                    raise ValueError(f"expected HTTP 400 for {field}={value!r}")
    print(f"PASS openai-python {openai.__version__}: chat format and generation fields")


if __name__ == "__main__":
    main()
