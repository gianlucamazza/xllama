#!/usr/bin/env python3
# Copyright (c) 2024 Gianluca Mazza
# SPDX-License-Identifier: MIT
"""Check the chat response_format contract against a running LAN endpoint."""

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
        for options in ({}, {"response_format": {"type": "text"}}):
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
            None,
        ):
            # extra_body sends malformed cases verbatim, including JSON null.
            try:
                client.chat.completions.create(
                    **request, stream=True, extra_body={"response_format": invalid}
                )
            except openai.BadRequestError as exc:
                if exc.code != "unsupported_response_format" or exc.param != "response_format":
                    raise ValueError(f"unexpected error for {invalid!r}: {exc}") from exc
            else:
                raise ValueError(f"expected HTTP 400 for {invalid!r}")
    print(f"PASS openai-python {openai.__version__}: chat response_format")


if __name__ == "__main__":
    main()
