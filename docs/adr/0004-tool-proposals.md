# ADR 0004: Client-side tool proposals

Status: Accepted (2026-10-13)

Accepted with the architecture plan for the local Enclosure subject.

## Context

The LAN chat route rejected any body that named `tools`, `functions`, or
`tool_choice`, including an empty list and `tool_choice: none`. Enclosure's
subject is a tool-using agent. The client executes the tools. xllama must not:
the console process has no shell, no filesystem tool, and no outbound tool
call of its own.

The hand-rolled `ChatFormat` has no tool role and no schema. The GGUF already
carries a Jinja chat template, and llama.cpp `common/chat` can render that
template and parse the generation.

## Decision

A non-empty `tools` array, or a legacy `functions` array on its own, is a
proposal. Render and parse it through the resident GGUF template
(`common_chat_templates_apply`, `common_chat_parse`). Return `tool_calls`.
Do not execute a tool, a shell, the filesystem, or the network.

One request is one generation. No grammar constraint and no hidden retry.
`tool_choice: none` and `tools: []` stay on the plain `ChatFormat` path.
`functions` is translated into `tools`. Both keys together are 400. A named
`tool_choice` offers only that function and asks the template for a required
call, because the llama.cpp choice enum has no name.

An ONNX session returns 400 `tools require a GGUF chat template`. A GGUF with
no explicit chat template returns 400 `model has no chat template`. A template
that renders without the schema (no offered tool name in the prompt beyond the
transcript's own mentions) returns 400 `model chat template does not render tools`. Do not
fall back to `ChatFormat` with the schema pasted on.

A call cut off by the token cap is omitted: `finish_reason` is `length` and
`content` is the raw text. A complete call still wins as `tool_calls`. If the
parser rejects the output, the status stays 200 and the body is the raw text.
KV reuse stays off. Stops are the template's additional stops plus the
client's `stop`. The template's preserved tokens (LFM2's
`<|tool_call_start|>`, ...) are decoded as text so the parser sees them; every
other special token is still dropped.

The budget trimmer for this path is `fit_tool_transcript`, beside `fit_prompt`,
in `prompt_budget.h`. Both ask the resident tokenizer and reserve
`max(n_predict, 250) + 1`. A proposal never drops the schema or the final user
or tool message, and it drops an assistant message that carries `tool_calls`
together with the tool results that follow it.

agent-runtime stays a `/v1` pass-through. The GUI chat path stays tool-free.

## Consequences

The unsloth Phi-4-mini GGUF template reads the schema from a `tools` field on
the system message, not from `tools`, so phi4-mini returns 400 `model chat
template does not render tools` (console, 1.6.0.1209: before the check it
answered in prose with 200). Enclosure needs a catalogue model whose template
renders tools; [api-endpoint.md](../api-endpoint.md) lists the ones measured.

Without a grammar, nothing stops a model after its call. The LFM2 parser, like
upstream's, accepts nothing after `<|tool_call_end|>` (llama-server relies on its
lazy grammar for that), so `lfm25-1.2b-instruct`, which narrates after a correct
call, comes back as text. That is the cost of the no-grammar decision; the
measured models are in [api-endpoint.md](../api-endpoint.md).

The UWP app links a static `llama-common` (`uwp/llama-common-uwp.vcxproj`,
pre-built by `build-uwp.ps1`), not `llama.cpp/tools/server`.
`validate-api.sh tools` passed on 1.6.0.1210 (Series S, 2026-10-14); the
Enclosure T0 rerun is outside this repository.

`stream: true` stays unimplemented.

## Alternatives

Rejecting every tool key keeps the server from executing tools and also makes
a tool-using client impossible.

Executing tools inside xllama would put a shell on the console.

A hand-rolled Phi tool dialect drifts from the GGUF template the model was
trained with.

A grammar constraint is not a field on `Session::generate` and would be a
second parser beside the template's own.
