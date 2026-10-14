# ADR 0005: Constrain tool proposals with the template's own grammar

Status: Accepted (2026-10-14)

Accepted by the owner's delegation after the best-practice review below. It
supersedes the "No grammar constraint" sentence of
[ADR 0004](0004-tool-proposals.md); the rest of 0004 stands.

## Context

ADR 0004 generates a tool proposal once, unconstrained, and parses the result
with the GGUF template's parser. On Series S (1.6.0.1210, matrix in
[api-endpoint.md](../api-endpoint.md)) five catalogue models return correct
`tool_calls`, and the failures split in two:

- **Narration after a correct call.** `lfm25-1.2b-instruct` emits
  `<|tool_call_start|>[get_weather(city="Paris")]<|tool_call_end|>` and keeps
  talking. The LFM2 parser is `content + tool_calls + end`
  (`llama.cpp/common/parsers/lfm2.cpp`), so the reply comes back as text.
- **No call markup.** `qwen25-coder-*` answers with bare JSON instead of
  `<tool_call>` tags.

ADR 0004 rejected a grammar as "a second parser beside the template's own".
That premise is wrong: `common_chat_params.grammar` is generated from the same
PEG parser (`parser.build_grammar(builder, grammar_lazy)` in every
`common/parsers/*.cpp`), together with `grammar_lazy` and `grammar_triggers`.

What llama-server does with it, at the pinned `d812350`:

- `tools/server/server-common.cpp` copies `grammar`, `grammar_lazy`,
  `grammar_triggers`, `preserved_tokens` and `additional_stops` from the template
  into the sampling parameters.
- A lazy grammar's root is the trigger rules only
  (`common/peg-parser.cpp:1797-1811`). Before the trigger the grammar is passive
  (`src/llama-grammar.cpp:1362`, `:1407`); once the call rule completes, only
  end-of-generation is accepted. That is why llama-server never shows the
  narration above.
- LFM2 is lazy for `auto` and full for `required`, triggered by
  `<|tool_call_start|>`.
- `common_sampler` samples first, checks only the chosen token against the
  grammar, and does a full-vocabulary grammar pass only on rejection
  (`common/sampling.cpp`).

Elsewhere, vLLM constrains named/`required` choices and leaves `auto`
unconstrained unless `strict`; Ollama and LM Studio parse after generation and
fall back to text; OpenAI constrains only with `strict: true`. Nobody retries.

## Decision

For a tool proposal, apply the grammar the template returns: lazy with its
triggers for `auto`, full for `required` and a named choice. Plain chat stays
unconstrained. Still one generation and no retry. The parser, stops and
preserved tokens of ADR 0004 are unchanged.

The grammar sampler goes into the shared llama.cpp sampler chain
(`src/bridge/sampler_chain.h`) as an optional slot, with check-then-resample
semantics. CLI, bench, GUI and API keep one chain. A plain
`llama_sampler_init_grammar_lazy_patterns` at the head of the chain would run
a full-vocabulary pass on every post-trigger token, so the check-then-resample
step is part of the decision, not an optimisation.

## Consequences

- `lfm25-1.2b-instruct` stops at the end of its call and is expected to return
  `tool_calls`. That must be re-measured with `validate-api.sh tools` and the
  model matrix, not assumed.
- `qwen25-coder-*` returns calls only for `required` or a named choice. The
  pinned autoparser has no wide trigger for bare JSON, so `auto` keeps returning
  text.
- `phi4-mini` stays `400` (template does not render tools). A vetted template
  override would be a separate decision (a catalogue field pointing at a
  `.jinja`, through the host smoke → console → manifest ladder).
- Cost: decode tok/s with tools on Series S, measured before and after; the
  number goes in `bench/results/`, not in this ADR. Until the trigger the cost
  is a regex over the decoded text per token.
- `"Cannot specify grammar with tools"` stays upstream's rule: xllama exposes no
  client grammar.

## Alternatives

- **Keep no grammar** (ADR 0004 as is): no sampler change, and narrating or
  untagged models keep returning text.
- **Stop at the end marker** (`<|tool_call_end|>` as a stop): an approximation of
  the grammar's end that also cuts a reply mentioning the token and cannot
  express parallel calls. Rejected as a workaround.
- **Lenient trailing-text parsing**: diverges from upstream's parser. Rejected.
- **Retry on parse failure**: doubles latency on a 4-core console and hides the
  failure. Rejected, as in 0004.
