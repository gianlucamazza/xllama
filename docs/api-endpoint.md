# LAN HTTP endpoint (OpenAI-compatible)

**Status:** v1 — spike + chat (streaming via SSE on request) + UI parity surface (#118: preferences,
training status, images). Opt-in, **default OFF**. Developer Mode / LAN research only.
Not for public inbound.

xllama can expose its inference core (`xllama::Session`) as an HTTP endpoint on the local
network, so a PC on the same LAN can run inference on the console:
`http://<ip-xbox>:<port>/v1/chat/completions`. It is another front-end on the
same `xllama::Session` abstraction used by the chat UI, bench and CLI. Since
PR #161/#164 the Session lives in `xllama::session_hub()` — ONE process-wide
resident model **shared with the chat UI** ("never 2× model in RAM" holds
process-wide). Code lives in
`uwp/api-server.{h,cpp}` (WinRT `StreamSocketListener`), entirely under `#ifdef
XLLAMA_UWP`; the Linux/CLI build is unaffected.

## Release compatibility

For successful requests to `POST /v1/chat/completions`, choose the response
parser from the HTTP Content-Type. With streaming disabled, both versions
return one JSON completion for either request type.

| Request with `stream: true` | v1.7.0.0 (MSIX 1.7.0.1495)           | v1.8.0.0                                                   |
| --------------------------- | ------------------------------------ | ---------------------------------------------------------- |
| Plain chat                  | SSE as token pieces become available | SSE as token pieces become available                       |
| Tool proposal               | One JSON completion after generation | SSE after the complete reply has been generated and parsed |

On v1.7.0.0, an ONNX tool proposal or an oversized tool transcript
returns HTTP 400 with `error.type` and `error.message`. Clients reading that
package must treat `error.code` and `error.param` as optional. In v1.8.0.0,
the ONNX case carries `tools_unsupported_for_model` and parameter `model`;
a tool-transcript overflow carries `context_length_exceeded`, with the
parameter present only when the server can attribute the overflow.

[v1.7.0.0 API contract](https://github.com/gianlucamazza/xllama/blob/v1.7.0.0/docs/api-endpoint.md) ·
[v1.8.0 release notes](releases/v1.8.0.md)

## Enabling and controlling it

The normal path is **Settings → LAN API**: choose a port and toggle the endpoint
on or off. The listener starts, stops, or rebinds immediately without an app
restart, and the dialog shows its current state and base URL.

While the endpoint is on, the main screen shows a **LAN API panel** (header,
right): base URL `http://<console IP>:<port>/v1`, the in-flight request
(endpoint, model, elapsed), counters (requests, `busy` rejections), the last
request's time and tokens, and the last failed request for 10 minutes. The
counters live in `xllama::api::activity()` (`include/xllama/api_activity.h`),
recorded for chat, embeddings and image requests. A pad turn that finds the
slot taken shows "Waiting for LAN API request to finish..." instead of
stalling silently.

For operator automation, `api.flag` and `api-port.txt` in `LocalState` remain the
persistent contract. `api.flag` is not consumed: `App::OnLaunched` applies it,
and the Settings toggle writes or removes it. The listener coexists with the
live XAML chat UI.

- **Port:** `LocalState\api-port.txt` if present, else **11434** (Ollama's default port).
  Xbox silently drops traffic for ports in **[57344, 65535]** and reserves **11443** for the
  Device Portal, so the server only honors an override in the bindable app range
  **[1025, 49151]** (and not 11443); anything else, including text that is not a whole
  number, falls back to 11434 with a log line. Settings shows that same effective port.
- **Model:** taken from the request's `"model"` field; if absent, falls back to
  `LocalState\model.txt`. The resident Session may already exist before any
  request — the GUI pre-loads it when a model becomes Ready — and it is
  swapped when a request names a different model. A swap invalidates the
  GUI's KV-reuse state (`hub.generation`): its next turn silently falls back
  to a full prefill. A model id that is neither resident nor a loadable
  `LocalState\models\<id>` directory gets **`404`** "model '<id>' is not on this
  console; see GET /v1/models" (chat and embeddings), before any load is
  attempted. A GGUF model (or `options.num_ctx`) whose weights, context and
  compute buffers do not fit the app's memory limit gets **`400`**
  "session create failed: insufficient memory: …" with the breakdown, before
  any weight is read ([architecture](architecture.md#session-memory-admission)).
- **GGUF GPU layers:** a GGUF session gets the catalogue entry's `gpu_layers`
  (D3 default per model). `LocalState\gguf_gpu_layers.txt`, when present,
  overrides it for the GUI and this API alike; `0` forces the CPU. A change
  reloads the resident session. Gate and verdict:
  [gguf-gpu-decode.md](gguf-gpu-decode.md).
- **GGUF threads:** `LocalState\gguf_n_threads.txt`, when present and
  positive, sets llama.cpp thread count for this API and the GUI. Absent, 0,
  or an invalid value keeps auto-detect. Threads are a tuning knob, not load
  identity, so a change applies on the next session create.

**Foreground only.** The endpoint dies when the app leaves the foreground (UWP Process
Lifetime Management — no always-on system service). On Xbox this is stricter than on PC: if
the package is game-classified (Device Portal _"Treat UWP apps as games by default"_) it is
**suspended and terminated** in the background, and `ExtendedExecutionSession` does not keep
a listener alive. Keep the app foregrounded on the console while using the endpoint. Accepted:
this is Developer Mode research, not a hosted service.

> Verified against learn.microsoft.com (2024-2026): `privateNetworkClientServer` +
> `StreamSocketListener` is the documented path for LAN inbound; network isolation blocks
> only _same-host_ sockets, so no `CheckNetIsolation` loopback exemption is needed for
> requests coming **from another LAN device** (only for a localhost client on the console).

### Open WebUI

Use an **OpenAI API** connection. The Ollama connection type is not supported:
xllama has no `/api/version` or `/api/chat`.

1. On the console (Developer Mode), open xllama, enable **Settings → LAN API**,
   and keep the app in the foreground. The LAN API panel shows the base URL
   `http://<console IP>:<port>/v1` (port **11434** by default).
2. In Open WebUI, go to **Admin Settings → Connections → OpenAI API** and add a
   connection. URL `http://<console IP>:<port>/v1`, API key empty: the endpoint
   has no authentication, so use a trusted LAN only. If Open WebUI runs in
   Docker, use the console's LAN address, not localhost.
3. Select a tool-capable GGUF such as `lfm25-350m` for native function
   calling. For plain chat on ONNX, edit the model under **Admin → Models →
   Capabilities** and turn off **Builtin Tools**. Remove any tools attached to
   the chat as well.
4. If a GGUF runs out of context, enable fewer tool categories or try Open
   WebUI's experimental [Tool Search](https://docs.openwebui.com/features/extensibility/plugin/tools/#tool-search).
   It lets the client load larger tool definitions on demand in Native mode.
   The server validates the token budget for each request with the resident
   tokenizer.

**Troubleshooting**

- `tools require a GGUF chat template` (`error.code`
  `tools_unsupported_for_model` on ONNX) means the selected model cannot
  handle this tool proposal. Select a tool-capable GGUF or follow step 3 for
  plain chat.
- `prompt too long: the request needs …` (`error.code`
  `context_length_exceeded`) means the rendered prompt and reply reserve
  exceed the context. With `error.param` set to `tools`, reduce the tool
  inventory. With `messages`, shorten the messages.
- Model list empty or a connection error: `curl http://<console IP>:<port>/health`
  from the Open WebUI machine, check xllama is in the foreground, and that both
  are on the same network.
- `busy` (503): another request or console chat turn is running. Retry when it
  ends.
- `out of memory` (503): the request ran the console out of memory. Use a
  smaller model, context or batch. `internal error` (500): the server failed.
  `xllama.log` (`./scripts/deploy.sh get-log`) holds an `[xllama] api: <route>
failed` line with the cause. `malformed request body` (400) means only that a
  field had the wrong type or was missing.
- Ollama connection "verify" fails with "no such endpoint": use the OpenAI API
  connection type.

## Protocol (v1)

| Method    | Path                     | Behaviour                                                                                                                                                                           |
| --------- | ------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `GET`     | `/` or `/health`         | `200 {"status":"ok","service":"xllama",…}` plus build `version`, contract `api` and route `capabilities` ([Discovery](#discovery-health-and-v1models)).                             |
| `GET`     | `/v1/models`             | OpenAI model discovery: every servable on-device model, with `active`, `kind`, `role`, `n_ctx` and `capabilities` ([Discovery](#discovery-health-and-v1models)).                    |
| `GET`     | `/api/tags`              | Ollama model discovery — same list, Ollama shape.                                                                                                                                   |
| `POST`    | `/api/pull`              | Download a trusted chat or embedding model from the built-in model list (Ollama-style NDJSON progress), then load it.                                                               |
| `POST`    | `/v1/chat/completions`   | OpenAI-compatible chat completion; one JSON body, or **SSE** chunk events when `stream: true`.                                                                                      |
| `POST`    | `/api/embed`             | Ollama-style embeddings for one string or a batch, catalogue embedding GGUF/llama.cpp only.                                                                                         |
| `POST`    | `/api/embeddings`        | Deprecated Ollama single-prompt embeddings adapter.                                                                                                                                 |
| `POST`    | `/v1/embeddings`         | OpenAI embeddings shape; `encoding_format` accepts `float` or `base64`.                                                                                                             |
| `POST`    | `/v1/preferences`        | Append a preference sample (`label` + `messages[]`) to `training/samples.jsonl` — same contract as the UI rate op (#118).                                                           |
| `GET`     | `/v1/training/status`    | `result.done` / `progress.json` / last personalized `result.json` + usable sample count (#118).                                                                                     |
| `POST`    | `/v1/images/generations` | SD-Turbo image gen (`prompt`, `steps` 1–4, `seed`); returns OpenAI-ish `{data:[{b64_json,path}]}` (#118). Shares process-wide compute admission with chat, embeddings and training. |
| `OPTIONS` | _any_                    | CORS preflight (`204` + `Allow-Methods/Headers`) for browser clients.                                                                                                               |

### Discovery (`/health` and `/v1/models`)

`GET /health` keeps its original two fields first and adds what a client needs
to branch on (`include/xllama/api_discovery.h`):

```json
{
  "status": "ok",
  "service": "xllama",
  "version": "1.8.1.1805",
  "api": 1,
  "capabilities": {
    "tools": true,
    "tools_stream": true,
    "embeddings": true,
    "images": true,
    "pull": true,
    "preferences": true,
    "training_status": true,
    "reasoning_content": true,
    "reasoning_tokens": true,
    "cached_tokens": true
  },
  "sampling": ["temperature", "top_p", "top_k", "repetition_penalty", "seed"],
  "runtime": {
    "llama_cpp": "d812350",
    "patches": [
      "0001-uwp-appcontainer-guards",
      "0003-kv-cache-k-shift-unallocated-input",
      "0004-recurrent-state-fill-zero",
      "0005-required-tool-call-no-content-prefix"
    ]
  }
}
```

`version` is the package identity (the CI build stamps the fourth part). `api`
is the contract version; it increases when a documented response shape or
status mapping changes in a way a client must branch on. `capabilities` lists
what the installed build serves: the llama.cpp-only package reports
`images: false`. `sampling` lists the sampling fields a chat request may set
(`include/xllama/api_sampling.h`, the table the request is validated against),
so a client sends only those instead of assuming them per version. `runtime`
names the llama.cpp commit the build compiles (llama.cpp's own build info; the
commit only, since its build number counts the submodule history and reads 1 in
a shallow checkout) and the patches applied to it
(`include/xllama/llama_patches.h`, kept equal to `patches/` by a test), so a
client can tell whether an upstream fix is in the build it talks to. It is
absent from a build without llama.cpp.

Each `/v1/models` entry keeps OpenAI's `id`, `object`, `created` and `owned_by`
and adds non-standard fields that clients may ignore:

| Field          | Meaning                                                                                                                                                                 |
| -------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `active`       | `true` on the model the resident session holds (absent otherwise)                                                                                                       |
| `display`      | Catalogue display name (absent for a model outside the catalogue)                                                                                                       |
| `kind`         | `gguf`, `ort-genai` or `diffusion`                                                                                                                                      |
| `role`         | Catalogue role: `""`, `coding` or `embedding`                                                                                                                           |
| `n_ctx`        | Context the session uses: the resident session's for the active model, else what a session of that entry opens with (`catalogue_n_ctx`: 512 floor, ONNX capped at 8192) |
| `capabilities` | `tools` (the GGUF chat template renders a tool schema, the same check a tool proposal makes), `reasoning`, `embedding`                                                  |

`tools` is read from the GGUF header's chat template without loading weights,
and cached per file, size and write time.

### Model download (`POST /api/pull`)

The request is `{"model":"<catalogue-id-or-alias>","stream":true}`; `stream` defaults to
true. Pulls use only HTTPS entries bundled in the installed package with valid SHA-256 pins.
LocalState catalogue overrides, diffusion assets, arbitrary URLs, and unknown models are
rejected. Ollama embedding aliases (`bge-m3`, `nomic-embed-text-v2-moe`,
`qwen3-embedding:4b`) resolve to their catalogue IDs when those models are included.

Streaming returns chunked `application/x-ndjson`: status records report download bytes, then
model loading, followed by `{"status":"success"}`. A failure after streaming starts is an
NDJSON `{"error":"..."}` record. When the files to fetch do not fit on the LocalState
volume, that record is `Not enough storage for this model: needs N MB, M MB available`
and no bytes are downloaded. With `stream:false`, the endpoint waits and returns one
JSON result. The API pull gate spans transfer and model loading. A simultaneous
API pull receives HTTP 409 when its handler overlaps; the WinRT listener may
otherwise dispatch the handlers sequentially. GUI downloads, USB import and explicit
rollback share one downloader-wide writer permit. A competing writer is rejected
before any file or marker changes: non-streaming pulls return HTTP 409, streaming
pulls emit an error record, and the GUI reports a download-busy error. The permit
is released before completion callbacks that may chain provisioning. Existing
inference may finish during the download. After verification, the pull must
acquire process-wide compute admission before replacing/loading the resident
model. If another compute job is still active, the download may already be
complete when the API returns HTTP 503 (or a streaming error record); retry after
the active job finishes.

```bash
curl -N http://<ip-xbox>:11434/api/pull \
  -H 'Content-Type: application/json' \
  -d '{"model":"lfm25-350m"}'

curl -s http://<ip-xbox>:11434/api/pull \
  -H 'Content-Type: application/json' \
  -d '{"model":"bge-m3","stream":false}'
```

Discovery semantics: a model is **servable** when its `LocalState\models\<id>`
directory holds a base GGUF (any `*.gguf` except a bare runtime-LoRA
`adapter.gguf`) or an ORT GenAI layout (`genai_config.json` / `model.onnx`) —
predicate `model_dir_files_ready` in `include/xllama/model_provision.h`
(host-tested). Any listed `id` is valid as the request `model`: the hub swaps
the process-wide resident Session when the requested model differs from the
loaded one (first request on a new model pays the load, and the GUI's KV-reuse
state is invalidated — see Model above).

Request body (subset): `model`, `messages[]`, optional `max_completion_tokens` /
`max_tokens` (default 512; the former wins — `max_tokens` is the deprecated OpenAI
alias), the sampling fields `temperature`, `top_p`, `top_k`, `repetition_penalty`
(or llama.cpp's spelling `repeat_penalty`, not both) and `seed`, and `stop` (string
or array). Numbers are checked before the model loads, and a bad one is `400`
naming the field: the token count is a whole number from 1 to the catalogue `n_ctx`
minus 1, `temperature` is 0–2, `top_p` is 0–1, `top_k` is a whole number ≥ 0 (0
keeps every token), `repetition_penalty` is 0–2 (1 is no penalty, 0 drops the
stage), and `seed` is a whole number (a negative seed keeps the default,
llama.cpp's `-1`). An absent field keeps the default in `include/xllama/sampling.h`;
on ORT GenAI models the same values go to its search options (`ort_sampling.h`),
whose repetition penalty covers the whole sequence rather than llama.cpp's last 64
tokens. `stream: true` answers with SSE; see the streaming
paragraph under [Protocol (v1)](#protocol-v1). Most other fields are ignored.

Plain chat (no tool schema, `tools: []`, or `tool_choice` `"none"`) keeps
`ChatFormat::render_prompt` (`include/xllama/chat_prompt.h`). Roles are system, user,
and assistant, content is a string, and the transcript must end on a user turn, else
`400`. `chat_format_for(model)` selects the template and its stops. Client `stop`
values are added to those. When no system message is sent, a default one is injected.

A non-empty `tools` array, or a legacy `functions` array on its own, is a proposal.
`functions` is translated into `tools`. Sending both is `400`
(`tools and functions cannot be combined`). `tool_choice` accepts `"auto"`,
`"none"`, `"required"`, and `{"type":"function","function":{"name":"..."}}`.
`"none"` stays on the plain path. A named choice offers only that function.
`"required"` and a named choice constrain the reply to start with the call (reasoning
first, for a thinking model): no text can come before it (`patches/0005`, #563). A reply
that reaches `max_tokens` before the call completes is `finish_reason` `length`.
Content may be a string, null, or an array of `{type:"text",text}` parts joined
with no separator. Anything else, including an image part, is `400`. A `tool`
message requires `tool_call_id`. The last message must be `user` or `tool`.

xllama does not execute the tool, a shell, the filesystem, or the network. The
client does. One request is one generation, constrained by the template's grammar
(ADR 0005), with no hidden retry. The prompt and the parse go through the resident GGUF Jinja template
(`common_chat_templates_apply` / `common_chat_parse`). A proposal for an ONNX
model is rejected before the resident session is loaded or replaced. The API returns
`400` with `error.code` `tools_unsupported_for_model`, `error.param` `model`,
and a message beginning with `tools require a GGUF chat template`. A GGUF with no explicit chat template
returns `400` `model has no chat template`, and one that renders without the
schema returns `400` `model chat template does not render tools` (phi4-mini). That check compares the prompt rendered with and without `tools`, so replaying a call in the transcript does not trigger it. A template that raises on such a transcript (`gemma3-*`) returns `400` `model chat template failed: ...` instead. The server does not fall back to
`ChatFormat` with the schema pasted on. KV reuse stays off. Stops are the
template's additional stops plus the client's `stop`. The template's preserved tokens (LFM2's
`<|tool_call_start|>`, ...) are decoded as text so the parser sees them; every
other special token is still dropped.

xllama routes tool proposals through the GGUF template and parser. The
[catalogue's three SmolLM2 ONNX packages](../uwp/models/manifest.json) ship
without a declared chat template in the [release assets](https://github.com/gianlucamazza/xllama/releases/tag/models-v1). The original
[SmolLM2-1.7B-Instruct model](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct#function-calling)
documents function calling with a dedicated prompt and parser. The catalogue's
ONNX INT4 export remains unqualified for tool calling on Xbox.
The [ONNX Runtime GenAI C API documents guidance](https://onnxruntime.ai/docs/genai/api/c.html#ogageneratorparamssetguidance).
xllama's ONNX chat path uses plain chat formatting and its regular sampler.

`choices[0].message.tool_calls[]` is `{id, type:"function", function:{name, arguments}}`
with `arguments` a JSON string. Missing call ids become `call_1`, `call_2`, … in
parse order, counting only the empty ones. `content` is null only when there are
tool calls and no text. `finish_reason` is `tool_calls` when at least one call is
complete, even if generation hit the token cap. A call cut off by the cap is
omitted: `finish_reason` is `length` and `content` is the raw text. If the parser
rejects the output, the status is still 200 and `content` is the raw text with
`finish_reason` `stop` or `length`. `logprobs` stays null. Decision:
[ADR 0004](adr/0004-tool-proposals.md).

Measured on 1.6.0.1231 with the grammar (Series S, `temperature` 0, `max_tokens` 128,
one `get_weather` schema, named and `auto` choice; raw rows in
[api-tools-matrix-1231.json](audit/excellence/evidence/api-tools-matrix-1231.json),
before the grammar in
[api-tools-matrix-1210.json](audit/excellence/evidence/api-tools-matrix-1210.json)):

| Model                                                                                                             | Result                                                                                                                                                                                   |
| ----------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lfm25-230m`, `lfm25-350m`, `lfm25-1.2b-instruct`, `lfm2-2.6b`, `llama32-3b`, `minicpm5-1b`                       | `tool_calls` `get_weather({"city":"Paris"})` for both choices                                                                                                                            |
| `qwen25-coder-0.5b`; `qwen25-coder-3b` (benchmark; model under a non-commercial research licence; not catalogued) | Bare JSON instead of `<tool_call>` tags. The template's forced grammar allows text before the call, and the model never reaches the tag (still none at 512 tokens), so the reply is text |
| `lfm25-1.2b-thinking`                                                                                             | Still reasoning at the 128-token cap (`length`); needs a larger `max_tokens`                                                                                                             |
| `gemma3-270m`, `gemma3-1b`, `phi4-mini`                                                                           | `400` `model chat template does not render tools`                                                                                                                                        |

Generation follows the template's own grammar ([ADR 0005](adr/0005-tool-proposal-grammar.md)):
lazy for `auto` (free text until the template's call marker, then only the call
and end-of-generation), full for `required` or a named function. On the ten
requests whose call was identical with and without it, total latency moved from
25.8 s to 25.9 s (+0.4%, within run-to-run noise).

**Context budget.** Plain chat is fitted with `xllama::fit_prompt`
(`include/xllama/prompt_budget.h`) and the model's own tokenizer: the **oldest**
turns are dropped until the prompt plus the requested reply
(`max(n_predict, 250) + 1`) fits, and the trailing user message is never dropped.
A LAN tool proposal uses `fit_tool_template` in the same header, the same tokenizer,
and the same reserve. It never drops the tool schema or the final user or tool
message, and it drops an assistant message that carries `tool_calls` together
with the tool results that follow it. The selected render supplies the prompt,
grammar, stop sequences and preserved tokens together. Trimming can skip a
suffix that the template rejects. Dropped messages are
logged. When the protected tail still does not fit, the request is `400`. Plain chat says
the final user message needs the three numbers. A proposal keeps
`prompt too long: the request needs X tokens plus Y for the reply, but n_ctx is Z`
(Y is `n_predict`) and reports the effective reply reserve. On overflow, the server can also report the token count of
the retained transcript rendered without tools and how many tokens the tool
definitions add. The body has `error.type` `xllama_error` and `error.code`
`context_length_exceeded`. With both measurements available, `error.param`
is `tools` if only the added tokens exceed the input budget, or `messages` if
only the no-tools prompt exceeds it. The parameter is omitted when neither or
both components independently exceed the budget, the reply reserve exceeds
the context, or the baseline is unavailable.

Framing is exact ([ADR 0006](adr/0006-http-framing.md)). `Content-Length` must
be one plain decimal (repeats must agree), else `400`; any `Transfer-Encoding`
gets `411` (OpenAI SDKs send Content-Length). A body over 8 MiB is `413`, a head
over 64 KiB is `431`, and a body shorter than declared is `400` with nothing
dispatched. The head must arrive within 10 s and the body within 15 s, else `408`. Malformed JSON / wrong-typed fields get `400`, never
a dropped connection.

The reply follows the OpenAI shape: `id` (unique `chatcmpl-…`), `object: chat.completion`,
`created`, `model`, `choices[0]` (`message`, `finish_reason` — `length` only when the token
cap is hit, else `stop`; and `logprobs: null`, omitting which breaks openai-python/LangChain
Pydantic validation), and a `usage` block from `InferenceResult`: `prompt_tokens` is the prompt's length
(`n_prompt`), not the tokens prefilled after a KV prefix reuse (`n_p_eval`), and
`completion_tokens` is `n_eval`. `prompt_tokens_details.cached_tokens` (#595)
is the part of `prompt_tokens` the session did not prefill because its resident
KV already held it (#170a prefix reuse): `n_prompt - n_p_eval`, llama-server's
`timings.cache_n`. A GGUF session reuses a matching prefix on every path, tool
proposals included; an ONNX session prefills the whole prompt, so it is `0`.
`completion_tokens_details.reasoning_tokens`
(#585) is the part of `completion_tokens` generated inside the reasoning, tags
included: a token counts when any of its text lies in a block the splitter
below returns as `reasoning_content`, or in the reasoning a tool template's
parser extracted; a token that renders no text is not counted. It is always
present, `0` for a model that does not reason (OpenAI's shape). A `length` finish with `reasoning_tokens == completion_tokens`
ran out of tokens while reasoning; with fewer, the answer itself was cut off.

**Prefix reuse for clients.** Every request is a full prompt; the session keeps
the longest token prefix it shares with its resident KV and prefills the rest.
A client that sends the history unchanged (append-only: the previous request,
the assistant reply, the new turn) pays only for the new turn and the reply's
re-rendered tail. Text the client does not echo byte for byte costs one
re-prefill from that point: a `<think>` block stripped from the reply, tool
calls the template re-renders differently, or an edited earlier turn.
`cached_tokens` shows what was reused.

`stream: true` answers with Server-Sent Events instead of one JSON body: one
`chat.completion.chunk` per token piece (`choices[0].delta.content`), a final chunk carrying
`finish_reason` and `usage`, then `data: [DONE]`. The response is `Content-Type:
text/event-stream` over chunked transfer (no `Content-Length`) and the socket closes at the
end. Plain chat emits raw token pieces using `GenerateParams::on_token`. The writer
holds back any suffix that could still complete a stop sequence and drops the match.
A failure after the head travels as an `{"error":{...}}` event.

A stream that has written nothing for 5 s (`kSseKeepaliveInterval`), as during a
long prefill, gets an SSE comment line, `: keepalive` (#581). Clients skip
comments. The write is how the server notices a client that left: a failed
write sets the request's abort flag, which ends the prefill inside its batch
(#580) and releases the session and the compute permit. With TCP the first
write after the peer closed usually still succeeds, so a disconnect is seen
within about two intervals. `validate-api.sh disconnect` measures it, for plain
chat and for a streamed tool proposal.

**Reasoning (#564).** For a thinking model (`capabilities.reasoning` in
`/v1/models`), the `<think>…</think>` block is returned apart from the answer,
as llama-server's `--reasoning-format deepseek`, vLLM's reasoning parsers and
the DeepSeek API do:

- JSON replies carry it in `message.reasoning_content`, absent when the model
  wrote none.
- SSE carries it in `delta.reasoning_content` chunks, with `delta.content` for
  the answer.
- A tool proposal's message carries it in both shapes, beside `tool_calls`.

`content` never contains the tags. One splitter (`thinking_split.h`) decides
for every path and for the GUI, which shows the content and keeps the reasoning
out of the chat history. A reply cut off mid-thought has an empty `content`
and the reasoning so far. Other models' text is returned unchanged, including
a literal `<think>`. Whether to keep `reasoning_content` in the next request's
history is the client's choice; xllama renders past assistant turns through the
template as sent.

A tool proposal with `stream: true` opens the stream once the request checks
pass, before generation. The SSE head and the assistant role go out first, and
the keepalive above covers the prefill (#594). It then buffers generation,
applies the GGUF template parser, and sends a delta with the assembled content
and any indexed `delta.tool_calls[]`, the finish chunk with usage, and
`data: [DONE]`. The same raw-text fallback as the JSON path applies when parsing
fails or a tool call is cut off. Request errors are found before the head and
keep the normal HTTP JSON error response:

- bad JSON;
- an unknown model;
- `tools` + `functions`;
- a template that fails to render;
- a prompt too long.

A failure in generation or parsing comes after the head, so it travels as an
`{"error":{...}}` event, as on plain chat. Before 1.10.0 it was an HTTP `500`.

Measured for plain chat on
Series S (2026-10-16, `lfm25-1.2b-instruct`, 52-token reply): first byte at 11 ms against
1.90 s for the same completion without `stream`, both 1.90 s in total.

### Embeddings (`/api/embed`, `/api/embeddings`, `/v1/embeddings`)

Embedding requests use the same resident `Session` and single-slot mutex as chat. Inputs in
a batch are encoded serially, preserving order and avoiding a second model or a large
multi-sequence activation allocation. The supported inference path is GGUF via llama.cpp;
ORT GenAI sessions and ranking-only models return an explicit unsupported error. Use an
embedding model already present in `LocalState\models\<id>` or download one from the
built-in model list with `/api/pull`.

`/api/embed` accepts `input` as a string or string array (up to 128 items), optional `truncate` (default true),
`dimensions`, and an Ollama `options` object. `options.num_ctx` can lower the configured
context and may recreate the resident session when it differs from the active context; it
cannot exceed xllama's configured model limit. Omitting it restores the catalogue context,
even after a previous request lowered that context. Other sampling options have no
meaning for embeddings and are ignored. The response follows Ollama's model/embeddings and
duration/token-count fields. `/api/embeddings` adapts the legacy `prompt` string to an
`embedding` response. `/v1/embeddings` accepts the same scalar or batch `input` and returns
OpenAI `data[]`, indices, and token usage; `base64` serializes little-endian float32 values.
All returned vectors are L2-normalized after optional dimension truncation. Context overflow
with `truncate:false`, empty inputs, impossible dimensions, and unsupported backend/model
combinations return errors.

```bash
curl -s http://<ip-xbox>:11434/api/embed \
  -H 'Content-Type: application/json' \
  -d '{"model":"embed-bge-m3","input":["first document","second document"]}'

curl -s http://<ip-xbox>:11434/v1/embeddings \
  -H 'Content-Type: application/json' \
  -d '{"model":"embed-nomic-v2-moe","input":"search_query: find relevant documents","encoding_format":"base64"}'
```

**Available embedding models** (catalogue ids, LAN API only):

- **embed-bge-m3**: BAAI/bge-m3 (Q8_0, 1024 dim, 8192 ctx). Dense retrieval, no required prefix. Not Matryoshka — `dimensions` must be 0 or 1024.
- **embed-nomic-v2-moe**: nomic-ai/nomic-embed-text-v2-moe (Q8_0, 768 dim, 512 ctx). Client must prepend `search_query:` or `search_document:` to inputs.

Qwen3-Embedding-4B Q4_K_M is not in the catalogue: its Release embedding smoke peaked at 4411 MiB, above the 3584 MiB host gate. The listed models pass host Release smoke and the Series S contract check.
BGE-M3 (`embed-bge-m3`) and Nomic Embed Text v2 MoE (`embed-nomic-v2-moe`)
passed the Series S contract check only; throughput and retrieval quality are
not yet measured.
[Catalogue status](model-matrix.md#g-embedding-api-catalogue-host-and-series-s-validated) owns
the current validation evidence.

**Ollama name aliases**: The API also accepts `bge-m3` and `nomic-embed-text-v2-moe`. The `qwen3-embedding:4b` alias is reserved for a future catalogue entry.

**Model swap**: Embedding requests share the single-slot `session_hub()` with chat. An embedding call that names a different model swaps the resident chat model out. The next chat turn will prefill cleanly (KV is cleared on swap).

**Dimensions**: `dimensions=0` (the default) returns the model's native width. The catalogue BGE-M3 API accepts only 0 or 1024; other widths return 400. Nomic v2 MoE accepts reduced widths from 1 through 768 and re-normalizes the retained prefix. This API width policy is distinct from the generic SDK normalization helper.

**Context limits**: Each model opens at its catalogue `n_ctx` (BGE-M3: 8192, Nomic MoE: 512). The `options.num_ctx` field cannot exceed that limit or go below 32. The effective input limit is the smaller of the context and the logical embedding batch
(default 2048 tokens): **BGE-M3 accepts at most 2048 tokens**, including tokenizer
special tokens, despite its 8192-token catalogue context; Nomic accepts at most 512.
A smaller `options.num_ctx` reduces this limit; an omitted value restores the
catalogue context after an earlier small-context request. Non-causal sequences
run in one physical microbatch. Inputs beyond
that limit are truncated when `truncate=true` (default), or rejected with 400 when
`truncate=false`.

Embedding models retain their own pooling behavior from GGUF metadata. The API does not inject task prefixes automatically: clients must prepend the appropriate instruction strings themselves. Models can be downloaded only from the built-in model list; `/api/show` and full Ollama model-management parity are not implemented.

### Preferences (`POST /v1/preferences`)

Same validation as the UI **rate** op. Appends one JSONL line to
`LocalState\training\samples.jsonl`.

```bash
curl -s http://<ip-xbox>:11434/v1/preferences \
  -H 'Content-Type: application/json' \
  -d '{"label":"like","messages":[
        {"role":"user","content":"hi"},
        {"role":"assistant","content":"hello"}
      ]}'
# → {"ok":true,"label":"like","path":"training/samples.jsonl"}
```

`label` ∈ `like|dislike|correction|implicit`. For `correction`, optional
`preferred_assistant` string. Invalid label or empty messages → `400`.

### Training status (`GET /v1/training/status`)

Read-only snapshot of on-device train progress (headless or in-app):

```bash
curl -s http://<ip-xbox>:11434/v1/training/status
# → {"state":"idle"|"running"|"ok"|"fail","usable_samples":N, ...}
```

Optional fields: `result_done`, `progress` (object from
`training/progress.json`), `result` (parsed
`training/out/personalized/result.json` when present). Does **not** start a
job — use Settings / autopilot `start_train` for that.

### Images (`POST /v1/images/generations`)

In-process SD-Turbo with the same contract as the Image dialog
(`include/xllama/diffusion/request.h`): `steps` clamped to 1–4, `seed` any
uint32 (above it clamps; a negative seed keeps 42). The same seed reproduces the
same image from the dialog, the API and autopilot.
Shares process-wide compute admission with chat → `503` when busy.

```bash
curl -s http://<ip-xbox>:11434/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a red sports car on a mountain road","steps":1,"seed":42}'
# → {"created":…,"data":[{"b64_json":"…","path":"diffuse-out.png"}]}
```

Requires `sd-turbo-fp16` provisioned (auto-download on first UI Generate still
applies when the user opens Image; the API does not download the model for you).

## Concurrency

`Session::generate()` is single-slot / non-concurrent. `ConnectionReceived`
only queues a bounded thread-pool work item (`kApiMaxInFlightConnections` in
[api_connection.h](../include/xllama/api_connection.h)); serving the socket
inline would hide the permit from every later LAN request until the current
reply ends. Compute requests first acquire the process-wide RAII permit in
[ADR 0008](adr/0008-process-compute-admission.md), then acquire
`session_hub().mtx` with `try_lock`: a request arriving while another request,
GUI turn, image generation, or device training is active gets **HTTP 503**
`{"error":{"message":"busy"}}` for OpenAI-compatible routes and
`{"error":"busy"}` for native Ollama embedding routes. The slot covers the
whole process, including a streamed `/v1/chat/completions` reply (the permit
stays held until the last SSE event, and is released if the client
disconnects mid-stream). Native `/api/embed`, `/api/embeddings`,
and `/api/pull` errors always expose a non-empty string in `error`;
`/v1/embeddings` exposes an error object with `message` and `type`. One
exception: while the session **pre-load** is active (right after a model becomes
Ready), API compute admission waits — bounded, ≤15 s, only while
`hub.preloading` is set — instead of bouncing the client's very first request.
Stopping the endpoint closes the listener only; the Session is hub-owned and is
NOT released (the chat UI may be using it).

From a LAN host (console address is `XBOX_IP` / `API_PORT`, never hardcoded):

```bash
source ~/.config/xllama/xbox-env
# (b) streamed reply holds the permit: second request is 503, stream ends [DONE]
./scripts/validate-api.sh admission
# (a) while a GUI turn is generating, the same URL must be 503:
curl -sS -o /tmp/xllama-gui-busy.json -w '%{http_code}\n' \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL:-lfm25-350m}\",\"messages\":[{\"role\":\"user\",\"content\":\"ping\"}],\"max_tokens\":8}" \
  "http://${XBOX_IP}:${API_PORT:-11434}/v1/chat/completions"
```

## Not in scope / do not do

- No `internetClientServer` / public inbound — `privateNetworkClientServer` (already in the
  manifest) covers LAN only.
- No `llama.cpp/tools/server` or cpp-httplib imported into the MSIX.
- The endpoint is **unauthenticated** — only expose it on a trusted LAN.
- ~~Memory note: the server's `Session` is independent of the UI's…~~ **Done
  (PR #161/#164)**: server and UI share the one `session_hub()` Session behind
  the same mutex — two loaded copies can no longer happen.

## Validation

See `scripts/validate-api.sh` (`spike|discovery|chat|stream|admission|budget|tools|framing|embed|pull|prefs|train|all`). Run it **from another host on the LAN**,
not from a client on the console itself — cross-device inbound needs no loopback exemption,
but a same-host localhost client would (`CheckNetIsolation`). Spike gate first (`GET /` → 200
proves the bind survives the Series S firewall/PLM), then chat / prefs / train as needed. `tools` checks the tool-proposal contract:
a named `tool_choice` returns a `get_weather` call with JSON arguments, a `tool`
result is answered, `tools` + `functions` is `400`, `tool_choice: "none"`
stays plain chat, and an oversized `tools` array is `400` with the
`prompt too long: the request needs` prefix and `context_length_exceeded`.
The gate also checks an indexed SSE tool call and the streamed answer to its
result, including usage, finish reason and `[DONE]`.
`TOOLS_MODEL` (default `MODEL`) must be a GGUF with a chat template; set
`ORT_MODEL` to an ONNX id to also check its `400`.
`all` includes pull and embeddings; provision `embed-nomic-v2-moe` first, or set
`EMBED_MODEL=embed-bge-m3`. Set `XLLAMA_API_EVIDENCE_DIR` to retain the embedding
requests and responses. The embedding gate verifies native widths, finite unit vectors,
float/base64 parity, errors, and restoration of the catalogue context. Each successful
embedding request logs `[embedding]` with model, input/token counts, width, duration,
and `peak_ws_mb` (process lifetime peak from the existing platform collector).
Restart between models when measuring separate cold peaks.

`scripts/measure-api-ttfb.py` (#421) measures timing as a LAN client sees it,
for repeated streamed `/v1/chat/completions` requests. For each attempt it
records the connection, the complete response header, the first body byte, the
first non-empty content delta and `[DONE]`, all measured from the start of the
request. It writes `xllama-api-ttfb/1` JSON. Failed attempts are kept with
their error: timeouts, non-200 status, stream error records, malformed SSE, no
content or a missing `[DONE]`. These times include the network, HTTP handling
and any model load. They are not the in-app `ttft_ms` and not a throughput
result. The hardware label is a declaration made by the operator.

Images are not in `all` (need SD-Turbo on device; use the curl example above). Chat round-trip:

```bash
curl -s http://<ip-xbox>:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<catalogue-id>","messages":[{"role":"user","content":"Say hi"}]}'
```
