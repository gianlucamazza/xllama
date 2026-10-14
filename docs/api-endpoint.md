# LAN HTTP endpoint (OpenAI-compatible)

**Status:** v1 — spike + non-streaming chat + UI parity surface (#118: preferences,
training status, images). Opt-in, **default OFF**. Dev Mode / LAN research only.
Not for the Store, not for public inbound.

xllama can expose its inference core (`xllama::Session`) as an HTTP endpoint on the local
network, so a PC on the same LAN can run inference on the console:
`http://<ip-xbox>:<port>/v1/chat/completions`. It is another front-end on the
same `xllama::Session` abstraction used by the chat UI, bench and CLI. Since
PR #161/#164 the Session lives in `xllama::session_hub()` — ONE process-wide
resident model **shared with the chat UI** ("never 2× model in RAM" holds
process-wide). Code lives in
`uwp/api-server.{h,cpp}` (WinRT `StreamSocketListener`), entirely under `#ifdef
XLLAMA_UWP`; the Linux/CLI build is unaffected.

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
  **[1025, 49151]** (and not 11443); anything else falls back to 11434 with a log line.
- **Model:** taken from the request's `"model"` field; if absent, falls back to
  `LocalState\model.txt`. The resident Session may already exist before any
  request — the GUI pre-loads it when a model becomes Ready — and it is
  swapped when a request names a different model. A swap invalidates the
  GUI's KV-reuse state (`hub.generation`): its next turn silently falls back
  to a full prefill. A model id that is neither resident nor a loadable
  `LocalState\models\<id>` directory gets **`404`** "model '<id>' is not on this
  console; see GET /v1/models" (chat and embeddings), before any load is
  attempted.
- **GGUF GPU layers:** a GGUF session gets the catalogue entry's `gpu_layers`
  (D3 default per model). `LocalState\gguf_gpu_layers.txt`, when present,
  overrides it for the GUI and this API alike; `0` forces the CPU. A change
  reloads the resident session. Gate and verdict:
  [gguf-gpu-decode.md](gguf-gpu-decode.md).

**Foreground only.** The endpoint dies when the app leaves the foreground (UWP Process
Lifetime Management — no always-on system service). On Xbox this is stricter than on PC: if
the package is game-classified (Device Portal _"Treat UWP apps as games by default"_) it is
**suspended and terminated** in the background, and `ExtendedExecutionSession` does not keep
a listener alive. Keep the app foregrounded on the console while using the endpoint. Accepted:
this is Dev Mode research, not a hosted service.

> Verified against learn.microsoft.com (2024-2026): `privateNetworkClientServer` +
> `StreamSocketListener` is the documented path for LAN inbound; network isolation blocks
> only _same-host_ sockets, so no `CheckNetIsolation` loopback exemption is needed for
> requests coming **from another LAN device** (only for a localhost client on the console).

## Protocol (v1)

| Method    | Path                     | Behaviour                                                                                                                                         |
| --------- | ------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| `GET`     | `/` or `/health`         | `200 {"status":"ok","service":"xllama"}` — the spike/liveness probe.                                                                              |
| `GET`     | `/v1/models`             | OpenAI model discovery — every servable on-device model; non-standard `"active": true` marks the one currently loaded.                            |
| `GET`     | `/api/tags`              | Ollama model discovery — same list, Ollama shape.                                                                                                 |
| `POST`    | `/api/pull`              | Pull a trusted catalogue chat/embedding model with Ollama-style NDJSON progress; load it when complete.                                           |
| `POST`    | `/v1/chat/completions`   | OpenAI-compatible chat completion, **non-streaming**.                                                                                             |
| `POST`    | `/api/embed`             | Ollama-style embeddings for one string or a batch, catalogue embedding GGUF/llama.cpp only.                                                       |
| `POST`    | `/api/embeddings`        | Deprecated Ollama single-prompt embeddings adapter.                                                                                               |
| `POST`    | `/v1/embeddings`         | OpenAI embeddings shape; `encoding_format` accepts `float` or `base64`.                                                                           |
| `POST`    | `/v1/preferences`        | Append a preference sample (`label` + `messages[]`) to `training/samples.jsonl` — same contract as the UI rate op (#118).                         |
| `GET`     | `/v1/training/status`    | `result.done` / `progress.json` / last personalized `result.json` + usable sample count (#118).                                                   |
| `POST`    | `/v1/images/generations` | SD-Turbo image gen (`prompt`, `steps` 1–4, `seed`); returns OpenAI-ish `{data:[{b64_json,path}]}` (#118). Shares the single-slot mutex with chat. |
| `OPTIONS` | _any_                    | CORS preflight (`204` + `Allow-Methods/Headers`) for browser clients.                                                                             |

### Model pulling (`POST /api/pull`)

The request is `{"model":"<catalogue-id-or-alias>","stream":true}`; `stream` defaults to
true. Pulls use only HTTPS entries bundled in the installed package with valid SHA-256 pins.
LocalState catalogue overrides, diffusion assets, arbitrary URLs, and unknown models are
rejected. Ollama embedding aliases (`bge-m3`, `nomic-embed-text-v2-moe`,
`qwen3-embedding:4b`) resolve to their catalogue IDs when those models are included.

Streaming returns chunked `application/x-ndjson`: status records report download bytes, then
model loading, followed by `{"status":"success"}`. A failure after streaming starts is an
NDJSON `{"error":"..."}` record. With `stream:false`, the endpoint waits and returns one
JSON result. The API pull gate spans transfer and model loading. A simultaneous
API pull receives HTTP 409 when its handler overlaps; the WinRT listener may
otherwise dispatch the handlers sequentially. GUI downloads, USB import and explicit
rollback share one downloader-wide writer permit. A competing writer is rejected
before any file or marker changes: non-streaming pulls return HTTP 409, streaming
pulls emit an error record, and the GUI reports a download-busy error. The permit
is released before completion callbacks that may chain provisioning. Existing
inference may finish during the download; after verification, the pull waits for the shared
session lock, replaces the resident model, and reports success only after loading it.

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
alias), `temperature`, `top_p`, `seed`, and `stop` (string or array). Numbers are
checked before the model loads, and a bad one is `400` naming the field: the token
count is a whole number from 1 to the catalogue `n_ctx` minus 1, `temperature` is
0–2, `top_p` is 0–1, and `seed` is a whole number (a negative seed keeps the
default, llama.cpp's `-1`). `stream: true`
is not implemented. Most other fields are ignored.

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
Content may be a string, null, or an array of `{type:"text",text}` parts joined
with no separator. Anything else, including an image part, is `400`. A `tool`
message requires `tool_call_id`. The last message must be `user` or `tool`.

xllama does not execute the tool, a shell, the filesystem, or the network. The
client does. One request is one generation, constrained by the template's grammar
(ADR 0005), with no hidden retry. The prompt and the parse go through the resident GGUF Jinja template
(`common_chat_templates_apply` / `common_chat_parse`). An ONNX session returns
`400` `tools require a GGUF chat template`. A GGUF with no explicit chat template
returns `400` `model has no chat template`, and one that renders without the
schema returns `400` `model chat template does not render tools` (phi4-mini). The server does not fall back to
`ChatFormat` with the schema pasted on. KV reuse stays off. Stops are the
template's additional stops plus the client's `stop`. The template's preserved tokens (LFM2's
`<|tool_call_start|>`, ...) are decoded as text so the parser sees them; every
other special token is still dropped.

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

| Model                                                                                       | Result                                                                                                                                                                                   |
| ------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `lfm25-230m`, `lfm25-350m`, `lfm25-1.2b-instruct`, `lfm2-2.6b`, `llama32-3b`, `minicpm5-1b` | `tool_calls` `get_weather({"city":"Paris"})` for both choices                                                                                                                            |
| `qwen25-coder-0.5b`, `qwen25-coder-3b`                                                      | Bare JSON instead of `<tool_call>` tags. The template's forced grammar allows text before the call, and the model never reaches the tag (still none at 512 tokens), so the reply is text |
| `lfm25-1.2b-thinking`                                                                       | Still reasoning at the 128-token cap (`length`); needs a larger `max_tokens`                                                                                                             |
| `gemma3-270m`, `gemma3-1b`, `phi4-mini`                                                     | `400` `model chat template does not render tools`                                                                                                                                        |

Generation follows the template's own grammar ([ADR 0005](adr/0005-tool-proposal-grammar.md)):
lazy for `auto` (free text until the template's call marker, then only the call
and end-of-generation), full for `required` or a named function. On the ten
requests whose call was identical with and without it, total latency moved from
25.8 s to 25.9 s (+0.4%, within run-to-run noise).

**Context budget.** Plain chat is fitted with `xllama::fit_prompt`
(`include/xllama/prompt_budget.h`) and the model's own tokenizer: the **oldest**
turns are dropped until the prompt plus the requested reply
(`max(n_predict, 250) + 1`) fits, and the trailing user message is never dropped.
A tool proposal uses `fit_tool_transcript` in the same header, the same tokenizer,
and the same reserve. It never drops the tool schema or the final user or tool
message, and it drops an assistant message that carries `tool_calls` together
with the tool results that follow it. Dropped messages are logged, not signalled.
When the protected tail still does not fit, the request is `400`. Plain chat says
the final user message needs the three numbers. A proposal says
`prompt too long: the request needs X tokens plus Y for the reply, but n_ctx is Z`,
where Y is `n_predict`. An oversized input is a client error; it used to surface
as a `500` from the generator instead.

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
`completion_tokens` is `n_eval`.

`stream: true` is **not** implemented in v1 (always returns the full completion). The
`GenerateParams::on_token` hook is the seam for adding SSE later.

### Embeddings (`/api/embed`, `/api/embeddings`, `/v1/embeddings`)

Embedding requests use the same resident `Session` and single-slot mutex as chat. Inputs in
a batch are encoded serially, preserving order and avoiding a second model or a large
multi-sequence activation allocation. The supported inference path is GGUF via llama.cpp;
ORT GenAI sessions and ranking-only models return an explicit unsupported error. Use an
embedding model already present in `LocalState\models\<id>` or pull one from the bundled
catalogue with `/api/pull`.

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

Qwen3-Embedding-4B Q4_K_M is not in the catalogue: its Release embedding smoke peaked at 4411 MiB, above the 3584 MiB host gate. The listed models pass host Release smoke and Series S contract validation.
[Catalogue status](model-matrix.md#g-embedding-api-catalogue-host-and-series-s-validated) owns
the current validation evidence and remaining throughput/retrieval-quality work.

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

Embedding models retain their own pooling behavior from GGUF metadata. The API does not inject task prefixes automatically: clients must prepend the appropriate instruction strings themselves. Model pulling is catalogue-only; `/api/show` and full Ollama model-management parity are not implemented.

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

In-process SD-Turbo with the same clamps as the Image dialog (`steps` 1–4).
Shares the single-slot mutex with chat → `503` when busy.

```bash
curl -s http://<ip-xbox>:11434/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{"prompt":"a red sports car on a mountain road","steps":1,"seed":42}'
# → {"created":…,"data":[{"b64_json":"…","path":"diffuse-out.png"}]}
```

Requires `sd-turbo-fp16` provisioned (auto-download on first UI Generate still
applies when the user opens Image; the API does not download the model for you).

## Concurrency

`Session::generate()` is single-slot / non-concurrent. Requests acquire the
process-wide `session_hub().mtx` with `try_lock`: a request arriving while
another request **or a chat-UI turn** is generating gets **HTTP 503**
`{"error":{"message":"busy"}}` for OpenAI-compatible routes and
`{"error":"busy"}` for native Ollama embedding routes. The slot covers the
whole process. Native `/api/embed`, `/api/embeddings`, and `/api/pull` errors
always expose a non-empty string in `error`; `/v1/embeddings` exposes an error
object with `message` and `type`. One exception: while the session **pre-load** is holding
the hub (right after a model becomes Ready), `acquire_hub_or_busy()` waits —
bounded, ≤15 s, only while `hub.preloading` is set — instead of bouncing the
client's very first request. Stopping the endpoint closes the listener only;
the Session is hub-owned and is NOT released (the chat UI may be using it).

## Not in scope / do not do

- No `internetClientServer` / public inbound — `privateNetworkClientServer` (already in the
  manifest) covers LAN only.
- No `llama.cpp/tools/server` or cpp-httplib imported into the MSIX.
- The endpoint is **unauthenticated** — only expose it on a trusted LAN.
- ~~Memory note: the server's `Session` is independent of the UI's…~~ **Done
  (PR #161/#164)**: server and UI share the one `session_hub()` Session behind
  the same mutex — two loaded copies can no longer happen.

## Validation

See `scripts/validate-api.sh` (`spike|chat|budget|tools|framing|embed|pull|prefs|train|all`). Run it **from another host on the LAN**,
not from a client on the console itself — cross-device inbound needs no loopback exemption,
but a same-host localhost client would (`CheckNetIsolation`). Spike gate first (`GET /` → 200
proves the bind survives the Series S firewall/PLM), then chat / prefs / train as needed. `tools` checks the tool-proposal contract:
a named `tool_choice` returns a `get_weather` call with JSON arguments, a `tool`
result is answered, `tools` + `functions` is `400`, and `tool_choice: "none"`
stays plain chat. `TOOLS_MODEL` (default `MODEL`) must be a GGUF with a chat
template; set `ORT_MODEL` to an ONNX id to also check its `400`.
`all` includes pull and embeddings; provision `embed-nomic-v2-moe` first, or set
`EMBED_MODEL=embed-bge-m3`. Set `XLLAMA_API_EVIDENCE_DIR` to retain the embedding
requests and responses. The embedding gate verifies native widths, finite unit vectors,
float/base64 parity, errors, and restoration of the catalogue context. Each successful
embedding request logs `[embedding]` with model, input/token counts, width, duration,
and `peak_ws_mb` (process lifetime peak from the existing platform collector).
Restart between models when measuring separate cold peaks.

Images are not in `all` (need SD-Turbo on device; use the curl example above). Chat round-trip:

```bash
curl -s http://<ip-xbox>:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<catalogue-id>","messages":[{"role":"user","content":"Say hi"}]}'
```
