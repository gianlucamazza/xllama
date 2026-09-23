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
restart, and the dialog shows its current state.

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
  to a full prefill.

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
| `POST`    | `/api/embed`             | Ollama-style embeddings for one string or a batch, GGUF/llama.cpp only.                                                                          |
| `POST`    | `/api/embeddings`        | Deprecated Ollama single-prompt embeddings adapter.                                                                                              |
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
JSON result. Only one pull runs at a time; a concurrent pull receives HTTP 409. Existing
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

Request body (subset): `model`, `messages[]` (`role` ∈ system/user/assistant, `content`),
optional `max_completion_tokens` / `max_tokens` (default 512; the former wins — `max_tokens`
is the deprecated OpenAI alias), `temperature`, `top_p`, `seed` (reproducibility), and
`stop` (string or array, added to the format's own stops). `messages` must be a JSON array
and contain a trailing user turn, else `400`. When no `system` message is sent, a default
one is injected (small instruct models degrade with an empty system turn). Other unknown
Most unsupported fields are ignored for compatibility, but tool-related fields
(`tools`, `functions`, `tool_choice`) are rejected. xllama never executes model-requested
tools, shell commands, filesystem operations, or arbitrary network calls. `messages[]` is mapped to
`ChatFormat::render_prompt(system, history, final_user)` (`include/xllama/chat_prompt.h`);
`chat_format_for(model)` selects the template and stop sequences.

**Context budget.** The prompt is fitted to the session's `n_ctx` with the same
primitive the chat UI uses (`xllama::fit_prompt`, `include/xllama/prompt_budget.h`) and the
model's own tokenizer: the **oldest** `messages[]` entries are dropped until the prompt plus
the requested reply (`max(max_tokens, 250)` tokens) fits. Dropped entries are logged, not
signalled in the response. When the trailing user message alone cannot fit, the request is
refused with `400` and a body naming the three numbers — a client's oversized input is a
client error, and it used to surface as a `500` from the generator instead.

Only `Content-Length` framing is supported — a `Transfer-Encoding: chunked` request gets
`411` (OpenAI SDKs send Content-Length). Malformed JSON / wrong-typed fields get `400`, never
a dropped connection.

The reply follows the OpenAI shape: `id` (unique `chatcmpl-…`), `object: chat.completion`,
`created`, `model`, `choices[0]` (`message`, `finish_reason` — `length` only when the token
cap is hit, else `stop`; and `logprobs: null`, omitting which breaks openai-python/LangChain
Pydantic validation), and a `usage` block from `InferenceResult` (`n_p_eval` / `n_eval`).

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
cannot exceed xllama's configured model limit. Other sampling options have no
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

Qwen3-Embedding-4B Q4_K_M is not in the catalogue: its Release embedding smoke peaked at 4411 MiB, above the 3584 MiB host gate. The two listed models passed host Release smoke; Xbox memory and throughput still need a device run.

**Ollama name aliases**: The API also accepts `bge-m3` and `nomic-embed-text-v2-moe`. The `qwen3-embedding:4b` alias is reserved for a future catalogue entry.

**Model swap**: Embedding requests share the single-slot `session_hub()` with chat. An embedding call that names a different model swaps the resident chat model out. The next chat turn will prefill cleanly (KV is cleared on swap).

**Dimensions**: `dimensions=0` (the default) returns the model's native width. Non-zero values are accepted only when the model declares Matryoshka support. BGE-M3 is not Matryoshka; requesting any dimensions other than 0 or 1024 returns 400.

**Context limits**: Each model opens at its catalogue `n_ctx` (BGE-M3: 8192, Nomic MoE: 512). The `options.num_ctx` field cannot exceed that limit or go below 32. Inputs longer than context are truncated when `truncate=true` (default), or rejected with 400 when `truncate=false`.

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
`{"error":{"message":"busy"}}` — the Ollama single-slot semantics, widened to
the whole process. One exception: while the session **pre-load** is holding
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

See `scripts/validate-api.sh` (`spike|chat|embed|prefs|train|all`). Run it **from another host on the LAN**,
not from a client on the console itself — cross-device inbound needs no loopback exemption,
but a same-host localhost client would (`CheckNetIsolation`). Spike gate first (`GET /` → 200
proves the bind survives the Series S firewall/PLM), then chat / prefs / train as needed.
Images are not in `all` (need SD-Turbo on device; use the curl example above). Chat round-trip:

```bash
curl -s http://<ip-xbox>:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<catalogue-id>","messages":[{"role":"user","content":"Say hi"}]}'
```
