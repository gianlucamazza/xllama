# LAN HTTP endpoint (OpenAI-compatible)

**Status:** v1 — spike + non-streaming chat. Opt-in, **default OFF**. Dev Mode / LAN
research only. Not for the Store, not for public inbound.

xllama can expose its inference core (`xllama::Session`) as an HTTP endpoint on the local
network, so a PC on the same LAN can run inference on the console:
`http://<ip-xbox>:<port>/v1/chat/completions`. It is another front-end on the
same `xllama::Session` abstraction used by the chat UI, bench and CLI, with its
own lazily created Session instance and lifecycle. Code lives in
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
  `LocalState\model.txt`. The `Session` is created lazily on the first request and reused;
  it is re-created only when a request asks for a different model.

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

| Method    | Path                   | Behaviour                                                             |
| --------- | ---------------------- | --------------------------------------------------------------------- |
| `GET`     | `/` or `/health`       | `200 {"status":"ok","service":"xllama"}` — the spike/liveness probe.  |
| `GET`     | `/v1/models`           | OpenAI model discovery — lists the current/served model.              |
| `GET`     | `/api/tags`            | Ollama model discovery — same model, Ollama shape.                    |
| `POST`    | `/v1/chat/completions` | OpenAI-compatible chat completion, **non-streaming**.                 |
| `OPTIONS` | _any_                  | CORS preflight (`204` + `Allow-Methods/Headers`) for browser clients. |

Request body (subset): `model`, `messages[]` (`role` ∈ system/user/assistant, `content`),
optional `max_completion_tokens` / `max_tokens` (default 512; the former wins — `max_tokens`
is the deprecated OpenAI alias), `temperature`, `top_p`, `seed` (reproducibility), and
`stop` (string or array, added to the format's own stops). `messages` must be a JSON array
and contain a trailing user turn, else `400`. When no `system` message is sent, a default
one is injected (small instruct models degrade with an empty system turn). Other unknown
fields (`n`, penalties, `tools`, …) are ignored, not rejected. `messages[]` is mapped to
`ChatFormat::render_prompt(system, history, final_user)` (`include/xllama/chat_prompt.h`);
`chat_format_for(model)` selects the template and stop sequences.

Only `Content-Length` framing is supported — a `Transfer-Encoding: chunked` request gets
`411` (OpenAI SDKs send Content-Length). Malformed JSON / wrong-typed fields get `400`, never
a dropped connection.

The reply follows the OpenAI shape: `id` (unique `chatcmpl-…`), `object: chat.completion`,
`created`, `model`, `choices[0]` (`message`, `finish_reason` — `length` only when the token
cap is hit, else `stop`; and `logprobs: null`, omitting which breaks openai-python/LangChain
Pydantic validation), and a `usage` block from `InferenceResult` (`n_p_eval` / `n_eval`).

`stream: true` is **not** implemented in v1 (always returns the full completion). The
`GenerateParams::on_token` hook is the seam for adding SSE later.

## Concurrency

`Session::generate()` is single-slot / non-concurrent. The server holds one shared
`Session` behind a `std::mutex` with `try_lock`: a request arriving while another is being
served gets **HTTP 503** `{"error":{"message":"busy"}}` — Ollama single-slot semantics.
Stopping the endpoint first closes the listener, then releases its Session after
the active request (if any) leaves the single-slot mutex.

## Not in scope / do not do

- No `internetClientServer` / public inbound — `privateNetworkClientServer` (already in the
  manifest) covers LAN only.
- No `llama.cpp/tools/server` or cpp-httplib imported into the MSIX.
- The endpoint is **unauthenticated** — only expose it on a trusted LAN.
- Memory note: the server's `Session` is independent of the UI's. On large models (e.g. the
  3B H4 line, ~1.8 GB) two loaded copies may not fit; the follow-up is to share the
  controller's `m_session` behind the same mutex.

## Validation

See `scripts/validate-api.sh` (`spike|chat|all`). Run it **from another host on the LAN**,
not from a client on the console itself — cross-device inbound needs no loopback exemption,
but a same-host localhost client would (`CheckNetIsolation`). Spike gate first (`GET /` → 200
proves the bind survives the Series S firewall/PLM), then a chat round-trip:

```bash
curl -s http://<ip-xbox>:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<catalogue-id>","messages":[{"role":"user","content":"Say hi"}]}'
```
