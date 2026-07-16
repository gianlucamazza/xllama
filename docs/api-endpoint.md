# LAN HTTP endpoint (OpenAI-compatible)

**Status:** v1 — spike + non-streaming chat. Opt-in, **default OFF**. Dev Mode / LAN
research only. Not for the Store, not for public inbound.

xllama can expose its inference core (`xllama::Session`) as an HTTP endpoint on the local
network, so a PC on the same LAN can run inference on the console:
`http://<ip-xbox>:<port>/v1/chat/completions`. It is **another front-end on the same
`Session`** used by the chat UI, bench and CLI — not a separate inference path. Code lives
in `uwp/api-server.{h,cpp}` (WinRT `StreamSocketListener`), entirely under `#ifdef
XLLAMA_UWP`; the Linux/CLI build is unaffected.

## Enabling it

Drop an empty `api.flag` in the app's `LocalState` (via Device Portal / WDP, same channel
as the other flags). Unlike the headless `bench.flag`/`diffuse.flag`, `api.flag` is **not
consumed**: the server is persistent and coexists with the live XAML chat UI. It starts
from `App::OnLaunched` on a detached MTA thread and stays bound for the app lifetime.

- **Port:** `LocalState\api-port.txt` if present, else **11434** (Ollama's default port).
- **Model:** taken from the request's `"model"` field; if absent, falls back to
  `LocalState\model.txt`. The `Session` is created lazily on the first request and reused;
  it is re-created only when a request asks for a different model.

The endpoint dies when the app is suspended/closed (UWP PLM — no always-on system service).
That is accepted: this is research on Dev Mode, not a hosted service.

## Protocol (v1)

| Method | Path                   | Behaviour                                                            |
| ------ | ---------------------- | -------------------------------------------------------------------- |
| `GET`  | `/` or `/health`       | `200 {"status":"ok","service":"xllama"}` — the spike/liveness probe. |
| `POST` | `/v1/chat/completions` | OpenAI-compatible chat completion, **non-streaming**.                |

Request body (subset): `model`, `messages[]` (`role` ∈ system/user/assistant, `content`),
optional `max_tokens` (default 512), `temperature`, `top_p`. `messages[]` is mapped to
`ChatFormat::render_prompt(system, history, final_user)` (`include/xllama/chat_prompt.h`);
`chat_format_for(model)` selects the template and stop sequences. The reply carries
`choices[0].message.content`, `finish_reason` (`stop`/`length`) and a `usage` block from
`InferenceResult` (`n_p_eval` / `n_eval`).

`stream: true` is **not** implemented in v1 (always returns the full completion). The
`GenerateParams::on_token` hook is the seam for adding SSE later.

## Concurrency

`Session::generate()` is single-slot / non-concurrent. The server holds one shared
`Session` behind a `std::mutex` with `try_lock`: a request arriving while another is being
served gets **HTTP 503** `{"error":{"message":"busy"}}` — Ollama single-slot semantics.

## Not in scope / do not do

- No `internetClientServer` / public inbound — `privateNetworkClientServer` (already in the
  manifest) covers LAN only.
- No `llama.cpp/tools/server` or cpp-httplib imported into the MSIX.
- The endpoint is **unauthenticated** — only expose it on a trusted LAN.
- Memory note: the server's `Session` is independent of the UI's. On large models (e.g. the
  3B H4 line, ~1.8 GB) two loaded copies may not fit; the follow-up is to share the
  controller's `m_session` behind the same mutex.

## Validation

See `scripts/validate-api.sh`. Spike gate first (`GET /` → 200 proves the bind survives the
Series S firewall/PLM), then a chat round-trip:

```bash
curl -s http://<ip-xbox>:11434/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<catalogue-id>","messages":[{"role":"user","content":"Say hi"}]}'
```
