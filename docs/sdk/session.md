# Session API

## `xllama::Session`

The `Session` class owns a loaded model and tokenizer. It supports multi-turn generation via KV-cache reuse and GGUF text embeddings through the llama.cpp backend.

```cpp
struct Session {
    static Backend backend_for(const SessionParams& params);
    static std::unique_ptr<Session> create(const SessionParams& params, std::string* err = nullptr);
    virtual InferenceResult generate(const GenerateParams& params) = 0;
    virtual int count_tokens(const std::string& prompt) = 0;
    virtual EmbeddingResult embed(const EmbeddingParams& params);
    virtual int context_length() const;
    virtual bool can_context_shift() const;
    virtual bool save_state(const std::string& path, std::string* err = nullptr,
                            std::size_t max_bytes = 0);
    virtual bool load_state(const std::string& path, std::string* err = nullptr);
    virtual ~Session() = default;
};
```

`backend_for()` resolves the backend that `create()` would use without loading
the model.

Thread safety: serialize generation, embedding, token counting and state operations
on a Session. The GUI and LAN API hold `SessionHub::mtx` through each operation;
`embed()` clears chat KV, so the next chat turn requires a clean prefill.
The default backend implementation returns an unsuccessful embedding result;
its `context_length()` returns 0. The llama.cpp session reports the context it
opened with, which can be below `SessionParams::n_ctx` (capped at the model's
trained context). Budget prompts against `context_length()`, not the request.

`save_state()` refuses a state larger than `max_bytes` (0 means the loader's
512 MB ceiling) before it copies or writes anything. The chat UI passes the
KV snapshot pool cap (`kKvStoreMaxBytes`), so a long dense context is not
saved. Its next visit prefills normally.

## `xllama::SessionHub`

The single process-wide owner of the resident model. Both the GUI and the LAN API lock it for the duration of a turn.

```cpp
struct SessionHub {
    std::mutex mtx;
    Session* ensure_locked(const std::string& model_id, const SessionParams& sp, std::string* err = nullptr);
    void reset_locked();
};
```

**SDK usage** — create your own instance:

```cpp
xllama::SessionHub hub;
std::lock_guard<std::mutex> lock(hub.mtx);
auto* session = hub.ensure_locked("my-model", params, &err);
// Keep the lock while using session; a reset/model swap invalidates the pointer.
```

**Global instance** (backward compatible):

```cpp
auto& hub = xllama::session_hub();
```

## Embedding contracts

`embedding.h` defines `EmbeddingParams` (`input`, `dimensions=0`, `truncate=true`)
and `EmbeddingResult` (`success`, `embedding`, `n_tokens`, `error_msg`).
`Session::embed()` tokenizes the raw input, honors model pooling metadata, limits
it to `min(context_length(), logical batch)`, then returns a normalized vector.
The effective token count includes tokenizer special tokens. Invalid dimensions,
unsupported backend/encoder-decoder/ranking models and untruncated oversized
input return `success=false` with `error_msg`.

`trim_tokens_for_pooling` retains the prefix for mean/CLS and the suffix for
last/none pooling. `normalize_embedding` truncates dimensions and L2-normalizes;
it rejects empty, non-finite or zero-norm vectors. `embedding_base64` encodes
float32 values as little-endian bytes. The SDK helper does not enforce the
catalogue-specific BGE width rule; the [LAN adapter](../api-endpoint.md) does.

A host smoke uses repeated `-p` inputs:

```bash
./build/linux-release/bin/xllama-cli --embed -m /path/model.gguf \
  -p "search_query: find a red car" -p "search_query: find a red car"
```

The caller supplies task prefixes. This verifies encoding and repeated-input
stability, not retrieval quality or cross-platform parity.

## `xllama::SessionParams`

Configuration for creating a persistent session.

| Field              | Default | Description                                                      |
| ------------------ | ------- | ---------------------------------------------------------------- |
| `model_path`       | —       | Path to model                                                    |
| `n_ctx`            | 2048    | Context size; GGUF caps it at the model's trained context        |
| `n_threads`        | 0       | Thread count (0=auto)                                            |
| `n_batch`          | 0       | llama.cpp logical prefill batch (0=default 2048)                 |
| `n_ubatch`         | 0       | llama.cpp physical prefill chunk (0=default 512)                 |
| `backend`          | Auto    | ORTGenAI, LlamaCpp, or Auto (inspects model)                     |
| `n_gpu_layers`     | 0       | llama.cpp GPU layers (0=CPU)                                     |
| `lora_path`        | ""      | GGUF LoRA adapter                                                |
| `lora_scale`       | 1.0f    | LoRA scale                                                       |
| `kv_q8`            | false   | Quantize KV cache to q8_0                                        |
| `dml_warmup`       | true    | Warm-up DML models at load time                                  |
| `prompt_lookup`    | false   | Draft-free prompt-lookup speculative decoding                    |
| `prefix_snapshots` | true    | Hybrid prefix snapshots; `false` or LocalState file `0` opts out |

For GGUF catalogue entries, `resolve_session_params` joins the relative LoRA
adapter name to the model directory using the host filesystem separator. An
empty model directory leaves the adapter name relative.

On UWP, interactive GGUF chat (GUI and LAN API) reads
`LocalState\gguf_n_threads.txt` through `resolve_session_params`. Absent, 0,
an invalid value, or one above `kGgufNThreadsMax` (ggml's thread-array size)
keeps `n_threads` at 0 (auto). `bench_threads.txt`
remains bench-only. Threads are a tuning knob, not load identity
([ADR 0007](../adr/0007-session-spec.md)).

## `xllama::GenerateParams`

Parameters for a single generation turn.

| Field                | Default    | Description                                 |
| -------------------- | ---------- | ------------------------------------------- |
| `prompt`             | —          | Input text                                  |
| `n_predict`          | 96         | Max tokens to generate                      |
| `temperature`        | 0.8        | Sampling temperature                        |
| `top_p`              | 0.9        | Top-p sampling                              |
| `top_k`              | 40         | Top-k sampling                              |
| `repetition_penalty` | 1.1        | Repetition penalty                          |
| `seed`               | 0xFFFFFFFF | Random seed                                 |
| `stop_sequences`     | []         | Stop strings                                |
| `reuse_kv`           | false      | Enable KV-cache reuse (continuous decoding) |
| `reset_kv`           | false      | Reset KV cache for this turn                |
| `n_keep`             | 0          | Tokens pinned across context shifts         |
| `turn_close`         | ""         | Turn terminator a shift evicts up to        |
| `expected_kv_epoch`  | 0          | Continuation guard (`kv_epoch()`); 0 = off  |
| `on_token`           | —          | Callback per generated token                |
| `on_status`          | —          | Callback for status changes                 |
| `abort_flag`         | —          | Atomic flag for early termination           |

### KV-Cache Reuse Modes

| `reuse_kv` | `reset_kv` | Behavior                                                                |
| ---------- | ---------- | ----------------------------------------------------------------------- |
| `false`    | —          | Stateless turn: full context prefill                                    |
| `true`     | `false`    | Reuse supported prefix; hybrid tails restore a saved prefix if possible |
| `true`     | `true`     | Reset: full context prefill, subsequent turns reuse                     |

`Session::kv_epoch()` identifies the resident KV content. It is unique across
sessions, changes on every KV change that is not a continuation, and stays the
same across successful continuations. A caller that shares the session (the GUI
next to the LAN API) records it after its turn and passes it as
`expected_kv_epoch` on the next continuation: on a mismatch `generate` returns
`success=false, n_eval=0` before touching the KV, and the caller retries with
the full prompt. A full prompt that cannot fit `n_ctx` is refused before the
KV is rewound, so it never costs the resident conversation.

A full-prompt turn (LAN API, regenerate) diffs the new tokens against the
resident KV. Non-hybrid caches rewind the divergent tail (#170a). Hybrid
caches (LFM2) cannot always rewind: after the prompt head the session keeps
an in-memory prefix checkpoint, the partial state that `seq_rm` cannot rebuild
(the recurrent cells only; `PARTIAL_ONLY`). On a long prompt the checkpoint
sits a few tokens before the prompt end, then that tail prefills. If the new
prompt extends the checkpoint's tokens and the attention cells up to it are
still resident, the recurrent state is restored, the attention KV is truncated
to the checkpoint and only the suffix prefills — including an exact
regenerate, whose output matches a cold run. A checkpoint that does not apply
is dropped and the full prompt prefills; a context shift drops it too.

A continuation that would overflow `n_ctx` shifts the context (#169) when the
cache supports it: the oldest tokens after `n_keep` are evicted and the rest is
RoPE-shifted down. Eviction ends right after a complete `turn_close` (the
format's turn terminator), so it drops whole turns and never half a message or
a template marker. Without a boundary that frees enough room, or with an empty
`turn_close`, the continuation fails with `n_eval == 0` (#173) and the caller
retries with the full prompt, where `fit_prompt` drops whole turns. On hybrid
caches the recurrent state keeps what it absorbed from evicted tokens
(llama.cpp's approximation); a recall probe on LFM2.5, Granite-4.0-H-Tiny and
Qwen2.5-Coder-3B did not recall a fact from an evicted turn.
`SessionParams::prefix_snapshots` defaults on;
`LocalState\hybrid-prefix-snapshots.txt` = `0` opts out. Disk `load_state` is
still not a delta-prefill guarantee — it includes generated tokens, so a
hybrid tail that cannot rewind falls back to a full prefill. See
[KV architecture](../architecture.md#kv-cache-reuse-both-backends)
and the separate prefix-reuse/fallback [console probes](../console-validation-runbook.md#controlled-model-writer-and-kv-fallback-probes).
