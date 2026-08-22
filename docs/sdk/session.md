# Session API

## `xllama::Session`

The `Session` class owns a loaded model and tokenizer. It supports multi-turn generation via KV-cache reuse.

```cpp
struct Session {
    static std::unique_ptr<Session> create(const SessionParams& params, std::string* err = nullptr);
    virtual InferenceResult generate(const GenerateParams& params) = 0;
    virtual int count_tokens(const std::string& prompt) = 0;
    virtual bool can_context_shift() const;
    virtual bool save_state(const std::string& path, std::string* err = nullptr);
    virtual bool load_state(const std::string& path, std::string* err = nullptr);
    virtual ~Session() = default;
};
```

Thread safety: `generate()` must not be called concurrently.

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
auto* session = hub.ensure_locked("my-model", params, &err);
```

**Global instance** (backward compatible):

```cpp
auto& hub = xllama::session_hub();
```

## `xllama::SessionParams`

Configuration for creating a persistent session.

| Field           | Default | Description                                      |
| --------------- | ------- | ------------------------------------------------ |
| `model_path`    | —       | Path to model                                    |
| `n_ctx`         | 2048    | Context size                                     |
| `n_threads`     | 0       | Thread count (0=auto)                            |
| `n_batch`       | 0       | llama.cpp logical prefill batch (0=default 2048) |
| `n_ubatch`      | 0       | llama.cpp physical prefill chunk (0=default 512) |
| `backend`       | Auto    | ORTGenAI, LlamaCpp, or Auto (inspects model)     |
| `n_gpu_layers`  | 0       | llama.cpp GPU layers (0=CPU)                     |
| `lora_path`     | ""      | GGUF LoRA adapter                                |
| `lora_scale`    | 1.0f    | LoRA scale                                       |
| `kv_q8`         | false   | Quantize KV cache to q8_0                        |
| `dml_warmup`    | true    | Warm-up DML models at load time                  |
| `prompt_lookup` | false   | Draft-free prompt-lookup speculative decoding    |

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
| `on_token`           | —          | Callback per generated token                |
| `on_status`          | —          | Callback for status changes                 |
| `abort_flag`         | —          | Atomic flag for early termination           |

### KV-Cache Reuse Modes

| `reuse_kv` | `reset_kv` | Behavior                                               |
| ---------- | ---------- | ------------------------------------------------------ |
| `false`    | —          | Stateless turn: full context prefill                   |
| `true`     | `false`    | Continuation: delta-only prefill (multi-turn TTFT win) |
| `true`     | `true`     | Reset: full context prefill, subsequent turns reuse    |
