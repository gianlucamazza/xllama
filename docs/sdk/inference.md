# Inference API

## `xllama::InferenceParams`

Configuration for a single inference call. Used by the CLI, benchmarks, and `run_inference()`.

| Field | Default | Description |
|-------|---------|-------------|
| `model_path` | — | Path to model (Linux: absolute; UWP: filename in LocalFolder) |
| `prompt` | — | Input text |
| `n_predict` | 128 | Max tokens to generate |
| `n_ctx` | 2048 | Context size |
| `max_length_override` | 0 | Override max_length (0=derive, -1=saturate to n_ctx, >0=explicit) |
| `n_threads` | 0 | Thread count (0=auto) |
| `run_index` | 0 | Bench repetition index (0=not a bench) |
| `n_batch` | 0 | llama.cpp logical prefill batch (0=default 2048) |
| `n_ubatch` | 0 | llama.cpp physical prefill chunk (0=default 512) |
| `kv_q8` | false | Quantize KV cache to q8_0 |
| `temperature` | 0.8 | Sampling temperature |
| `top_p` | 0.9 | Top-p sampling |
| `top_k` | 40 | Top-k sampling |
| `repetition_penalty` | 1.1 | Repetition penalty |
| `seed` | 0xFFFFFFFF | Random seed |
| `greedy` | false | Deterministic argmax decode |
| `system_prompt` | "" | System message (empty = built-in default) |
| `dump_logits_path` | "" | Dump logit vector for parity harness |
| `stop_sequences` | [] | Stop strings |
| `chat_template` | false | Wrap prompt with chat template |
| `lora_path` | "" | GGUF LoRA adapter path |
| `lora_scale` | 1.0f | LoRA scale |
| `prompt_lookup` | false | Draft-free prompt-lookup speculative decoding |
| `on_status` | — | Callback for status changes |
| `on_token` | — | Callback per generated token |
| `echo_stdout` | false | Stream to stdout (interactive CLI) |
| `abort_flag` | — | Atomic flag for early termination |

Probe flags (require `XLLAMA_BUILD_PROBES=ON`): `run_membw`, `run_diskbw`, `run_gpubw`, `run_gpugemv`, `run_ramceil`.

## `xllama::InferenceResult`

Result of an inference call.

| Field | Type | Description |
|-------|------|-------------|
| `success` | bool | Whether generation succeeded |
| `t_load_ms` | double | Model load time (ms) |
| `t_p_eval_ms` | double | Prefill time (ms) |
| `t_eval_ms` | double | Decode time (ms) |
| `n_p_eval` | int | Prefill tokens |
| `n_eval` | int | Decode tokens |
| `ended_with_stop` | bool | Stopped on a stop sequence |
| `max_length` | int | Max length actually requested |
| `peak_ws_mb` | size_t | Peak working set (MB) |
| `gpu_mem_mb` | size_t | Per-process GPU memory (CurrentUsage) |
| `gpu_budget_mb` | size_t | OS-granted GPU budget |
| `n_drafted` | int | Speculative drafts (0 if prompt_lookup off) |
| `n_spec_accepted` | int | Speculative accepted |
| `output_text` | std::string | Generated text |
| `error_msg` | std::string | Error description on failure |

## `xllama::run_inference()`

Synchronous inference function.

```cpp
InferenceResult run_inference(const InferenceParams& params);
```

Path semantics:
- Linux: `model_path` is an absolute filesystem path.
- UWP: `model_path` is relative to `ApplicationData::LocalFolder\models\`.

## `xllama::write_bench_csv()`

Write a benchmark CSV row to the resolved local path.

```cpp
void write_bench_csv(const InferenceParams& params, const InferenceResult& res,
                     const char* host_label = nullptr);
```