# Routing Policy

## `xllama::RoutingSettings`

Configuration for EP (execution provider) routing.

| Field | Default | Description |
|-------|---------|-------------|
| `mode` | CpuOnly | CpuOnly, GpuOnly, or Auto |
| `token_threshold` | 1550 | Token count threshold for GPU routing |
| `cpu_model` | "" | CPU model path |
| `gpu_model` | "" | GPU model path |

## `xllama::RoutingDecision`

Result of a routing decision.

| Field | Type | Description |
|-------|------|-------------|
| `active_model` | std::string | Selected model path |
| `use_gpu` | bool | Whether GPU is active |
| `token_count` | int | Token count at decision time |

## `xllama::RoutingPolicy` (SDK-configurable)

Bundles every routing gate behind `std::function` callbacks. Override any gate without patching the header.

```cpp
struct RoutingPolicy {
    std::function<bool(std::string_view gpu_model)> dml_text_model_ok_fn;
    std::function<RoutingDecision(const RoutingSettings&, int, bool, bool)> decide_fn;
    std::function<bool(const std::wstring& kind)> allow_kind_fn;
    std::function<bool(const std::wstring& kind)> reuse_kv_kind_fn;
    std::function<bool(std::string_view model)> reuse_kv_model_fn;

    // Convenience wrappers:
    bool dml_text_model_ok(std::string_view gpu_model) const;
    RoutingDecision decide(const RoutingSettings& s, int n_tok, bool base_is_gguf,
                           bool gpu_available) const;
    bool routing_allowed_for_kind(const std::wstring& kind) const;
    bool kv_reuse_allowed_for_kind(const std::wstring& kind) const;
    bool kv_reuse_supported_for_model(std::string_view model) const;
};
```

**Custom routing example:**

```cpp
xllama::RoutingPolicy policy;
policy.dml_text_model_ok_fn = [](std::string_view m) { return true; };
policy.decide_fn = [](const xllama::RoutingSettings& s, int n, bool gguf, bool gpu) {
    xllama::RoutingDecision d;
    d.use_gpu = gpu;
    d.active_model = gpu ? s.gpu_model : s.cpu_model;
    return d;
};
```

**Free-function wrappers** (default behavior, backward compatible):

```cpp
bool ok = xllama::dml_text_model_ok("smollm2-360m-dml-fp16-v2");
auto decision = xllama::decide_routing(settings, n_tok, base_is_gguf, gpu_available);
bool allowed = xllama::routing_allowed_for_kind(L"ort-genai");
bool reuse = xllama::kv_reuse_supported_for_model("smollm2-360m-cpu-int4");
```