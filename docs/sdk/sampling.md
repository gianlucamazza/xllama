# Sampling API

## `xllama::SamplingConfig`

Shared sampling configuration. Both `InferenceParams` and `GenerateParams` expose these values.

```cpp
struct SamplingConfig {
    float temperature = 0.8f;
    float top_p = 0.9f;
    int top_k = 40;
    float repetition_penalty = 1.1f;
    uint32_t seed = 0xFFFFFFFF;
    bool greedy = false;

    bool is_greedy() const;
};

bool same_chain(const SamplingConfig& a, const SamplingConfig& b);
```

## Sampling Defaults

```cpp
namespace sampling_defaults {
    inline constexpr float kTemperature = 0.8f;
    inline constexpr float kTopP = 0.9f;
    inline constexpr int kTopK = 40;
    inline constexpr float kRepetitionPenalty = 1.1f;
    inline constexpr int kPenaltyLastN = 64;
    inline constexpr uint32_t kSeed = 0xFFFFFFFF;
}
```
