# Miscellaneous APIs

## Cancel Policy

```cpp
enum class CancelTarget {
    None,     // nothing is running
    Image,    // diffuse-cancel.flag
    Training, // the training abort flag
    Text,     // the inference abort flag
};

CancelTarget cancel_target(bool image_running, bool training_running, bool text_running);
```

Precedence: image > training > text.

## JSON Utilities

```cpp
std::string json_escape(const std::string& s);
bool json_read_string(const std::string& s, size_t& pos, std::string& out);
```

## Speculative Decoding (Phase 15 W2)

```cpp
inline constexpr int kSpecDraftKDefault = 2;
inline constexpr int kSpecNgramDefault = 2;

std::vector<int32_t> prompt_lookup_draft(const std::vector<int32_t>& history,
                                         int n_gram = kSpecNgramDefault,
                                         int k_max = kSpecDraftKDefault);
```

## Personalization (Phase 11)

```cpp
struct PersonalizeSpec {
    std::string base_model;
    std::string dataset_path = "training/samples.jsonl";
    std::string out_dir = "training/out/personalized";
    std::string name = "personalized";
    int last_block = -1;
    int epochs = 8;
    float learning_rate = 2e-4f;
    int n_ctx_train = 256;
    std::string eval_prompt;
    std::string eval_expect_contains;
};

bool build_personalize_job(const PersonalizeSpec& spec, TrainingJob& out,
                           std::string* err = nullptr);
std::vector<std::string> last_block_param_filter(int last_block);
int guess_last_block_from_model_id(const std::string& model_id);
int count_usable_preference_samples(const std::string& samples_path);
```

## Preference Capture

```cpp
bool preference_label_valid(const std::string& label);
std::string format_preference_sample_jsonl(
    const std::string& label,
    const std::vector<std::pair<std::string, std::string>>& messages,
    const std::string& preferred_assistant = {},
    const std::string& ts_iso = {});
bool append_preference_sample_file(const std::string& path, const std::string& jsonl_line);
inline const char* kPreferenceSamplesRelPath = "training/samples.jsonl";
```
