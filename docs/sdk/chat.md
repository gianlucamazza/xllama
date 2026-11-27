# Chat Prompt API

## System Prompts

```cpp
inline constexpr const char kDefaultSystemPrompt[] = "You are a helpful AI assistant.";
inline constexpr const char kCodingSystemPrompt[] =
    "You are a concise coding assistant. Prefer correct, complete code and brief explanations.";
```

## `xllama::ChatFormat`

Chat template format detection and rendering.

```cpp
enum class ChatFormatKind { ChatML, Gemma, Llama3, Phi3, Granite };

enum class SystemStyle { DedicatedTurn, MergeIntoFirstUser };

struct ChatTurn {
    std::string user;
    std::string assistant;
};

struct ChatFormat {
    ChatFormatKind kind = ChatFormatKind::ChatML;
    std::string turn_open;
    std::string turn_close;
    std::string role_sep;
    std::string user_tag;
    std::string assistant_tag;
    std::string system_tag;
    std::string system_sep;
    SystemStyle system_style = SystemStyle::DedicatedTurn;
    std::vector<std::string> stop_sequences;
    std::string gen_suffix;
    std::string bos;
    bool strip_thinking_content = false;
    bool reasoning_opened = false; // the prefill opens <think> (Qwen3.5, MiniCPM5 on)
    std::string_view reasoning_prefill() const;

    std::string render_prompt(const std::string& system,
                              const std::vector<ChatTurn>& history,
                              const std::string& final_user) const;
    std::string render_delta(const std::string& user, bool prev_ended_with_stop) const;
    std::string render_system_prefix(const std::string& system) const;
    std::string postprocess_output(std::string text) const;
};

ChatFormat chat_format_for(const std::string& model_id, bool thinking);
ChatFormat chat_format_for(const std::string& model_id); // thinking = model_is_thinking
```

`thinking` selects the template's thinking or no-think path where it has both
(Qwen3 / Qwen3.5, MiniCPM5, SmolLM3). Take it from `decide_reasoning`
(`xllama/reasoning_control.h`), which honours a request's switch only where
`model_supports_reasoning_control` is true and otherwise keeps
`model_is_thinking`.

Dedicated-turn formats (ChatML, Llama-3, Phi-3, Gemma 4, Granite) emit a
system turn even when `system` is empty. Granite 4.x IBM jinja inserts a
default system message in that case; this renderer does not. CLI, GUI, and
the LAN API always pass `kDefaultSystemPrompt` (or the coding default).

## Model Format Detection

```cpp
bool model_is_qwen(const std::string& model_id);
bool model_is_qwen3(const std::string& model_id);
bool model_is_qwen35(const std::string& model_id);
bool model_is_gemma(const std::string& model_id);
bool model_is_llama(const std::string& model_id);
bool model_is_phi(const std::string& model_id);
bool model_is_granite(const std::string& model_id);
bool model_is_thinking(const std::string& model_id);
bool model_is_minicpm5(const std::string& model_id);
```

## Thinking Model Helpers

```cpp
std::string qwen_no_think_gen_suffix(const std::string& model_id);
std::string strip_empty_thinking_tags(std::string text);
std::string strip_thinking_blocks(std::string text);
bool apply_stop_sequences(std::string& output, const std::vector<std::string>& stops);
```
