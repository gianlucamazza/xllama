# KV Store

## `xllama::KvStore`

On-disk KV snapshot pool for conversation persistence.

```cpp
struct KvStore {
    std::string dir;
    static bool valid_id(const std::string& id);
    std::string path_for(const std::string& id) const;
    void erase(const std::string& id) const;
    void prune(size_t max_files, uint64_t max_bytes) const;
    uint64_t total_bytes() const;
};

inline constexpr size_t kKvStoreMaxFiles = 3;
inline constexpr uint64_t kKvStoreMaxBytes = 192ull << 20; // 192 MiB
```

A snapshot is ~12 KiB per resident token. The pool is capped by file count and total bytes.

## `xllama::fit_prompt()`

Exact token-budget trimmer. Enforces the context budget where the tokenizer is available.

```cpp
using TokenCounter = std::function<int(const std::string&)>;

struct PromptFit {
    std::string prompt;     // rendered, ready for Session::generate
    int n_tokens = 0;       // exact token count (-1 when no counter given)
    int dropped = 0;        // oldest turns dropped to make it fit
    bool fits = false;      // false = even the trailing message does not fit
};

PromptFit fit_prompt(const ChatFormat& fmt, const std::string& system,
                     const std::vector<ChatTurn>& turns, const std::string& user_text,
                     int n_ctx, int n_predict, const TokenCounter& count);
```

## `xllama::fit_tool_transcript()`

The same budget for a client-side tool proposal ([ADR 0004](../adr/0004-tool-proposals.md)).
The prompt is rendered by the GGUF template through `render`, then counted by the resident
tokenizer; the reply reserve is the same as `fit_prompt`.

```cpp
using ToolPromptRenderer = std::function<std::string(const std::vector<ChatMessage>& messages,
                                                     const std::vector<ToolSpec>& tools)>;

struct ToolTranscriptFit {
    std::string prompt;
    std::vector<ChatMessage> messages; // what `prompt` was rendered from
    int n_tokens = 0;                  // exact count of `prompt` (-1 with no counter)
    int dropped = 0;                   // messages removed, not groups
    bool fits = false;
};

ToolTranscriptFit fit_tool_transcript(const std::vector<ChatMessage>& messages,
                                      const std::vector<ToolSpec>& tools, int n_ctx, int n_predict,
                                      const ToolPromptRenderer& render, const TokenCounter& count);
```

The schema, every system message, and the final user or tool message are never dropped. An
assistant message carrying `tool_calls` drops together with the tool results that follow it.
