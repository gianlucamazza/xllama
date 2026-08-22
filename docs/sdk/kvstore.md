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