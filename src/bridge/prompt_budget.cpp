// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "xllama/prompt_budget.h"

#include "xllama/routing_policy.h" // kReservedGenerationTokens

#include <algorithm>
#include <cstddef>
#include <exception>
#include <utility>

namespace xllama {

std::int64_t prompt_reply_reserve(int n_predict) {
    return std::max<std::int64_t>(n_predict, kReservedGenerationTokens) + 1;
}

bool budget_fits(int prompt_tokens, int n_predict, int n_ctx) {
    return prompt_tokens >= 0 &&
           std::int64_t{prompt_tokens} + prompt_reply_reserve(n_predict) <= n_ctx;
}

PromptFit fit_prompt(const ChatFormat& fmt, const std::string& system,
                     const std::vector<ChatTurn>& turns, const std::string& user_text, int n_ctx,
                     int n_predict, const TokenCounter& count) {
    PromptFit out;
    if (!count) {
        out.prompt = fmt.render_prompt(system, turns, user_text);
        out.n_tokens = -1;
        out.fits = false;
        return out;
    }
    // The reply's room is floored inside budget_fits: a caller asking for fewer
    // tokens than the floor still gets the floor, so a low n_predict cannot be
    // used to smuggle a context-filling prompt past the budget.
    const size_t n = turns.size();

    // Render with the first |drop| turns removed. The trailing user_text always
    // stays: a single oversized message is a user-visible error, not something to
    // silently truncate.
    auto render_dropping = [&](size_t drop) {
        const std::vector<ChatTurn> kept(turns.begin() + static_cast<std::ptrdiff_t>(drop),
                                         turns.end());
        return fmt.render_prompt(system, kept, user_text);
    };
    // Mirrors LlamaSession::generate: kv + prompt + 1 must fit n_ctx, and the reply
    // needs `reserve` slots on top.
    auto fits_at = [&](size_t drop, std::string* prompt, int* tokens) {
        *prompt = render_dropping(drop);
        *tokens = count(*prompt);
        return budget_fits(*tokens, n_predict, n_ctx);
    };

    // Dropping a turn can only shrink the prompt, so "fits" is monotone in |drop|
    // and the smallest fitting drop can be bisected. Walking one turn at a time
    // would tokenize once per dropped turn — fine for a handful, quadratic in
    // rendered bytes for a long conversation, and this runs on the turn's critical
    // path. Probe exponentially, then bisect: O(log n) tokenizations.
    if (fits_at(0, &out.prompt, &out.n_tokens)) {
        out.fits = true;
        return out;
    }
    if (n == 0) {
        // Nothing to drop: the trailing message alone is too big. out already holds
        // its render and count — the caller needs both to report the three numbers.
        out.fits = false;
        return out;
    }
    size_t lo = 0; // known NOT to fit
    size_t hi = 1; // candidate
    std::string probe_prompt;
    int probe_tokens = 0;
    bool found = false;
    while (hi <= n) {
        if (fits_at(hi, &probe_prompt, &probe_tokens)) {
            found = true;
            break;
        }
        lo = hi;
        if (hi == n)
            break;
        hi = std::min(n, hi * 2);
    }
    if (!found) {
        // Even with no history at all the prompt is too big: report it with the
        // numbers so the caller can tell the user which three they are up against.
        out.prompt = std::move(probe_prompt);
        out.n_tokens = probe_tokens;
        out.dropped = static_cast<int>(n);
        out.fits = false;
        return out;
    }
    // (lo, hi] brackets the answer: lo does not fit, hi does.
    while (hi - lo > 1) {
        const size_t mid = lo + (hi - lo) / 2;
        std::string mid_prompt;
        int mid_tokens = 0;
        if (fits_at(mid, &mid_prompt, &mid_tokens)) {
            hi = mid;
            probe_prompt = std::move(mid_prompt);
            probe_tokens = mid_tokens;
        } else {
            lo = mid;
        }
    }
    out.prompt = std::move(probe_prompt);
    out.n_tokens = probe_tokens;
    out.dropped = static_cast<int>(hi);
    out.fits = true;
    return out;
}

namespace {

struct DropGroup {
    std::vector<size_t> indexes;
};

size_t protected_tail(const std::vector<ChatMessage>& messages) {
    if (messages.empty())
        return 0;
    size_t begin = messages.size() - 1;
    if (messages.back().role != ChatRole::Tool)
        return begin;
    while (begin > 0 && messages[begin - 1].role == ChatRole::Tool)
        --begin;
    if (begin > 0 && messages[begin - 1].role == ChatRole::Assistant &&
        !messages[begin - 1].tool_calls.empty())
        --begin;
    return begin;
}

std::vector<DropGroup> drop_groups(const std::vector<ChatMessage>& messages, size_t tail_begin) {
    std::vector<DropGroup> groups;
    for (size_t i = 0; i < tail_begin; ++i) {
        const ChatMessage& message = messages[i];
        if (message.role == ChatRole::System)
            continue;
        DropGroup group;
        group.indexes.push_back(i);
        if (message.role == ChatRole::Assistant && !message.tool_calls.empty()) {
            size_t next = i + 1;
            while (next < tail_begin && messages[next].role == ChatRole::Tool) {
                group.indexes.push_back(next);
                ++next;
            }
            i = next - 1;
        }
        groups.push_back(std::move(group));
    }
    return groups;
}

struct Kept {
    std::vector<ChatMessage> messages;
    int dropped = 0;
};

Kept take_groups(const std::vector<ChatMessage>& messages, const std::vector<DropGroup>& groups,
                 size_t tail_begin, size_t drop_groups) {
    std::vector<char> drop(messages.size(), 0);
    const size_t n_drop = std::min(drop_groups, groups.size());
    for (size_t g = 0; g < n_drop; ++g) {
        for (size_t index : groups[g].indexes)
            drop[index] = 1;
    }
    // A tool whose assistant call was dropped is part of that pair.
    bool seen_assistant = false;
    bool assistant_kept = false;
    for (size_t i = 0; i < tail_begin && i < messages.size(); ++i) {
        const ChatMessage& message = messages[i];
        if (message.role == ChatRole::Assistant && !message.tool_calls.empty()) {
            seen_assistant = true;
            assistant_kept = !drop[i];
        } else if (message.role == ChatRole::Tool && seen_assistant && !assistant_kept) {
            drop[i] = 1;
        }
    }
    Kept kept;
    for (size_t i = 0; i < messages.size(); ++i) {
        if (drop[i]) {
            ++kept.dropped;
            continue;
        }
        kept.messages.push_back(messages[i]);
    }
    return kept;
}

// Tokenize the template's existing baseline only for a rejected prompt.
void count_baseline(ToolTranscriptFit& fit, const TokenCounter& count) {
    if (fit.n_tokens >= 0 && fit.rendered.prompt_without_tools) {
        try {
            fit.n_tokens_without_tools = count(*fit.rendered.prompt_without_tools);
        } catch (const std::exception&) {
            // The accepted render count is still enough to diagnose overflow.
        }
    }
}

} // namespace

ToolTranscriptFit fit_tool_transcript(const std::vector<ChatMessage>& messages,
                                      const std::vector<ToolSpec>& tools, int n_ctx, int n_predict,
                                      const ToolPromptRenderer& render, const TokenCounter& count) {
    if (!render)
        return {};

    const size_t tail = protected_tail(messages);
    const std::vector<DropGroup> groups = drop_groups(messages, tail);
    auto probe = [&](size_t n_drop) {
        Kept kept = take_groups(messages, groups, tail, n_drop);
        ToolTranscriptFit fit;
        fit.rendered = render(kept.messages, tools);
        fit.messages = std::move(kept.messages);
        fit.dropped = kept.dropped;
        if (fit.rendered.ok && count) {
            fit.n_tokens = count(fit.rendered.prompt);
            fit.fits = budget_fits(fit.n_tokens, n_predict, n_ctx);
        }
        return fit;
    };

    ToolTranscriptFit best = probe(0);
    if (best.fits || !best.rendered.ok || !count)
        return best;

    // A template can reject a suffix that starts partway through a turn.
    // Look backwards from the probe point within the unclassified interval.
    auto probe_valid_suffix = [&](size_t& index, size_t lower) {
        while (index > lower) {
            ToolTranscriptFit candidate = probe(index);
            if (candidate.rendered.ok)
                return candidate;
            --index;
        }
        return ToolTranscriptFit{};
    };

    const size_t n = groups.size();
    size_t lo = 0;
    size_t hi = std::min<size_t>(1, n);
    while (hi > lo) {
        size_t index = hi;
        ToolTranscriptFit candidate = probe_valid_suffix(index, lo);
        if (candidate.rendered.ok) {
            best = std::move(candidate);
            if (best.fits) {
                hi = index;
                break;
            }
        }
        lo = hi;
        if (hi == n)
            break;
        hi = std::min(n, hi * 2);
    }
    if (!best.fits) {
        count_baseline(best, count);
        return best;
    }

    while (hi - lo > 1) {
        const size_t mid = lo + (hi - lo) / 2;
        size_t index = mid;
        ToolTranscriptFit candidate = probe_valid_suffix(index, lo);
        if (candidate.fits) {
            hi = index;
            best = std::move(candidate);
        } else {
            lo = mid;
        }
    }
    return best;
}

} // namespace xllama
