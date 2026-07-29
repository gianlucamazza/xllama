// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "xllama/prompt_budget.h"

#include "xllama/routing_policy.h" // kReservedGenerationTokens

#include <algorithm>
#include <utility>

namespace xllama {

PromptFit fit_prompt(const ChatFormat& fmt, const std::string& system, std::vector<ChatTurn> turns,
                     const std::string& user_text, int n_ctx, int n_predict,
                     const TokenCounter& count) {
    PromptFit out;
    if (!count) {
        out.prompt = fmt.render_prompt(system, turns, user_text);
        out.n_tokens = -1;
        out.fits = false;
        return out;
    }
    // The reply's room, floored: a caller asking for fewer tokens than the floor
    // still gets the floor, so a low n_predict cannot be used to smuggle a
    // context-filling prompt past the budget.
    const int reserve = std::max(n_predict, kReservedGenerationTokens);
    for (;;) {
        out.prompt = fmt.render_prompt(system, turns, user_text);
        out.n_tokens = count(out.prompt);
        // Mirrors LlamaSession::generate: kv + prompt + 1 must fit n_ctx, and the
        // reply needs `reserve` slots on top.
        if (out.n_tokens + reserve + 1 <= n_ctx) {
            out.fits = true;
            return out;
        }
        if (turns.empty()) {
            out.fits = false; // only the new message left, and it is too big
            return out;
        }
        turns.erase(turns.begin());
        ++out.dropped;
    }
}

} // namespace xllama
