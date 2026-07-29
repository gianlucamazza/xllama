// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "xllama/prompt_budget.h"

#include "xllama/routing_policy.h" // kReservedGenerationTokens

#include <algorithm>
#include <cstddef>

namespace xllama {

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
    // The reply's room, floored: a caller asking for fewer tokens than the floor
    // still gets the floor, so a low n_predict cannot be used to smuggle a
    // context-filling prompt past the budget.
    const int reserve = std::max(n_predict, kReservedGenerationTokens);
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
        return *tokens + reserve + 1 <= n_ctx;
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

} // namespace xllama
