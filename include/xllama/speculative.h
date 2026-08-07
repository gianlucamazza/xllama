// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Phase 15 W2 — draft-free prompt-lookup helpers (#210).
// Pure functions only: no llama.h, no I/O. Host-testable without a GGUF.
// The decode-loop integration (verify batch, rollback) lives elsewhere.
#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace xllama {

// Shipping pre-gate setting (phase15-spec-pregate): k=2. Not a starting point
// for sweeps — at k=4 prompt-lookup drops below the 1.4× gate on code.
inline constexpr int kSpecDraftKDefault = 2;

// Lookup n-gram length. 2 matches short-repeat coding patterns without needing
// a long shared history before the first hit (n=12 would starve k=2 drafts).
inline constexpr int kSpecNgramDefault = 2;

// Draft up to |k_max| tokens by finding the last earlier occurrence of the
// trailing |n_gram| tokens of |history| and copying what followed that match.
//
// |history| is the full token sequence so far, with the latest accepted (or
// just-sampled) token as the final element. Empty draft means "decline" —
// the caller must fall back to plain single-token decode (the property that
// keeps open-chat regimes at 1.00× instead of paying for bad drafts).
//
// Rules (locked by tests/test_speculative.cpp):
//   * history shorter than n_gram, or n_gram/k_max < 1 → empty
//   * no earlier match → empty
//   * multiple matches → the last (most recent) earlier one wins
//   * match with no following tokens → empty
//   * draft length is min(k_max, tokens available after the match)
inline std::vector<int32_t> prompt_lookup_draft(const std::vector<int32_t>& history,
                                                int n_gram = kSpecNgramDefault,
                                                int k_max = kSpecDraftKDefault) {
    std::vector<int32_t> draft;
    if (n_gram < 1 || k_max < 1)
        return draft;
    const std::size_t n = history.size();
    const std::size_t ng = static_cast<std::size_t>(n_gram);
    if (n < ng)
        return draft;

    // Pattern = trailing n_gram tokens. Search earlier windows only: the last
    // valid start index is n - n_gram - 1 (anything at n - n_gram is ourselves).
    if (n == ng)
        return draft; // only one window in the whole history

    const int32_t* pat = history.data() + (n - ng);
    // i walks start indices of candidate windows; signed so i>=0 is natural.
    for (std::ptrdiff_t i = static_cast<std::ptrdiff_t>(n - ng) - 1; i >= 0; --i) {
        bool match = true;
        for (std::size_t k = 0; k < ng; ++k) {
            if (history[static_cast<std::size_t>(i) + k] != pat[k]) {
                match = false;
                break;
            }
        }
        if (!match)
            continue;

        const std::size_t follow = static_cast<std::size_t>(i) + ng;
        if (follow >= n)
            return draft; // match at the very end with nothing after it
        const std::size_t available = n - follow;
        const std::size_t take = std::min(static_cast<std::size_t>(k_max), available);
        draft.assign(history.begin() + static_cast<std::ptrdiff_t>(follow),
                     history.begin() + static_cast<std::ptrdiff_t>(follow + take));
        return draft;
    }
    return draft;
}

} // namespace xllama
