// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// #169 / #527: host-testable shift admission. Does not load a GGUF.

#include "xllama/context_shift.h"

#include <doctest/doctest.h>

using namespace xllama;

TEST_CASE("context_shift_allowed: SWA or cannot-shift is fail-fast") {
    CHECK_FALSE(context_shift_allowed(false, 0));
    CHECK_FALSE(context_shift_allowed(true, 1));
    CHECK_FALSE(context_shift_allowed(true, 8));
    CHECK_FALSE(context_shift_allowed(false, 8));
}

TEST_CASE("context_shift_allowed: attention can_shift and no SWA is #169") {
    CHECK(context_shift_allowed(true, 0));
}

TEST_CASE("context_shift_allowed: hybrid is not a third gate") {
    // LFM2 catalogue models and granitehybrid both report memory_can_shift=1
    // with n_swa=0. Folding is_hybrid into the predicate would change shipping
    // long-chat behavior. Document the current formula only.
    const bool hybrid_like_memory_can_shift = true;
    const int n_swa = 0;
    CHECK(context_shift_allowed(hybrid_like_memory_can_shift, n_swa));
}

TEST_CASE("turn_aligned_discard: evicts whole turns, never half a message") {
    using xllama::turn_aligned_discard;
    // keep = 2 (system head). Turns end with the two-token terminator {9, 0}.
    //                 0  1 | 2  3  9  0 | 4  5  6  9  0 | 7  8
    const std::vector<int> t{1, 1, 2, 3, 9, 0, 4, 5, 6, 9, 0, 7, 8};
    const std::vector<int> close{9, 0};
    // Boundaries after keep: discard 4 (ends at index 6) and 9 (ends at 11).
    CHECK(turn_aligned_discard(t, 2, 3, 1, close) == 4u); // smallest >= target
    CHECK(turn_aligned_discard(t, 2, 5, 1, close) == 9u);
    CHECK(turn_aligned_discard(t, 2, 10, 3, close) == 9u); // none >= target: largest >= min
    CHECK_FALSE(turn_aligned_discard(t, 2, 10, 10, close).has_value()); // nothing fits
    CHECK_FALSE(turn_aligned_discard(t, 2, 3, 1, std::vector<int>{}).has_value());
    // A terminator straddling the pinned head does not count.
    CHECK(turn_aligned_discard(std::vector<int>{9, 0, 5, 9, 0}, 1, 1, 1, close) == 4u);
}
