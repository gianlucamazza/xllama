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
