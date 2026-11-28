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
    // long-chat behavior. Instead, context_shift_mode selects Rebuild for them.
    const bool hybrid_like_memory_can_shift = true;
    const int n_swa = 0;
    CHECK(context_shift_allowed(hybrid_like_memory_can_shift, n_swa));
}

TEST_CASE("context_shift_mode: selects None, InPlace, or Rebuild") {
    CHECK(context_shift_mode(false, 0, false, false) == ContextShiftMode::None);
    CHECK(context_shift_mode(false, 0, true, false) == ContextShiftMode::None);
    CHECK(context_shift_mode(true, 4, true, false) == ContextShiftMode::None);
    CHECK(context_shift_mode(true, 4, false, true) == ContextShiftMode::None);
    CHECK(context_shift_mode(true, 0, false, false) == ContextShiftMode::InPlace);
    CHECK(context_shift_mode(true, 0, true, false) == ContextShiftMode::Rebuild);
    CHECK(context_shift_mode(true, 0, false, true) == ContextShiftMode::Rebuild);
    CHECK(context_shift_mode(true, 0, true, true) == ContextShiftMode::Rebuild);
}

TEST_CASE("shift_rebuild_checkpoint: picks longest checkpoint at or below keep") {
    CHECK_FALSE(shift_rebuild_checkpoint({}, 100).has_value());
    CHECK_FALSE(shift_rebuild_checkpoint({512, 1024}, 0).has_value());
    CHECK_FALSE(shift_rebuild_checkpoint({0, 512, 1024}, 256).has_value());
    CHECK(shift_rebuild_checkpoint({512, 1024, 1536}, 512) == 0u);
    CHECK(shift_rebuild_checkpoint({512, 1536, 1024}, 1200) == 2u);
    CHECK(shift_rebuild_checkpoint({0, 64, 512}, 64) == 1u);
}

TEST_CASE("shift_rebuild_capture_points: includes keep inside (kv_keep, head) without duplicates") {
    CHECK(shift_rebuild_capture_points(0, 64, 600) == std::vector<std::size_t>{64, 512, 592});
    CHECK(shift_rebuild_capture_points(64, 64, 600) == std::vector<std::size_t>{512, 592});
    CHECK(shift_rebuild_capture_points(0, 512, 600) == std::vector<std::size_t>{512, 592});
    CHECK(shift_rebuild_capture_points(0, 0, 600) == std::vector<std::size_t>{512, 592});
    CHECK(shift_rebuild_capture_points(0, 595, 600) == std::vector<std::size_t>{512, 592});
    CHECK(shift_rebuild_capture_points(0, 4, 8).empty());
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
