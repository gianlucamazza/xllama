// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// Host-testable prefix-snapshot planner. Hybrid-shaped fixture: can_rewind=false.

#include "xllama/prefix_snapshot.h"

#include <doctest/doctest.h>

#include <cstdint>
#include <vector>

using namespace xllama;

TEST_CASE("prefix snapshot: first turn clears") {
    const std::vector<int32_t> resident;
    const std::vector<int32_t> prompt{1, 2, 3};
    const size_t common = common_kv_prefix(resident, prompt);
    CHECK(common == 0);
    const auto plan = plan_prefix_snapshot(common, resident.size(), prompt.size(),
                                           /*snapshot_n=*/0, /*snapshot_is_prefix=*/false,
                                           /*can_rewind=*/true);
    CHECK(plan.action == PrefixSnapshotAction::Clear);
    CHECK(plan.kv_keep == 0);
}

TEST_CASE("prefix snapshot: extension keeps resident (and the saved snapshot)") {
    const std::vector<int32_t> resident{1, 2, 3, 4, 5};
    const std::vector<int32_t> prompt{1, 2, 3, 4, 5, 6, 7};
    const std::vector<int32_t> snapshot{1, 2, 3};
    const size_t common = common_kv_prefix(resident, prompt);
    CHECK(common == resident.size());
    CHECK(tokens_are_prefix(snapshot, prompt));
    const auto plan = plan_prefix_snapshot(common, resident.size(), prompt.size(), snapshot.size(),
                                           true, /*can_rewind=*/false);
    CHECK(plan.action == PrefixSnapshotAction::KeepResident);
    CHECK(plan.kv_keep == resident.size());
}

TEST_CASE("prefix snapshot: non-extension restores a matching hybrid prefix") {
    const std::vector<int32_t> resident{1, 2, 3, 4, 5}; // prompt + generated tail
    const std::vector<int32_t> prompt{1, 2, 3, 8, 9};   // diverges after the saved prefix
    const std::vector<int32_t> snapshot{1, 2, 3};
    const size_t common = common_kv_prefix(resident, prompt);
    CHECK(common == 3);
    CHECK(tokens_are_prefix(snapshot, prompt));
    const auto plan = plan_prefix_snapshot(common, resident.size(), prompt.size(), snapshot.size(),
                                           true, /*can_rewind=*/false);
    CHECK(plan.action == PrefixSnapshotAction::RestoreSnapshot);
    CHECK(plan.kv_keep == snapshot.size());
}

TEST_CASE("prefix snapshot: non-extension with no matching snapshot clears") {
    const std::vector<int32_t> resident{1, 2, 3, 4, 5};
    const std::vector<int32_t> prompt{9, 10, 11};
    const std::vector<int32_t> snapshot{1, 2, 3};
    const size_t common = common_kv_prefix(resident, prompt);
    CHECK(common == 0);
    CHECK_FALSE(tokens_are_prefix(snapshot, prompt));
    const auto plan = plan_prefix_snapshot(common, resident.size(), prompt.size(), snapshot.size(),
                                           false, /*can_rewind=*/false);
    CHECK(plan.action == PrefixSnapshotAction::Clear);
    CHECK(plan.kv_keep == 0);
}

TEST_CASE("prefix snapshot: non-hybrid KV reuse still rewinds the tail") {
    const std::vector<int32_t> resident{1, 2, 3, 4, 5};
    const std::vector<int32_t> prompt{1, 2, 3, 8, 9};
    const std::vector<int32_t> snapshot{1, 2, 3};
    const size_t common = common_kv_prefix(resident, prompt);
    CHECK(common == 3);
    const auto plan = plan_prefix_snapshot(common, resident.size(), prompt.size(), snapshot.size(),
                                           true, /*can_rewind=*/true);
    CHECK(plan.action == PrefixSnapshotAction::RewindTail);
    CHECK(plan.kv_keep == common);
}

TEST_CASE("prefix snapshot: hybrid regenerate clears so the last token reseeds") {
    const std::vector<int32_t> resident{1, 2, 3, 4, 5};
    const std::vector<int32_t> prompt{1, 2, 3};
    const std::vector<int32_t> snapshot{1, 2, 3};
    const size_t common = common_kv_prefix(resident, prompt);
    CHECK(common == prompt.size() - 1);
    CHECK(tokens_are_prefix(snapshot, prompt));
    const auto plan = plan_prefix_snapshot(common, resident.size(), prompt.size(), snapshot.size(),
                                           true, /*can_rewind=*/false);
    // Restoring the full prompt would skip the last-token logit seed; hybrid
    // cannot drop that last cell after a full-state restore.
    CHECK(plan.action == PrefixSnapshotAction::Clear);
    CHECK(plan.kv_keep == 0);
}

TEST_CASE("prefix snapshot: capture_n leaves tail slack on a long prompt") {
    CHECK(prefix_snapshot_capture_n(3) == 3);
    CHECK(prefix_snapshot_capture_n(kPrefixSnapshotTailSlack) == kPrefixSnapshotTailSlack);
    CHECK(prefix_snapshot_capture_n(kPrefixSnapshotTailSlack + 1) == 1);
    CHECK(prefix_snapshot_capture_n(32) == 32 - kPrefixSnapshotTailSlack);
    CHECK(prefix_snapshot_capture_n(32, 0) == 32);
}

TEST_CASE("prefix snapshot: an exact hybrid regenerate restores the slacked checkpoint") {
    // The checkpoint sits before the tail of the same prompt; restoring it and
    // prefilling the tail is the cold head/tail computation, so no Clear.
    const std::vector<int32_t> last_full{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const std::vector<int32_t> resident{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14};
    const std::vector<int32_t>& prompt = last_full;
    const size_t snap_n = prefix_snapshot_capture_n(last_full.size());
    REQUIRE(snap_n < last_full.size());
    const std::vector<int32_t> snapshot(last_full.begin(),
                                        last_full.begin() + static_cast<std::ptrdiff_t>(snap_n));
    const auto plan =
        plan_prefix_snapshot(common_kv_prefix(resident, prompt), resident.size(), prompt.size(),
                             snapshot.size(), tokens_are_prefix(snapshot, prompt), false);
    CHECK(plan.action == PrefixSnapshotAction::RestoreSnapshot);
    CHECK(plan.kv_keep == snap_n);
}

TEST_CASE("prefix snapshot: diverging tail restores a slacked head") {
    const std::vector<int32_t> last_full{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
    const std::vector<int32_t> resident{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13};
    const size_t snap_n = prefix_snapshot_capture_n(last_full.size());
    REQUIRE(snap_n < last_full.size());
    const std::vector<int32_t> snapshot(last_full.begin(),
                                        last_full.begin() + static_cast<std::ptrdiff_t>(snap_n));
    std::vector<int32_t> prompt(snapshot);
    prompt.insert(prompt.end(), {99, 100, 101});
    CHECK_FALSE(tokens_are_prefix(last_full, prompt));
    CHECK(tokens_are_prefix(snapshot, prompt));
    const size_t common = common_kv_prefix(resident, prompt);
    const auto whole = plan_prefix_snapshot(common, resident.size(), prompt.size(),
                                            last_full.size(), tokens_are_prefix(last_full, prompt),
                                            /*can_rewind=*/false);
    CHECK(whole.action == PrefixSnapshotAction::Clear);
    const auto plan = plan_prefix_snapshot(common, resident.size(), prompt.size(), snapshot.size(),
                                           true, /*can_rewind=*/false);
    CHECK(plan.action == PrefixSnapshotAction::RestoreSnapshot);
    CHECK(plan.kv_keep == snapshot.size());
}

TEST_CASE("prefix snapshot: opt-out knob") {
    CHECK(prefix_snapshots_enabled(kPrefixSnapshotsKnobAbsent));
    CHECK(prefix_snapshots_enabled(1));
    CHECK_FALSE(prefix_snapshots_enabled(0));
}

TEST_CASE("prefix snapshot: empty snapshot is not a prefix") {
    const std::vector<int32_t> empty;
    const std::vector<int32_t> seq{1, 2};
    CHECK_FALSE(tokens_are_prefix(empty, seq));
    CHECK_FALSE(tokens_are_prefix(seq, empty));
}

TEST_CASE("prefix snapshot: a partial checkpoint needs its attention prefix still resident") {
    // The checkpoint carries only the recurrent state; the attention cells
    // [0, n) come from the resident KV. If the resident tokens already diverge
    // before n, restoring would pair the right recurrent state with the wrong
    // attention cells.
    const std::vector<int32_t> snapshot{1, 2, 3, 4, 5, 6};
    const std::vector<int32_t> prompt{1, 2, 3, 4, 5, 6, 7, 8};
    const std::vector<int32_t> resident_ok{1, 2, 3, 4, 5, 6, 9, 9};
    const std::vector<int32_t> resident_bad{1, 2, 3, 9, 9, 9, 9, 9};
    REQUIRE(tokens_are_prefix(snapshot, prompt));
    const auto ok = plan_prefix_snapshot(common_kv_prefix(resident_ok, prompt), resident_ok.size(),
                                         prompt.size(), snapshot.size(), true, false);
    CHECK(ok.action == PrefixSnapshotAction::RestoreSnapshot);
    CHECK(ok.kv_keep == snapshot.size());
    const auto bad =
        plan_prefix_snapshot(common_kv_prefix(resident_bad, prompt), resident_bad.size(),
                             prompt.size(), snapshot.size(), true, false);
    CHECK(bad.action == PrefixSnapshotAction::Clear);
}
