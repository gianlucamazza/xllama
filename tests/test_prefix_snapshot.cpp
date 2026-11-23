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

TEST_CASE("prefix snapshot: hybrid regenerate restores the exact prompt") {
    const std::vector<int32_t> resident{1, 2, 3, 4, 5};
    const std::vector<int32_t> prompt{1, 2, 3};
    const std::vector<int32_t> snapshot{1, 2, 3};
    const size_t common = common_kv_prefix(resident, prompt);
    CHECK(common == prompt.size() - 1);
    CHECK(tokens_are_prefix(snapshot, prompt));
    const auto plan = plan_prefix_snapshot(common, resident.size(), prompt.size(), snapshot.size(),
                                           true, /*can_rewind=*/false);
    CHECK(plan.action == PrefixSnapshotAction::RestoreSnapshot);
    CHECK(plan.kv_keep == prompt.size()); // n_p_eval == 0; logits came with the blob
}

TEST_CASE("prefix snapshot: empty snapshot is not a prefix") {
    const std::vector<int32_t> empty;
    const std::vector<int32_t> seq{1, 2};
    CHECK_FALSE(tokens_are_prefix(empty, seq));
    CHECK_FALSE(tokens_are_prefix(seq, empty));
}
