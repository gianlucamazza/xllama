// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// Host-testable prefix-snapshot planner. Hybrid-shaped fixture: can_rewind=false.

#include "xllama/prefix_snapshot.h"

#include <doctest/doctest.h>

#include <cstdint>
#include <vector>

using namespace xllama;

// The single-checkpoint shape these cases were written for: one checkpoint of
// |snapshot_n| tokens, offered to the planner only when it is a prefix of the
// prompt, as LlamaSession filters it.
static PrefixSnapshotPlan plan_one(size_t common, size_t resident_n, size_t prompt_n,
                                   size_t snapshot_n, bool snapshot_is_prefix, bool can_rewind) {
    std::vector<size_t> checkpoints;
    if (snapshot_is_prefix && snapshot_n > 0)
        checkpoints.push_back(snapshot_n);
    return plan_prefix_snapshot(common, resident_n, prompt_n, checkpoints, can_rewind);
}

TEST_CASE("prefix snapshot: first turn clears") {
    const std::vector<int32_t> resident;
    const std::vector<int32_t> prompt{1, 2, 3};
    const size_t common = common_kv_prefix(resident, prompt);
    CHECK(common == 0);
    const auto plan = plan_one(common, resident.size(), prompt.size(),
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
    const auto plan = plan_one(common, resident.size(), prompt.size(), snapshot.size(), true,
                               /*can_rewind=*/false);
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
    const auto plan = plan_one(common, resident.size(), prompt.size(), snapshot.size(), true,
                               /*can_rewind=*/false);
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
    const auto plan = plan_one(common, resident.size(), prompt.size(), snapshot.size(), false,
                               /*can_rewind=*/false);
    CHECK(plan.action == PrefixSnapshotAction::Clear);
    CHECK(plan.kv_keep == 0);
}

TEST_CASE("prefix snapshot: non-hybrid KV reuse still rewinds the tail") {
    const std::vector<int32_t> resident{1, 2, 3, 4, 5};
    const std::vector<int32_t> prompt{1, 2, 3, 8, 9};
    const std::vector<int32_t> snapshot{1, 2, 3};
    const size_t common = common_kv_prefix(resident, prompt);
    CHECK(common == 3);
    const auto plan = plan_one(common, resident.size(), prompt.size(), snapshot.size(), true,
                               /*can_rewind=*/true);
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
    const auto plan = plan_one(common, resident.size(), prompt.size(), snapshot.size(), true,
                               /*can_rewind=*/false);
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
    const auto plan = plan_one(common_kv_prefix(resident, prompt), resident.size(), prompt.size(),
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
    const auto whole = plan_one(common, resident.size(), prompt.size(), last_full.size(),
                                tokens_are_prefix(last_full, prompt),
                                /*can_rewind=*/false);
    CHECK(whole.action == PrefixSnapshotAction::Clear);
    const auto plan = plan_one(common, resident.size(), prompt.size(), snapshot.size(), true,
                               /*can_rewind=*/false);
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
    const auto ok = plan_one(common_kv_prefix(resident_ok, prompt), resident_ok.size(),
                             prompt.size(), snapshot.size(), true, false);
    CHECK(ok.action == PrefixSnapshotAction::RestoreSnapshot);
    CHECK(ok.kv_keep == snapshot.size());
    const auto bad = plan_one(common_kv_prefix(resident_bad, prompt), resident_bad.size(),
                              prompt.size(), snapshot.size(), true, false);
    CHECK(bad.action == PrefixSnapshotAction::Clear);
}

// #601: several checkpoints of one prefill. A prompt that shares only part of
// the prefix restores the longest checkpoint it still shares.
TEST_CASE("prefix checkpoints: a new conversation with the same prefix restores the longest "
          "shared checkpoint") {
    // Last prefill: 3000 tokens, checkpoints on the grid and at the head.
    const std::vector<size_t> saved = prefix_checkpoint_positions(0, 3000);
    REQUIRE(saved == std::vector<size_t>{512, 1024, 1536, 2048, 3000 - kPrefixSnapshotTailSlack});
    // The new prompt shares 1100 tokens (system prompt and tools), then differs.
    const size_t common = 1100;
    const auto plan = plan_prefix_snapshot(common, /*resident_n=*/3050, /*prompt_n=*/2400, saved,
                                           /*can_rewind=*/false);
    CHECK(plan.action == PrefixSnapshotAction::RestoreSnapshot);
    CHECK(plan.kv_keep == 1024);
    CHECK(saved[plan.checkpoint] == 1024);
}

TEST_CASE("prefix checkpoints: common falling between two points picks the lower one") {
    const std::vector<size_t> saved{512, 1024, 1536};
    const auto plan = plan_prefix_snapshot(1535, 2000, 1800, saved, false);
    CHECK(plan.action == PrefixSnapshotAction::RestoreSnapshot);
    CHECK(plan.kv_keep == 1024);
    CHECK(plan.checkpoint == 1);
}

TEST_CASE("prefix checkpoints: order of the list does not matter") {
    const std::vector<size_t> saved{1536, 512, 1024};
    const auto plan = plan_prefix_snapshot(1600, 2000, 1800, saved, false);
    CHECK(plan.kv_keep == 1536);
    CHECK(plan.checkpoint == 0);
}

TEST_CASE("prefix checkpoints: a prompt shorter than every checkpoint, or sharing less than "
          "the first, clears") {
    const std::vector<size_t> saved{512, 1024};
    CHECK(plan_prefix_snapshot(400, 2000, 1800, saved, false).action ==
          PrefixSnapshotAction::Clear);
    CHECK(plan_prefix_snapshot(511, 2000, 512, saved, false).action == PrefixSnapshotAction::Clear);
}

TEST_CASE("prefix checkpoints: an exact regenerate still restores the slacked head") {
    const size_t prompt_n = 3000;
    const std::vector<size_t> saved = prefix_checkpoint_positions(0, prompt_n);
    // Same prompt again: common is capped at prompt_n - 1.
    const auto plan = plan_prefix_snapshot(prompt_n - 1, prompt_n + 40, prompt_n, saved, false);
    CHECK(plan.action == PrefixSnapshotAction::RestoreSnapshot);
    CHECK(plan.kv_keep == prompt_n - kPrefixSnapshotTailSlack);
}

TEST_CASE("prefix checkpoints: the grid is dense to 2048, then doubles") {
    const size_t prompt_n = 70000;
    const std::vector<size_t> points = prefix_checkpoint_positions(0, prompt_n);
    CHECK(points == std::vector<size_t>{512, 1024, 1536, 2048, 4096, 8192, 16384, 32768, 65536,
                                        prompt_n - kPrefixSnapshotTailSlack});
    // Every grid point is a multiple of the 512 ubatch (#172).
    for (size_t i = 0; i + 1 < points.size(); ++i)
        CHECK(points[i] % kPrefixCheckpointStep == 0);
}

TEST_CASE("prefix checkpoints: a head on a grid point is captured once, as the head") {
    // 1032 - 8 = 1024: the head coincides with a grid point. It is listed once,
    // last, so LlamaSession takes it as the head, which no budget refuses.
    CHECK(prefix_checkpoint_positions(0, 1024 + kPrefixSnapshotTailSlack) ==
          std::vector<size_t>{512, 1024});
}

TEST_CASE("prefix checkpoints: positions skip what is already cached") {
    CHECK(prefix_checkpoint_positions(1024, 3000) ==
          std::vector<size_t>{1536, 2048, 3000 - kPrefixSnapshotTailSlack});
    CHECK(prefix_checkpoint_positions(1100, 3000) ==
          std::vector<size_t>{1536, 2048, 3000 - kPrefixSnapshotTailSlack});
    // A short prompt, slack 0, or a head already cached: no split prefill.
    CHECK(prefix_checkpoint_positions(0, kPrefixSnapshotTailSlack).empty());
    CHECK(prefix_checkpoint_positions(0, 3000, 0).empty());
    CHECK(prefix_checkpoint_positions(2995, 3000).empty());
    // Below one step only the head is captured, as before #601.
    CHECK(prefix_checkpoint_positions(0, 300) ==
          std::vector<size_t>{300 - kPrefixSnapshotTailSlack});
}

TEST_CASE("prefix checkpoints: the grid holds a bounded number of bytes") {
    const size_t mib = 1024u * 1024u;
    CHECK(prefix_checkpoint_fits(0, 160u * 1024u));                 // LFM2.5-1.2B
    CHECK(prefix_checkpoint_fits(10 * 160u * 1024u, 160u * 1024u)); // a full LFM2 grid
    CHECK_FALSE(prefix_checkpoint_fits(0, 55 * mib));               // Granite-4.0-H-Tiny
    CHECK(prefix_checkpoint_fits(0, kPrefixCheckpointGridBytes));
    CHECK_FALSE(prefix_checkpoint_fits(1, kPrefixCheckpointGridBytes));
    CHECK_FALSE(prefix_checkpoint_fits(kPrefixCheckpointGridBytes, 1));
}
