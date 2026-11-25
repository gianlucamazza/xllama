// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Hybrid-cache prefix snapshots — plan when a full-prompt turn can keep the
// resident KV, rewind a tail, restore a saved post-prefill state, or clear.
//
// Pure functions, no llama.h. LlamaSession captures a partial-state
// checkpoint (llama_state_seq_get_data_ext with PARTIAL_ONLY: the recurrent
// cells only, as llama-server's context checkpoints) after the prompt head,
// leaving a short tail to prefill so a later prompt that diverges only there
// can restore. A restore puts the recurrent state back and truncates the
// still-resident attention KV to the checkpoint, so it needs that attention
// prefix to be intact. Non-hybrid caches never need the blob:
// RewindTail is the existing #170a path. Host tests pass can_rewind=false
// as a hybrid-shaped fixture.
#pragma once

#include <algorithm>
#include <cstddef>
#include <vector>

namespace xllama {

enum class PrefixSnapshotAction {
    Clear,           // no usable resident or snapshot; full prefill
    KeepResident,    // new prompt extends resident KV (including generated)
    RewindTail,      // seq_rm can drop the divergent tail (non-hybrid)
    RestoreSnapshot, // seq_rm refused; saved prefix matches the new prompt
};

struct PrefixSnapshotPlan {
    PrefixSnapshotAction action = PrefixSnapshotAction::Clear;
    // Tokens already in cache after applying action. Prefill prompt[kv_keep:].
    // RestoreSnapshot keeps kv_keep < prompt_n so the last token always
    // prefills and reseeds logits (#170a). An exact regenerate restores too:
    // checkpoint plus tail is the same computation as the cold head/tail
    // prefill, so it reproduces the cold output.
    size_t kv_keep = 0;
};

// LocalState\hybrid-prefix-snapshots.txt: absent (-1) or any non-zero → on
// (the default). 0 opts out; hybrid sessions then full-re-prefill when
// seq_rm refuses, matching pre-snapshot behaviour.
inline constexpr int kPrefixSnapshotsKnobAbsent = -1;

// Tokens left unsaved at the end of a long hybrid prefill. A later prompt
// that diverges only in that tail can still restore the saved head. 0 keeps
// today's capture-the-whole-prompt behaviour.
inline constexpr size_t kPrefixSnapshotTailSlack = 8;

inline bool prefix_snapshots_enabled(int knob) {
    return knob != 0;
}

// How many leading tokens of a full prompt the hybrid snapshot should cover.
// A short prompt (or slack 0) captures the whole sequence — today's behaviour.
inline size_t prefix_snapshot_capture_n(size_t prompt_n, size_t slack = kPrefixSnapshotTailSlack) {
    if (slack == 0 || prompt_n <= slack)
        return prompt_n;
    return prompt_n - slack;
}

// Shared leading tokens, capped at prompt.size()-1 so a rewind path always
// prefills at least the last token (its logits seed the first sample).
template <typename T>
inline size_t common_kv_prefix(const std::vector<T>& resident, const std::vector<T>& prompt) {
    if (prompt.empty())
        return 0;
    const size_t max_keep = prompt.size() - 1;
    size_t n = 0;
    while (n < resident.size() && n < max_keep && resident[n] == prompt[n])
        ++n;
    return n;
}

template <typename T>
inline bool tokens_are_prefix(const std::vector<T>& prefix, const std::vector<T>& seq) {
    if (prefix.empty() || prefix.size() > seq.size())
        return false;
    return std::equal(prefix.begin(), prefix.end(), seq.begin());
}

// |common| is common_kv_prefix(resident, prompt).
// |can_rewind| is true when llama_memory_seq_rm of the divergent tail would
// succeed (or no tail needs removing). Hybrid LFM2 caches pass false after a
// refused seq_rm; tests pass false as the hybrid-shaped fixture.
inline PrefixSnapshotPlan plan_prefix_snapshot(size_t common, size_t resident_n, size_t prompt_n,
                                               size_t snapshot_n, bool snapshot_is_prefix,
                                               bool can_rewind) {
    PrefixSnapshotPlan plan;
    if (prompt_n == 0)
        return plan;

    // Pure extension of the resident sequence: keep generated tokens, do not
    // restore (that would drop the tail the new prompt still needs).
    if (resident_n > 0 && common >= resident_n) {
        plan.action = PrefixSnapshotAction::KeepResident;
        plan.kv_keep = resident_n;
        return plan;
    }

    // Divergent tail on a cache that can rewind — existing #170a behaviour.
    if (resident_n > 0 && common > 0 && can_rewind) {
        plan.action = PrefixSnapshotAction::RewindTail;
        plan.kv_keep = common;
        return plan;
    }

    // Hybrid (or empty resident after a failed rewind): restore the saved
    // post-prefill prefix instead of prefilling from scratch. Snapshot tokens
    // are a prefix of the last prefilled prompt (possibly a few tokens short
    // of the end), not the generated tail. Only restore when the new prompt
    // is strictly longer — then the suffix prefills and seeds logits.
    // The checkpoint holds no attention state: the attention cells [0, n) must
    // still be resident and equal to the prompt's, i.e. common >= n.
    if (snapshot_n > 0 && snapshot_is_prefix && snapshot_n < prompt_n && common >= snapshot_n) {
        plan.action = PrefixSnapshotAction::RestoreSnapshot;
        plan.kv_keep = snapshot_n;
        return plan;
    }

    plan.action = PrefixSnapshotAction::Clear;
    return plan;
}

} // namespace xllama
