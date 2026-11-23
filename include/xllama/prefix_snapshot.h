// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Hybrid-cache prefix snapshots — plan when a full-prompt turn can keep the
// resident KV, rewind a tail, restore a saved post-prefill state, or clear.
//
// Pure functions, no llama.h. LlamaSession captures the blob with
// llama_state_get_data after a successful prefill (logits + memory) and
// restores it with llama_state_set_data when seq_rm refuses a tail rewind.
// Non-hybrid caches never need the blob: RewindTail is the existing #170a
// path. Host tests pass can_rewind=false as a hybrid-shaped fixture.
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
    // RestoreSnapshot may set this equal to prompt_n (logits came with the
    // snapshot, so a regenerate need not decode a dummy token).
    size_t kv_keep = 0;
};

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
    // are the last prefilled prompt, not the generated tail.
    if (snapshot_n > 0 && snapshot_is_prefix) {
        plan.action = PrefixSnapshotAction::RestoreSnapshot;
        plan.kv_keep = snapshot_n;
        return plan;
    }

    plan.action = PrefixSnapshotAction::Clear;
    return plan;
}

} // namespace xllama
