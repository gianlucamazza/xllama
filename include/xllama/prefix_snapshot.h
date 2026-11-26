// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Hybrid-cache prefix snapshots — plan when a full-prompt turn can keep the
// resident KV, rewind a tail, restore a saved post-prefill state, or clear.
//
// Pure functions, no llama.h. LlamaSession captures partial-state checkpoints
// (llama_state_seq_get_data_ext with PARTIAL_ONLY: the recurrent cells only,
// as llama-server's context checkpoints) at a few points of a long prefill:
// a fixed grid (prefix_checkpoint_positions) and a slacked head just before
// the prompt end. A later prompt that shares only part of the prefix — a new
// conversation with the same system prompt and tools, an edited turn —
// restores the longest checkpoint it still shares (#601). A restore puts the
// recurrent state back and truncates the still-resident attention KV to the
// checkpoint, so it needs that attention prefix to be intact. Non-hybrid caches
// never need a checkpoint: RewindTail is the existing #170a path. Host tests
// pass can_rewind=false as a hybrid-shaped fixture.
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
    // RestoreSnapshot: index of the checkpoint to restore, in the list given.
    size_t checkpoint = 0;
};

// LocalState\hybrid-prefix-snapshots.txt: absent (-1) or any non-zero → on
// (the default). 0 opts out; hybrid sessions then full-re-prefill when
// seq_rm refuses, matching pre-snapshot behaviour.
inline constexpr int kPrefixSnapshotsKnobAbsent = -1;

// Tokens left unsaved at the end of a long hybrid prefill. A later prompt
// that diverges only in that tail can still restore the saved head. 0 keeps
// today's capture-the-whole-prompt behaviour.
inline constexpr size_t kPrefixSnapshotTailSlack = 8;

// The checkpoint grid (#601): every kPrefixCheckpointStep tokens up to
// kPrefixCheckpointDense, then doubling. A multiple of the physical ubatch
// (512, #172), so splitting a prefill there yields the same ubatches as not
// splitting it. A prefix of at least one step that two prompts share is
// reused at least by half, and whole when it ends on a grid point; the count
// grows with log2 of the context (10 points below 64k tokens).
inline constexpr size_t kPrefixCheckpointStep = 512;
inline constexpr size_t kPrefixCheckpointDense = 2048;

// Bytes the grid checkpoints of one session may hold together. The memory
// admission does not count checkpoints, so the grid stays small next to the
// console budget: a few hundred KiB each on LFM2, where every point fits; on
// a model whose recurrent state is large (Granite-4.0-H-Tiny, 55 MiB) no grid
// point fits and only the slacked head is held, as before #601.
inline constexpr size_t kPrefixCheckpointGridBytes = 32u * 1024u * 1024u;

// Whether a grid checkpoint of |blob_bytes| may join |held_grid_bytes|.
inline bool prefix_checkpoint_fits(size_t held_grid_bytes, size_t blob_bytes) {
    return blob_bytes <= kPrefixCheckpointGridBytes &&
           held_grid_bytes <= kPrefixCheckpointGridBytes - blob_bytes;
}

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

// Where one full-prompt hybrid prefill captures, ascending: the grid points
// in (kv_keep, head), then |head| itself (prefix_snapshot_capture_n). Points
// at or below kv_keep are already cached. Empty when the head is not before
// the prompt end: a short prompt, or slack 0, captures the whole prompt
// after its prefill instead.
inline std::vector<size_t> prefix_checkpoint_positions(size_t kv_keep, size_t prompt_n,
                                                       size_t slack = kPrefixSnapshotTailSlack) {
    std::vector<size_t> points;
    const size_t head = prefix_snapshot_capture_n(prompt_n, slack);
    if (head >= prompt_n || head <= kv_keep)
        return points;
    for (size_t n = kPrefixCheckpointStep; n < head;
         n = n < kPrefixCheckpointDense ? n + kPrefixCheckpointStep : n * 2) {
        if (n > kv_keep)
            points.push_back(n);
    }
    points.push_back(head);
    return points;
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
// |checkpoints| are the token counts of the saved checkpoints whose tokens are
// a prefix of this prompt, in any order.
inline PrefixSnapshotPlan plan_prefix_snapshot(size_t common, size_t resident_n, size_t prompt_n,
                                               const std::vector<size_t>& checkpoints,
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

    // Hybrid (or empty resident after a failed rewind): restore the longest
    // saved checkpoint the prompt still shares instead of prefilling from
    // scratch. Only a checkpoint strictly shorter than the prompt restores, so
    // the suffix prefills and seeds logits. A checkpoint holds no attention
    // state: the attention cells [0, n) must still be resident and equal to the
    // prompt's, i.e. common >= n.
    bool found = false;
    for (size_t i = 0; i < checkpoints.size(); ++i) {
        const size_t n = checkpoints[i];
        if (n == 0 || n >= prompt_n || n > common)
            continue;
        if (!found || n > plan.kv_keep) {
            plan.kv_keep = n;
            plan.checkpoint = i;
            found = true;
        }
    }
    if (found) {
        plan.action = PrefixSnapshotAction::RestoreSnapshot;
        return plan;
    }

    plan.action = PrefixSnapshotAction::Clear;
    plan.kv_keep = 0;
    return plan;
}

} // namespace xllama
