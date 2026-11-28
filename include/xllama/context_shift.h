// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// #169 / #723 context-shift admission and mode. Host-testable helpers used by
// LlamaSession. Combines llama.cpp's llama_memory_can_shift with xllama's SWA
// guard and hybrid/recurrent rebuild selection.
#pragma once

#include "xllama/prefix_snapshot.h"

#include <algorithm>
#include <cstddef>
#include <optional>
#include <vector>

namespace xllama {

enum class ContextShiftMode {
    None,    // cannot shift (fail-fast #173)
    InPlace, // attention-only: seq_rm + seq_add
    Rebuild, // hybrid/recurrent: restore checkpoint <= keep or clear, then re-prefill
};

// True when a continuation overflow may evict oldest tokens (#169) instead of
// fail-fast (#173). |memory_can_shift| is llama_memory_can_shift after the
// context exists. |n_swa| is llama_model_n_swa; a sliding window makes the
// resident range not [0, kv_len) and the shift arithmetic would lie.
//
// Hybrid / recurrent caches (LFM2, granitehybrid/Mamba2) also return true here
// when their attention half can shift, so long chats stay in the shift regime
// instead of failing fast (#173). However, context_shift_mode selects Rebuild
// for them: llama_memory_recurrent::seq_rm does not erase history absorbed into
// the SSM/conv state and seq_add only moves the tail cell's pos, so in-place
// eviction would leave evicted turns in the recurrent state (#723).
inline bool context_shift_allowed(bool memory_can_shift, int n_swa) {
    return memory_can_shift && n_swa == 0;
}

inline ContextShiftMode context_shift_mode(bool memory_can_shift, int n_swa, bool is_hybrid,
                                           bool is_recurrent) {
    if (!context_shift_allowed(memory_can_shift, n_swa))
        return ContextShiftMode::None;
    return (is_hybrid || is_recurrent) ? ContextShiftMode::Rebuild : ContextShiftMode::InPlace;
}

// Index in |checkpoint_ns| of the longest saved partial-state checkpoint with
// 0 < n <= keep. Rebuilding a hybrid/recurrent shift from that checkpoint
// skips re-prefilling the pinned prefix up to n; nullopt clears from 0.
inline std::optional<std::size_t>
shift_rebuild_checkpoint(const std::vector<std::size_t>& checkpoint_ns, std::size_t keep) {
    if (keep == 0)
        return std::nullopt;
    std::optional<std::size_t> best;
    for (std::size_t i = 0; i < checkpoint_ns.size(); ++i) {
        const std::size_t n = checkpoint_ns[i];
        if (n == 0 || n > keep)
            continue;
        if (!best || n > checkpoint_ns[*best])
            best = i;
    }
    return best;
}

// Capture positions during a hybrid/recurrent shift rebuild: the standard
// grid + slacked head in (kv_keep, total_n), plus |keep| itself when it lies
// strictly in (kv_keep, head) so subsequent shifts can restore the pinned
// prefix instead of re-prefilling it from token 0.
inline std::vector<std::size_t>
shift_rebuild_capture_points(std::size_t kv_keep, std::size_t keep, std::size_t total_n,
                             std::size_t slack = kPrefixSnapshotTailSlack) {
    std::vector<std::size_t> points = prefix_checkpoint_positions(kv_keep, total_n, slack);
    if (points.empty() || keep <= kv_keep || keep >= points.back())
        return points;
    const auto it = std::lower_bound(points.begin(), points.end(), keep);
    if (it == points.end() || *it != keep)
        points.insert(it, keep);
    return points;
}

// How many resident tokens past |keep| a #169 shift evicts so that it ends on
// a turn boundary: right after a complete |turn_close| (the format's turn
// terminator, tokenized) — never inside a message or a template marker.
//
// |target| is the amortised amount (upstream evicts at least half the movable
// region, so a long chat does not shift on every turn); |minimum| is what the
// overflowing turn needs. Returns the smallest boundary that evicts at least
// |target|, else the largest that evicts at least |minimum|, else nullopt: no
// boundary fits and the caller fails fast (#173) so the full-prompt retry drops
// whole turns through fit_prompt instead.
template <typename T>
inline std::optional<std::size_t>
turn_aligned_discard(const std::vector<T>& tokens, std::size_t keep, std::size_t target,
                     std::size_t minimum, const std::vector<T>& turn_close) {
    if (turn_close.empty() || keep >= tokens.size())
        return std::nullopt;
    std::optional<std::size_t> best_below_target;
    const std::size_t p = turn_close.size();
    for (std::size_t i = keep; i + p <= tokens.size(); ++i) {
        if (!std::equal(turn_close.begin(), turn_close.end(), tokens.begin() + i))
            continue;
        const std::size_t discard = i + p - keep;
        if (discard >= target)
            return discard;
        if (discard >= minimum)
            best_below_target = discard;
    }
    return best_below_target;
}

} // namespace xllama
