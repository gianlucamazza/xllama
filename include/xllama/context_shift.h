// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// #169 context-shift admission. Host-testable predicate used by LlamaSession.
// Combines llama.cpp's llama_memory_can_shift with xllama's SWA guard.
#pragma once

#include <algorithm>
#include <cstddef>
#include <optional>
#include <vector>

namespace xllama {

// True when a continuation overflow may evict oldest tokens (#169) instead of
// fail-fast (#173). |memory_can_shift| is llama_memory_can_shift after the
// context exists. |n_swa| is llama_model_n_swa; a sliding window makes the
// resident range not [0, kv_len) and the shift arithmetic would lie.
//
// Hybrid / recurrent caches (LFM2, granitehybrid/Mamba2) can still report
// memory_can_shift=1: llama.cpp's llama_memory_hybrid::get_can_shift returns
// the attention half, and llama_memory_recurrent::seq_add only updates the
// tail cell's pos. Front-drop seq_rm does not erase absorbed SSM/recurrent
// history. That is llama.cpp's documented approximation, not a second xllama
// gate. Do not fold is_hybrid into this predicate: LFM2 catalogue models
// already shift. Measured on host (2026-11-25, recall probe in test_session.cpp
// and test_granite_hybrid_snapshot.cpp): after shifts evicted a turn holding a
// code word, LFM2.5-1.2B, Granite-4.0-H-Tiny and attention-only
// Qwen2.5-Coder-3B all failed to recall it, so the absorbed history did not
// carry the evicted fact. One probe, not a proof (#527 / M0 E2).
inline bool context_shift_allowed(bool memory_can_shift, int n_swa) {
    return memory_can_shift && n_swa == 0;
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
