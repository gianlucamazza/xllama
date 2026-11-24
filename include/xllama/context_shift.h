// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// #169 context-shift admission. Host-testable predicate used by LlamaSession.
// Combines llama.cpp's llama_memory_can_shift with xllama's SWA guard.
#pragma once

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
// already shift. A granitehybrid overflow vs cold comparison is a follow-up
// measurement, not a default change (#527 / M0 E2).
inline bool context_shift_allowed(bool memory_can_shift, int n_swa) {
    return memory_can_shift && n_swa == 0;
}

} // namespace xllama
