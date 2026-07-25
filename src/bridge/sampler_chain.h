// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// Internal: the llama.cpp sampler chain, assembled in exactly one place.
// Needs llama.h, so it stays out of the public include/ tree.
#pragma once

#include "llama.h"
#include "xllama/sampling.h"

namespace xllama {

// Add the sampling stages to |chain| in the order that defines our behaviour.
// ORDER MATTERS and is why this is shared rather than copied: penalties must see
// raw logits, top_k narrows before top_p renormalises, and temp applies last
// before the distribution draw. run_inference (CLI, bench) and Session (GUI, LAN
// API) built two different orders before #125.
inline void add_sampler_stages(llama_sampler* chain, const SamplingConfig& sc) {
    if (sc.is_greedy()) {
        llama_sampler_chain_add(chain, llama_sampler_init_greedy());
        return;
    }
    if (sc.repetition_penalty > 0.0f) {
        llama_sampler_chain_add(chain,
                                llama_sampler_init_penalties(sampling_defaults::kPenaltyLastN,
                                                             sc.repetition_penalty, 0.0f, 0.0f));
    }
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(sc.top_k));
    llama_sampler_chain_add(chain, llama_sampler_init_top_p(sc.top_p, 1));
    llama_sampler_chain_add(chain, llama_sampler_init_temp(sc.temperature));
    llama_sampler_chain_add(chain, llama_sampler_init_dist(sc.seed));
}

} // namespace xllama
