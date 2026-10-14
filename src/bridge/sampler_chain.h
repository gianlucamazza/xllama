// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// Internal: the llama.cpp sampler chain, assembled in exactly one place.
// Needs llama.h, so it stays out of the public include/ tree.
#pragma once

#include "llama.h"
#include "xllama/grammar_spec.h"
#include "xllama/sampling.h"

#include <cctype>
#include <exception>
#include <string>
#include <vector>

namespace xllama {

// Add the sampling stages to |chain| in the order that defines our behaviour.
// ORDER MATTERS and is why this is shared rather than copied: penalties must see
// raw logits, top_k narrows before top_p renormalises, and temp applies last
// before the distribution draw. run_inference (CLI, bench) and Session (GUI, LAN
// API) built two different orders before #125.
// |vocab| is the model's vocabulary: the penalties sampler needs its size to
// bound the backend-side penalty buffer (llama.cpp b10258 moved n_vocab into
// the sampler's own construction, upstream #26520).
inline void add_sampler_stages(llama_sampler* chain, const SamplingConfig& sc,
                               const llama_vocab* vocab) {
    if (sc.is_greedy()) {
        llama_sampler_chain_add(chain, llama_sampler_init_greedy());
        return;
    }
    if (sc.repetition_penalty > 0.0f) {
        llama_sampler_chain_add(chain,
                                llama_sampler_init_penalties(llama_vocab_n_tokens(vocab),
                                                             sampling_defaults::kPenaltyLastN,
                                                             sc.repetition_penalty, 0.0f, 0.0f));
    }
    llama_sampler_chain_add(chain, llama_sampler_init_top_k(sc.top_k));
    llama_sampler_chain_add(chain, llama_sampler_init_top_p(sc.top_p, 1));
    llama_sampler_chain_add(chain, llama_sampler_init_temp(sc.temperature));
    llama_sampler_chain_add(chain, llama_sampler_init_dist(sc.seed));
}

// The grammar stage for a tool proposal (ADR 0005), built as common/sampling.cpp
// builds llama-server's: lazy with its trigger patterns and tokens, or full.
// Kept outside the chain: decode_loop checks one token against it and applies
// it to the whole vocabulary only on rejection. Returns nullptr when the
// grammar does not parse; the caller fails the generation, it never falls back.
inline llama_sampler* make_grammar_sampler(const llama_vocab* vocab, const GrammarSpec& spec) {
    if (spec.empty())
        return nullptr;
    if (!spec.lazy)
        return llama_sampler_init_grammar(vocab, spec.gbnf.c_str(), "root");
    std::vector<const char*> patterns;
    patterns.reserve(spec.trigger_patterns.size());
    for (const std::string& p : spec.trigger_patterns)
        patterns.push_back(p.c_str());
    std::vector<llama_token> tokens(spec.trigger_tokens.begin(), spec.trigger_tokens.end());
    return llama_sampler_init_grammar_lazy_patterns(vocab, spec.gbnf.c_str(), "root",
                                                    patterns.data(), patterns.size(), tokens.data(),
                                                    tokens.size());
}

// A full grammar's root starts at the template's generation prompt, which is
// already in the prompt, so it consumes those tokens first (common/sampling.cpp
// does the same). Returns false when the grammar rejects them.
inline bool prefill_grammar(llama_sampler* grammar, const llama_vocab* vocab,
                            const std::string& generation_prompt) {
    if (generation_prompt.empty())
        return true;
    const int32_t n =
        -llama_tokenize(vocab, generation_prompt.c_str(),
                        static_cast<int32_t>(generation_prompt.size()), nullptr, 0, false, true);
    std::vector<llama_token> tokens(static_cast<size_t>(n > 0 ? n : 0));
    if (n <= 0 || llama_tokenize(vocab, generation_prompt.c_str(),
                                 static_cast<int32_t>(generation_prompt.size()), tokens.data(), n,
                                 false, true) != n)
        return n == 0;
    try {
        for (size_t i = 0; i < tokens.size(); ++i) {
            char piece[256] = {};
            const int len =
                llama_token_to_piece(vocab, tokens[i], piece, sizeof(piece) - 1, 0, true);
            // Some tokenizers put a space before the first special token.
            if (i == 0 && len > 0 && std::isspace(static_cast<unsigned char>(piece[0])) &&
                !std::isspace(static_cast<unsigned char>(generation_prompt[0])))
                continue;
            llama_sampler_accept(grammar, tokens[i]);
        }
    } catch (const std::exception&) {
        return false;
    }
    return true;
}

} // namespace xllama
