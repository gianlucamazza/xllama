// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace xllama {

// The grammar a GGUF chat template returns for a tool proposal (ADR 0005),
// with its triggers already in sampler form: patterns are regexes (a template
// WORD is escaped, a full pattern anchored), tokens are vocabulary ids.
struct GrammarSpec {
    std::string gbnf;
    // Lazy: passive until a trigger matches, then the call rule, then only
    // end-of-generation. Full (required / named choice): from the first token.
    bool lazy = false;
    std::vector<std::string> trigger_patterns;
    std::vector<int32_t> trigger_tokens;
    // Text the template already put after the prompt; a full grammar consumes
    // it first, since its root starts there.
    std::string generation_prompt;
    bool empty() const {
        return gbnf.empty();
    }
};

} // namespace xllama
