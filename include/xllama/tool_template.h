// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Render and parse a tool-proposal turn through the resident GGUF chat
// template. The implementation is the only translation unit that includes
// llama.cpp `common/chat.h`. A null model with a Jinja override is the host
// test seam; production passes Session::resident_llama_model().
#pragma once

#include "xllama/grammar_spec.h"
#include "xllama/tool_transcript.h"

#include <string>
#include <vector>

namespace xllama {

struct ToolTemplateResult {
    bool ok = false;
    std::string prompt;
    std::string error;
    std::vector<std::string> stop_sequences;
    // Special tokens the template's parser reads (LFM2 <|tool_call_start|>,
    // ...). The decode must emit them as text or the parser never sees a call.
    std::vector<std::string> preserved_tokens;
    // The template's grammar for this request (ADR 0005); empty when the
    // template has none for it.
    GrammarSpec grammar;
};

struct ToolParseResult {
    // The template's parser rejected the output. The caller returns the raw
    // generation as text; this is not an HTTP error and not a second sample.
    bool parser_threw = false;
    std::string content;
    std::vector<IncomingCall> calls;
};

ToolTemplateResult render_tool_prompt(const void* llama_model, const std::string& jinja_override,
                                      const std::vector<ChatMessage>& messages,
                                      const std::vector<ToolSpec>& tools, ToolChoice choice,
                                      const std::string& forced_name);

ToolParseResult parse_tool_generation(const void* llama_model, const std::string& jinja_override,
                                      const std::vector<ChatMessage>& messages,
                                      const std::vector<ToolSpec>& tools, ToolChoice choice,
                                      const std::string& forced_name, const std::string& raw_text,
                                      bool hit_cap);

} // namespace xllama
