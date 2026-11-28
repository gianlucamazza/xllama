// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "xllama/chat_prompt.h"
#include "xllama/ggml_d3d12.h"
#include "xllama/llama_raii.h"
#include "xllama/session.h"
#include "xllama/tool_transcript.h"

#if __has_include("chat-peg-parser.h") || __has_include("llama-ext.h")
    #error "xllama exports private llama.cpp headers"
#endif
#if __has_include("ggml-backend-impl.h")
    #error "xllama exports private ggml headers"
#endif

int main() {
    using namespace xllama;
    // A consumer can bind and link the original public function signatures.
    ThinkingSplit (*split)(std::string_view) = &split_thinking;
    std::vector<TextSpan> (*spans)(std::string_view) = &thinking_spans;
    ChoiceAssembly (*assemble)(const std::string&, const std::string&,
                               const std::vector<IncomingCall>&, bool, bool, const std::string&) =
        &assemble_choice;
    const std::string raw = "<think>plan</think>answer";
    if (split(raw).content != "answer" || spans(raw).size() != 1 ||
        assemble(raw, "", {}, false, true, "").reasoning != "plan")
        return 1;
    // Keep positional aggregate initialization through the former last field.
    const ChatFormat format{ChatFormatKind::ChatML,     "", "", "",    "", "",   "",  "", "",
                            SystemStyle::DedicatedTurn, {}, "", false, "", true, true};
    if (format.reasoning_prefill() != kThinkOpenTag)
        return 1;
    return Session::backend_for({}) == Backend::LlamaCpp ? 0 : 1;
}
