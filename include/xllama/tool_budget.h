// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// Actionable 400 text when a tool proposal cannot fit n_ctx. WinRT-free so the
// host suite can pin the prefix, the breakdown, and param "tools" vs "messages"
// (empty when attribution is inconclusive).
#pragma once

#include <string>

namespace xllama {

inline constexpr const char* kContextLengthExceeded = "context_length_exceeded";
inline constexpr const char* kToolBudgetParamTools = "tools";
inline constexpr const char* kToolBudgetParamMessages = "messages";

struct ToolBudgetError {
    std::string message;
    // Set only when exactly one of the baseline and the tool delta exceeds the budget.
    const char* param = "";
    const char* code = kContextLengthExceeded;
};

// The baseline is the same transcript rendered without tools. A negative
// baseline leaves the contribution and parameter unknown.
ToolBudgetError tool_prompt_too_long(int n_tokens, int n_tokens_without_tools, int n_predict,
                                     int n_ctx);

} // namespace xllama
