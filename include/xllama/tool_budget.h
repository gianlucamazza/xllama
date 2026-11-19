// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// Actionable 400 text when a tool proposal cannot fit n_ctx. WinRT-free so the
// host suite can pin the prefix, the breakdown, and param "tools" vs "messages".
#pragma once

#include <string>

namespace xllama {

inline constexpr const char* kContextLengthExceeded = "context_length_exceeded";
inline constexpr const char* kToolBudgetParamTools = "tools";
inline constexpr const char* kToolBudgetParamMessages = "messages";

struct ToolBudgetError {
    std::string message;
    // "tools" when the rendered definitions alone do not fit; otherwise "messages".
    const char* param = kToolBudgetParamMessages;
    const char* code = kContextLengthExceeded;
};

// Exact prefix that validate-api.sh and the docs grep for. Do not change.
std::string tool_prompt_too_long_prefix(int n_tokens, int n_predict, int n_ctx);

// Prefix, then how many tokens are tool definitions vs messages. When the
// definitions alone exceed the room, a hint to send fewer tools (Open WebUI:
// Capabilities → Builtin Tools off, or Function Calling → Legacy).
ToolBudgetError tool_prompt_too_long(int n_tokens, int n_tool_tokens, int n_predict, int n_ctx);

} // namespace xllama
