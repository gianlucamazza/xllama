// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "xllama/tool_budget.h"

#include "xllama/prompt_budget.h"

#include <algorithm>
#include <string>

namespace xllama {

std::string tool_prompt_too_long_prefix(int n_tokens, int n_predict, int n_ctx) {
    return "prompt too long: the request needs " + std::to_string(n_tokens) + " tokens plus " +
           std::to_string(n_predict) + " for the reply, but n_ctx is " + std::to_string(n_ctx);
}

ToolBudgetError tool_prompt_too_long(int n_tokens, int n_tool_tokens, int n_predict, int n_ctx) {
    ToolBudgetError out;
    out.message = tool_prompt_too_long_prefix(n_tokens, n_predict, n_ctx);
    if (n_tool_tokens < 0) {
        out.message += " Tool-definition tokens could not be counted.";
        out.param = "";
        return out;
    }
    const int n_message_tokens = n_tokens >= 0 ? std::max(0, n_tokens - n_tool_tokens) : 0;
    out.message += ". Tool definitions use " + std::to_string(n_tool_tokens) +
                   " tokens; messages use " + std::to_string(n_message_tokens) + " tokens.";
    if (!budget_fits(n_tool_tokens, n_predict, n_ctx)) {
        out.message +=
            " The client sent more tool definitions than fit. Send fewer tools, or disable the "
            "client's built-in tools (e.g. Open WebUI: model Capabilities → Builtin Tools off, or "
            "Function Calling → Legacy).";
        out.param = kToolBudgetParamTools;
    } else {
        out.param = kToolBudgetParamMessages;
    }
    return out;
}

} // namespace xllama
