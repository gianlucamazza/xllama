// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "xllama/tool_budget.h"

#include "xllama/prompt_budget.h"

#include <algorithm>
#include <string>

namespace xllama {

ToolBudgetError tool_prompt_too_long(int n_tokens, int n_tokens_without_tools, int n_predict,
                                     int n_ctx) {
    ToolBudgetError out;
    out.message = prompt_too_long_message("request", n_tokens, n_predict, n_ctx);
    out.message += ". " + std::to_string(prompt_reply_reserve(n_predict)) +
                   " tokens are reserved for generation.";
    if (!budget_fits(0, n_predict, n_ctx)) {
        out.message += " The reply reserve exceeds the context. Use a larger context";
        out.message += budget_fits(0, 0, n_ctx) ? " or lower max_tokens." : ".";
        return out;
    }
    if (n_tokens < 0 || n_tokens_without_tools < 0) {
        out.message += " Tool-definition tokens could not be counted.";
        return out;
    }
    const int tool_delta = n_tokens - n_tokens_without_tools;
    out.message +=
        " Without tools the prompt uses " + std::to_string(n_tokens_without_tools) + " tokens.";
    if (tool_delta > 0) {
        out.message += " Tool definitions add " + std::to_string(tool_delta) + " tokens.";
    }

    const bool baseline_fits = budget_fits(n_tokens_without_tools, n_predict, n_ctx);
    const bool delta_fits = budget_fits(std::max(0, tool_delta), n_predict, n_ctx);
    if (baseline_fits && !delta_fits) {
        out.param = kToolBudgetParamTools;
        out.message += " The client sent more tool definitions than fit. Send fewer tools, shorten "
                       "their definitions or disable unused built-in tools in the client.";
    } else if (!baseline_fits && delta_fits) {
        out.param = kToolBudgetParamMessages;
        out.message += tool_delta < 0 ? " Shorten the messages."
                                      : " The messages exceed the budget without tools.";
    } else if (!baseline_fits) {
        out.message += " Shorten both the messages and the tool definitions or use a larger "
                       "context.";
    } else {
        out.message += " Shorten the messages or reduce the tool definitions.";
    }
    return out;
}

} // namespace xllama
