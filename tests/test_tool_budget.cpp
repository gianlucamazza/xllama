// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "xllama/prompt_budget.h"
#include "xllama/routing_policy.h"
#include "xllama/tool_budget.h"

#include <string>

using namespace xllama;

TEST_CASE("tool budget error keeps the exact prompt-too-long prefix") {
    const ToolBudgetError err = tool_prompt_too_long(8123, 7500, 512, 2048);
    const std::string prefix = tool_prompt_too_long_prefix(8123, 512, 2048);
    CHECK(prefix ==
          "prompt too long: the request needs 8123 tokens plus 512 for the reply, but n_ctx is "
          "2048");
    CHECK(err.message.compare(0, prefix.size(), prefix) == 0);
    CHECK(std::string(err.code) == kContextLengthExceeded);
}

TEST_CASE("tool budget error names tool-definition vs message token counts") {
    const ToolBudgetError err = tool_prompt_too_long(1000, 800, 512, 2048);
    CHECK(err.message.find("Tool definitions use 800 tokens") != std::string::npos);
    CHECK(err.message.find("messages use 200 tokens") != std::string::npos);
}

TEST_CASE("tool budget param is tools only when the definitions alone overflow") {
    const int floor = kReservedGenerationTokens;
    const ToolBudgetError tools =
        tool_prompt_too_long(3000, 2500, floor, /*n_ctx=*/floor + 1 + 100);
    CHECK(std::string(tools.param) == kToolBudgetParamTools);
    CHECK(tools.message.find("more tool definitions than fit") != std::string::npos);
    CHECK(tools.message.find("Builtin Tools") != std::string::npos);
    CHECK(tools.message.find("Function Calling") != std::string::npos);

    const ToolBudgetError messages = tool_prompt_too_long(3000, 100, 512, 2048);
    CHECK(std::string(messages.param) == kToolBudgetParamMessages);
    CHECK(messages.message.find("more tool definitions than fit") == std::string::npos);
    CHECK(budget_fits(100, 512, 2048));
}

TEST_CASE("tool budget treats a missing tool count as messages, not tools") {
    const ToolBudgetError err = tool_prompt_too_long(3000, -1, 512, 2048);
    CHECK(std::string(err.param) == kToolBudgetParamMessages);
    CHECK(err.message.find("Tool definitions use 0 tokens") != std::string::npos);
    CHECK(err.message.find("more tool definitions than fit") == std::string::npos);
}
