// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "xllama/chat_prompt.h"

using namespace xllama;

TEST_CASE("qwen detection") {
    CHECK(model_is_qwen("qwen35-0.8b"));
    CHECK(model_is_qwen("Qwen3.5-0.8B-Q4_K_M.gguf"));
    CHECK_FALSE(model_is_qwen("lfm25-350m"));
    CHECK_FALSE(model_is_qwen("smollm2-360m-cpu-int4"));
}

TEST_CASE("qwen no-think generation suffix") {
    const std::string suffix = qwen_no_think_gen_suffix("qwen35-0.8b");
    CHECK_FALSE(suffix.empty());
    CHECK(suffix.find("qwen35") == std::string::npos);
    CHECK(suffix.find('\n') != std::string::npos);
    CHECK(qwen_no_think_gen_suffix("lfm25-350m").empty());
}

TEST_CASE("strip empty thinking tags") {
    const std::string block = std::string("<think>") + "\n\n" + "</think>";
    CHECK(strip_empty_thinking_tags(block + "\n\nCiao!") == "Ciao!");
    CHECK(strip_empty_thinking_tags("  \n" + block + "  \n  Risposta") == "Risposta");
    CHECK(strip_empty_thinking_tags("plain text") == "plain text");
}