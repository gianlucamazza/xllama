// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#include "xllama/chat_prompt.h"
#include "xllama/thinking_split.h"

#include <doctest/doctest.h>

#include <string>
#include <vector>

using xllama::split_thinking;
using xllama::ThinkingStreamSplitter;

TEST_CASE("split_thinking separates reasoning from the answer") {
    const auto a = split_thinking("<think>\nthe user greets me\n</think>\n\nHello!");
    CHECK(a.reasoning == "the user greets me");
    CHECK(a.content == "Hello!");

    const auto none = split_thinking("  just an answer \n");
    CHECK(none.reasoning.empty());
    CHECK(none.content == "just an answer");

    const auto two = split_thinking("<think>a</think>x <think>b</think>y");
    CHECK(two.reasoning == "a\n\nb");
    CHECK(two.content == "x y");

    // The template (or a stop) ate the opener.
    const auto opened = split_thinking("still thinking</think>answer");
    CHECK(opened.reasoning == "still thinking");
    CHECK(opened.content == "answer");

    // Cut off mid-thought: no answer, the reasoning so far.
    const auto cut = split_thinking("<think>going on and on");
    CHECK(cut.reasoning == "going on and on");
    CHECK(cut.content.empty());

    // A stray closer after a balanced block is a tag, not a boundary.
    const auto stray = split_thinking("<think>a</think>answer</think>");
    CHECK(stray.reasoning == "a");
    CHECK(stray.content == "answer");

    const auto empty = split_thinking("<think>\n\n</think>\n\nok");
    CHECK(empty.reasoning.empty());
    CHECK(empty.content == "ok");
}

TEST_CASE("strip_thinking_blocks is split_thinking's content") {
    for (const std::string text :
         {"<think>r</think>a", "r</think>a", "<think>r", "a</think>", "plain", "<think>x</think>"})
        CHECK(xllama::strip_thinking_blocks(text) == split_thinking(text).content);
}

namespace {

xllama::ThinkingSplit stream(const std::string& text, size_t piece) {
    ThinkingStreamSplitter s;
    xllama::ThinkingSplit all;
    for (size_t i = 0; i < text.size(); i += piece) {
        const auto out = s.push(std::string_view(text).substr(i, piece));
        all.reasoning += out.reasoning;
        all.content += out.content;
    }
    const auto tail = s.finish();
    all.reasoning += tail.reasoning;
    all.content += tail.content;
    return all;
}

std::string trim(const std::string& s) {
    const size_t b = s.find_first_not_of(" \n");
    const size_t e = s.find_last_not_of(" \n");
    return b == std::string::npos ? std::string() : s.substr(b, e - b + 1);
}

} // namespace

TEST_CASE("ThinkingStreamSplitter matches split_thinking for any piece size") {
    const std::vector<std::string> texts{
        "<think>\nplan the answer\n</think>\n\nThe answer is 4.",
        "No reasoning here, only <b>text</b> and a < sign.",
        "<think>a</think>answer</think> tail",
        "<think>cut off mid-thought",
        "<thin and <think>real</think> ok",
    };
    for (const std::string& text : texts) {
        const auto batch = split_thinking(text);
        for (size_t piece : {1, 2, 3, 5, 8, 64}) {
            const auto s = stream(text, piece);
            CAPTURE(text);
            CAPTURE(piece);
            CHECK(trim(s.content) == batch.content);
            CHECK(trim(s.reasoning).find(batch.reasoning) != std::string::npos);
            CHECK(s.content.find("<think>") == std::string::npos);
            CHECK(s.content.find("</think>") == std::string::npos);
        }
    }
}

TEST_CASE("ThinkingStreamSplitter holds a partial tag and releases ordinary text") {
    ThinkingStreamSplitter s;
    auto out = s.push("Hello <th");
    CHECK(out.content == "Hello ");
    out = s.push("ere");
    CHECK(out.content == "<there");
    out = s.push(" end</thi");
    CHECK(out.content == " end");
    out = s.finish();
    CHECK(out.content == "</thi");
}

TEST_CASE("ChatFormat::split_output keeps what postprocess_output removes") {
    xllama::ChatFormat thinking;
    thinking.strip_thinking_content = true;
    const auto s = thinking.split_output("<think>plan</think>\n\nanswer");
    CHECK(s.reasoning == "plan");
    CHECK(s.content == "answer");
    CHECK(thinking.postprocess_output("<think>plan</think>\n\nanswer") == "answer");

    xllama::ChatFormat plain; // a non-thinking model keeps a literal tag as text
    const auto p = plain.split_output("<think>x</think>y");
    CHECK(p.reasoning.empty());
    CHECK(p.content == "<think>x</think>y");
}

TEST_CASE("thinking_spans covers each block with its tags, by split_thinking's rules") {
    using xllama::thinking_spans;
    const std::string two = "<think>a</think>x <think>b</think>y";
    const auto s = thinking_spans(two);
    REQUIRE(s.size() == 2);
    CHECK(two.substr(s[0].begin, s[0].end - s[0].begin) == "<think>a</think>");
    CHECK(two.substr(s[1].begin, s[1].end - s[1].begin) == "<think>b</think>");

    const auto opened = thinking_spans("still thinking</think>answer");
    REQUIRE(opened.size() == 1);
    CHECK(opened[0].begin == 0);
    CHECK(opened[0].end == std::string("still thinking</think>").size());

    const std::string cut = "ok <think>going on";
    const auto c = thinking_spans(cut);
    REQUIRE(c.size() == 1);
    CHECK(c[0].begin == 3);
    CHECK(c[0].end == cut.size());

    CHECK(thinking_spans("no reasoning here").empty());
}

namespace {

xllama::ReasoningTokenCounter counter_of(const std::vector<std::string>& pieces) {
    xllama::ReasoningTokenCounter c;
    for (const std::string& p : pieces)
        c.push(p);
    return c;
}

} // namespace

TEST_CASE("ReasoningTokenCounter counts the tokens inside the reasoning blocks") {
    // 5 reasoning tokens (tags included), 3 answer tokens.
    const auto c =
        counter_of({"<think>", "the", " user", " greets", "</think>", "\n\n", "Hel", "lo"});
    CHECK(c.count(true) == 5);
    // A non-thinking model: a literal tag is content.
    CHECK(c.count(false) == 0);

    // Cut off while reasoning: every token is reasoning (#585's case).
    const auto cut = counter_of({"<think>", "step", " one", " step", " two"});
    CHECK(cut.count(true) == 5);

    // A token that straddles the closer counts as reasoning.
    const auto straddle = counter_of({"<think>", "a</th", "ink>ans", "wer"});
    CHECK(straddle.count(true) == 3);

    // No reasoning at all.
    CHECK(counter_of({"just", " an", " answer"}).count(true) == 0);
    CHECK(xllama::ReasoningTokenCounter{}.count(true) == 0);
}

TEST_CASE("ReasoningTokenCounter counts a template parser's reasoning where it occurs") {
    const auto c = counter_of(
        {"<|channel|>", "anal", "ysis", "<|message|>", "plan", " it", "<|end|>", "answer"});
    CHECK(c.count(false, "plan it") == 2);
    // Inside a <think> block already counted: no double count.
    const auto t = counter_of({"<think>", "plan", "</think>", "ok"});
    CHECK(t.count(true, "plan") == 3);
    // Text the parser normalised and the reply does not hold verbatim: not counted.
    CHECK(c.count(false, "something else") == 0);
}
