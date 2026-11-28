// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// #648: per-request reasoning control. The policy table (which model can
// switch which way), the request parser (reasoning_effort and
// chat_template_kwargs.enable_thinking), and what plain chat and a tool
// proposal render for the effective decision, against the real templates.
#include "xllama/chat_prompt.h"
#include "xllama/reasoning_control.h"
#include "xllama/session.h"
#include "xllama/thinking_split.h"
#include "xllama/tool_request.h"
#include "xllama/tool_template.h"

#include <doctest/doctest.h>

#include <cstdlib>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

using namespace xllama;

namespace {

std::string read_template(const char* name) {
    std::ifstream file(std::string(LLAMA_TEMPLATES_DIR) + "/" + name);
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}

bool ends_with(const std::string& text, const std::string& suffix) {
    return text.size() >= suffix.size() &&
           text.compare(text.size() - suffix.size(), suffix.size(), suffix) == 0;
}

ReasoningFields effort(ReasoningSwitch s) {
    ReasoningFields f;
    f.effort = s;
    return f;
}

std::string plain_prompt(const std::string& model, bool thinking) {
    return chat_format_for(model, thinking).render_prompt("sys", {{"hi", "hello"}}, "ping please");
}

} // namespace

TEST_CASE("reasoning control: which models a request can switch") {
    struct Row {
        const char* id;
        bool reasons; // model_is_thinking: the default
        bool control; // model_supports_reasoning_control
    };
    const Row rows[] = {
        // Hybrid templates with a thinking and a no-think path.
        {"qwen3-4b", false, true},
        {"gemma4-e2b", false, true},
        {"gemma-4-E2B-it-Q3_K_S.gguf", false, true},
        {"qwen3-1.7b", false, true},
        {"Qwen3-4B-Q4_K_M.gguf", false, true},
        {"qwen35-0.8b", false, true},
        {"minicpm5-1b-Q4_K_M.gguf", false, true},
        {"smollm3-3b", false, true},
        // Thinking-only: no off switch.
        {"lfm25-1.2b-thinking", true, false},
        {"lfm25-8b-a1b", true, false},
        {"qwen3-4b-thinking-2507", true, false},
        // No thinking path.
        {"qwen3-4b-instruct-2507", false, false},
        {"Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf", false, false},
        {"lfm25-350m", false, false},
        {"qwen25-coder-1.5b", false, false},
        {"llama32-3b", false, false},
        {"gemma4-e4b", false, false}, // different checkpoint/template
        {"smollm2-360m-cpu-int4", false, false},
        {"embed-bge-m3", false, false},
    };
    for (const Row& row : rows) {
        CAPTURE(row.id);
        CHECK(model_is_thinking(row.id) == row.reasons);
        CHECK(model_supports_reasoning_control(row.id) == row.control);
    }
}

TEST_CASE("reasoning control: decision per model family and request") {
    using S = ReasoningSwitch;
    const char* controllable[] = {"qwen3-4b", "qwen35-0.8b", "minicpm5-1b", "smollm3-3b",
                                  "gemma4-e2b"};
    for (const char* id : controllable) {
        CAPTURE(id);
        CHECK_FALSE(decide_reasoning(id, ReasoningFields{}).thinking); // default: off
        CHECK_FALSE(decide_reasoning(id, effort(S::Off)).thinking);
        const ReasoningDecision on = decide_reasoning(id, effort(S::On));
        CHECK(on.error.empty());
        CHECK(on.thinking);
    }

    // Thinking-only: restating is fine, turning it off is refused.
    for (const char* id : {"lfm25-1.2b-thinking", "lfm25-8b-a1b"}) {
        CAPTURE(id);
        CHECK(decide_reasoning(id, ReasoningFields{}).thinking);
        CHECK(decide_reasoning(id, effort(S::On)).thinking);
        const ReasoningDecision off = decide_reasoning(id, effort(S::Off));
        CHECK(std::string(off.code) == kReasoningUnsupportedCode);
        CHECK(std::string(off.param) == "reasoning_effort");
        CHECK(off.error.find("always reasons") != std::string::npos);
        CHECK(off.error.find(id) != std::string::npos);
    }

    // No reasoning: off restates, on is refused.
    for (const char* id : {"lfm25-350m", "gemma3-270m", "qwen25-coder-1.5b"}) {
        CAPTURE(id);
        CHECK(decide_reasoning(id, effort(S::Off)).error.empty());
        ReasoningFields on;
        on.enable_thinking = S::On;
        const ReasoningDecision refused = decide_reasoning(id, on);
        CHECK(std::string(refused.code) == kReasoningUnsupportedCode);
        CHECK(std::string(refused.param) == "chat_template_kwargs.enable_thinking");
        CHECK(refused.error.find("does not reason") != std::string::npos);
    }
}

TEST_CASE("reasoning control: request parser, both spellings, conflicts and wrong types") {
    const auto decide = [](const char* model, const std::string& extra) {
        return decide_reasoning(
            model,
            read_reasoning_fields(std::string(R"({"model":"m","messages":[])") + extra + "}"));
    };
    CHECK(reasoning_effort_switch("none") == ReasoningSwitch::Off);
    for (const char* level : {"low", "medium", "high"})
        CHECK(reasoning_effort_switch(level) == ReasoningSwitch::On);
    for (const char* other : {"minimal", "xhigh", "None", ""})
        CHECK(reasoning_effort_switch(other) == ReasoningSwitch::Invalid);

    CHECK(decide("qwen3-4b", R"(,"reasoning_effort":"high")").thinking);
    CHECK_FALSE(decide("qwen3-4b", R"(,"reasoning_effort":"none")").thinking);
    CHECK(decide("qwen3-4b", R"(,"chat_template_kwargs":{"enable_thinking":true})").thinking);
    CHECK_FALSE(
        decide("qwen3-4b", R"(,"chat_template_kwargs":{"enable_thinking":false})").thinking);
    // null is "not given".
    CHECK(decide("lfm25-8b-a1b", R"(,"reasoning_effort":null)").thinking);
    CHECK(decide("lfm25-8b-a1b", R"(,"chat_template_kwargs":null)").error.empty());
    CHECK(decide("lfm25-8b-a1b", R"(,"chat_template_kwargs":{"enable_thinking":null})")
              .error.empty());
    // Both spellings must agree.
    CHECK(decide("qwen3-4b",
                 R"(,"reasoning_effort":"low","chat_template_kwargs":{"enable_thinking":true})")
              .thinking);
    const ReasoningDecision conflict =
        decide("qwen3-4b",
               R"(,"reasoning_effort":"none","chat_template_kwargs":{"enable_thinking":true})");
    CHECK(conflict.error == "reasoning_effort and chat_template_kwargs.enable_thinking disagree");
    CHECK(std::string(conflict.code) == "invalid_value");

    struct Bad {
        const char* extra;
        const char* param;
    };
    const Bad bad[] = {
        {R"(,"reasoning_effort":"max")", "reasoning_effort"},
        {R"(,"reasoning_effort":true)", "reasoning_effort"},
        {R"(,"reasoning_effort":{"effort":"low"})", "reasoning_effort"},
        {R"(,"chat_template_kwargs":true)", "chat_template_kwargs"},
        {R"(,"chat_template_kwargs":[])", "chat_template_kwargs"},
        {R"(,"chat_template_kwargs":{"enable_thinking":"yes"})",
         "chat_template_kwargs.enable_thinking"},
        {R"(,"chat_template_kwargs":{"enable_thinking":1})",
         "chat_template_kwargs.enable_thinking"},
        // A template variable xllama would silently drop.
        {R"(,"chat_template_kwargs":{"preserve_thinking":true})", "chat_template_kwargs"},
    };
    for (const Bad& b : bad) {
        CAPTURE(b.extra);
        const ReasoningDecision d = decide("qwen3-4b", b.extra);
        CHECK_FALSE(d.error.empty());
        CHECK(std::string(d.code) == "invalid_value");
        CHECK(std::string(d.param) == b.param);
    }
    // A body that is not an object has no reasoning fields; the handler reports it.
    CHECK(read_reasoning_fields("not json").effort == ReasoningSwitch::Absent);
}

TEST_CASE("reasoning control: plain chat renders the effective decision") {
    const std::string empty_think = "<think>\n\n</think>\n\n";
    SUBCASE("Qwen3: off renders the empty think block, on leaves <think> to the model") {
        CHECK(ends_with(plain_prompt("qwen3-4b", false), "<|im_start|>assistant\n" + empty_think));
        const ChatFormat on = chat_format_for("qwen3-4b", true);
        CHECK(ends_with(plain_prompt("qwen3-4b", true), "<|im_start|>assistant\n"));
        CHECK(on.strip_thinking_content);
        CHECK_FALSE(on.reasoning_opened);
        // The default is the model's own behaviour.
        CHECK(chat_format_for("qwen3-4b").gen_suffix ==
              chat_format_for("qwen3-4b", false).gen_suffix);
        const ThinkingSplit split = on.split_output("<think>plan</think>answer");
        CHECK(split.reasoning == "plan");
        CHECK(split.content == "answer");
    }
    SUBCASE("Qwen3.5 and MiniCPM5 open <think> in the prompt when on") {
        for (const char* id : {"qwen35-0.8b", "minicpm5-1b-Q4_K_M.gguf"}) {
            CAPTURE(id);
            CHECK(ends_with(plain_prompt(id, false), "assistant\n" + empty_think));
            const ChatFormat on = chat_format_for(id, true);
            CHECK(ends_with(plain_prompt(id, true), "<|im_start|>assistant\n<think>\n"));
            REQUIRE(on.reasoning_opened);
            CHECK(on.reasoning_prefill() == "<think>");
            // The reply starts inside the block; one cut off mid-thought is reasoning.
            ThinkingSplit split = on.split_output("plan</think>answer");
            CHECK(split.reasoning == "plan");
            CHECK(split.content == "answer");
            split = on.split_output("still planning");
            CHECK(split.reasoning == "still planning");
            CHECK(split.content.empty());
            // Streamed and counted the same way.
            ThinkingStreamSplitter stream(on.reasoning_prefill());
            ThinkingSplit streamed = stream.push("plan</thi");
            const ThinkingSplit rest = stream.push("nk>answer");
            streamed.reasoning += rest.reasoning;
            streamed.content += rest.content;
            CHECK(streamed.reasoning == "plan");
            CHECK(streamed.content == "answer");
            ReasoningTokenCounter counter(on.reasoning_prefill());
            for (const char* piece : {"pl", "an", "</think>", "ans", "wer"})
                counter.push(piece);
            CHECK(counter.count(true) == 3);
        }
        CHECK(chat_format_for("minicpm5-1b", true).bos == "<s>");
    }
    SUBCASE("SmolLM3: the template's /think mode, golden from its Jinja") {
        // HuggingFaceTB-SmolLM3-3B.jinja: "Reasoning Mode: /think", no empty
        // think block in past assistant turns or in the generation prompt.
        const std::string on = plain_prompt("smollm3-3b", true);
        CHECK(on.find("Reasoning Mode: /think\n") != std::string::npos);
        CHECK(on.find("<|im_start|>assistant\nhello<|im_end|>\n") != std::string::npos);
        CHECK(ends_with(on, "<|im_start|>assistant\n"));
        CHECK(on.find("</think>") == std::string::npos);
        const std::string off = plain_prompt("smollm3-3b", false);
        CHECK(off.find("Reasoning Mode: /no_think\n") != std::string::npos);
        CHECK(off.find("<|im_start|>assistant\n<think>\n\n</think>\nhello<|im_end|>\n") !=
              std::string::npos);
        CHECK(ends_with(off, "<|im_start|>assistant\n<think>\n\n</think>\n"));
        CHECK(chat_format_for("smollm3-3b", true).strip_thinking_content);
    }
    SUBCASE("a template without a switch renders the same either way") {
        CHECK(plain_prompt("lfm25-350m", true) == plain_prompt("lfm25-350m", false));
        CHECK(plain_prompt("gemma3-270m", true) == plain_prompt("gemma3-270m", false));
    }
}

TEST_CASE("reasoning control: a tool proposal renders the same decision as plain chat") {
    std::vector<ChatMessage> msgs(1);
    msgs[0].role = ChatRole::User;
    msgs[0].content = "ping please";
    ToolSpec ping;
    ping.name = "ping";
    ping.description = "reply";
    ping.parameters_json = R"({"type":"object","properties":{"text":{"type":"string"}}})";
    const auto tool_prompt = [&](const std::string& jinja, bool thinking) {
        const ToolTemplateResult r =
            render_tool_prompt(nullptr, thinking, jinja, msgs, {ping}, ToolChoice::Auto, "");
        REQUIRE_MESSAGE(r.ok, r.error);
        return r.prompt;
    };
    struct Case {
        const char* jinja;
        const char* id;
    };
    for (const Case& c :
         {Case{"Qwen-Qwen3-0.6B.jinja", "qwen3-4b"}, Case{"Qwen3.5-4B.jinja", "qwen35-0.8b"},
          Case{"openbmb-MiniCPM5-1B.jinja", "minicpm5-1b"}}) {
        CAPTURE(c.jinja);
        const std::string jinja = read_template(c.jinja);
        REQUIRE_FALSE(jinja.empty());
        for (const bool thinking : {false, true}) {
            CAPTURE(thinking);
            // The request decides; the generation prompt ends where plain chat's does.
            const ReasoningDecision d = decide_reasoning(
                c.id, effort(thinking ? ReasoningSwitch::On : ReasoningSwitch::Off));
            REQUIRE(d.error.empty());
            CHECK(ends_with(tool_prompt(jinja, d.thinking),
                            "<|im_start|>assistant\n" +
                                chat_format_for(c.id, d.thinking).gen_suffix));
        }
    }
    // The parser follows the same value: reasoning is split off with it on.
    const std::string qwen = read_template("Qwen-Qwen3-0.6B.jinja");
    const ToolParseResult parsed = parse_tool_generation(
        nullptr, true, qwen, msgs, {ping}, ToolChoice::Auto, "", "<think>plan</think>pong", false);
    CHECK_FALSE(parsed.parser_threw);
    CHECK(parsed.reasoning == "plan");
    CHECK(parsed.content == "pong");

    // LFM2.5 Thinking cannot be turned off: the request is refused before any
    // render, and its default keeps reasoning.
    const ReasoningDecision lfm = decide_reasoning("lfm25-8b-a1b", effort(ReasoningSwitch::Off));
    CHECK(std::string(lfm.code) == kReasoningUnsupportedCode);
    const std::string a1b = read_template("LFM2.5-8B-A1B.jinja");
    REQUIRE_FALSE(a1b.empty());
    CHECK_FALSE(ends_with(tool_prompt(a1b, decide_reasoning("lfm25-8b-a1b", {}).thinking),
                          "<think>\n\n</think>\n\n"));
}

TEST_CASE("reasoning control: Gemma 4 E2B thought channel is request-local") {
    const auto on = chat_format_for("gemma4-e2b", true);
    const auto off = chat_format_for("gemma4-e2b");
    CHECK(on.render_system_prefix("sys") == "<|turn>system\n<|think|>\nsys<turn|>\n");
    CHECK(off.render_system_prefix("sys") == "<|turn>system\nsys<turn|>\n");
    CHECK(on.render_delta("next", true) == off.render_delta("next", true));
    CHECK(on.reasoning_prefill().empty());
    const std::string raw = "<|channel>thought\nplan<channel|>answer";
    const auto buffered = on.split_output(raw);
    CHECK(buffered.content == "answer");
    CHECK(buffered.reasoning == "plan");
    CHECK(on.split_output("<|channel>thought\npartial").content.empty());
    CHECK(on.split_output("<think>literal</think>").content == "<think>literal</think>");
    CHECK(chat_format_for("qwen3-4b", true).split_output(raw).content == raw);
    for (size_t boundary = 0; boundary <= raw.size(); ++boundary) {
        ThinkingStreamSplitter stream(on.reasoning_prefill(), on.thinking_markup);
        const auto first = stream.push(std::string_view(raw).substr(0, boundary));
        const auto second = stream.push(std::string_view(raw).substr(boundary));
        const auto tail = stream.finish();
        CHECK(first.content + second.content + tail.content == buffered.content);
        CHECK(first.reasoning + second.reasoning + tail.reasoning == buffered.reasoning);
    }
    ReasoningTokenCounter counter(on.reasoning_prefill(), on.thinking_markup);
    for (const auto* piece : {"<|channel>", "thought", "\n", "plan", "<channel|>", "answer"})
        counter.push(piece);
    CHECK(counter.count(true) == 5);
    const auto assembled =
        assemble_choice(raw, "answer", {}, false, true, "plan", on.thinking_markup);
    CHECK(assembled.content == "answer");
    CHECK(assembled.reasoning == "plan");
    const auto continuation =
        thinking_prefill_at_end("response<|channel>thought\n", on.thinking_markup);
    REQUIRE_FALSE(continuation.empty());
    const auto cut =
        assemble_choice("partial", "", {}, true, true, "", on.thinking_markup, continuation);
    CHECK(cut.content.empty());
    CHECK(cut.reasoning == "partial");
    ReasoningTokenCounter resumed(continuation, on.thinking_markup);
    resumed.push("partial");
    CHECK(resumed.count(true) == 1);
}

TEST_CASE("reasoning control: Gemma 4 E2B catalogue template renders plain and tool requests") {
    const char* path = std::getenv("XLLAMA_TEST_GEMMA4_GGUF");
    if (!path || !*path)
        return;
    const std::string jinja = gguf_chat_template(path);
    REQUIRE_FALSE(jinja.empty());
    std::vector<ChatMessage> messages(4);
    messages[0].role = ChatRole::System;
    messages[0].content = "sys";
    messages[1].role = ChatRole::User;
    messages[1].content = "hi";
    messages[2].role = ChatRole::Assistant;
    messages[2].content = "hello";
    messages[3].role = ChatRole::User;
    messages[3].content = "ping please";
    ToolSpec ping;
    ping.name = "ping";
    ping.parameters_json = R"({"type":"object","properties":{}})";
    for (const bool on : {false, true}) {
        const auto decision =
            decide_reasoning("gemma4-e2b", effort(on ? ReasoningSwitch::On : ReasoningSwitch::Off));
        REQUIRE(decision.error.empty());
        const auto plain = render_tool_prompt(nullptr, decision.thinking, jinja, messages, {},
                                              ToolChoice::Auto, "");
        REQUIRE_MESSAGE(plain.ok, plain.error);
        CHECK(plain.prompt == plain_prompt("gemma4-e2b", on));
        const auto tools = render_tool_prompt(nullptr, decision.thinking, jinja, messages, {ping},
                                              ToolChoice::Auto, "");
        REQUIRE_MESSAGE(tools.ok, tools.error);
        CHECK((tools.prompt.find("<|think|>") != std::string::npos) == on);
    }
}

TEST_CASE("reasoning control: Gemma 4 E2B GGUF generates both request modes") {
    const char* path = std::getenv("XLLAMA_TEST_GEMMA4_GGUF");
    if (!path || !*path)
        return;
    SessionParams settings;
    settings.model_path = path;
    settings.backend = Backend::LlamaCpp;
    settings.n_ctx = 512;
    settings.n_threads = 4;
    std::string error;
    auto session = Session::create(settings, &error);
    REQUIRE_MESSAGE(session != nullptr, error);
    for (const bool on : {false, true}) {
        CAPTURE(on);
        const auto fmt = chat_format_for("gemma4-e2b", on);
        GenerateParams gp;
        gp.prompt =
            fmt.render_prompt("You are a helpful AI assistant.", {},
                              "A bat and a ball cost one dollar ten cents together. The bat costs "
                              "one dollar more than the ball. How many cents does the ball cost?");
        gp.n_predict = 48;
        gp.temperature = 0;
        gp.reset_kv = true;
        gp.stop_sequences = fmt.stop_sequences;
        ThinkingStreamSplitter stream(fmt.reasoning_prefill(), fmt.thinking_markup);
        ReasoningTokenCounter tokens(fmt.reasoning_prefill(), fmt.thinking_markup);
        ThinkingSplit emitted;
        gp.on_token = [&](std::string_view piece) {
            tokens.push(piece);
            const auto part = stream.push(piece);
            emitted.reasoning += part.reasoning;
            emitted.content += part.content;
        };
        const auto generated = session->generate(gp);
        REQUIRE_MESSAGE(generated.success, generated.error_msg);
        const auto tail = stream.finish();
        emitted.reasoning += tail.reasoning;
        emitted.content += tail.content;
        const auto final = fmt.split_output(generated.output_text);
        if (!on)
            CHECK_FALSE(final.content.empty());
        if (final.content.empty())
            CHECK(generated.hit_token_limit(gp.n_predict));
        CHECK(fmt.postprocess_output(emitted.content) == final.content);
        CHECK(emitted.content.find("<|channel>") == std::string::npos);
        CHECK(final.reasoning.empty() == !on);
        CHECK(emitted.reasoning.empty() == !on);
        CHECK((tokens.count(fmt.strip_thinking_content) > 0) == on);
    }
}
