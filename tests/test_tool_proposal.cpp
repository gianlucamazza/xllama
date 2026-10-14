// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "xllama/api_policy.h"
#include "xllama/prompt_budget.h"
#include "xllama/routing_policy.h"
#include "xllama/tool_request.h"
#include "xllama/tool_template.h"
#include "xllama/tool_transcript.h"

#include <algorithm>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

namespace {

xllama::ChatMessage message(xllama::ChatRole role, std::string content) {
    xllama::ChatMessage out;
    out.role = role;
    out.content = std::move(content);
    return out;
}

xllama::ToolSpec ping_tool() {
    xllama::ToolSpec tool;
    tool.name = "ping";
    tool.description = "reply";
    tool.parameters_json = R"({"type":"object","properties":{"text":{"type":"string"}}})";
    return tool;
}

const char* kToolTemplate = R"({%- if tools -%}
{%- for tool in tools -%}
{{ tool.function.name }}
{%- endfor -%}
{%- endif -%}
{%- for message in messages -%}
<|im_start|>{{ message.role }}
{{ message.content }}
{%- if message.tool_calls -%}
{%- for tc in message.tool_calls -%}
<tool_call>
{"name": "{{ tc.function.name }}", "arguments": {%- if tc.function.arguments is mapping -%}{{ tc.function.arguments | tojson }}{%- else -%}{{ tc.function.arguments }}{%- endif -%}}
</tool_call>
{%- endfor -%}
{%- endif -%}
<|im_end|>
{%- endfor -%}
{%- if add_generation_prompt -%}
<|im_start|>assistant
{%- endif -%})";

std::string render_marked(const std::vector<xllama::ChatMessage>& messages,
                          const std::vector<xllama::ToolSpec>& tools) {
    std::string out = "TOOLS";
    for (const xllama::ToolSpec& tool : tools)
        out += tool.name;
    for (const xllama::ChatMessage& message : messages) {
        out += "|";
        out += xllama::chat_role_name(message.role);
        out += ":";
        out += message.content;
        for (const xllama::ToolCall& call : message.tool_calls)
            out += call.name;
    }
    return out;
}

} // namespace

TEST_CASE("tool policy classifies proposals instead of rejecting every key") {
    xllama::ApiToolFields absent;
    CHECK(xllama::classify_api_tools(absent).disposition == xllama::ApiToolDisposition::None);

    xllama::ApiToolFields empty_tools;
    empty_tools.has_tools = true;
    empty_tools.tools_empty = true;
    CHECK(xllama::classify_api_tools(empty_tools).disposition == xllama::ApiToolDisposition::None);

    xllama::ApiToolFields choice_none;
    choice_none.has_tools = true;
    choice_none.tools_empty = false;
    choice_none.choice = xllama::ApiToolChoice::None;
    CHECK(xllama::classify_api_tools(choice_none).disposition == xllama::ApiToolDisposition::None);

    xllama::ApiToolFields proposal;
    proposal.has_tools = true;
    proposal.tools_empty = false;
    CHECK(xllama::classify_api_tools(proposal).disposition == xllama::ApiToolDisposition::Proposal);

    xllama::ApiToolFields legacy;
    legacy.has_functions = true;
    legacy.functions_empty = false;
    CHECK(xllama::classify_api_tools(legacy).disposition == xllama::ApiToolDisposition::Legacy);

    xllama::ApiToolFields both;
    both.has_tools = true;
    both.tools_empty = false;
    both.has_functions = true;
    both.functions_empty = false;
    const xllama::ApiToolClass invalid = xllama::classify_api_tools(both);
    CHECK(invalid.disposition == xllama::ApiToolDisposition::Invalid);
    CHECK(std::string(invalid.error) == "tools and functions cannot be combined");
}

TEST_CASE("tool request admits text content and rejects a transcript that cannot be completed") {
    const char* tools =
        R"("tools":[{"type":"function","function":{"name":"ping","parameters":{"type":"object"}}}]})";

    const xllama::ToolRequest parts = xllama::parse_tool_request(
        std::string(
            R"({"messages":[{"role":"user","content":[{"type":"text","text":"ab"},{"type":"text","text":"c"}]}],)") +
        tools);
    CHECK(parts.disposition == xllama::ApiToolDisposition::Proposal);
    REQUIRE(parts.messages.size() == 1);
    CHECK(parts.messages[0].content == "abc");

    const xllama::ToolRequest empty = xllama::parse_tool_request(
        std::string(R"({"messages":[{"role":"user","content":null}],)") + tools);
    CHECK(empty.error.empty());
    REQUIRE(empty.messages.size() == 1);
    CHECK(empty.messages[0].content.empty());

    const xllama::ToolRequest image = xllama::parse_tool_request(
        std::string(
            R"({"messages":[{"role":"user","content":[{"type":"image_url","image_url":{}}]}],)") +
        tools);
    CHECK(image.disposition == xllama::ApiToolDisposition::Invalid);
    CHECK(image.error == "unsupported message content");

    const xllama::ToolRequest unnamed = xllama::parse_tool_request(
        R"({"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"parameters":{}}}]})");
    CHECK(unnamed.error == "tool is missing a name");

    const xllama::ToolRequest no_id = xllama::parse_tool_request(
        std::string(R"({"messages":[{"role":"tool","content":"x"}],)") + tools);
    CHECK(no_id.error == "tool message requires tool_call_id");

    const xllama::ToolRequest trailing_assistant = xllama::parse_tool_request(
        std::string(R"({"messages":[{"role":"assistant","content":"x"}],)") + tools);
    CHECK(trailing_assistant.error == "last message must be user or tool");

    const xllama::ToolRequest legacy = xllama::parse_tool_request(
        R"({"messages":[{"role":"user","content":"hi"}],"functions":[{"name":"ping","parameters":{"type":"object"}}]})");
    CHECK(legacy.disposition == xllama::ApiToolDisposition::Legacy);
    REQUIRE(legacy.tools.size() == 1);
    CHECK(legacy.tools[0].name == "ping");

    const xllama::ToolRequest both = xllama::parse_tool_request(
        R"({"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"ping","parameters":{}}}],"functions":[{"name":"ping"}]})");
    CHECK(both.error == "tools and functions cannot be combined");
}

TEST_CASE("tool choice assembly emits calls, synthesized ids, and drops a truncated call") {
    xllama::IncomingCall call;
    call.name = "think";
    call.arguments = R"({ "observation" : "x" })";
    const xllama::ChoiceAssembly one = xllama::assemble_choice("", "", {call}, false);
    CHECK(one.finish_reason == "tool_calls");
    CHECK(one.content_null);
    REQUIRE(one.tool_calls.size() == 1);
    CHECK(one.tool_calls[0].id == "call_1");
    CHECK(one.tool_calls[0].name == "think");
    CHECK(one.tool_calls[0].arguments_json == R"({"observation":"x"})");

    xllama::IncomingCall named = call;
    named.id = "keep";
    xllama::IncomingCall second;
    second.name = "ping";
    second.arguments = R"({"text":"pong"})";
    const xllama::ChoiceAssembly two =
        xllama::assemble_choice("note", "note", {named, second}, false);
    CHECK(two.finish_reason == "tool_calls");
    CHECK_FALSE(two.content_null);
    CHECK(two.content == "note");
    REQUIRE(two.tool_calls.size() == 2);
    CHECK(two.tool_calls[0].id == "keep");
    CHECK(two.tool_calls[1].id == "call_1");

    const xllama::ChoiceAssembly text = xllama::assemble_choice("hello", "", {}, false);
    CHECK(text.finish_reason == "stop");
    CHECK(text.content == "hello");
    CHECK(text.tool_calls.empty());

    xllama::IncomingCall cut = call;
    cut.complete = false;
    const xllama::ChoiceAssembly truncated = xllama::assemble_choice("PARTIAL", "", {cut}, true);
    CHECK(truncated.finish_reason == "length");
    CHECK(truncated.content == "PARTIAL");
    CHECK_FALSE(truncated.content_null);
    CHECK(truncated.tool_calls.empty());
}

TEST_CASE("tool transcript budget drops an old call pair and keeps the schema and the final turn") {
    std::vector<xllama::ChatMessage> messages;
    messages.push_back(message(xllama::ChatRole::System, "S"));
    messages.push_back(message(xllama::ChatRole::User, "U1"));
    xllama::ChatMessage assistant = message(xllama::ChatRole::Assistant, "");
    xllama::ToolCall think;
    think.id = "call_9";
    think.name = "think";
    think.arguments_json = "{}";
    assistant.tool_calls.push_back(think);
    messages.push_back(assistant);
    xllama::ChatMessage tool = message(xllama::ChatRole::Tool, "R");
    tool.tool_call_id = "call_9";
    messages.push_back(tool);
    messages.push_back(message(xllama::ChatRole::User, "U2"));
    const std::vector<xllama::ToolSpec> tools{ping_tool()};

    const auto count = [](const std::string& text) { return static_cast<int>(text.size()); };
    const std::string full = render_marked(messages, tools);
    std::vector<xllama::ChatMessage> minimal{messages.front(), messages.back()};
    const std::string minimal_prompt = render_marked(minimal, tools);
    const int room = xllama::kReservedGenerationTokens + 1;

    const xllama::ToolTranscriptFit kept =
        xllama::fit_tool_transcript(messages, tools, static_cast<int>(full.size()) + room,
                                    xllama::kReservedGenerationTokens, render_marked, count);
    CHECK(kept.fits);
    CHECK(kept.dropped == 0);
    CHECK(kept.prompt.find("think") != std::string::npos);
    CHECK(kept.prompt.find("ping") != std::string::npos);

    const xllama::ToolTranscriptFit trimmed =
        xllama::fit_tool_transcript(messages, tools, static_cast<int>(minimal_prompt.size()) + room,
                                    xllama::kReservedGenerationTokens, render_marked, count);
    CHECK(trimmed.fits);
    CHECK(trimmed.prompt.find("ping") != std::string::npos);
    CHECK(trimmed.prompt.find("system:S") != std::string::npos);
    CHECK(trimmed.prompt.find("user:U2") != std::string::npos);
    CHECK(trimmed.prompt.find("think") == std::string::npos);
    CHECK(trimmed.prompt.find("|tool:") == std::string::npos);
    CHECK(trimmed.dropped == 3);

    const xllama::ToolTranscriptFit overflow = xllama::fit_tool_transcript(
        messages, tools, static_cast<int>(minimal_prompt.size()) + room - 1,
        xllama::kReservedGenerationTokens, render_marked, count);
    CHECK_FALSE(overflow.fits);
    CHECK(overflow.n_tokens == static_cast<int>(minimal_prompt.size()));
    CHECK(overflow.prompt.find("ping") != std::string::npos);
}

TEST_CASE("GGUF jinja template renders a tool name and parses a call") {
    xllama::ChatMessage user = message(xllama::ChatRole::User, "ping please");
    const std::vector<xllama::ToolSpec> tools{ping_tool()};
    const xllama::ToolTemplateResult rendered = xllama::render_tool_prompt(
        nullptr, kToolTemplate, {user}, tools, xllama::ToolChoice::Auto, "");
    REQUIRE_MESSAGE(rendered.ok, rendered.error);
    CHECK(rendered.prompt.find("ping") != std::string::npos);

    const std::string output =
        "<tool_call>\n{\"name\": \"ping\", \"arguments\": {\"text\": \"pong\"}}\n</tool_call>";
    const xllama::ToolParseResult parsed = xllama::parse_tool_generation(
        nullptr, kToolTemplate, {user}, tools, xllama::ToolChoice::Auto, "", output, false);
    REQUIRE_FALSE(parsed.parser_threw);
    REQUIRE(parsed.calls.size() == 1);
    CHECK(parsed.calls[0].name == "ping");
    CHECK(parsed.calls[0].complete);
    const xllama::ChoiceAssembly choice =
        xllama::assemble_choice(output, parsed.content, parsed.calls, false);
    CHECK(choice.finish_reason == "tool_calls");
    REQUIRE(choice.tool_calls.size() == 1);
    CHECK(choice.tool_calls[0].name == "ping");
    CHECK(choice.tool_calls[0].arguments_json.find("pong") != std::string::npos);
}

TEST_CASE("LFM2.5 tool calls need their special-token markers in the decoded text") {
    // Console regression (1.6.0.1208): lfm25-350m emitted a correct call, but the
    // decode dropped <|tool_call_start|> as a special token, so the parser saw
    // plain text. The template names the markers; the decode must keep them.
    std::ifstream file(std::string(LLAMA_TEMPLATES_DIR) + "/LFM2.5-Instruct.jinja");
    REQUIRE(file.good());
    const std::string jinja((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
    xllama::ChatMessage user = message(xllama::ChatRole::User, "ping please");
    const std::vector<xllama::ToolSpec> tools{ping_tool()};
    const xllama::ToolTemplateResult rendered =
        xllama::render_tool_prompt(nullptr, jinja, {user}, tools, xllama::ToolChoice::Auto, "");
    REQUIRE_MESSAGE(rendered.ok, rendered.error);
    const auto& kept = rendered.preserved_tokens;
    CHECK(std::find(kept.begin(), kept.end(), "<|tool_call_start|>") != kept.end());
    CHECK(std::find(kept.begin(), kept.end(), "<|tool_call_end|>") != kept.end());

    const xllama::ToolParseResult marked = xllama::parse_tool_generation(
        nullptr, jinja, {user}, tools, xllama::ToolChoice::Auto, "",
        "<|tool_call_start|>[ping(text=\"pong\")]<|tool_call_end|>", false);
    REQUIRE(marked.calls.size() == 1);
    CHECK(marked.calls[0].name == "ping");

    const xllama::ToolParseResult stripped =
        xllama::parse_tool_generation(nullptr, jinja, {user}, tools, xllama::ToolChoice::Auto, "",
                                      "[ping(text=\"pong\")]", false);
    CHECK(stripped.calls.empty());
}

TEST_CASE("a template that ignores tools is refused instead of answering without the schema") {
    // Phi-4-mini's GGUF template (unsloth Q4_K_M) reads the schema from a system
    // message field, never from `tools`. Rendering succeeds and the model answers
    // in prose (console, 1.6.0.1209); the proposal must be a 400 instead.
    const std::string phi4 =
        "{% for message in messages %}{% if message['role'] == 'system' and 'tools' in message "
        "and message['tools'] is not none %}{{ '<|' + message['role'] + '|>' + message['content'] "
        "+ '<|tool|>' + message['tools'] + '<|/tool|>' + '<|end|>' }}{% else %}{{ '<|' + "
        "message['role'] + '|>' + message['content'] + '<|end|>' }}{% endif %}{% endfor %}{% if "
        "add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}";
    const std::vector<xllama::ToolSpec> tools{ping_tool()};
    // The user names the tool: a mention in the transcript is not the schema.
    const xllama::ToolTemplateResult refused = xllama::render_tool_prompt(
        nullptr, phi4, {message(xllama::ChatRole::User, "call ping please")}, tools,
        xllama::ToolChoice::Auto, "");
    CHECK_FALSE(refused.ok);
    CHECK(refused.error == "model chat template does not render tools");

    const xllama::ToolTemplateResult rendered = xllama::render_tool_prompt(
        nullptr, kToolTemplate, {message(xllama::ChatRole::User, "call ping please")}, tools,
        xllama::ToolChoice::Auto, "");
    CHECK_MESSAGE(rendered.ok, rendered.error);
}
