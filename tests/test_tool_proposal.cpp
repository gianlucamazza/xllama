// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "xllama/api_policy.h"
#include "xllama/prompt_budget.h"
#include "xllama/routing_policy.h"
#include "xllama/session.h"
#include "xllama/tool_budget.h"
#include "xllama/tool_request.h"
#include "xllama/tool_template.h"
#include "xllama/tool_transcript.h"

#include <algorithm>
#include <cstdlib>
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

// LFM2-2.6B's GGUF template, reduced: it prints the schema and the tool result but
// not the assistant's calls.
const char* kNoCallsTemplate = R"({%- if tools -%}
{%- for tool in tools -%}
{{ tool.function.name }}
{%- endfor -%}
{%- endif -%}
{%- for message in messages -%}
<|im_start|>{{ message.role }}
{{ message.content }}<|im_end|>
{%- endfor -%}
{%- if add_generation_prompt -%}
<|im_start|>assistant
{%- endif -%})";

// Phi-4-mini's GGUF template (unsloth Q4_K_M) reads the schema from a system
// message field, never from `tools`.
const char* kPhi4Template =
    "{% for message in messages %}{% if message['role'] == 'system' and 'tools' in message "
    "and message['tools'] is not none %}{{ '<|' + message['role'] + '|>' + message['content'] "
    "+ '<|tool|>' + message['tools'] + '<|/tool|>' + '<|end|>' }}{% else %}{{ '<|' + "
    "message['role'] + '|>' + message['content'] + '<|end|>' }}{% endif %}{% endfor %}{% if "
    "add_generation_prompt %}{{ '<|assistant|>' }}{% endif %}";

// A replayed call to ping, answered by a tool message that names it.
std::vector<xllama::ChatMessage> replayed_ping() {
    xllama::ChatMessage assistant = message(xllama::ChatRole::Assistant, "");
    xllama::ToolCall call;
    call.id = "call_1";
    call.name = "ping";
    call.arguments_json = R"({"text":"hi"})";
    assistant.tool_calls.push_back(call);
    xllama::ChatMessage tool = message(xllama::ChatRole::Tool, "pong-result");
    tool.tool_call_id = "call_1";
    tool.name = "ping";
    return {message(xllama::ChatRole::User, "go"), assistant, tool};
}

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

// Includes the schema body so a 39-tool fixture can match the ~7.5k-token
// Open WebUI shape without copying that client's text.
std::string render_schema(const std::vector<xllama::ChatMessage>& messages,
                          const std::vector<xllama::ToolSpec>& tools) {
    std::string out;
    for (const xllama::ToolSpec& tool : tools) {
        out += tool.name;
        out += tool.description;
        out += tool.parameters_json;
    }
    for (const xllama::ChatMessage& message : messages) {
        out += "|";
        out += xllama::chat_role_name(message.role);
        out += ":";
        out += message.content;
    }
    return out;
}

std::vector<xllama::ToolSpec> synthetic_tools(int n) {
    std::vector<xllama::ToolSpec> tools;
    tools.reserve(static_cast<size_t>(n));
    const std::string pad(160, 'x');
    for (int i = 0; i < n; ++i) {
        xllama::ToolSpec tool;
        tool.name = "synth_" + std::to_string(i);
        tool.description = pad;
        tool.parameters_json = R"({"type":"object","properties":{"arg":{"type":"string"}}})";
        tools.push_back(std::move(tool));
    }
    return tools;
}

std::string synthetic_tools_json(int n) {
    std::string out = "[";
    for (int i = 0; i < n; ++i) {
        if (i)
            out += ",";
        out += R"({"type":"function","function":{"name":"synth_)" + std::to_string(i) +
               R"(","description":")" + std::string(160, 'x') +
               R"(","parameters":{"type":"object","properties":{"arg":{"type":"string"}}}}})";
    }
    out += "]";
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

namespace {
xllama::ToolTemplateRenderer
budget_renderer(std::string (*render)(const std::vector<xllama::ChatMessage>&,
                                      const std::vector<xllama::ToolSpec>&)) {
    return [render](const auto& messages, const auto& tools) {
        xllama::ToolTemplateResult result;
        result.ok = true;
        result.prompt = render(messages, tools);
        result.prompt_without_tools = render(messages, {});
        return result;
    };
}
} // namespace

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

    const xllama::ToolTemplateFit kept = xllama::fit_tool_template(
        messages, tools, static_cast<int>(full.size()) + room, xllama::kReservedGenerationTokens,
        budget_renderer(render_marked), count);
    CHECK(kept.fits);
    CHECK(kept.dropped == 0);
    CHECK(kept.rendered.prompt.find("think") != std::string::npos);
    CHECK(kept.rendered.prompt.find("ping") != std::string::npos);
    CHECK(kept.n_tokens_without_tools == -1); // tool count is only computed on overflow

    const xllama::ToolTemplateFit trimmed = xllama::fit_tool_template(
        messages, tools, static_cast<int>(minimal_prompt.size()) + room,
        xllama::kReservedGenerationTokens, budget_renderer(render_marked), count);
    CHECK(trimmed.fits);
    CHECK(trimmed.rendered.prompt.find("ping") != std::string::npos);
    CHECK(trimmed.rendered.prompt.find("system:S") != std::string::npos);
    CHECK(trimmed.rendered.prompt.find("user:U2") != std::string::npos);
    CHECK(trimmed.rendered.prompt.find("think") == std::string::npos);
    CHECK(trimmed.rendered.prompt.find("|tool:") == std::string::npos);
    CHECK(trimmed.dropped == 3);
    CHECK(trimmed.n_tokens_without_tools == -1);

    // Four droppable messages make the search probe 0, 1, 2, 4, then 3.
    // Probe 3 fails. The selected render and its metadata must come from 4.
    std::vector<xllama::ChatMessage> history{
        message(xllama::ChatRole::User, "U1"), message(xllama::ChatRole::Assistant, "A1"),
        message(xllama::ChatRole::User, "U2"), message(xllama::ChatRole::Assistant, "A2"),
        message(xllama::ChatRole::User, "U3")};
    const auto render_with_metadata = [](const auto& kept, const auto&) {
        xllama::ToolTemplateResult render;
        render.ok = true;
        render.prompt = std::string(kept.size(), 'x');
        render.grammar.generation_prompt = std::to_string(kept.size());
        render.stop_sequences = {render.grammar.generation_prompt};
        render.preserved_tokens = {render.grammar.generation_prompt};
        return render;
    };
    const auto selected = xllama::fit_tool_template(
        history, tools, room + 1, xllama::kReservedGenerationTokens, render_with_metadata, count);
    REQUIRE(selected.fits);
    CHECK(selected.dropped == 4);
    CHECK(selected.rendered.prompt == "x");
    CHECK(selected.rendered.grammar.generation_prompt == "1");
    CHECK(selected.rendered.stop_sequences == std::vector<std::string>{"1"});
    CHECK(selected.rendered.preserved_tokens == std::vector<std::string>{"1"});

    const auto strict_renderer = [&](const auto& kept, const auto& offered) {
        auto rendered = render_with_metadata(kept, offered);
        if (kept.front().role == xllama::ChatRole::Assistant) {
            rendered.ok = false;
            rendered.error = "template requires the first turn to be user";
        }
        return rendered;
    };
    const auto strict = xllama::fit_tool_template(
        history, tools, room + 1, xllama::kReservedGenerationTokens, strict_renderer, count);
    REQUIRE(strict.fits);
    CHECK(strict.rendered.grammar.generation_prompt == "1");

    // The shortest suffix is a protected call and result that needs the user
    // turn before it. Dropping six messages fits; dropping seven is invalid.
    history.push_back(message(xllama::ChatRole::Assistant, "A3"));
    history.push_back(message(xllama::ChatRole::User, "U4"));
    history.push_back(assistant);
    history.push_back(tool);
    const auto tool_tail = xllama::fit_tool_template(
        history, tools, room + 3, xllama::kReservedGenerationTokens, strict_renderer, count);
    REQUIRE(tool_tail.fits);
    CHECK(tool_tail.dropped == 6);
    CHECK(tool_tail.rendered.grammar.generation_prompt == "3");
    REQUIRE(tool_tail.messages.size() == 3);
    CHECK(tool_tail.messages.front().content == "U4");
    CHECK(tool_tail.messages.back().tool_call_id == "call_9");

    const xllama::ToolTemplateFit overflow = xllama::fit_tool_template(
        messages, tools, static_cast<int>(minimal_prompt.size()) + room - 1,
        xllama::kReservedGenerationTokens, budget_renderer(render_marked), count);
    CHECK_FALSE(overflow.fits);
    CHECK(overflow.n_tokens == static_cast<int>(minimal_prompt.size()));
    CHECK(overflow.rendered.prompt.find("ping") != std::string::npos);
    const int tool_delta = static_cast<int>(render_marked(overflow.messages, tools).size() -
                                            render_marked(overflow.messages, {}).size());
    CHECK(overflow.n_tokens - overflow.n_tokens_without_tools == tool_delta);
    CHECK(overflow.n_tokens > overflow.n_tokens_without_tools);
}

TEST_CASE("a generated 39-tool fixture overflows with the schema kept and param tools") {
    const std::vector<xllama::ToolSpec> tools = synthetic_tools(39);
    const std::vector<xllama::ChatMessage> messages{message(xllama::ChatRole::User, "hi")};
    const auto count = [](const std::string& text) { return static_cast<int>(text.size()); };
    const std::string full = render_schema(messages, tools);
    const std::string without_tools = render_schema(messages, {});
    const int tool_delta = static_cast<int>(full.size() - without_tools.size());
    REQUIRE(tools.size() == 39);
    REQUIRE(tool_delta > 0);
    const int room = xllama::kReservedGenerationTokens + 1;
    const int n_ctx = tool_delta + room - 1;

    const xllama::ToolTemplateFit overflow =
        xllama::fit_tool_template(messages, tools, n_ctx, xllama::kReservedGenerationTokens,
                                  budget_renderer(render_schema), count);
    CHECK_FALSE(overflow.fits);
    CHECK(overflow.n_tokens - overflow.n_tokens_without_tools == tool_delta);
    CHECK(overflow.n_tokens == static_cast<int>(full.size()));
    CHECK(overflow.rendered.prompt.find("synth_0") != std::string::npos);
    CHECK(overflow.rendered.prompt.find("synth_38") != std::string::npos);
    CHECK(overflow.dropped == 0);

    const xllama::ToolBudgetError err =
        xllama::tool_prompt_too_long(overflow.n_tokens, overflow.n_tokens_without_tools,
                                     xllama::kReservedGenerationTokens, n_ctx);
    const std::string prefix = xllama::tool_prompt_too_long_prefix(
        overflow.n_tokens, xllama::kReservedGenerationTokens, n_ctx);
    CHECK(err.message.compare(0, prefix.size(), prefix) == 0);
    CHECK(std::string(err.param) == xllama::kToolBudgetParamTools);
    CHECK(std::string(err.code) == xllama::kContextLengthExceeded);
    CHECK(err.message.find("more tool definitions than fit") != std::string::npos);

    const std::string body =
        std::string(R"({"messages":[{"role":"user","content":"hi"}],"tools":)") +
        synthetic_tools_json(39) + "}";
    const xllama::ToolRequest parsed = xllama::parse_tool_request(body);
    CHECK(parsed.error.empty());
    CHECK(parsed.disposition == xllama::ApiToolDisposition::Proposal);
    CHECK(parsed.choice == xllama::ToolChoice::Auto);
    REQUIRE(parsed.messages.size() == 1);
    CHECK(parsed.messages[0].role == xllama::ChatRole::User);
    CHECK(parsed.tools.size() == 39);
}

TEST_CASE("a fitting tool request keeps every schema token and drops nothing") {
    const std::vector<xllama::ChatMessage> messages{message(xllama::ChatRole::User, "hi")};
    const std::vector<xllama::ToolSpec> tools{ping_tool()};
    const auto count = [](const std::string& text) { return static_cast<int>(text.size()); };
    const std::string full = render_marked(messages, tools);
    const int room = xllama::kReservedGenerationTokens + 1;
    const xllama::ToolTranscriptFit fit =
        xllama::fit_tool_transcript(messages, tools, static_cast<int>(full.size()) + room,
                                    xllama::kReservedGenerationTokens, render_marked, count);
    CHECK(fit.fits);
    CHECK(fit.dropped == 0);
    CHECK(fit.prompt == full);
    CHECK(fit.prompt.find("ping") != std::string::npos);
}

TEST_CASE("tool_choice none and an empty tools array stay on the plain chat path") {
    const xllama::ToolRequest none = xllama::parse_tool_request(
        R"({"messages":[{"role":"user","content":"hi"}],"tools":[{"type":"function","function":{"name":"ping","parameters":{"type":"object"}}}],"tool_choice":"none"})");
    CHECK(none.error.empty());
    CHECK(none.disposition == xllama::ApiToolDisposition::None);

    const xllama::ToolRequest empty =
        xllama::parse_tool_request(R"({"messages":[{"role":"user","content":"hi"}],"tools":[]})");
    CHECK(empty.error.empty());
    CHECK(empty.disposition == xllama::ApiToolDisposition::None);
}

TEST_CASE("fit_tool_transcript: no counter means no verdict") {
    const std::vector<xllama::ChatMessage> messages{message(xllama::ChatRole::User, "hi")};
    const std::vector<xllama::ToolSpec> tools{ping_tool()};
    const xllama::ToolTemplateFit fit = xllama::fit_tool_template(
        messages, tools, 2048, 512, budget_renderer(render_marked), nullptr);
    CHECK_FALSE(fit.fits);
    CHECK(fit.n_tokens == -1);
    CHECK(fit.n_tokens_without_tools == -1);
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

TEST_CASE("a thinking model's reasoning stays out of a tool proposal's content") {
    xllama::IncomingCall call;
    call.name = "ping";
    call.arguments = "{}";
    const std::string reasoning = "<think>the user wants ping</think>";

    const xllama::ChoiceAssembly with_call =
        xllama::assemble_choice("", reasoning, {call}, false, true);
    CHECK(with_call.tool_calls.size() == 1);
    CHECK(with_call.content_null);
    CHECK(with_call.content.empty());

    const xllama::ChoiceAssembly answer =
        xllama::assemble_choice(reasoning + "pong", "", {}, false, true);
    CHECK(answer.content == "pong");
    CHECK(answer.finish_reason == "stop");

    const xllama::ChoiceAssembly cut_off =
        xllama::assemble_choice("<think>still going", "", {}, true, true);
    CHECK(cut_off.content.empty());
    CHECK(cut_off.finish_reason == "length");

    // Other models keep their text as it is.
    const xllama::ChoiceAssembly other = xllama::assemble_choice(reasoning, "", {}, false);
    CHECK(other.content == reasoning);
}

TEST_CASE("a template that ignores tools is refused instead of answering without the schema") {
    // Rendering Phi-4's template succeeds and the model answers in prose (console,
    // 1.6.0.1209); the proposal must be a 400 instead.
    const std::vector<xllama::ToolSpec> tools{ping_tool()};
    // The user names the tool: a mention in the transcript is not the schema.
    const xllama::ToolTemplateResult refused = xllama::render_tool_prompt(
        nullptr, kPhi4Template, {message(xllama::ChatRole::User, "call ping please")}, tools,
        xllama::ToolChoice::Auto, "");
    CHECK_FALSE(refused.ok);
    CHECK(refused.error == "model chat template does not render tools");

    const xllama::ToolTemplateResult rendered = xllama::render_tool_prompt(
        nullptr, kToolTemplate, {message(xllama::ChatRole::User, "call ping please")}, tools,
        xllama::ToolChoice::Auto, "");
    CHECK_MESSAGE(rendered.ok, rendered.error);
}

TEST_CASE("replaying a tool call is not mistaken for a template that ignores tools") {
    const std::vector<xllama::ToolSpec> tools{ping_tool()};
    const std::vector<xllama::ChatMessage> replay = replayed_ping();

    SUBCASE("the tool message's name is not printed") {
        const xllama::ToolTemplateResult rendered = xllama::render_tool_prompt(
            nullptr, kToolTemplate, replay, tools, xllama::ToolChoice::Auto, "");
        REQUIRE_MESSAGE(rendered.ok, rendered.error);
        CHECK(rendered.prompt.find("pong-result") != std::string::npos);
    }
    SUBCASE("the template does not print the assistant's calls") {
        const xllama::ToolTemplateResult rendered = xllama::render_tool_prompt(
            nullptr, kNoCallsTemplate, replay, tools, xllama::ToolChoice::Auto, "");
        REQUIRE_MESSAGE(rendered.ok, rendered.error);
        CHECK(rendered.prompt.find("pong-result") != std::string::npos);
    }
    SUBCASE("a template that ignores tools is still refused") {
        const xllama::ToolTemplateResult refused = xllama::render_tool_prompt(
            nullptr, kPhi4Template, replay, tools, xllama::ToolChoice::Auto, "");
        CHECK_FALSE(refused.ok);
    }
}

namespace {
std::string read_template(const char* name) {
    std::ifstream file(std::string(LLAMA_TEMPLATES_DIR) + "/" + name);
    return std::string((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
}
} // namespace

TEST_CASE("Qwen3.5 and Llama-3.2 templates count overflowing tools as param tools") {
    // These templates raise on an empty message list. The tool-token delta must
    // use the kept transcript (system + protected tail), not render({}, tools).
    const char* names[] = {"Qwen3.5-4B.jinja", "meta-llama-Llama-3.2-3B-Instruct.jinja"};
    for (const char* name : names) {
        const std::string jinja = read_template(name);
        REQUIRE_FALSE(jinja.empty());
        const xllama::ToolTemplateResult empty_msgs = xllama::render_tool_prompt(
            nullptr, jinja, {}, synthetic_tools(1), xllama::ToolChoice::Auto, "");
        CHECK_FALSE(empty_msgs.ok);

        const auto render = [&](const std::vector<xllama::ChatMessage>& msgs,
                                const std::vector<xllama::ToolSpec>& tools) {
            return xllama::render_tool_prompt(nullptr, jinja, msgs, tools, xllama::ToolChoice::Auto,
                                              "");
        };
        const std::vector<xllama::ToolSpec> tools = synthetic_tools(39);
        const std::vector<xllama::ChatMessage> messages{message(xllama::ChatRole::User, "hi")};
        const auto count = [](const std::string& text) { return static_cast<int>(text.size()); };
        const std::string with = render(messages, tools).prompt;
        const std::string without = render(messages, {}).prompt;
        const int tool_delta = static_cast<int>(with.size() - without.size());
        REQUIRE(tool_delta > 0);
        const int room = xllama::kReservedGenerationTokens + 1;
        const int n_ctx = tool_delta + room - 1;

        const xllama::ToolTemplateFit overflow = xllama::fit_tool_template(
            messages, tools, n_ctx, xllama::kReservedGenerationTokens, render, count);
        CHECK_FALSE(overflow.fits);
        CHECK(overflow.n_tokens - overflow.n_tokens_without_tools == tool_delta);
        CHECK(overflow.rendered.prompt.find("synth_0") != std::string::npos);

        const xllama::ToolBudgetError err =
            xllama::tool_prompt_too_long(overflow.n_tokens, overflow.n_tokens_without_tools,
                                         xllama::kReservedGenerationTokens, n_ctx);
        CHECK(std::string(err.param) == xllama::kToolBudgetParamTools);
        CHECK(err.message.find("more tool definitions than fit") != std::string::npos);
        CHECK(err.message.find("built-in tools") != std::string::npos);

        const auto named = xllama::render_tool_prompt(nullptr, jinja, messages, tools,
                                                      xllama::ToolChoice::Named, "synth_0");
        REQUIRE(named.ok);
        REQUIRE(named.prompt_without_tools.has_value());
        CHECK(*named.prompt_without_tools == without);
        CHECK(named.prompt.find("synth_0") != std::string::npos);
        CHECK(named.prompt.find("synth_1") == std::string::npos);
    }
}

TEST_CASE("the template's grammar travels with the prompt: lazy for auto, full when forced") {
    // ADR 0005: the grammar comes from the same parser as the parse step, with
    // llama-server's trigger form (a WORD escaped into a regex).
    const std::string jinja = read_template("LFM2.5-Instruct.jinja");
    REQUIRE_FALSE(jinja.empty());
    const std::vector<xllama::ChatMessage> msgs{message(xllama::ChatRole::User, "ping please")};
    const std::vector<xllama::ToolSpec> tools{ping_tool()};

    const xllama::ToolTemplateResult automatic =
        xllama::render_tool_prompt(nullptr, jinja, msgs, tools, xllama::ToolChoice::Auto, "");
    REQUIRE_MESSAGE(automatic.ok, automatic.error);
    CHECK_FALSE(automatic.grammar.empty());
    CHECK(automatic.grammar.lazy);
    const auto& patterns = automatic.grammar.trigger_patterns;
    CHECK(std::find(patterns.begin(), patterns.end(), "<\\|tool_call_start\\|>") != patterns.end());
    CHECK(automatic.grammar.gbnf.find("ping") != std::string::npos);

    const xllama::ToolTemplateResult forced =
        xllama::render_tool_prompt(nullptr, jinja, msgs, tools, xllama::ToolChoice::Named, "ping");
    REQUIRE_MESSAGE(forced.ok, forced.error);
    CHECK_FALSE(forced.grammar.empty());
    CHECK_FALSE(forced.grammar.lazy);
}

TEST_CASE("a grammar-constrained tool call stops at its end marker (opt-in: XLLAMA_TOOL_MODEL)") {
    // A real LFM2.5 GGUF: the call must parse and nothing may follow it, which
    // is what lfm25-1.2b-instruct did unconstrained on the console (1.6.0.1210).
    const char* model_env = std::getenv("XLLAMA_TOOL_MODEL");
    if (!model_env)
        return;
    xllama::SessionParams sp;
    sp.model_path = model_env;
    sp.n_ctx = 2048;
    std::string err;
    auto session = xllama::Session::create(sp, &err);
    REQUIRE_MESSAGE(session, err);

    std::vector<xllama::ChatMessage> msgs{
        message(xllama::ChatRole::System, "You are a helpful assistant."),
        message(xllama::ChatRole::User, "Call ping with the text pong, then explain it.")};
    const std::vector<xllama::ToolSpec> tools{ping_tool()};
    const void* model = session->resident_llama_model();
    for (const xllama::ToolChoice choice : {xllama::ToolChoice::Auto, xllama::ToolChoice::Named}) {
        const xllama::ToolTemplateResult rendered =
            xllama::render_tool_prompt(model, "", msgs, tools, choice, "ping");
        REQUIRE_MESSAGE(rendered.ok, rendered.error);
        xllama::GenerateParams gp;
        gp.prompt = rendered.prompt;
        gp.n_predict = 96;
        gp.temperature = 0.0f;
        gp.stop_sequences = rendered.stop_sequences;
        gp.preserved_tokens = rendered.preserved_tokens;
        gp.grammar = rendered.grammar;
        const xllama::InferenceResult r = session->generate(gp);
        REQUIRE_MESSAGE(r.success, r.error_msg);
        MESSAGE("output: " << r.output_text);
        const xllama::ToolParseResult parsed = xllama::parse_tool_generation(
            model, "", msgs, tools, choice, "ping", r.output_text, false);
        CHECK_FALSE(parsed.parser_threw);
        REQUIRE(parsed.calls.size() == 1);
        CHECK(parsed.calls[0].name == "ping");
        CHECK(r.n_eval < gp.n_predict); // stopped at the call, not at the cap
    }
}
