// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "xllama/tool_transcript.h"

#include "xllama/thinking_split.h"

#include "xllama/chat_prompt.h"

#include <nlohmann/json.hpp>

#include <utility>

namespace xllama {
namespace {

bool json_object(const std::string& text) {
    try {
        return nlohmann::json::parse(text).is_object();
    } catch (const std::exception&) {
        return false;
    }
}

// OpenAI's `arguments` field is a string whose contents are JSON. An object
// the parser handed us as text is re-serialized so the field is that string.
std::string arguments_json_string(const std::string& raw) {
    if (raw.empty())
        return "{}";
    try {
        const nlohmann::json value = nlohmann::json::parse(raw);
        if (value.is_string()) {
            const std::string& inner = value.get_ref<const std::string&>();
            try {
                return nlohmann::json::parse(inner).dump();
            } catch (const std::exception&) {
                return nlohmann::json(inner).dump();
            }
        }
        return value.dump();
    } catch (const std::exception&) {
        return nlohmann::json(raw).dump();
    }
}

} // namespace

AdmitResult admit_transcript(const std::vector<ChatMessage>& messages,
                             const std::vector<ToolSpec>& tools, ToolChoice choice,
                             const std::string& forced_name) {
    AdmitResult out;
    if (messages.empty() ||
        (messages.back().role != ChatRole::User && messages.back().role != ChatRole::Tool)) {
        out.error = "last message must be user or tool";
        return out;
    }
    for (const ToolSpec& tool : tools) {
        if (tool.name.empty()) {
            out.error = "tool is missing a name";
            return out;
        }
        if (!json_object(tool.parameters_json.empty() ? "{}" : tool.parameters_json)) {
            out.error = "tool parameters must be a JSON object";
            return out;
        }
    }
    for (const ChatMessage& message : messages) {
        if (message.role == ChatRole::Tool && message.tool_call_id.empty()) {
            out.error = "tool message requires tool_call_id";
            return out;
        }
        if (message.role != ChatRole::Assistant && !message.tool_calls.empty()) {
            out.error = "tool_calls are only valid on assistant messages";
            return out;
        }
        for (const ToolCall& call : message.tool_calls) {
            if (call.name.empty()) {
                out.error = "tool call is missing a name";
                return out;
            }
        }
    }
    if (choice == ToolChoice::Named) {
        bool found = false;
        for (const ToolSpec& tool : tools) {
            if (tool.name == forced_name) {
                found = true;
                break;
            }
        }
        if (!found) {
            out.error = "tool_choice names an unknown tool";
            return out;
        }
    }
    out.ok = true;
    return out;
}

ChoiceAssembly assemble_choice(const std::string& raw_text, const std::string& parsed_text,
                               const std::vector<IncomingCall>& calls, bool hit_cap,
                               bool strip_thinking, const std::string& parsed_reasoning) {
    return assemble_choice(raw_text, parsed_text, calls, hit_cap, strip_thinking, parsed_reasoning,
                           ThinkingMarkup::ThinkTags, {});
}

ChoiceAssembly assemble_choice(const std::string& raw_text, const std::string& parsed_text,
                               const std::vector<IncomingCall>& calls, bool hit_cap,
                               bool strip_thinking, const std::string& parsed_reasoning,
                               ThinkingMarkup markup, std::string_view prefill) {
    ChoiceAssembly out;
    out.reasoning = parsed_reasoning;
    const auto text_of = [&](const std::string& text, std::string_view open = {}) {
        if (!strip_thinking)
            return text;
        ThinkingSplit split = open.empty() ? split_thinking(text, markup)
                                           : split_thinking(std::string(open) + text, markup);
        if (!split.reasoning.empty() && split.reasoning != parsed_reasoning)
            out.reasoning += (out.reasoning.empty() ? "" : "\n\n") + split.reasoning;
        return std::move(split.content);
    };
    int missing = 0;
    for (const IncomingCall& call : calls) {
        if (!call.complete || call.name.empty())
            continue;
        ToolCall emitted;
        if (call.id.empty())
            emitted.id = "call_" + std::to_string(++missing);
        else
            emitted.id = call.id;
        emitted.name = call.name;
        emitted.arguments_json = arguments_json_string(call.arguments);
        out.tool_calls.push_back(std::move(emitted));
    }
    if (out.tool_calls.empty()) {
        out.content = text_of(raw_text, prefill);
        out.finish_reason = hit_cap ? "length" : "stop";
        return out;
    }
    out.content = text_of(parsed_text);
    out.content_null = out.content.empty();
    out.finish_reason = "tool_calls";
    return out;
}

} // namespace xllama
