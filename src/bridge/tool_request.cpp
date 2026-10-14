// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "xllama/tool_request.h"

#include <nlohmann/json.hpp>

#include <utility>

namespace xllama {
namespace {

using json = nlohmann::json;

bool read_content(const json& message, std::string& content, std::string& error) {
    if (!message.contains("content") || message.at("content").is_null()) {
        content.clear();
        return true;
    }
    const json& value = message.at("content");
    if (value.is_string()) {
        content = value.get<std::string>();
        return true;
    }
    if (!value.is_array()) {
        error = "unsupported message content";
        return false;
    }
    content.clear();
    for (const json& part : value) {
        if (!part.is_object() || !part.contains("type") || !part.at("type").is_string() ||
            part.at("type") != "text") {
            error = "unsupported message content";
            return false;
        }
        if (!part.contains("text") || part.at("text").is_null())
            continue;
        if (!part.at("text").is_string()) {
            error = "unsupported message content";
            return false;
        }
        content += part.at("text").get<std::string>();
    }
    return true;
}

bool read_tool(const json& item, ToolSpec& tool, std::string& error) {
    if (!item.is_object()) {
        error = "tool is missing a name";
        return false;
    }
    if (item.contains("type") && item.at("type").is_string() && item.at("type") != "function") {
        error = "unsupported tool type";
        return false;
    }
    const json* function = &item;
    if (item.contains("function") && item.at("function").is_object())
        function = &item.at("function");
    if (!function->contains("name") || !function->at("name").is_string() ||
        function->at("name").get<std::string>().empty()) {
        error = "tool is missing a name";
        return false;
    }
    tool.name = function->at("name").get<std::string>();
    if (function->contains("description") && function->at("description").is_string())
        tool.description = function->at("description").get<std::string>();
    if (!function->contains("parameters") || function->at("parameters").is_null()) {
        tool.parameters_json = "{}";
        return true;
    }
    if (!function->at("parameters").is_object()) {
        error = "tool parameters must be a JSON object";
        return false;
    }
    tool.parameters_json = function->at("parameters").dump();
    return true;
}

bool read_arguments(const json& function, std::string& arguments, std::string& error) {
    if (!function.contains("arguments") || function.at("arguments").is_null()) {
        arguments = "{}";
        return true;
    }
    const json& value = function.at("arguments");
    if (value.is_string()) {
        arguments = value.get<std::string>();
        return true;
    }
    if (value.is_object() || value.is_array()) {
        arguments = value.dump();
        return true;
    }
    error = "tool call arguments must be a JSON string or object";
    return false;
}

bool read_message(const json& item, ChatMessage& message, std::string& error) {
    if (!item.is_object() || !item.contains("role") || !item.at("role").is_string()) {
        error = "unknown message role";
        return false;
    }
    const std::string role = item.at("role").get<std::string>();
    if (role == "system")
        message.role = ChatRole::System;
    else if (role == "user")
        message.role = ChatRole::User;
    else if (role == "assistant")
        message.role = ChatRole::Assistant;
    else if (role == "tool")
        message.role = ChatRole::Tool;
    else {
        error = "unknown message role";
        return false;
    }
    if (!read_content(item, message.content, error))
        return false;
    if (message.role == ChatRole::Tool) {
        if (!item.contains("tool_call_id") || !item.at("tool_call_id").is_string() ||
            item.at("tool_call_id").get<std::string>().empty()) {
            error = "tool message requires tool_call_id";
            return false;
        }
        message.tool_call_id = item.at("tool_call_id").get<std::string>();
        if (item.contains("name") && item.at("name").is_string())
            message.name = item.at("name").get<std::string>();
    }
    if (!item.contains("tool_calls") || item.at("tool_calls").is_null())
        return true;
    if (message.role != ChatRole::Assistant) {
        error = "tool_calls are only valid on assistant messages";
        return false;
    }
    if (!item.at("tool_calls").is_array()) {
        error = "tool_calls must be an array";
        return false;
    }
    for (const json& call : item.at("tool_calls")) {
        if (!call.is_object()) {
            error = "tool call is missing a name";
            return false;
        }
        const json* function = &call;
        if (call.contains("function") && call.at("function").is_object())
            function = &call.at("function");
        ToolCall out;
        if (call.contains("id") && call.at("id").is_string())
            out.id = call.at("id").get<std::string>();
        if (!function->contains("name") || !function->at("name").is_string() ||
            function->at("name").get<std::string>().empty()) {
            error = "tool call is missing a name";
            return false;
        }
        out.name = function->at("name").get<std::string>();
        if (!read_arguments(*function, out.arguments_json, error))
            return false;
        message.tool_calls.push_back(std::move(out));
    }
    return true;
}

ApiToolChoice read_choice(const json& root, std::string& forced, std::string& error) {
    if (!root.contains("tool_choice") || root.at("tool_choice").is_null())
        return ApiToolChoice::Absent;
    const json& choice = root.at("tool_choice");
    if (choice.is_string()) {
        const std::string text = choice.get<std::string>();
        if (text == "none")
            return ApiToolChoice::None;
        if (text == "auto")
            return ApiToolChoice::Auto;
        if (text == "required")
            return ApiToolChoice::Required;
        error = "invalid tool_choice";
        return ApiToolChoice::Invalid;
    }
    if (!choice.is_object() || !choice.contains("type") || !choice.at("type").is_string() ||
        choice.at("type") != "function" || !choice.contains("function") ||
        !choice.at("function").is_object()) {
        error = "invalid tool_choice";
        return ApiToolChoice::Invalid;
    }
    const json& function = choice.at("function");
    if (!function.contains("name") || !function.at("name").is_string() ||
        function.at("name").get<std::string>().empty()) {
        error = "invalid tool_choice";
        return ApiToolChoice::Invalid;
    }
    forced = function.at("name").get<std::string>();
    return ApiToolChoice::Named;
}

ToolChoice to_choice(ApiToolChoice choice) {
    switch (choice) {
    case ApiToolChoice::Required:
        return ToolChoice::Required;
    case ApiToolChoice::Named:
        return ToolChoice::Named;
    case ApiToolChoice::None:
        return ToolChoice::None;
    case ApiToolChoice::Absent:
    case ApiToolChoice::Auto:
    case ApiToolChoice::Invalid:
        return ToolChoice::Auto;
    }
    return ToolChoice::Auto;
}

bool read_tool_array(const json& array, std::vector<ToolSpec>& tools, std::string& error) {
    for (const json& item : array) {
        ToolSpec tool;
        if (!read_tool(item, tool, error))
            return false;
        tools.push_back(std::move(tool));
    }
    return true;
}

} // namespace

ToolRequest parse_tool_request(const std::string& body) {
    ToolRequest out;
    json root;
    try {
        root = json::parse(body);
    } catch (const std::exception&) {
        out.disposition = ApiToolDisposition::Invalid;
        out.error = "invalid JSON body";
        return out;
    }
    if (!root.is_object()) {
        out.disposition = ApiToolDisposition::Invalid;
        out.error = "invalid JSON body";
        return out;
    }

    ApiToolFields fields;
    if (root.contains("tools") && !root.at("tools").is_null()) {
        if (!root.at("tools").is_array()) {
            out.disposition = ApiToolDisposition::Invalid;
            out.error = "tools must be an array";
            return out;
        }
        fields.has_tools = true;
        fields.tools_empty = root.at("tools").empty();
    }
    if (root.contains("functions") && !root.at("functions").is_null()) {
        if (!root.at("functions").is_array()) {
            out.disposition = ApiToolDisposition::Invalid;
            out.error = "functions must be an array";
            return out;
        }
        fields.has_functions = true;
        fields.functions_empty = root.at("functions").empty();
    }
    std::string choice_error;
    fields.choice = read_choice(root, out.forced_name, choice_error);
    if (!choice_error.empty()) {
        out.disposition = ApiToolDisposition::Invalid;
        out.error = choice_error;
        return out;
    }

    const ApiToolClass classified = classify_api_tools(fields);
    out.disposition = classified.disposition;
    out.choice = to_choice(fields.choice);
    if (classified.disposition == ApiToolDisposition::Invalid) {
        out.error = classified.error;
        return out;
    }
    if (classified.disposition == ApiToolDisposition::None)
        return out;

    const json& schemas = classified.disposition == ApiToolDisposition::Legacy
                              ? root.at("functions")
                              : root.at("tools");
    if (!read_tool_array(schemas, out.tools, out.error)) {
        out.disposition = ApiToolDisposition::Invalid;
        return out;
    }
    if (!root.contains("messages") || !root.at("messages").is_array()) {
        out.disposition = ApiToolDisposition::Invalid;
        out.error = "missing or non-array 'messages'";
        return out;
    }
    for (const json& item : root.at("messages")) {
        ChatMessage message;
        if (!read_message(item, message, out.error)) {
            out.disposition = ApiToolDisposition::Invalid;
            return out;
        }
        out.messages.push_back(std::move(message));
    }
    const AdmitResult admitted =
        admit_transcript(out.messages, out.tools, out.choice, out.forced_name);
    if (!admitted.ok) {
        out.disposition = ApiToolDisposition::Invalid;
        out.error = admitted.error;
    }
    return out;
}

} // namespace xllama
