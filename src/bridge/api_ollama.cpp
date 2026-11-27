// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "xllama/api_ollama.h"

#include "xllama/api_sampling.h"
#include "xllama/chat_prompt.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <optional>
#include <unordered_set>
#include <utility>
#include <vector>

namespace xllama {
namespace {

using Json = nlohmann::json;

// Native Ollama names a tool call by function name and passes its arguments
// as an object. The template bridge reads the OpenAI transcript, so each call
// gets a request-local id and each tool result the id of the call it answers.
std::string normalize_messages(Json& messages, bool& has_tool_history) {
    struct PendingCall {
        std::string name;
        std::string id;
        Json result;
    };
    std::vector<PendingCall> pending;
    size_t received = 0;
    std::unordered_set<std::string> ids;
    size_t next_id = 0;
    Json ordered = Json::array();
    for (auto& message : messages) {
        if (!message.is_object() || !message.contains("role") || !message["role"].is_string())
            return "messages[].role must be a string";
        const auto role = message["role"].get<std::string>();
        if (role != "system" && role != "user" && role != "assistant" && role != "tool")
            return "unsupported messages[].role";
        if (message.contains("content") && !message["content"].is_string() &&
            !message["content"].is_null())
            return "messages[].content must be a string";
        if (message.contains("images") && !message["images"].is_null() &&
            !message["images"].empty())
            return "messages[].images is unsupported";
        if (role != "tool" && !pending.empty())
            return "missing tool result for an assistant call";
        if (message.contains("tool_calls") && !message["tool_calls"].is_null()) {
            if (role != "assistant" || !message["tool_calls"].is_array())
                return "tool_calls must be an array on an assistant message";
            for (auto& call : message["tool_calls"]) {
                if (!call.is_object() || !call.contains("function") ||
                    !call["function"].is_object())
                    return "invalid tool call function";
                const auto& fn = call["function"];
                if (!fn.contains("name") || !fn["name"].is_string() ||
                    fn["name"].get_ref<const std::string&>().empty())
                    return "tool call requires function.name";
                if (!fn.contains("arguments") || !fn["arguments"].is_object())
                    return "tool call function.arguments must be an object";
                std::string id;
                if (call.contains("id")) {
                    if (!call["id"].is_string())
                        return "tool call id must be a string";
                    id = call["id"].get<std::string>();
                }
                if (id.empty()) {
                    // Some GGUF templates require exactly nine alphanumeric bytes.
                    // The 8 MiB body bound keeps the ordinal within eight digits.
                    do {
                        id = "X" + std::to_string(10000000 + ++next_id);
                    } while (ids.count(id));
                }
                if (!ids.insert(id).second)
                    return "duplicate tool call id";
                call["id"] = id;
                pending.push_back({fn["name"].get<std::string>(), std::move(id), nullptr});
                has_tool_history = true;
            }
        }
        if (role == "tool") {
            if (!message.contains("tool_name") || !message["tool_name"].is_string())
                return "tool message requires tool_name";
            const auto& name = message["tool_name"].get_ref<const std::string&>();
            const auto match = std::find_if(pending.begin(), pending.end(), [&](const auto& call) {
                return call.name == name && call.result.is_null();
            });
            if (match == pending.end())
                return "tool result has no matching assistant call";
            message["name"] = match->name;
            message["tool_call_id"] = match->id;
            match->result = std::move(message);
            if (++received == pending.size()) {
                // Results for repeated names bind in call order, and some
                // templates bind results positionally: render the complete
                // group in the order the assistant issued its calls.
                for (auto& call : pending)
                    ordered.push_back(std::move(call.result));
                pending.clear();
                received = 0;
            }
        } else {
            ordered.push_back(std::move(message));
        }
    }
    if (!pending.empty())
        return "missing tool result for an assistant call";
    if (!ordered.empty() && ordered.back()["role"] != "user" && ordered.back()["role"] != "tool")
        return "last message must be user or tool";
    messages = std::move(ordered);
    return {};
}

bool is_sampling_key(const std::string& key) {
    return key == kRepeatPenaltyAlias || find_api_sampling_field(key) != nullptr;
}

// Native options become the chat body's own fields. Every key is checked here,
// before compute admission; one xllama cannot apply is refused, not ignored.
std::string normalize_options(const Json& options, Json& body) {
    if (!options.is_object())
        return "options must be an object";
    for (const auto& [key, value] : options.items()) {
        if (value.is_null())
            continue;
        if (key == "num_predict") {
            if (!value.is_number())
                return "options.num_predict must be an integer";
            // -1 is Ollama's "no limit": xllama keeps its bounded default.
            if (value != -1)
                body["max_completion_tokens"] = value;
        } else if (key == "stop") {
            const bool strings =
                value.is_string() ||
                (value.is_array() && std::all_of(value.begin(), value.end(),
                                                 [](const Json& s) { return s.is_string(); }));
            if (!strings)
                return "options.stop must be a string or an array of strings";
            body[key] = value;
        } else if (is_sampling_key(key)) {
            if (!value.is_number())
                return "options." + key + " must be a number";
            body[key] = value;
        } else {
            return "unsupported options." + key;
        }
    }
    // The same ranges as an OpenAI request (api_sampling.h), checked now.
    return validate_api_sampling([&body](std::string_view key) -> std::optional<double> {
        const auto it = body.find(std::string(key));
        if (it == body.end() || !it->is_number())
            return std::nullopt;
        return it->get<double>();
    });
}

} // namespace

bool ollama_model_id(std::string_view id) {
    return !id.empty() && id != "." && id != ".." &&
           id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
                                "0123456789._-") == std::string_view::npos;
}

bool ollama_chat_model(std::string_view id, bool gguf, bool embedding) {
    return gguf && !embedding && ollama_model_id(id);
}

OllamaRequest parse_ollama_request(std::string_view body, OllamaRoute route) {
    OllamaRequest result;
    const auto fail = [&](const std::string& error) {
        result.error = error;
        return result;
    };
    const Json input = Json::parse(body.begin(), body.end(), nullptr, false);
    if (!input.is_object())
        return fail("invalid JSON body");
    if (!input.contains("model") || !input["model"].is_string() ||
        !ollama_model_id(input["model"].get_ref<const std::string&>()))
        return fail("model must be an id from /api/tags");
    const std::string model = input["model"].get<std::string>();
    if (input.contains("stream") && !input["stream"].is_null()) {
        if (!input["stream"].is_boolean())
            return fail("stream must be boolean");
        result.stream = input["stream"].get<bool>();
    }
    // Fields that would change the reply in a way xllama cannot honour are
    // refused when set to anything but their default.
    const auto requested = [&](const char* key, const Json& default_value) {
        return input.contains(key) && !input[key].is_null() && input[key] != default_value;
    };
    for (const char* key : {"format", "suffix", "template"}) {
        if (requested(key, ""))
            return fail(std::string(key) + " is unsupported");
    }
    for (const char* key : {"images", "context"}) {
        if (requested(key, Json::array()))
            return fail(std::string(key) + " is unsupported");
    }
    for (const char* key : {"raw", "logprobs"}) {
        if (requested(key, false))
            return fail(std::string(key) + " is unsupported");
    }
    if (requested("top_logprobs", 0))
        return fail("top_logprobs is unsupported");
    // `think` may restate what the model does; changing it is #648.
    if (input.contains("think") && !input["think"].is_null() &&
        (!input["think"].is_boolean() || input["think"].get<bool>() != model_is_thinking(model)))
        return fail(kOllamaThinkUnsupported);

    Json normalized{{"model", model}};
    if (input.contains("options") && !input["options"].is_null()) {
        if (const auto error = normalize_options(input["options"], normalized); !error.empty())
            return fail(error);
    }
    Json messages = Json::array();
    if (route == OllamaRoute::Chat) {
        if (input.contains("messages") && !input["messages"].is_null()) {
            if (!input["messages"].is_array())
                return fail("messages must be an array");
            messages = input["messages"];
        }
        bool has_tool_history = false;
        if (const auto error = normalize_messages(messages, has_tool_history); !error.empty())
            return fail(error);
        if (input.contains("tools") && !input["tools"].is_null())
            normalized["tools"] = input["tools"];
        result.replay_without_tools =
            has_tool_history && (!normalized.contains("tools") || normalized["tools"].empty());
    } else {
        for (const char* key : {"prompt", "system"}) {
            if (input.contains(key) && !input[key].is_null() && !input[key].is_string())
                return fail(std::string(key) + " must be a string");
        }
        const bool has_prompt = input.contains("prompt") && input["prompt"].is_string() &&
                                !input["prompt"].get_ref<const std::string&>().empty();
        if (has_prompt) {
            if (input.contains("system") && input["system"].is_string())
                messages.push_back({{"role", "system"}, {"content", input["system"]}});
            messages.push_back({{"role", "user"}, {"content", input["prompt"]}});
        }
    }
    if (messages.empty()) {
        result.stream = false; // Ollama answers a load with one JSON object
        result.action = OllamaAction::Load;
    }
    normalized["messages"] = std::move(messages);
    result.body = normalized.dump();
    return result;
}

} // namespace xllama
