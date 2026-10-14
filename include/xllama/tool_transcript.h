// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// OpenAI chat transcript for a client-side tool proposal. WinRT-free and
// llama-free: the LAN handler parses JSON into these types, and the GGUF
// template bridge is the only place that touches llama.cpp.
#pragma once

#include <string>
#include <vector>

namespace xllama {

enum class ChatRole { System, User, Assistant, Tool };

inline const char* chat_role_name(ChatRole role) {
    switch (role) {
    case ChatRole::System:
        return "system";
    case ChatRole::User:
        return "user";
    case ChatRole::Assistant:
        return "assistant";
    case ChatRole::Tool:
        return "tool";
    }
    return "";
}

enum class ToolChoice { Auto, None, Required, Named };

// One function the client is willing to run. `parameters_json` is a JSON object.
struct ToolSpec {
    std::string name;
    std::string description;
    std::string parameters_json = "{}";
};

// One call, already in the shape OpenAI puts on the wire: `arguments_json`
// is a string, not a nested object.
struct ToolCall {
    std::string id;
    std::string name;
    std::string arguments_json;
};

struct ChatMessage {
    ChatRole role = ChatRole::User;
    std::string content;
    std::vector<ToolCall> tool_calls;
    std::string tool_call_id;
    std::string name;
};

// A call the parser found, before ids are synthesized and before a truncated
// call is dropped. `complete` is false when the token cap cut the call.
struct IncomingCall {
    std::string id;
    std::string name;
    std::string arguments;
    bool complete = true;
};

struct AdmitResult {
    bool ok = false;
    std::string error;
};

// Reject a transcript that cannot be completed: no trailing user/tool turn,
// a tool result with no id, a tool with no name or a non-object schema, or a
// named tool_choice that is not one of the schemas. Does not render or generate.
AdmitResult admit_transcript(const std::vector<ChatMessage>& messages,
                             const std::vector<ToolSpec>& tools, ToolChoice choice,
                             const std::string& forced_name);

struct ChoiceAssembly {
    bool content_null = false;
    std::string content;
    std::vector<ToolCall> tool_calls;
    std::string finish_reason; // "stop", "length", or "tool_calls"
};

// Build the OpenAI choice. A complete call wins over the token cap. A call the
// cap cut in half is not emitted: finish_reason is "length" and `raw_text` is
// the content. Missing ids become call_1, call_2, … in parse order. Arguments
// are always a JSON string. `content` is null only when calls were emitted and
// the model wrote no text (`parsed_text` empty).
ChoiceAssembly assemble_choice(const std::string& raw_text, const std::string& parsed_text,
                               const std::vector<IncomingCall>& calls, bool hit_cap);

} // namespace xllama
