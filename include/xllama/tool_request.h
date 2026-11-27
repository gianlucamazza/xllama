// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Parse an OpenAI chat body into a tool transcript, or into the plain-chat
// turns ChatFormat renders. The LAN handler keeps WinRT for the HTTP frame;
// these parsers are what host tests exercise for roles, content arrays, null
// content, and legacy `functions`.
#pragma once

#include "xllama/api_policy.h"
#include "xllama/chat_prompt.h"
#include "xllama/tool_transcript.h"

#include <string>
#include <vector>

namespace xllama {

struct ToolRequest {
    ApiToolDisposition disposition = ApiToolDisposition::None;
    std::string error;
    std::vector<ChatMessage> messages;
    std::vector<ToolSpec> tools;
    ToolChoice choice = ToolChoice::Auto;
    std::string forced_name;
};

// `error` is set on a malformed body. Plain chat (`None`) does not parse
// message content here: parse_plain_chat does, with the same content rules.
// |replay_without_tools| (a native Ollama transcript that replays tool calls
// and results but offers no tools, api_ollama.h) turns what would be plain
// chat into a proposal with ToolChoice::None, so the GGUF template renders the
// tool turns plain chat cannot.
ToolRequest parse_tool_request(const std::string& body, bool replay_without_tools = false);

// A plain-chat messages[] folded into ChatFormat::render_prompt's inputs. Both
// parsers read a message the same way: `developer` (OpenAI's newer name for
// the system role) is a system message, and content is a string, null, or an
// array of {type:"text",text} parts joined with no separator. Any other part,
// an image included, sets `error` to "unsupported message content". System and
// developer contents join with "\n" in order; the trailing user message is
// `final_user` (empty when the transcript does not end on one). Roles plain
// chat cannot render (`tool`, unknown names) are skipped, as before.
struct PlainChatRequest {
    std::string error;
    std::string system;
    std::vector<ChatTurn> history;
    std::string final_user;
};

PlainChatRequest parse_plain_chat(const std::string& body);

} // namespace xllama
