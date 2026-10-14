// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Parse an OpenAI chat body into a tool transcript. The LAN handler keeps
// WinRT for the HTTP frame; this parser is what host tests exercise for
// content arrays, null content, and legacy `functions`.
#pragma once

#include "xllama/api_policy.h"
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
// message content: that path still uses the string-only splitter.
ToolRequest parse_tool_request(const std::string& body);

} // namespace xllama
