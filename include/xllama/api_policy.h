// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#pragma once

namespace xllama {

// How a chat request names tools. The JSON parser fills this; the classifier
// stays free of WinRT and of llama.cpp so host tests can pin it.
enum class ApiToolChoice {
    Absent,
    None,
    Auto,
    Required,
    Named,
    Invalid,
};

// Presence of the OpenAI tool keys. Empty arrays are present and empty:
// `tools: []` is not a proposal.
struct ApiToolFields {
    bool has_tools = false;
    bool tools_empty = true;
    bool has_functions = false;
    bool functions_empty = true;
    ApiToolChoice choice = ApiToolChoice::Absent;
};

// What the chat handler should do. xllama still has no tool executor: a
// proposal is text in the HTTP response, and the client runs the tool.
enum class ApiToolDisposition {
    None,     // plain chat (no schema to offer, or tool_choice "none")
    Proposal, // `tools` to render and parse
    Legacy,   // `functions` translated into tools, then the same path
    Invalid,  // both keys, or a choice that names nothing
};

struct ApiToolClass {
    ApiToolDisposition disposition = ApiToolDisposition::None;
    // Static string when disposition is Invalid. Empty otherwise.
    const char* error = "";
};

// `tools` and `functions` together are invalid. `tool_choice: none` and an
// empty tool list fall through to plain chat. `required` or a named function
// with no schema is invalid.
inline ApiToolClass classify_api_tools(const ApiToolFields& fields) {
    ApiToolClass out;
    if (fields.has_tools && fields.has_functions) {
        out.disposition = ApiToolDisposition::Invalid;
        out.error = "tools and functions cannot be combined";
        return out;
    }
    if (fields.choice == ApiToolChoice::Invalid) {
        out.disposition = ApiToolDisposition::Invalid;
        out.error = "invalid tool_choice";
        return out;
    }
    if (fields.choice == ApiToolChoice::None)
        return out;

    const bool tools = fields.has_tools && !fields.tools_empty;
    const bool functions = fields.has_functions && !fields.functions_empty;
    const bool forcing =
        fields.choice == ApiToolChoice::Required || fields.choice == ApiToolChoice::Named;
    if (forcing && !tools && !functions) {
        out.disposition = ApiToolDisposition::Invalid;
        out.error = "tool_choice requires a tool";
        return out;
    }
    if (functions) {
        out.disposition = ApiToolDisposition::Legacy;
        return out;
    }
    if (tools)
        out.disposition = ApiToolDisposition::Proposal;
    return out;
}

} // namespace xllama
