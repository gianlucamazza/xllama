// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#pragma once

namespace xllama {

/// Parsed tool-related request fields. The transport parser owns JSON typing;
/// this small policy stays host-testable and has no WinRT dependency.
struct ApiToolFields {
    bool tools = false;
    bool functions = false;
    bool tool_choice = false;
};

/// xllama has no tool executor. Any advertised tool capability is rejected so
/// clients cannot mistake local inference for permission to cause side effects.
inline bool api_tool_execution_requested(const ApiToolFields& fields) {
    return fields.tools || fields.functions || fields.tool_choice;
}

} // namespace xllama
