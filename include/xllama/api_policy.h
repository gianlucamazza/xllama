// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#pragma once

#include <cmath>
#include <string>

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

// max_tokens / max_completion_tokens as the JSON parser hands them over (a
// double), checked before any model load (F01). A whole number from 1 to
// n_ctx - 1: the reply plus at least one prompt token must fit the context.
// `error` is empty when `value` is usable.
struct ApiTokenLimit {
    int value = 0;
    std::string error;
};

inline ApiTokenLimit validate_max_tokens(const char* key, double v, int n_ctx) {
    ApiTokenLimit out;
    const std::string name(key);
    if (!std::isfinite(v) || v != std::floor(v))
        out.error = name + " must be a whole number";
    else if (v < 1)
        out.error = name + " must be at least 1";
    else if (v >= n_ctx)
        out.error =
            name + " must be less than the model context (n_ctx " + std::to_string(n_ctx) + ")";
    else
        out.value = static_cast<int>(v);
    return out;
}

// A sampling number from the JSON body, checked before any model load (F01):
// finite and inside [lo, hi], and whole when `whole`. Converting an
// out-of-range double to an integer is undefined behaviour, so the range check
// comes before any cast.
inline std::string validate_api_number(const char* key, double v, double lo, double hi,
                                       bool whole) {
    const std::string name(key);
    if (!std::isfinite(v) || (whole && v != std::floor(v)))
        return name + (whole ? " must be a whole number" : " must be a finite number");
    if (v < lo || v > hi) {
        auto text = [](double x) {
            std::string s = std::to_string(x);
            s.erase(s.find_last_not_of('0') + 1);
            if (!s.empty() && s.back() == '.')
                s.pop_back();
            return s;
        };
        return name + " must be between " + text(lo) + " and " + text(hi);
    }
    return std::string();
}

// Clamp a JSON number into [lo, hi] before converting it, for fields whose
// contract is "clamped, never rejected" (image steps and seed). NaN takes `lo`.
inline int clamp_api_int(double v, int lo, int hi) {
    if (!(v >= lo))
        return lo;
    if (v > hi)
        return hi;
    return static_cast<int>(v);
}

} // namespace xllama
