// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <string>
#include <variant>

namespace xllama {

// The only explicit chat response_format both backends satisfy. /health lists
// it under "response_formats".
inline constexpr char kTextResponseFormat[] = "text";

// A chat request's `response_format`, as the JSON parser saw it.
struct ApiResponseFormat {
    enum class Kind { Absent, Null, Object, Other };
    Kind kind = Kind::Absent;
    std::size_t members = 0;     // keys of the object
    bool type_is_string = false; // `type` present and a string
    std::string type;
};

// Absent or null (OpenAI's "not given") is plain text. An object must be
// exactly {"type":"text"}: an extra key such as a JSON schema would be a
// constraint this server silently drops, so it is refused instead.
inline bool api_response_format_supported(const ApiResponseFormat& f) {
    using Kind = ApiResponseFormat::Kind;
    if (f.kind == Kind::Absent || f.kind == Kind::Null)
        return true;
    return f.kind == Kind::Object && f.members == 1 && f.type_is_string &&
           f.type == kTextResponseFormat;
}

// Values of OpenAI generation fields. Absent and JSON null both map to nullptr.
// Strings, arrays and objects map to monostate so they cannot match a default.
using ApiGenerationValue = std::variant<std::nullptr_t, bool, double, std::monostate>;

template <typename Lookup> const char* unsupported_openai_generation_field(Lookup lookup) {
    static constexpr struct {
        const char* key;
        ApiGenerationValue value;
    } defaults[] = {{"n", 1.0},
                    {"logprobs", false},
                    {"top_logprobs", 0.0},
                    {"presence_penalty", 0.0},
                    {"frequency_penalty", 0.0}};
    for (const auto& field : defaults) {
        const ApiGenerationValue value = lookup(field.key);
        if (!std::holds_alternative<std::nullptr_t>(value) && value != field.value)
            return field.key;
    }
    return nullptr;
}

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

// The smallest context a request may load (options.num_ctx, chat and
// embeddings alike).
inline constexpr int kMinContextOverride = 32;

// options.num_ctx only reduces a load: a whole number from 32 to the
// catalogue context |configured| (catalogue_n_ctx), which it never raises.
// A different value is a different load (same_load), so the hub reloads.
inline std::string validate_context_override(double value, int configured) {
    return validate_api_number("options.num_ctx", value, kMinContextOverride, configured, true);
}

// The reply limit of a request that sets none: 512, or half the context when
// a reduced context is 1024 or less, so the reply reserve (prompt_budget.h)
// still leaves the prompt room.
inline int default_max_tokens(int n_ctx) {
    return n_ctx > 1024 ? 512 : std::max(1, n_ctx / 2);
}

} // namespace xllama
