// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// The sampling fields a LAN API chat request may set, in one table (#609).
//
// The table is the single source for three things: what a request is
// validated against (before any model load, F01), what it copies into the
// generation parameters, and what /health reports as accepted
// (api_discovery.cpp). The defaults stay in xllama/sampling.h; an absent field
// keeps them, so a request that sets none of these behaves as before.
//
// WinRT-free: the server passes a lookup over its JSON object, and host tests
// pass one over a map.
#pragma once

#include "xllama/api_policy.h"

#include <array>
#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <string_view>

namespace xllama {

struct ApiSamplingField {
    const char* key;
    double lo, hi;
    bool whole;
};

// temperature and top_p use OpenAI's ranges. top_k 0 keeps every token, as in
// llama.cpp. repetition_penalty: 1 is no penalty, 0 drops the stage
// (sampler_chain.h); above 2 a model only degrades. A seed is a whole number,
// and a negative one (llama.cpp's "-1 = random") keeps the default seed.
inline constexpr std::array<ApiSamplingField, 5> kApiSamplingFields = {{
    {"temperature", 0.0, 2.0, false},
    {"top_p", 0.0, 1.0, false},
    {"top_k", 0.0, 2147483647.0, true},
    {"repetition_penalty", 0.0, 2.0, false},
    {"seed", -2147483648.0, 4294967295.0, true},
}};

// llama.cpp's server spelling of repetition_penalty, accepted as an alias.
inline constexpr const char* kRepeatPenaltyAlias = "repeat_penalty";

// A request's number for |key|, or nullopt when the key is absent.
using ApiNumberLookup = std::function<std::optional<double>(std::string_view key)>;

// The value a request gives for |field|, through the alias for
// repetition_penalty. Both spellings at once is an error (|error| set).
inline std::optional<double> api_sampling_value(const ApiNumberLookup& get,
                                                const ApiSamplingField& field, std::string& error) {
    std::optional<double> value = get(field.key);
    if (std::string_view(field.key) == "repetition_penalty") {
        const std::optional<double> alias = get(kRepeatPenaltyAlias);
        if (value && alias) {
            error = std::string("pass only one of repetition_penalty and ") + kRepeatPenaltyAlias;
            return std::nullopt;
        }
        if (alias)
            value = alias;
    }
    return value;
}

// Empty when every sampling field present is valid, else the first error,
// naming the field as the request spelled it.
inline std::string validate_api_sampling(const ApiNumberLookup& get) {
    for (const ApiSamplingField& field : kApiSamplingFields) {
        std::string error;
        const std::optional<double> value = api_sampling_value(get, field, error);
        if (!error.empty())
            return error;
        if (!value)
            continue;
        const bool via_alias = std::string_view(field.key) == "repetition_penalty" &&
                               !get(field.key) && get(kRepeatPenaltyAlias);
        error = validate_api_number(via_alias ? kRepeatPenaltyAlias : field.key, *value, field.lo,
                                    field.hi, field.whole);
        if (!error.empty())
            return error;
    }
    return std::string();
}

// Copies the fields validate_api_sampling accepted into |params| (any struct
// with SamplingConfig's field names: GenerateParams, InferenceParams,
// SamplingConfig). Absent fields keep what |params| already holds.
template <class Params> void apply_api_sampling(const ApiNumberLookup& get, Params& params) {
    std::string error;
    for (const ApiSamplingField& field : kApiSamplingFields) {
        const std::optional<double> value = api_sampling_value(get, field, error);
        if (!value)
            continue;
        const std::string_view key(field.key);
        if (key == "temperature")
            params.temperature = static_cast<float>(*value);
        else if (key == "top_p")
            params.top_p = static_cast<float>(*value);
        else if (key == "top_k")
            params.top_k = static_cast<int>(*value);
        else if (key == "repetition_penalty")
            params.repetition_penalty = static_cast<float>(*value);
        else if (key == "seed" && *value >= 0)
            params.seed = static_cast<uint32_t>(*value);
    }
}

} // namespace xllama
