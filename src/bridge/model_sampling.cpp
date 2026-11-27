// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#include "xllama/model_sampling.h"

#include "xllama/api_policy.h"
#include "xllama/api_sampling.h"

#include <nlohmann/json.hpp>

#include <cmath>
#include <string_view>

namespace xllama {

const char* sampling_source_name(SamplingSource source) {
    switch (source) {
    case SamplingSource::Model:
        return "model";
    case SamplingSource::User:
        return "user";
    case SamplingSource::Default:
        break;
    }
    return "default";
}

namespace {

template <typename T>
void layer(const std::optional<T>& value, SamplingSource source, T& out, SamplingSource& from) {
    if (value) {
        out = *value;
        from = source;
    }
}

void apply_layer(const SamplingOverrides& o, SamplingSource source, ResolvedSampling& r) {
    layer(o.temperature, source, r.temperature, r.temperature_source);
    layer(o.top_p, source, r.top_p, r.top_p_source);
    layer(o.top_k, source, r.top_k, r.top_k_source);
    layer(o.repetition_penalty, source, r.repetition_penalty, r.repetition_penalty_source);
}

const ApiSamplingField* field_named(std::string_view key) {
    for (const ApiSamplingField& field : kApiSamplingFields)
        if (key == field.key)
            return &field;
    return nullptr;
}

} // namespace

ResolvedSampling resolve_sampling(const SamplingOverrides& model, const SamplingOverrides& user) {
    ResolvedSampling r;
    apply_layer(model, SamplingSource::Model, r);
    apply_layer(user, SamplingSource::User, r);
    return r;
}

std::optional<SamplingOverrides> parse_sampling_overrides(std::string_view json_object,
                                                          std::string* error) {
    auto fail = [error](std::string why) -> std::optional<SamplingOverrides> {
        if (error)
            *error = std::move(why);
        return std::nullopt;
    };
    nlohmann::json obj;
    try {
        obj = nlohmann::json::parse(json_object);
    } catch (const std::exception&) {
        return fail("sampling is not JSON");
    }
    if (!obj.is_object())
        return fail("sampling is not an object");
    SamplingOverrides out;
    for (auto it = obj.begin(); it != obj.end(); ++it) {
        const std::string& key = it.key();
        const ApiSamplingField* field = field_named(key);
        // The seed is per request: a model has no default seed.
        if (!field || key == "seed")
            return fail("sampling." + key + " is not a model sampling field");
        if (!it->is_number())
            return fail("sampling." + key + " must be a number");
        const double value = it->get<double>();
        const std::string bad =
            validate_api_number(key.c_str(), value, field->lo, field->hi, field->whole);
        if (!bad.empty())
            return fail("sampling." + bad);
        if (key == "temperature")
            out.temperature = static_cast<float>(value);
        else if (key == "top_p")
            out.top_p = static_cast<float>(value);
        else if (key == "top_k")
            out.top_k = static_cast<int>(value);
        else if (key == "repetition_penalty")
            out.repetition_penalty = static_cast<float>(value);
    }
    return out;
}

} // namespace xllama
