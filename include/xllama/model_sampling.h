// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// Sampling defaults per model, and the one order in which they apply.
//
// A model can carry recommended sampling values, as its card or its
// publisher's generation config states them. They live in the catalogue
// (uwp/models/manifest.json, an entry's optional "sampling" object), because
// llama.cpp does not read sampling defaults from a GGUF, and a client should
// not keep its own copy. Every surface applies the same layers, each over the
// one before:
//
//   global (xllama/sampling.h)  <  model (catalogue)  <  user (Settings)  <  request
//
// The request layer is the LAN API's (xllama/api_sampling.h,
// apply_api_sampling); it is applied last onto what resolve_sampling gives, so
// an absent request field keeps the model's value. A field left unset in a
// layer inherits from the layer below. The seed is not a model default: it
// stays per request and per Settings.
//
// WinRT-free and host-tested (tests/test_model_sampling.cpp).
#pragma once

#include "xllama/sampling.h"

#include <optional>
#include <string>
#include <string_view>

namespace xllama {

// Values one layer sets; nullopt inherits from the layer below.
struct SamplingOverrides {
    std::optional<float> temperature;
    std::optional<float> top_p;
    std::optional<int> top_k;
    std::optional<float> repetition_penalty;

    bool empty() const {
        return !temperature && !top_p && !top_k && !repetition_penalty;
    }
    bool operator==(const SamplingOverrides& o) const {
        return temperature == o.temperature && top_p == o.top_p && top_k == o.top_k &&
               repetition_penalty == o.repetition_penalty;
    }
};

enum class SamplingSource { Default, Model, User };

// "default", "model" or "user", as /v1/models reports a value's source.
const char* sampling_source_name(SamplingSource source);

// The effective values and the layer each one came from.
struct ResolvedSampling {
    float temperature = sampling_defaults::kTemperature;
    float top_p = sampling_defaults::kTopP;
    int top_k = sampling_defaults::kTopK;
    float repetition_penalty = sampling_defaults::kRepetitionPenalty;
    SamplingSource temperature_source = SamplingSource::Default;
    SamplingSource top_p_source = SamplingSource::Default;
    SamplingSource top_k_source = SamplingSource::Default;
    SamplingSource repetition_penalty_source = SamplingSource::Default;
};

// The global defaults, then |model|, then |user|.
ResolvedSampling resolve_sampling(const SamplingOverrides& model,
                                  const SamplingOverrides& user = {});

// Copies the four resolved values into |params| (GenerateParams,
// InferenceParams, SamplingConfig: any struct with those field names). The
// seed and greedy are left alone.
template <class Params> void apply_resolved_sampling(const ResolvedSampling& r, Params& params) {
    params.temperature = r.temperature;
    params.top_p = r.top_p;
    params.top_k = r.top_k;
    params.repetition_penalty = r.repetition_penalty;
}

// Parses a catalogue entry's "sampling" object. Only the four keys are
// allowed, each a number in the range a request may give
// (kApiSamplingFields): a model default the API would refuse is no default.
// nullopt and |error| set (naming the field) on anything else.
std::optional<SamplingOverrides> parse_sampling_overrides(std::string_view json_object,
                                                          std::string* error);

} // namespace xllama
