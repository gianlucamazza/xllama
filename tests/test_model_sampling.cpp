// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// Per-model sampling defaults and the order in which the layers apply.

#include "xllama/api_sampling.h"
#include "xllama/model_sampling.h"

#include <doctest/doctest.h>
#include <nlohmann/json.hpp>

#include <fstream>
#include <map>
#include <sstream>
#include <string>

using namespace xllama;

TEST_CASE("model sampling: nothing set keeps the global defaults") {
    const ResolvedSampling r = resolve_sampling({});
    CHECK(r.temperature == sampling_defaults::kTemperature);
    CHECK(r.top_p == sampling_defaults::kTopP);
    CHECK(r.top_k == sampling_defaults::kTopK);
    CHECK(r.repetition_penalty == sampling_defaults::kRepetitionPenalty);
    CHECK(r.temperature_source == SamplingSource::Default);
    CHECK(std::string(sampling_source_name(r.top_k_source)) == "default");
}

TEST_CASE("model sampling: the model overrides the defaults, the user overrides the model") {
    SamplingOverrides model;
    model.temperature = 0.1f;
    model.top_k = 50;
    model.repetition_penalty = 1.05f;
    SamplingOverrides user;
    user.temperature = 0.7f;
    const ResolvedSampling r = resolve_sampling(model, user);
    CHECK(r.temperature == doctest::Approx(0.7f));
    CHECK(r.temperature_source == SamplingSource::User);
    CHECK(r.top_k == 50);
    CHECK(r.top_k_source == SamplingSource::Model);
    CHECK(r.repetition_penalty == doctest::Approx(1.05f));
    CHECK(r.top_p == sampling_defaults::kTopP); // neither layer set it
    CHECK(r.top_p_source == SamplingSource::Default);
    CHECK(std::string(sampling_source_name(r.repetition_penalty_source)) == "model");
}

TEST_CASE("model sampling: a request field wins, an absent one keeps the model's value") {
    SamplingOverrides model;
    model.temperature = 0.1f;
    model.top_k = 50;
    SamplingConfig params;
    apply_resolved_sampling(resolve_sampling(model), params);
    // The LAN API applies the request last, onto the resolved values.
    const std::map<std::string, double> request = {{"temperature", 0.9}};
    apply_api_sampling(
        [&](std::string_view key) -> std::optional<double> {
            const auto it = request.find(std::string(key));
            return it == request.end() ? std::nullopt : std::optional<double>(it->second);
        },
        params);
    CHECK(params.temperature == doctest::Approx(0.9f));
    CHECK(params.top_k == 50);
    CHECK(params.seed == sampling_defaults::kSeed); // not a model default
}

TEST_CASE("model sampling: parse a catalogue sampling object") {
    std::string error;
    const auto parsed = parse_sampling_overrides(
        R"({"temperature": 0.1, "top_k": 50, "repetition_penalty": 1.05})", &error);
    REQUIRE_MESSAGE(parsed, error);
    CHECK(parsed->temperature == doctest::Approx(0.1f));
    CHECK(parsed->top_k == 50);
    CHECK(parsed->repetition_penalty == doctest::Approx(1.05f));
    CHECK_FALSE(parsed->top_p);
    const auto empty = parse_sampling_overrides("{}", &error);
    REQUIRE(empty);
    CHECK(empty->empty());
}

TEST_CASE("model sampling: a value the API would refuse is no model default") {
    const char* bad[] = {
        R"({"temperature": 3})",  // above the API range
        R"({"top_k": 1.5})",      // not whole
        R"({"top_p": "0.9"})",    // not a number
        R"({"seed": 7})",         // per request, not per model
        R"({"min_p": 0.05})",     // not a field xllama samples with
        R"([0.1])",               // not an object
        R"({"temperature": 0.1)", // not JSON
    };
    for (const char* text : bad) {
        std::string error;
        CHECK_MESSAGE(!parse_sampling_overrides(text, &error), text);
        CHECK_MESSAGE(!error.empty(), text);
    }
    std::string error;
    parse_sampling_overrides(R"({"temperature": 3})", &error);
    CHECK(error.find("sampling.temperature") == 0);
}

TEST_CASE("model sampling: every sampling object in the bundled catalogue parses") {
    std::ifstream in(MANIFEST_JSON_PATH);
    REQUIRE(in);
    std::stringstream text;
    text << in.rdbuf();
    const nlohmann::json root = nlohmann::json::parse(text.str());
    int checked = 0;
    for (const nlohmann::json& entry : root.at("models")) {
        if (!entry.contains("sampling"))
            continue;
        std::string error;
        const std::string name = entry.value("name", std::string("?"));
        CHECK_MESSAGE(parse_sampling_overrides(entry.at("sampling").dump(), &error),
                      (name + ": " + error));
        ++checked;
    }
    MESSAGE("catalogue entries with sampling: " << checked);
}
