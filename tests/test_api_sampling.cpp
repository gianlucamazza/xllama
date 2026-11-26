// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "xllama/api_sampling.h"
#include "xllama/sampling.h"

#include <cmath>
#include <map>
#include <string>

// #609: a client could not set top_k or repetition_penalty, and a top_k sent
// before was dropped silently.

namespace {

xllama::ApiNumberLookup lookup(const std::map<std::string, double>& body) {
    return [&body](std::string_view key) -> std::optional<double> {
        auto it = body.find(std::string(key));
        if (it == body.end())
            return std::nullopt;
        return it->second;
    };
}

} // namespace

TEST_CASE("apply_api_sampling: absent fields keep the defaults") {
    const std::map<std::string, double> body;
    xllama::SamplingConfig sc;
    CHECK(xllama::validate_api_sampling(lookup(body)).empty());
    xllama::apply_api_sampling(lookup(body), sc);
    CHECK(sc.temperature == xllama::sampling_defaults::kTemperature);
    CHECK(sc.top_k == xllama::sampling_defaults::kTopK);
    CHECK(sc.repetition_penalty == xllama::sampling_defaults::kRepetitionPenalty);
    CHECK(sc.seed == xllama::sampling_defaults::kSeed);
}

TEST_CASE("apply_api_sampling: a model card's settings reach the sampler config") {
    const std::map<std::string, double> body{
        {"temperature", 0.2}, {"top_k", 80}, {"repetition_penalty", 1.05}, {"seed", 7}};
    xllama::SamplingConfig sc;
    REQUIRE(xllama::validate_api_sampling(lookup(body)).empty());
    xllama::apply_api_sampling(lookup(body), sc);
    CHECK(sc.temperature == doctest::Approx(0.2));
    CHECK(sc.top_k == 80);
    CHECK(sc.repetition_penalty == doctest::Approx(1.05));
    CHECK(sc.seed == 7u);
    CHECK(sc.top_p == xllama::sampling_defaults::kTopP);
}

TEST_CASE("repeat_penalty is llama.cpp's spelling of repetition_penalty") {
    const std::map<std::string, double> alias{{"repeat_penalty", 1.1}};
    xllama::SamplingConfig sc;
    REQUIRE(xllama::validate_api_sampling(lookup(alias)).empty());
    xllama::apply_api_sampling(lookup(alias), sc);
    CHECK(sc.repetition_penalty == doctest::Approx(1.1));

    const std::map<std::string, double> both{{"repeat_penalty", 1.1}, {"repetition_penalty", 1.2}};
    CHECK(xllama::validate_api_sampling(lookup(both)) ==
          "pass only one of repetition_penalty and repeat_penalty");
    const std::map<std::string, double> bad_alias{{"repeat_penalty", 3}};
    CHECK(xllama::validate_api_sampling(lookup(bad_alias)) ==
          "repeat_penalty must be between 0 and 2");
}

TEST_CASE("validate_api_sampling: ranges and whole numbers, before any cast") {
    using M = std::map<std::string, double>;
    CHECK(xllama::validate_api_sampling(lookup(M{{"top_k", 1.5}})) ==
          "top_k must be a whole number");
    CHECK(xllama::validate_api_sampling(lookup(M{{"top_k", -1}})) ==
          "top_k must be between 0 and 2147483647");
    CHECK(xllama::validate_api_sampling(lookup(M{{"top_k", 1e12}})) ==
          "top_k must be between 0 and 2147483647");
    CHECK(xllama::validate_api_sampling(lookup(M{{"repetition_penalty", -0.5}})) ==
          "repetition_penalty must be between 0 and 2");
    CHECK(xllama::validate_api_sampling(lookup(M{{"temperature", 2.5}})) ==
          "temperature must be between 0 and 2");
    CHECK(xllama::validate_api_sampling(lookup(M{{"top_p", std::nan("")}})) ==
          "top_p must be a finite number");
    CHECK(xllama::validate_api_sampling(lookup(M{{"top_k", 0}})).empty()); // keeps every token
}

TEST_CASE("a negative seed keeps the default seed") {
    const std::map<std::string, double> body{{"seed", -1}};
    xllama::SamplingConfig sc;
    REQUIRE(xllama::validate_api_sampling(lookup(body)).empty());
    xllama::apply_api_sampling(lookup(body), sc);
    CHECK(sc.seed == xllama::sampling_defaults::kSeed);
}
