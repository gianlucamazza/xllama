// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "xllama/api_policy.h"

#include <cmath>
#include <limits>

// F01: the audit sent max_tokens -1, 1.5 and 2147483647 and got 200 for all three.

TEST_CASE("validate_max_tokens: whole numbers from 1 to n_ctx - 1") {
    using xllama::validate_max_tokens;
    CHECK(validate_max_tokens("max_tokens", 1, 2048).value == 1);
    CHECK(validate_max_tokens("max_tokens", 2047, 2048).value == 2047);
    CHECK(validate_max_tokens("max_tokens", 2047, 2048).error.empty());

    CHECK(validate_max_tokens("max_tokens", -1, 2048).error == "max_tokens must be at least 1");
    CHECK(validate_max_tokens("max_tokens", 0, 2048).error == "max_tokens must be at least 1");
    CHECK(validate_max_tokens("max_tokens", 1.5, 2048).error ==
          "max_tokens must be a whole number");
    CHECK(validate_max_tokens("max_completion_tokens", 2147483647.0, 2048).error ==
          "max_completion_tokens must be less than the model context (n_ctx 2048)");
    CHECK(validate_max_tokens("max_tokens", 2048, 2048).error ==
          "max_tokens must be less than the model context (n_ctx 2048)");
    CHECK_FALSE(validate_max_tokens("max_tokens", std::numeric_limits<double>::infinity(), 2048)
                    .error.empty());
    CHECK_FALSE(validate_max_tokens("max_tokens", std::nan(""), 2048).error.empty());
}

TEST_CASE("validate_api_number: range before any cast") {
    using xllama::validate_api_number;
    CHECK(validate_api_number("temperature", 0.7, 0, 2, false).empty());
    CHECK(validate_api_number("temperature", 2.5, 0, 2, false) ==
          "temperature must be between 0 and 2");
    CHECK(validate_api_number("top_p", -0.1, 0, 1, false) == "top_p must be between 0 and 1");
    CHECK(validate_api_number("seed", 42, -2147483648.0, 4294967295.0, true).empty());
    CHECK(validate_api_number("seed", -1, -2147483648.0, 4294967295.0, true).empty());
    CHECK(validate_api_number("seed", 1.5, -2147483648.0, 4294967295.0, true) ==
          "seed must be a whole number");
    CHECK(validate_api_number("seed", 1e20, -2147483648.0, 4294967295.0, true) ==
          "seed must be between -2147483648 and 4294967295");
    CHECK(validate_api_number("temperature", std::nan(""), 0, 2, false) ==
          "temperature must be a finite number");
}

TEST_CASE("api_response_format_supported: text, absent or null only") {
    using Kind = xllama::ApiResponseFormat::Kind;
    xllama::ApiResponseFormat f;
    CHECK(xllama::api_response_format_supported(f)); // absent
    f.kind = Kind::Null;                             // OpenAI's "not given"
    CHECK(xllama::api_response_format_supported(f));
    f = {Kind::Object, 1, true, "text"};
    CHECK(xllama::api_response_format_supported(f));
    // A schema next to "text" would be a constraint silently dropped.
    f = {Kind::Object, 2, true, "text"};
    CHECK_FALSE(xllama::api_response_format_supported(f));
    f = {Kind::Object, 1, true, "json_object"};
    CHECK_FALSE(xllama::api_response_format_supported(f));
    f = {Kind::Object, 1, false, ""}; // type is not a string
    CHECK_FALSE(xllama::api_response_format_supported(f));
    f = {Kind::Object, 0, false, ""};
    CHECK_FALSE(xllama::api_response_format_supported(f));
    f.kind = Kind::Other; // "text" as a bare string, a number, an array
    CHECK_FALSE(xllama::api_response_format_supported(f));
}
