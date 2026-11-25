// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#include "xllama/local_knob.h"

#include <doctest/doctest.h>

#include <climits>
#include <string>

using xllama::parse_knob_int;
using xllama::parse_knob_int_or;

TEST_CASE("parse_knob_int accepts a plain integer with surrounding whitespace") {
    CHECK(parse_knob_int("6") == 6);
    CHECK(parse_knob_int("  12\r\n") == 12);
    CHECK(parse_knob_int("\t-1 ") == -1);
    CHECK(parse_knob_int("+3") == 3);
    CHECK(parse_knob_int("0") == 0);
}

TEST_CASE("parse_knob_int rejects text atoi would turn into 0 or a prefix") {
    CHECK_FALSE(parse_knob_int("").has_value());
    CHECK_FALSE(parse_knob_int("   \n").has_value());
    CHECK_FALSE(parse_knob_int("auto").has_value());
    CHECK_FALSE(parse_knob_int("6 threads").has_value());
    CHECK_FALSE(parse_knob_int("1.5").has_value());
    CHECK_FALSE(parse_knob_int("-").has_value());
    CHECK_FALSE(parse_knob_int("0x10").has_value());
}

TEST_CASE("parse_knob_int rejects values outside int") {
    CHECK(parse_knob_int(std::to_string(INT_MAX)) == INT_MAX);
    CHECK(parse_knob_int(std::to_string(INT_MIN)) == INT_MIN);
    CHECK_FALSE(parse_knob_int("2147483648").has_value());
    CHECK_FALSE(parse_knob_int("-2147483649").has_value());
    CHECK_FALSE(parse_knob_int("99999999999999999999").has_value());
}

TEST_CASE("parse_knob_int_or falls back when the knob is absent or unparsable") {
    CHECK(parse_knob_int_or("", -1) == -1);
    CHECK(parse_knob_int_or("auto", -1) == -1);
    CHECK(parse_knob_int_or("4\n", -1) == 4);
    // 0 is a real setting for gguf_gpu_layers.txt (all layers on CPU); only a
    // parsed "0" may produce it.
    CHECK(parse_knob_int_or("0", -1) == 0);
}
