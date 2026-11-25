// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#include "xllama/local_knob.h"

#include <doctest/doctest.h>

#include <climits>
#include <filesystem>
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

TEST_CASE("normalize_knob_text drops CRLF and trailing line ends, keeps the body") {
    using xllama::normalize_knob_text;
    CHECK(normalize_knob_text("1\r\n") == "1");
    CHECK(normalize_knob_text("line one\r\nline two \r\n\r\n") == "line one\nline two");
    CHECK(normalize_knob_text("  keep leading") == "  keep leading");
    CHECK(normalize_knob_text("a\rb") == "a\rb"); // a lone CR is data
    CHECK(normalize_knob_text("") == "");
}

namespace {
// On Linux resolve_local_path is the identity, so an absolute path is a
// LocalState name for the host build.
std::string temp_knob_path(const char* leaf) {
    return (std::filesystem::temp_directory_path() / leaf).string();
}
} // namespace

TEST_CASE("LocalState text round trip, absent file and fallbacks") {
    const std::string path = temp_knob_path("xllama-test-knob.txt");
    std::filesystem::remove(path);
    CHECK_FALSE(xllama::read_local_file(path).has_value());
    CHECK(xllama::read_local_text(path).empty());
    CHECK(xllama::read_local_text_or(path, "dflt") == "dflt");
    CHECK(xllama::read_local_int(path, -1) == -1);

    REQUIRE(xllama::write_local_text(path, "line one\r\nline two\r\n"));
    REQUIRE(xllama::read_local_file(path).has_value());
    CHECK(*xllama::read_local_file(path) == "line one\nline two");

    REQUIRE(xllama::write_local_text(path, ""));
    CHECK(xllama::read_local_file(path).has_value()); // present but empty
    CHECK(xllama::read_local_text_or(path, "dflt") == "dflt");

    REQUIRE(xllama::write_local_text(path, " 12\n"));
    CHECK(xllama::read_local_int(path, -1) == 12);
    CHECK(xllama::read_local_int_in(path, 1, 1, 50) == 12);
    CHECK(xllama::read_local_int_in(path, 1, 1, 10) == 1); // out of range
    REQUIRE(xllama::write_local_text(path, "auto"));
    CHECK(xllama::read_local_int(path, -1) == -1);
    CHECK(xllama::read_local_int_in(path, 1, 1, 50) == 1);
    std::filesystem::remove(path);
}

TEST_CASE("write_local_text reports a file it cannot create") {
    CHECK_FALSE(xllama::write_local_text(temp_knob_path("no-such-dir/x.txt"), "x"));
}
