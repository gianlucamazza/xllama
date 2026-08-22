// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "xllama/json_utils.h"

#include <doctest/doctest.h>
#include <string>

TEST_CASE("json_escape: quotes and backslash") {
    CHECK(xllama::json_escape("hello") == "hello");
    CHECK(xllama::json_escape("say \"hi\"") == "say \\\"hi\\\"");
    CHECK(xllama::json_escape("path\\to\\file") == "path\\\\to\\\\file");
}

TEST_CASE("json_escape: control chars → \\uXXXX") {
    CHECK(xllama::json_escape("a\nb") == "a\\nb");
    CHECK(xllama::json_escape("a\tb") == "a\\tb");
    CHECK(xllama::json_escape("a\rb") == "a\\rb");
    CHECK(xllama::json_escape("a\bb") == "a\\bb");
    CHECK(xllama::json_escape("a\fb") == "a\\fb");
    // Other control chars → \uXXXX
    std::string ctrl01;
    ctrl01 += '\x01';
    CHECK(xllama::json_escape("a" + ctrl01 + "b") == "a\\u0001b");
    std::string ctrl1f;
    ctrl1f += '\x1f';
    CHECK(xllama::json_escape("a" + ctrl1f + "b") == "a\\u001fb");
}

TEST_CASE("json_escape: mixed content") {
    std::string input = "line1\nline2\ttab \"quote\" back\\slash";
    std::string expected = "line1\\nline2\\ttab \\\"quote\\\" back\\\\slash";
    CHECK(xllama::json_escape(input) == expected);
}

TEST_CASE("json_read_string: basic escapes") {
    std::string json = "\"hello \\\"world\\\"\"";
    size_t pos = 0;
    // Skip opening quote
    ++pos;
    std::string out;
    CHECK(xllama::json_read_string(json, pos, out));
    CHECK(out == "hello \"world\"");
    // pos should be past closing quote
    CHECK(json[pos] == '\0');
}

TEST_CASE("json_read_string: all standard escapes") {
    std::string json = "\"\\n\\t\\r\\\\\\\"\\b\\f\"";
    size_t pos = 0;
    ++pos;
    std::string out;
    CHECK(xllama::json_read_string(json, pos, out));
    CHECK(out.size() == 7);
    CHECK(out[0] == '\n');
    CHECK(out[1] == '\t');
    CHECK(out[2] == '\r');
    CHECK(out[3] == '\\');
    CHECK(out[4] == '"');
    CHECK(out[5] == '\b');
    CHECK(out[6] == '\f');
}

TEST_CASE("json_read_string: \\uXXXX decode") {
    std::string json = "\"hello\\u0020world\"";
    size_t pos = 0;
    ++pos;
    std::string out;
    CHECK(xllama::json_read_string(json, pos, out));
    CHECK(out == "hello world");
}

TEST_CASE("json_read_string: \\uXXXX with UTF-8") {
    // é = U+00E9 → UTF-8: 0xC3 0xA9
    std::string json = "\"caf\\u00e9\"";
    size_t pos = 0;
    ++pos;
    std::string out;
    CHECK(xllama::json_read_string(json, pos, out));
    CHECK(out == "caf\u00e9");
}

TEST_CASE("json_read_string: surrogate pair → BMP") {
    // 😀 = U+1F600 = high surrogate D83D + low surrogate DE00
    std::string json = "\"\\ud83d\\ude00\"";
    size_t pos = 0;
    ++pos;
    std::string out;
    CHECK(xllama::json_read_string(json, pos, out));
    // U+1F600 encoded as UTF-8: F0 9F 98 80
    CHECK(out.size() == 4);
    CHECK(out[0] == '\xf0');
    CHECK(out[1] == '\x9f');
    CHECK(out[2] == '\x98');
    CHECK(out[3] == '\x80');
}

TEST_CASE("json_read_string: lone surrogate → replacement") {
    std::string json = "\"\\ud800\"";
    size_t pos = 0;
    ++pos;
    std::string out;
    CHECK(xllama::json_read_string(json, pos, out));
    CHECK(out == "\xef\xbf\xbd"); // U+FFFD
}

TEST_CASE("json_read_string: unterminated returns false") {
    std::string json = "\"hello";
    size_t pos = 0;
    ++pos;
    std::string out;
    CHECK_FALSE(xllama::json_read_string(json, pos, out));
}

TEST_CASE("json_read_string: unknown escape → lenient passthrough") {
    std::string json = "\"\\x\"";
    size_t pos = 0;
    ++pos;
    std::string out;
    CHECK(xllama::json_read_string(json, pos, out));
    CHECK(out == "x");
}

TEST_CASE("round-trip: escape → read_string") {
    std::string original = "hello \"world\"\n\t\001\x1f";
    std::string escaped = xllama::json_escape(original);
    // The escaped string is a valid JSON string value (with surrounding quotes)
    std::string json = "\"" + escaped + "\"";
    size_t pos = 0;
    ++pos; // skip opening "
    std::string decoded;
    CHECK(xllama::json_read_string(json, pos, decoded));
    CHECK(decoded == original);
}

TEST_CASE("round-trip: escape → read_string with UTF-8") {
    std::string original = "caf\u00e9 \u2603"; // café ☃
    std::string escaped = xllama::json_escape(original);
    std::string json = "\"" + escaped + "\"";
    size_t pos = 0;
    ++pos;
    std::string decoded;
    CHECK(xllama::json_read_string(json, pos, decoded));
    CHECK(decoded == original);
}