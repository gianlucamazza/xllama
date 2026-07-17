// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

#include "xllama/preference_capture.h"

#include <doctest/doctest.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>

TEST_CASE("preference_label_valid accepts known labels") {
    CHECK(xllama::preference_label_valid("like"));
    CHECK(xllama::preference_label_valid("dislike"));
    CHECK(xllama::preference_label_valid("correction"));
    CHECK(xllama::preference_label_valid("implicit"));
    CHECK_FALSE(xllama::preference_label_valid("meh"));
    CHECK_FALSE(xllama::preference_label_valid(""));
}

TEST_CASE("format_preference_sample_jsonl shapes a JSONL line") {
    std::vector<std::pair<std::string, std::string>> msgs = {
        {"user", "hello \"world\""},
        {"assistant", "hi\nthere"},
    };
    const std::string line =
        xllama::format_preference_sample_jsonl("like", msgs, {}, "2026-07-17T00:00:00Z");
    REQUIRE(!line.empty());
    CHECK(line.find("\"label\":\"like\"") != std::string::npos);
    CHECK(line.find("\"ts\":\"2026-07-17T00:00:00Z\"") != std::string::npos);
    CHECK(line.find("hello \\\"world\\\"") != std::string::npos);
    CHECK(line.find("hi\\nthere") != std::string::npos);
    CHECK(line.front() == '{');
    CHECK(line.back() == '}');
}

TEST_CASE("format_preference_sample_jsonl rejects empty messages") {
    CHECK(xllama::format_preference_sample_jsonl("like", {}).empty());
}

TEST_CASE("append_preference_sample_file writes and appends") {
    auto path = std::filesystem::temp_directory_path() / "xllama-pref-test.jsonl";
    std::error_code ec;
    std::filesystem::remove(path, ec);
    const auto line1 = xllama::format_preference_sample_jsonl(
        "like", {{"user", "a"}, {"assistant", "b"}}, {}, "t1");
    const auto line2 = xllama::format_preference_sample_jsonl(
        "dislike", {{"user", "c"}, {"assistant", "d"}}, {}, "t2");
    CHECK(xllama::append_preference_sample_file(path.string(), line1));
    CHECK(xllama::append_preference_sample_file(path.string(), line2));
    std::ifstream in(path);
    std::string all((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    CHECK(all.find("\"label\":\"like\"") != std::string::npos);
    CHECK(all.find("\"label\":\"dislike\"") != std::string::npos);
    // two lines
    const auto n = std::count(all.begin(), all.end(), '\n');
    CHECK(n >= 2);
    std::filesystem::remove(path, ec);
}
