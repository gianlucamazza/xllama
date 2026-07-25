// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#ifdef _WIN32
    #include "xllama/utf8_utils.h"

TEST_CASE("UTF-8 roundtrip: ASCII") {
    std::string original = "Hello, World!";
    auto w = xllama::utf8_to_wstring(original);
    auto back = xllama::wstring_to_utf8(w);
    CHECK(back == original);
}

TEST_CASE("UTF-8 roundtrip: empty") {
    std::string original;
    auto w = xllama::utf8_to_wstring(original);
    auto back = xllama::wstring_to_utf8(w);
    CHECK(back == original);
}

TEST_CASE("UTF-8 roundtrip: multi-byte") {
    std::string original = u8"\u00e9\u00e0\u00fc"; // éàü
    auto w = xllama::utf8_to_wstring(original);
    auto back = xllama::wstring_to_utf8(w);
    CHECK(back == original);
}

#else
TEST_CASE("UTF-8 utils stub on non-Windows") {
    CHECK(true); // no-op on Linux
}
#endif
