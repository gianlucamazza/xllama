// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// Strict integer parse for operator knobs (LocalState `*.txt` files and env
// vars). `atoi` turned "auto" or "6 threads" into 0 or 6, and 0 is a real
// setting for some knobs (gguf_gpu_layers.txt = all layers on the CPU), so a
// typo silently changed placement. A knob is a whole decimal integer with
// optional surrounding whitespace, or it is absent.
#pragma once

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>

namespace xllama {

inline std::optional<int> parse_knob_int(std::string_view text) {
    constexpr std::string_view kSpace = " \t\r\n";
    const std::size_t first = text.find_first_not_of(kSpace);
    if (first == std::string_view::npos)
        return std::nullopt;
    const std::size_t last = text.find_last_not_of(kSpace);
    const std::string s(text.substr(first, last - first + 1));
    // strtol accepts leading whitespace and a "0x" prefix only with base 0;
    // base 10 plus the end check below leaves exactly [+-]digits.
    const char c0 = s[0];
    if (!(c0 == '+' || c0 == '-' || (c0 >= '0' && c0 <= '9')))
        return std::nullopt;
    errno = 0;
    char* end = nullptr;
    const long v = std::strtol(s.c_str(), &end, 10);
    if (end == s.c_str() || *end != '\0' || errno == ERANGE || v < INT_MIN || v > INT_MAX)
        return std::nullopt;
    return static_cast<int>(v);
}

inline int parse_knob_int_or(std::string_view text, int fallback) {
    return parse_knob_int(text).value_or(fallback);
}

} // namespace xllama
