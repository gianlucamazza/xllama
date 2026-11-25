// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// LocalState operator files: the one place that reads, writes and parses them.
//
// Knobs (`gguf_gpu_layers.txt`, `d3d12-rope.txt`, `diffuse-steps.txt`, …),
// headless inputs (`prompt.txt`, `model.txt`) and job progress files used to
// be read by nine copies with different trim rules and `atoi`, which turned
// "auto" or "6 threads" into 0 or 6. 0 is a real setting for some knobs
// (gguf_gpu_layers.txt = all layers on the CPU), so a typo silently changed
// placement. Knob, input and job-status files now go through these helpers
// on every surface. Streaming result files (bench CSVs written row by row) and
// settings.json (MainPage's own serializer) are outputs with their own writers.
//
// Contract:
//  - A name is a path relative to LocalState (UTF-8, may contain `\\`
//    subdirectories); on Linux it resolves relative to the working directory
//    (resolve_local_path), which is what host tests use.
//  - Text is read as bytes, CRLF becomes LF, and trailing spaces and line
//    ends are dropped. Leading whitespace and interior text are kept.
//  - An integer knob is a whole decimal integer with optional surrounding
//    whitespace, or it is absent; anything else is the documented fallback.
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

// The text normalisation read_local_file applies to raw file bytes.
inline std::string normalize_knob_text(std::string_view raw) {
    std::string out;
    out.reserve(raw.size());
    for (std::size_t i = 0; i < raw.size(); ++i) {
        if (raw[i] == '\r' && i + 1 < raw.size() && raw[i + 1] == '\n')
            continue;
        out.push_back(raw[i]);
    }
    while (!out.empty() && (out.back() == '\n' || out.back() == '\r' || out.back() == ' '))
        out.pop_back();
    return out;
}

// Contents of LocalState/|name|, normalised; nullopt when it cannot be opened.
std::optional<std::string> read_local_file(const std::string& name);

// read_local_file, with an absent file as "".
std::string read_local_text(const std::string& name);

// read_local_text, with an absent or empty file as |fallback|.
std::string read_local_text_or(const std::string& name, const std::string& fallback);

// Replace LocalState/|name| with |bytes| exactly (no normalisation). False when
// the file cannot be created or fully written.
bool write_local_text(const std::string& name, std::string_view bytes);

// Integer knob: |fallback| when absent or unparsable.
int read_local_int(const std::string& name, int fallback);

// Integer knob bounded to [lo, hi]: |fallback| when absent, unparsable or
// outside the range.
int read_local_int_in(const std::string& name, int fallback, int lo, int hi);

} // namespace xllama
