// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#include "xllama/local_knob.h"

#include "xllama/path_utils.h"
#ifdef XLLAMA_UWP
    #include "xllama/utf8_utils.h"
#endif

#include <cstdio>
#include <string>

namespace xllama {
namespace {

// Binary mode on both platforms: read_local_file normalises line ends itself,
// so the result does not depend on the CRT's text-mode translation.
FILE* open_local(const std::string& name, bool for_write) {
    const std::string path = resolve_local_path(name);
#ifdef XLLAMA_UWP
    return _wfopen(utf8_to_wstring(path).c_str(), for_write ? L"wb" : L"rb");
#else
    return std::fopen(path.c_str(), for_write ? "wb" : "rb");
#endif
}

} // namespace

std::optional<std::string> read_local_file(const std::string& name) {
    FILE* f = open_local(name, false);
    if (!f)
        return std::nullopt;
    std::string raw;
    char buf[4096];
    std::size_t n;
    while ((n = std::fread(buf, 1, sizeof(buf), f)) > 0)
        raw.append(buf, n);
    std::fclose(f);
    return normalize_knob_text(raw);
}

std::string read_local_text(const std::string& name) {
    return read_local_file(name).value_or(std::string());
}

std::string read_local_text_or(const std::string& name, const std::string& fallback) {
    std::string text = read_local_text(name);
    return text.empty() ? fallback : text;
}

bool write_local_text(const std::string& name, std::string_view bytes) {
    FILE* f = open_local(name, true);
    if (!f)
        return false;
    const bool ok = std::fwrite(bytes.data(), 1, bytes.size(), f) == bytes.size();
    return std::fclose(f) == 0 && ok;
}

int read_local_int(const std::string& name, int fallback) {
    return parse_knob_int_or(read_local_text(name), fallback);
}

int read_local_int_in(const std::string& name, int fallback, int lo, int hi) {
    const std::optional<int> v = parse_knob_int(read_local_text(name));
    return (v && *v >= lo && *v <= hi) ? *v : fallback;
}

} // namespace xllama
