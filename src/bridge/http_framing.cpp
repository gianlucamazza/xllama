// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "xllama/http_framing.h"

#include <cctype>
#include <limits>

namespace xllama {
namespace {

std::string lower(std::string_view s) {
    std::string out(s);
    for (char& c : out)
        c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
    return out;
}

std::string_view trim(std::string_view s) {
    while (!s.empty() && (s.front() == ' ' || s.front() == '\t'))
        s.remove_prefix(1);
    while (!s.empty() && (s.back() == ' ' || s.back() == '\t'))
        s.remove_suffix(1);
    return s;
}

// A token character (RFC 9110 tchar): header names are made only of these.
bool tchar(char c) {
    return std::isalnum(static_cast<unsigned char>(c)) ||
           std::string_view("!#$%&'*+-.^_`|~").find(c) != std::string_view::npos;
}

// Strictly decimal, no sign, no overflow.
bool parse_length(std::string_view s, size_t* out) {
    if (s.empty())
        return false;
    size_t v = 0;
    for (char c : s) {
        if (c < '0' || c > '9')
            return false;
        const size_t d = static_cast<size_t>(c - '0');
        if (v > (std::numeric_limits<size_t>::max() - d) / 10)
            return false;
        v = v * 10 + d;
    }
    *out = v;
    return true;
}

HttpHead fail(int status, const char* error) {
    HttpHead h;
    h.status = status;
    h.error = error;
    return h;
}

} // namespace

size_t http_head_end(std::string_view data) {
    const size_t at = data.find("\r\n\r\n");
    return at == std::string_view::npos ? std::string_view::npos : at + 4;
}

HttpHead parse_http_head(std::string_view data, size_t max_body) {
    const size_t end = http_head_end(data);
    if (end == std::string_view::npos)
        return data.size() >= kHttpMaxHead ? fail(431, "request head too large")
                                           : fail(400, "incomplete request head");
    if (end > kHttpMaxHead)
        return fail(431, "request head too large");

    HttpHead head;
    head.body_start = end;
    const std::string_view text = data.substr(0, end - 4);
    const size_t line_end = text.find("\r\n");
    const std::string_view line = text.substr(0, line_end);

    // METHOD SP target SP HTTP/x.y, exactly three parts.
    const size_t sp1 = line.find(' ');
    const size_t sp2 = sp1 == std::string_view::npos ? sp1 : line.find(' ', sp1 + 1);
    if (sp1 == 0 || sp2 == std::string_view::npos || sp2 == sp1 + 1 ||
        line.find(' ', sp2 + 1) != std::string_view::npos ||
        line.substr(sp2 + 1).substr(0, 5) != "HTTP/")
        return fail(400, "malformed request line");
    for (char c : line.substr(0, sp1))
        if (!tchar(c))
            return fail(400, "malformed request line");
    head.method = std::string(line.substr(0, sp1));
    head.path = std::string(line.substr(sp1 + 1, sp2 - sp1 - 1));
    if (const size_t q = head.path.find('?'); q != std::string::npos)
        head.path.erase(q);

    bool have_length = false;
    size_t pos = line_end == std::string_view::npos ? text.size() : line_end + 2;
    while (pos < text.size()) {
        size_t eol = text.find("\r\n", pos);
        if (eol == std::string_view::npos)
            eol = text.size();
        const std::string_view field = text.substr(pos, eol - pos);
        pos = eol + 2;
        const size_t colon = field.find(':');
        if (colon == 0 || colon == std::string_view::npos)
            return fail(400, "malformed header field");
        const std::string_view name = field.substr(0, colon);
        for (char c : name)
            if (!tchar(c)) // also rejects "Content-Length :" and folded lines
                return fail(400, "malformed header field");
        const std::string key = lower(name);
        const std::string_view value = trim(field.substr(colon + 1));
        if (key == "transfer-encoding")
            return fail(411, "chunked transfer-encoding not supported; send Content-Length");
        if (key == "content-length") {
            size_t length = 0;
            if (!parse_length(value, &length))
                return fail(400, "invalid Content-Length");
            if (have_length && length != head.content_length)
                return fail(400, "conflicting Content-Length");
            have_length = true;
            head.content_length = length;
        }
    }
    if (head.content_length > max_body)
        return fail(413, "request body too large");
    return head;
}

} // namespace xllama
