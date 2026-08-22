// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Minimal JSON string helpers: escaping and parsing with full \uXXXX decode
// (including surrogate pairs → UTF-8). Replaces the 7 duplicated json_escape
// implementations across the tree and adds \u support to the training job
// parser (previously a correctness gap: \uXXXX was passed through verbatim).
//
// Header-only, WinRT-free, host-testable.

#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace xllama {

// Escape a UTF-8 string for JSON output: " → \", \ → \\, control chars → \uXXXX.
// This is the canonical implementation — all callers should use this one.
inline std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        case '\b':
            out += "\\b";
            break;
        case '\f':
            out += "\\f";
            break;
        default:
            if (c < 0x20) {
                char buf[8];
                std::snprintf(buf, sizeof(buf), "\\u%04x", c);
                out += buf;
            } else {
                out += static_cast<char>(c);
            }
        }
    }
    return out;
}

// Decode a single JSON escape sequence starting at s[pos] (the char after '\').
// Returns the decoded character(s) and advances pos past the escape.
// For \uXXXX, writes the UTF-8 codepoint(s) to `out` and advances pos by 6
// (the backslash + 'u' + 4 hex digits). Returns false on malformed escape.
static inline bool json_decode_escape(const std::string& s, size_t& pos, std::string& out) {
    if (pos >= s.size())
        return false;
    char e = s[pos++];
    switch (e) {
    case '"':
    case '\\':
    case '/':
        out.push_back(e);
        break;
    case 'n':
        out.push_back('\n');
        break;
    case 't':
        out.push_back('\t');
        break;
    case 'r':
        out.push_back('\r');
        break;
    case 'b':
        out.push_back('\b');
        break;
    case 'f':
        out.push_back('\f');
        break;
    case 'u': {
        if (pos + 4 > s.size())
            return false;
        unsigned cp = 0;
        for (int i = 0; i < 4; ++i) {
            char h = s[pos++];
            cp <<= 4;
            if (h >= '0' && h <= '9')
                cp |= (h - '0');
            else if (h >= 'a' && h <= 'f')
                cp |= (h - 'a' + 10);
            else if (h >= 'A' && h <= 'F')
                cp |= (h - 'A' + 10);
            else
                return false;
        }
        // Surrogate pair: high surrogate followed by low surrogate
        if (cp >= 0xD800 && cp <= 0xDBFF) {
            if (pos + 6 <= s.size() && s[pos] == '\\' && s[pos + 1] == 'u') {
                pos += 2; // skip \u
                unsigned cp2 = 0;
                bool ok = true;
                for (int i = 0; i < 4; ++i) {
                    char h = s[pos++];
                    cp2 <<= 4;
                    if (h >= '0' && h <= '9')
                        cp2 |= (h - '0');
                    else if (h >= 'a' && h <= 'f')
                        cp2 |= (h - 'a' + 10);
                    else if (h >= 'A' && h <= 'F')
                        cp2 |= (h - 'A' + 10);
                    else {
                        ok = false;
                        break;
                    }
                }
                if (ok && cp2 >= 0xDC00 && cp2 <= 0xDFFF) {
                    // Combine: U+10000 + (high-0xD800)<<10 + (low-0xDC00)
                    cp = 0x10000 + ((cp - 0xD800) << 10) + (cp2 - 0xDC00);
                } else {
                    // Malformed surrogate — emit replacement char
                    cp = 0xFFFD;
                }
            } else {
                // Lone surrogate — emit replacement char
                cp = 0xFFFD;
            }
        }
        // Encode codepoint as UTF-8
        if (cp < 0x80) {
            out.push_back(static_cast<char>(cp));
        } else if (cp < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (cp >> 6)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else if (cp < 0x10000) {
            out.push_back(static_cast<char>(0xE0 | (cp >> 12)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (cp >> 18)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((cp >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (cp & 0x3F)));
        }
        break;
    }
    default:
        // Unknown escape — emit literally (lenient, matches existing parser behavior)
        out.push_back(e);
        break;
    }
    return true;
}

// Parse a JSON string value starting at s[pos] (pos must point to the char
// AFTER the opening '"'). Advances pos to the char AFTER the closing '"'.
// Returns true on success, false on unterminated string or bad escape.
inline bool json_read_string(const std::string& s, size_t& pos, std::string& out) {
    out.clear();
    while (pos < s.size()) {
        char c = s[pos++];
        if (c == '"')
            return true;
        if (c == '\\' && pos <= s.size()) {
            if (!json_decode_escape(s, pos, out))
                return false;
        } else {
            out.push_back(c);
        }
    }
    return false; // unterminated
}

} // namespace xllama