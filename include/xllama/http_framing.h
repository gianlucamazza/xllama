// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// HTTP/1.1 request framing for the LAN API (ADR 0006, audit F02). WinRT-free so
// the host suite can test it; the UWP transport only moves bytes and enforces
// the deadlines below.
#pragma once

#include <chrono>
#include <cstddef>
#include <string>
#include <string_view>

namespace xllama {

inline constexpr size_t kHttpMaxHead = 64 * 1024;       // request line + headers
inline constexpr size_t kHttpMaxBody = 8 * 1024 * 1024; // larger is 413, not clamped
// Measured on Series S (2026-10-14): an 8 MiB body arrives in 1.3–1.4 s over
// the LAN. The body deadline is about 10× that, enough for a ~5 Mbit/s client;
// the head is a few hundred bytes. Past either, the request is 408.
inline constexpr std::chrono::seconds kHttpHeadDeadline{10};
inline constexpr std::chrono::seconds kHttpBodyDeadline{15};

struct HttpHead {
    // 0 when the head is valid; otherwise the HTTP status to answer with
    // (400, 411, 413 or 431) and `error` says why.
    int status = 0;
    std::string error;
    std::string method;
    std::string path; // query string removed
    size_t content_length = 0;
    size_t body_start = 0; // offset of the body in the bytes parsed
};

// Offset just past "\r\n\r\n", or npos while the head is incomplete.
size_t http_head_end(std::string_view data);

// Parses the request line and headers of `data`, which must contain the whole
// head. Exact framing: complete header names (case-insensitive), one decimal
// Content-Length (repeats must agree), no Transfer-Encoding, and a body no
// larger than `max_body`.
HttpHead parse_http_head(std::string_view data, size_t max_body = kHttpMaxBody);

} // namespace xllama
