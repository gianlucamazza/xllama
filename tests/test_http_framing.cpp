// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "xllama/http_framing.h"

#include <string>

using xllama::parse_http_head;

namespace {
std::string head(const std::string& fields) {
    return "POST /v1/preferences?x=1 HTTP/1.1\r\nHost: xbox\r\n" + fields + "\r\n";
}
} // namespace

TEST_CASE("http head: a well-formed request parses exactly") {
    const std::string req = head("Content-Length: 5\r\n") + "hello";
    const auto h = parse_http_head(req);
    CHECK(h.status == 0);
    CHECK(h.method == "POST");
    CHECK(h.path == "/v1/preferences");
    CHECK(h.content_length == 5);
    CHECK(req.substr(h.body_start) == "hello");
    // Case-insensitive name, optional whitespace, a repeat that agrees.
    CHECK(parse_http_head(head("content-length:   7  \r\nCONTENT-LENGTH: 7\r\n")).content_length ==
          7);
    // No Content-Length: an empty body (GET, or a POST with nothing to send).
    CHECK(parse_http_head("GET / HTTP/1.1\r\n\r\n").status == 0);
    CHECK(parse_http_head("GET / HTTP/1.1\r\n\r\n").content_length == 0);
}

TEST_CASE("http head: Content-Length is one exact decimal (F02)") {
    // The audit sent a negative length and got 200.
    for (const char* bad :
         {"-1", "+5", "5x", " ", "", "0x10", "1e3", "5, 5", "99999999999999999999999"})
        CHECK_MESSAGE(
            parse_http_head(head(std::string("Content-Length: ") + bad + "\r\n")).status == 400,
            bad);
    CHECK(parse_http_head(head("Content-Length: 5\r\nContent-Length: 6\r\n")).error ==
          "conflicting Content-Length");
    // A substring is not the header: the old reader matched "x-content-length:".
    CHECK(parse_http_head(head("X-Content-Length: 9\r\n")).content_length == 0);
}

TEST_CASE("http head: limits are refusals, not clamps") {
    CHECK(parse_http_head(head("Content-Length: 8388608\r\n")).status == 0);
    CHECK(parse_http_head(head("Content-Length: 8388609\r\n")).status == 413);
    CHECK(parse_http_head(head("Content-Length: 5\r\n"), 4).status == 413);
    CHECK(parse_http_head(head("Transfer-Encoding: chunked\r\n")).status == 411);
    CHECK(parse_http_head(head("Transfer-Encoding: gzip\r\n")).status == 411);
    const std::string huge = "GET / HTTP/1.1\r\nX: " + std::string(xllama::kHttpMaxHead, 'a');
    CHECK(parse_http_head(huge).status == 431);
}

TEST_CASE("http head: malformed request lines and fields are 400") {
    for (const char* bad :
         {"GET /\r\n\r\n", "GET  / HTTP/1.1\r\n\r\n", "GET / HTTP/1.1 x\r\n\r\n",
          " GET / HTTP/1.1\r\n\r\n", "GET / FTP/1.1\r\n\r\n", "GET / HTTP/1.1\r\nNoColon\r\n\r\n",
          "GET / HTTP/1.1\r\nContent-Length : 3\r\n\r\n", "GET / HTTP/1.1\r\n folded: x\r\n\r\n",
          "GET / HTTP/1.1\r\n"})
        CHECK_MESSAGE(parse_http_head(bad).status == 400, bad);
}

TEST_CASE("http_head_end finds the terminator or reports an incomplete head") {
    CHECK(xllama::http_head_end("GET / HTTP/1.1\r\n\r\nbody") == 18);
    CHECK(xllama::http_head_end("GET / HTTP/1.1\r\n") == std::string::npos);
}
