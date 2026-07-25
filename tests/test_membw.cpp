// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "xllama/membw.h"

#include <string>

using namespace xllama;

TEST_CASE("membw: small buffer yields plausible positive bandwidths") {
    // Tiny buffer + 2 iterations keeps the test fast; we assert shape, not a
    // specific GB/s (that is hardware-dependent and measured, not unit-tested).
    MembwResult r = measure_membw(/*bytes=*/1u << 20, /*iterations=*/2, /*threads=*/1);
    CHECK(r.threads == 1);
    CHECK(r.iterations == 2);
    CHECK(r.buffer_bytes > 0);
    CHECK(r.read_gbs > 0.0);
    CHECK(r.copy_gbs > 0.0);
    CHECK(r.triad_gbs > 0.0);
    // Sanity ceiling: no CPU moves petabytes/s — catches a bogus zero-time div.
    CHECK(r.read_gbs < 100000.0);
    CHECK(r.copy_gbs < 100000.0);
    CHECK(r.triad_gbs < 100000.0);
}

TEST_CASE("membw: threads<=0 auto-detects at least one worker") {
    MembwResult r = measure_membw(1u << 20, 1, /*threads=*/0);
    CHECK(r.threads >= 1);
}

TEST_CASE("membw: buffer is rounded to a whole number of doubles") {
    // An odd byte count must not crash and must round down to a double multiple.
    MembwResult r = measure_membw(/*bytes=*/(1u << 20) + 3, 1, 1);
    CHECK(r.buffer_bytes % sizeof(double) == 0);
}

TEST_CASE("membw: CSV row matches the header column count") {
    MembwResult r = measure_membw(1u << 20, 1, 1);
    std::string header = membw_csv_header();
    std::string row = format_membw_row(r, "unittest");
    auto commas = [](const std::string& s) {
        int n = 0;
        for (char c : s)
            if (c == ',')
                ++n;
        return n;
    };
    CHECK(commas(header) == commas(row));
    CHECK(row.find("unittest") != std::string::npos);
    CHECK(row.back() == '\n');
}
