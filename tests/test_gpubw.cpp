// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "xllama/gpubw.h"

#include <string>
#include <vector>

using namespace xllama;

TEST_CASE("gpubw: pattern is deterministic and checksum is pure XOR") {
    constexpr std::size_t n = 1024;
    std::vector<std::uint32_t> a(n), b(n);
    gpubw_fill_pattern(a.data(), n);
    gpubw_fill_pattern(b.data(), n);
    for (std::size_t i = 0; i < n; ++i)
        CHECK(a[i] == b[i]);
    CHECK(a[0] == gpubw_pattern_word(0));
    CHECK(a[1] == gpubw_pattern_word(1));

    const std::uint32_t c = gpubw_checksum_words(a.data(), n);
    std::uint32_t expect = 0;
    for (std::size_t i = 0; i < n; ++i)
        expect ^= a[i];
    CHECK(c == expect);

    // Associativity: split reduction matches whole-buffer XOR.
    const std::uint32_t left = gpubw_checksum_words(a.data(), n / 2);
    const std::uint32_t right = gpubw_checksum_words(a.data() + n / 2, n - n / 2);
    CHECK((left ^ right) == c);
}

TEST_CASE("gpubw: kill gate helper requires real D3D12 success") {
    GpubwResult fail{};
    fail.read_gbs = 200.0;
    fail.checksum_ok = true;
    fail.d3d12_ran = false;
    CHECK_FALSE(gpubw_passes_kill_gate(fail));

    GpubwResult low{};
    low.read_gbs = 50.0;
    low.checksum_ok = true;
    low.d3d12_ran = true;
    CHECK_FALSE(gpubw_passes_kill_gate(low));
    CHECK(kGpubwKillReadGbs == 100.0);

    GpubwResult pass{};
    pass.read_gbs = 100.0;
    pass.checksum_ok = true;
    pass.d3d12_ran = true;
    CHECK(gpubw_passes_kill_gate(pass));
}

TEST_CASE("gpubw: CSV row matches header column count") {
    GpubwResult r;
    r.buffer_bytes = 64 * 1024 * 1024;
    r.iterations = 3;
    r.read_gbs = 12.5;
    r.checksum_ok = true;
    r.d3d12_ran = false;
    r.checksum = 1;
    r.expected_checksum = 1;
    r.error_msg = "d3d12 unavailable on this platform";
    std::string header = gpubw_csv_header();
    std::string row = format_gpubw_row(r, "unittest");
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

TEST_CASE("gpubw: measure_gpubw on non-D3D12 host reports unavailable") {
    // On Linux CI this exercises the shipped entry point without claiming a
    // GPU bandwidth number (hardware-dependent; console is the gate).
    GpubwResult r = measure_gpubw(/*bytes=*/1u << 20, /*iterations=*/1);
    CHECK(r.buffer_bytes >= sizeof(std::uint32_t));
    CHECK(r.iterations == 1);
#if !defined(_WIN32)
    CHECK_FALSE(r.d3d12_ran);
    CHECK_FALSE(r.error_msg.empty());
    CHECK_FALSE(gpubw_passes_kill_gate(r));
#endif
}
