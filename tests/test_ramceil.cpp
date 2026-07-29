// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "xllama/ramceil.h"

#include <string>
#include <vector>

using namespace xllama;

TEST_CASE("ramceil: a small limit commits and reports every step") {
    // 4 MB in 1 MB steps: small enough to be safe on any CI host, large enough
    // to exercise the loop. We assert shape, not a ceiling — the ceiling is a
    // measured platform number, not a unit-testable constant.
    RamCeilResult r = probe_ram_ceiling(/*step_mb=*/1, /*limit_mb=*/4, /*floor_avail_mb=*/0);
    CHECK(r.step_mb == 1);
    CHECK(r.limit_mb == 4);
    CHECK(r.steps.size() == 4);
    CHECK(r.max_committed_mb == 4);
    CHECK(r.stop_reason == "limit");
}

TEST_CASE("ramceil: committed total advances by exactly one step per row") {
    RamCeilResult r = probe_ram_ceiling(2, 8, 0);
    std::size_t expected = 0;
    for (const RamCeilStep& s : r.steps) {
        expected += 2;
        CHECK(s.committed_mb == expected);
        CHECK(s.ok);
    }
}

TEST_CASE("ramceil: the sink sees every step, in order, as it happens") {
    // The sink is the durability mechanism: on console a PLM kill can end the
    // probe at any step, and only rows already handed to the sink survive.
    std::vector<std::size_t> seen;
    RamCeilResult r =
        probe_ram_ceiling(1, 3, 0, [&](const RamCeilStep& s) { seen.push_back(s.committed_mb); });
    CHECK(seen == std::vector<std::size_t>{1, 2, 3});
    CHECK(seen.size() == r.steps.size());
}

TEST_CASE("ramceil: a limit below one step commits nothing and still reports") {
    RamCeilResult r = probe_ram_ceiling(/*step_mb=*/8, /*limit_mb=*/4, 0);
    CHECK(r.steps.empty());
    CHECK(r.max_committed_mb == 0);
    CHECK(r.stop_reason == "limit");
}

TEST_CASE("ramceil: an impossible floor stops after the first step") {
    // floor_avail_mb far above any real machine: the first sample is already
    // under it, so the probe must stop rather than keep committing.
    RamCeilResult r = probe_ram_ceiling(1, 64, /*floor_avail_mb=*/1024u * 1024u * 1024u);
    CHECK(r.steps.size() == 1);
    CHECK(r.max_committed_mb == 1);
    CHECK(r.stop_reason == "avail_floor");
}

TEST_CASE("ramceil: step_mb of zero falls back to the default instead of looping") {
    RamCeilResult r = probe_ram_ceiling(/*step_mb=*/0, /*limit_mb=*/0, 0);
    CHECK(r.step_mb == 256);
    CHECK(r.steps.empty());
}

TEST_CASE("ramceil: CSV row matches the header column count") {
    RamCeilResult r = probe_ram_ceiling(1, 1, 0);
    REQUIRE(r.steps.size() == 1);
    std::string header = ramceil_csv_header();
    std::string row = format_ramceil_row(r.steps.front(), "unittest");
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
