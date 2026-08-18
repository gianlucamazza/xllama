// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "xllama/diskbw.h"

#include <cstdio>
#include <string>

using namespace xllama;

namespace {

// Small scratch file so the tests exercise shape, not the device (GB/s is
// hardware-dependent and measured on target, not unit-tested).
struct ScratchFile {
    std::string path = "diskbw-unittest.bin";
    ScratchFile(std::size_t bytes) {
        std::string err;
        ok = ensure_diskbw_file(path, bytes, &err);
    }
    ~ScratchFile() {
        std::remove(path.c_str());
    }
    bool ok = false;
};

} // namespace

TEST_CASE("diskbw: sequential read over a small file yields positive bandwidth") {
    ScratchFile f(4u << 20);
    REQUIRE(f.ok);
    DiskbwResult r = measure_diskbw(f.path, 4u << 20, 1u << 20, /*random=*/false,
                                    /*threads=*/1, /*iterations=*/2, /*try_unbuffered=*/false);
    CHECK(r.error_msg.empty());
    CHECK(r.threads == 1);
    CHECK(!r.random);
    CHECK(!r.unbuffered);
    CHECK(r.file_bytes == 4u << 20);
    CHECK(r.read_gbs_first > 0.0);
    CHECK(r.read_gbs_best >= r.read_gbs_first);
    // Sanity ceiling: catches a bogus zero-time division.
    CHECK(r.read_gbs_best < 100000.0);
}

TEST_CASE("diskbw: random pattern and multiple threads cover the whole file") {
    ScratchFile f(8u << 20);
    REQUIRE(f.ok);
    DiskbwResult r = measure_diskbw(f.path, 8u << 20, 1u << 20, /*random=*/true,
                                    /*threads=*/2, /*iterations=*/1, /*try_unbuffered=*/false);
    CHECK(r.error_msg.empty());
    CHECK(r.random);
    CHECK(r.file_bytes == 8u << 20);
    CHECK(r.read_gbs_best > 0.0);
}

TEST_CASE("diskbw: block size is rounded up to 1 MiB") {
    ScratchFile f(4u << 20);
    REQUIRE(f.ok);
    DiskbwResult r = measure_diskbw(f.path, 4u << 20, (1u << 20) + 3, false, 1, 1, false);
    CHECK(r.error_msg.empty());
    CHECK(r.block_bytes == 2u << 20);
}

TEST_CASE("diskbw: missing file reports an error, not a crash") {
    DiskbwResult r = measure_diskbw("no-such-file.bin", 4u << 20, 1u << 20, false, 1, 1, false);
    CHECK(!r.error_msg.empty());
    CHECK(r.read_gbs_best == 0.0);
}

TEST_CASE("diskbw: ensure_diskbw_file reuses an existing file of the right size") {
    ScratchFile f(2u << 20);
    REQUIRE(f.ok);
    std::string err;
    CHECK(ensure_diskbw_file(f.path, 2u << 20, &err)); // second call: reuse, no rewrite
    CHECK(err.empty());
}

TEST_CASE("diskbw: CSV row matches the header column count") {
    ScratchFile f(2u << 20);
    REQUIRE(f.ok);
    DiskbwResult r = measure_diskbw(f.path, 2u << 20, 1u << 20, false, 1, 1, false);
    std::string header = diskbw_csv_header();
    std::string row = format_diskbw_row(r, "unittest");
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
