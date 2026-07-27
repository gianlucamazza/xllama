// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "xllama/kv_store.h"

#include <doctest/doctest.h>
#include <filesystem>
#include <fstream>
#include <thread>

namespace {

std::filesystem::path fresh_dir(const char* name) {
    auto d = std::filesystem::temp_directory_path() / name;
    std::error_code ec;
    std::filesystem::remove_all(d, ec);
    std::filesystem::create_directories(d);
    return d;
}

void write_bytes(const std::filesystem::path& p, size_t n) {
    std::ofstream f(p, std::ios::binary);
    const std::string chunk(n, 'x');
    f << chunk;
}

} // namespace

TEST_CASE("KvStore::valid_id refuses anything that is not a plain identifier") {
    using xllama::KvStore;
    CHECK(KvStore::valid_id("a1b2-c3d4_e5"));
    CHECK_FALSE(KvStore::valid_id(""));
    // Path traversal is the reason this function exists: ids come from the UI.
    CHECK_FALSE(KvStore::valid_id(".."));
    CHECK_FALSE(KvStore::valid_id("../../etc/passwd"));
    CHECK_FALSE(KvStore::valid_id("a/b"));
    CHECK_FALSE(KvStore::valid_id("a\\b"));
    CHECK_FALSE(KvStore::valid_id("a b"));
    CHECK_FALSE(KvStore::valid_id(std::string(65, 'a')));
}

TEST_CASE("KvStore::path_for stays inside the directory") {
    const auto dir = fresh_dir("xllama-kvstore-path");
    const xllama::KvStore store{dir.string()};
    const std::string p = store.path_for("conv-1");
    CHECK(p.find(dir.string()) == 0);
    CHECK(p.substr(p.size() - 3) == ".kv");
    CHECK(store.path_for("../escape").empty());
    CHECK(xllama::KvStore{""}.path_for("conv-1").empty());
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

TEST_CASE("KvStore::prune keeps the newest snapshots within both caps") {
    const auto dir = fresh_dir("xllama-kvstore-prune");
    const xllama::KvStore store{dir.string()};

    // Distinct mtimes: the eviction order is least-recently-modified first.
    for (const char* id : {"old", "mid", "new"}) {
        write_bytes(store.path_for(id), 1024);
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    // An interrupted save leaves a .tmp behind; pruning must clear it.
    write_bytes(dir / "leftover.kv.tmp", 4096);

    store.prune(/*max_files=*/2, /*max_bytes=*/xllama::kKvStoreMaxBytes);
    CHECK_FALSE(std::filesystem::exists(store.path_for("old")));
    CHECK(std::filesystem::exists(store.path_for("mid")));
    CHECK(std::filesystem::exists(store.path_for("new")));
    CHECK_FALSE(std::filesystem::exists(dir / "leftover.kv.tmp"));
    CHECK(store.total_bytes() == 2048);

    // The byte cap bites before the file cap when snapshots are large.
    store.prune(/*max_files=*/10, /*max_bytes=*/1500);
    CHECK(store.total_bytes() == 1024);
    CHECK(std::filesystem::exists(store.path_for("new")));

    store.erase("new");
    CHECK(store.total_bytes() == 0);

    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
}

TEST_CASE("KvStore tolerates a missing directory") {
    const xllama::KvStore store{
        (std::filesystem::temp_directory_path() / "xllama-kv-absent").string()};
    CHECK(store.total_bytes() == 0);
    store.prune(xllama::kKvStoreMaxFiles, xllama::kKvStoreMaxBytes); // must not throw
    store.erase("whatever");
}
