// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#include "xllama/model_write.h"

#include <chrono>
#include <doctest/doctest.h>
#include <iterator>
#include <thread>

namespace {
struct ModelDir {
    std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("xllama-writer-" +
         std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    ModelDir() {
        std::filesystem::create_directories(path);
    }
    ~ModelDir() {
        std::error_code ec;
        std::filesystem::remove_all(path, ec);
    }
    void put(const std::string& name, const std::string& data) {
        std::ofstream(path / name, std::ios::binary) << data;
    }
    std::string read(const std::string& name) {
        std::ifstream f(path / name, std::ios::binary);
        return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
    }
};
} // namespace

TEST_CASE("Model writer rejects all competing writers without touching files") {
    ModelDir dir;
    dir.put(".complete", "original marker");
    dir.put("old.gguf", "original weights");
    dir.put("model.gguf.part", "first writer staging");
    xllama::ModelWriteGate gate;
    xllama::ModelWriteOperation active(gate, dir.path);
    REQUIRE(active.owns_lock());
    xllama::ModelWriteOperation rejected(gate, dir.path);
    std::error_code ec;
    CHECK_FALSE(rejected.begin(ec));
    CHECK(ec == std::errc::device_or_resource_busy);
    CHECK_FALSE(rejected.complete({L"new.gguf"}, ec));
    CHECK(dir.read(".complete") == "original marker");
    CHECK(dir.read("old.gguf") == "original weights");
    CHECK(dir.read("model.gguf.part") == "first writer staging");
    xllama::ModelWriteOperation nonexistent(gate, dir.path / "absent");
    CHECK_FALSE(nonexistent.begin(ec));
    CHECK_FALSE(std::filesystem::exists(dir.path / "absent"));
}

TEST_CASE("Model writer publishes only after success and preserves adapter and rollback files") {
    ModelDir dir;
    dir.put(".complete", "old marker");
    dir.put("OLD.GGUF", "old weights");
    dir.put("New.GGUF", "new weights");
    dir.put("ADAPTER.GGUF", "personalization");
    dir.put("OLD.GGUF.previous", "rollback weights");
    xllama::ModelWriteGate gate;
    xllama::ModelWriteOperation writer(gate, dir.path);
    std::error_code ec;
    REQUIRE(writer.begin(ec));
    CHECK_FALSE(std::filesystem::exists(dir.path / ".complete"));
    CHECK(dir.read("OLD.GGUF") == "old weights"); // no pruning before verified success
    REQUIRE(writer.complete({L"new.gguf"}, ec));
    CHECK_FALSE(std::filesystem::exists(dir.path / "OLD.GGUF"));
    CHECK(dir.read("New.GGUF") == "new weights");
    CHECK(dir.read("ADAPTER.GGUF") == "personalization");
    CHECK(dir.read("OLD.GGUF.previous") == "rollback weights");
    CHECK(dir.read(".complete") == "ok");
    writer.release();
    CHECK(gate.try_acquire().owns_lock()); // completion callback may chain a download
}

TEST_CASE("Model writer releases after failed preparation and exceptions") {
    ModelDir dir;
    dir.put("not-a-directory", "keep");
    xllama::ModelWriteGate gate;
    {
        xllama::ModelWriteOperation writer(gate, dir.path / "not-a-directory");
        std::error_code ec;
        CHECK_FALSE(writer.begin(ec));
        CHECK(ec);
    }
    CHECK(gate.try_acquire().owns_lock());
    try {
        xllama::ModelWriteOperation writer(gate, dir.path);
        std::error_code ec;
        REQUIRE(writer.begin(ec));
        throw std::runtime_error("transfer failed");
    } catch (const std::runtime_error&) {
    }
    CHECK(gate.try_acquire().owns_lock());
    CHECK_FALSE(std::filesystem::exists(dir.path / ".complete"));
    CHECK(dir.read("not-a-directory") == "keep");
}

TEST_CASE("Model writer permit moves and can release on another thread") {
    xllama::ModelWriteGate gate;
    auto first = gate.try_acquire();
    auto second = std::move(first);
    CHECK_FALSE(first.owns_lock());
    REQUIRE(second.owns_lock());
    std::thread worker([permit = std::move(second)]() mutable { permit.release(); });
    worker.join();
    CHECK_FALSE(second.owns_lock());
    CHECK(gate.try_acquire().owns_lock());
    xllama::ModelWriteGate other;
    auto a = gate.try_acquire();
    auto b = other.try_acquire();
    b = std::move(a);
    CHECK_FALSE(a.owns_lock());
    CHECK(other.try_acquire().owns_lock());
    CHECK_FALSE(gate.try_acquire().owns_lock());
}

TEST_CASE("Model writer admits exactly one simultaneous contender") {
    xllama::ModelWriteGate gate;
    std::atomic<int> ready{0}, attempted{0}, winners{0};
    std::atomic<bool> start{false};
    std::vector<std::thread> threads;
    for (int i = 0; i < 16; ++i) {
        threads.emplace_back([&] {
            ready.fetch_add(1);
            while (!start.load())
                std::this_thread::yield();
            auto permit = gate.try_acquire();
            if (permit.owns_lock())
                winners.fetch_add(1);
            attempted.fetch_add(1);
            while (attempted.load() != 16)
                std::this_thread::yield();
        });
    }
    while (ready.load() != 16)
        std::this_thread::yield();
    start.store(true);
    for (auto& thread : threads)
        thread.join();
    CHECK(winners.load() == 1);
    CHECK(gate.try_acquire().owns_lock());
}
