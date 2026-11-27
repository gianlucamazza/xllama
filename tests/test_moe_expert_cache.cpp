// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "xllama/moe_expert_cache.h"

#include <algorithm>
#include <atomic>
#include <limits>
#include <optional>
#include <stdexcept>
#include <system_error>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

using namespace xllama;

namespace {

struct Reader final : MoeExpertReader {
    enum class Mode { complete, short_read, over_report, exception, io_error };
    Mode mode = Mode::complete;
    std::atomic_bool* cancel_after_read = nullptr;
    std::vector<uint64_t> offsets;
    unsigned char salt = 0;
    MoeExpertCache* reenter = nullptr;

    unsigned char value(uint64_t offset) const {
        return static_cast<unsigned char>((offset + salt) % 251);
    }

    size_t read(uint64_t offset, unsigned char* dst, size_t bytes) override {
        offsets.push_back(offset);
        if (reenter)
            std::exchange(reenter, nullptr)->acquire(0, 0);
        if (mode == Mode::exception)
            throw std::runtime_error("injected failure");
        if (mode == Mode::io_error)
            throw std::system_error(std::make_error_code(std::errc::io_error));
        const size_t written = mode == Mode::short_read ? bytes - 1 : bytes;
        for (size_t i = 0; i < written; ++i)
            dst[i] = value(offset + i);
        if (cancel_after_read)
            cancel_after_read->store(true);
        return mode == Mode::over_report ? bytes + 1 : written;
    }
};

using Lease = decltype(std::declval<MoeExpertCache&>().acquire(0, 0));
static_assert(!std::is_copy_constructible_v<Lease> && !std::is_copy_assignable_v<Lease>);
static_assert(std::is_nothrow_move_constructible_v<Lease> &&
              std::is_nothrow_move_assignable_v<Lease>);
constexpr MoeExpertTensor f32{MoeRole::up_exps, GGML_TYPE_F32, 4, 2, 4, 0};
constexpr size_t slice_bytes = 32;
constexpr uint64_t file_bytes = 4 * slice_bytes;

MoeExpertCache cache_for(Reader& r, size_t slots) {
    return MoeExpertCache(r, file_bytes, &f32, 1, slots, slice_bytes,
                          MoeExpertCache::required_bytes(1, slots, slice_bytes));
}

void check_payload(const Lease& lease, const Reader& r, uint64_t offset, size_t length) {
    REQUIRE(lease.size() == length);
    REQUIRE(lease.data() != nullptr);
    for (size_t i = 0; i < length; ++i)
        REQUIRE(lease.data()[i] == r.value(offset + i));
}

} // namespace

TEST_CASE("moe_expert_cache: hits refresh deterministic LRU") {
    Reader r;
    auto c = cache_for(r, 2);
    for (auto id : {0, 1, 0, 2, 0, 1})
        c.acquire(0, id).reset();
    CHECK(r.offsets == std::vector<uint64_t>{0, 32, 64, 32});
    CHECK(c.counters().requests == 6);
    CHECK(c.counters().hits == 2);
    CHECK(c.counters().evictions == 2);
}

TEST_CASE("moe_expert_cache: eviction waits for the last lease to a repeated key") {
    Reader r;
    auto c = cache_for(r, 2);
    auto first = c.acquire(0, 0);
    auto alias = c.acquire(0, 0);
    auto second = c.acquire(0, 1);
    const auto* original = first.data();
    CHECK(alias.data() == original);
    CHECK_THROWS(c.acquire(0, 2));
    first.reset();
    first.reset();
    CHECK_THROWS(c.acquire(0, 2));
    CHECK(r.offsets == std::vector<uint64_t>{0, 32});
    check_payload(alias, r, 0, 32);
    check_payload(second, r, 32, 32);

    // Move assignment releases the previous expert held by the destination.
    second = std::move(alias);
    CHECK(alias.data() == nullptr);
    auto moved = std::move(second);
    CHECK(second.size() == 0);
    auto third = c.acquire(0, 2);
    CHECK(moved.data() == original);
    check_payload(moved, r, 0, 32);
    CHECK_THROWS(c.acquire(0, 3));
    moved.reset();
    auto fourth = c.acquire(0, 3);
    CHECK(fourth.data() == original);
    check_payload(fourth, r, 96, 32);
    check_payload(third, r, 64, 32);
    CHECK(r.offsets == std::vector<uint64_t>{0, 32, 64, 96});
}

TEST_CASE("moe_expert_cache: held bytes outlive the cache owner") {
    std::optional<Lease> held;
    Reader r;
    {
        auto c = cache_for(r, 1);
        held.emplace(c.acquire(0, 3));
    }
    check_payload(*held, r, 96, 32);
    held.reset();
}

TEST_CASE("moe_expert_cache: workers read coordinator-held leases across a barrier") {
    // Single-coordinator contract: acquire and release stay on this thread; workers
    // only read. Debug builds assert the coordinator thread on every pin/owner change.
    Reader r;
    auto c = cache_for(r, 2);
    auto a = c.acquire(0, 1);
    auto b = c.acquire(0, 2);
    std::vector<int> ok(2, 0);
    std::vector<std::thread> workers;
    const Lease* leases[] = {&a, &b};
    for (size_t w = 0; w < 2; ++w)
        workers.emplace_back([&, w] {
            const Lease& l = *leases[w];
            const uint64_t offset = (w + 1) * slice_bytes;
            bool same = l.size() == slice_bytes && l.data() != nullptr;
            for (size_t i = 0; same && i < slice_bytes; ++i)
                same = l.data()[i] == r.value(offset + i);
            ok[w] = same ? 1 : 0;
        });
    for (auto& t : workers)
        t.join();
    CHECK(ok == std::vector<int>{1, 1});
    a.reset();
    b.reset();
    CHECK(c.acquire(0, 3).size() == slice_bytes);
}

TEST_CASE("moe_expert_cache: partial, invalid and failed reads cannot publish a hit") {
    for (auto mode : {Reader::Mode::short_read, Reader::Mode::over_report, Reader::Mode::exception,
                      Reader::Mode::io_error}) {
        CAPTURE(static_cast<int>(mode));
        Reader r;
        auto c = cache_for(r, 1);
        c.acquire(0, 0).reset();
        r.mode = mode;
        CHECK_THROWS(c.acquire(0, 1));
        CHECK(c.counters().hits == 0);
        CHECK(c.counters().logical_read_bytes ==
              32u + (mode == Reader::Mode::short_read ? 31u : 0u));
        r.mode = Reader::Mode::complete;
        auto recovered = c.acquire(0, 1);
        check_payload(recovered, r, 32, 32);
        CHECK(r.offsets == std::vector<uint64_t>{0, 32, 32});
    }
}

TEST_CASE("moe_expert_cache: partial overwrite invalidates the previous key") {
    Reader r;
    auto c = cache_for(r, 1);
    c.acquire(0, 0).reset();
    r.mode = Reader::Mode::short_read;
    CHECK_THROWS(c.acquire(0, 1));
    CHECK_THROWS(c.acquire(0, 0));
    r.mode = Reader::Mode::complete;
    check_payload(c.acquire(0, 0), r, 0, 32);
    CHECK(r.offsets == std::vector<uint64_t>{0, 32, 0, 0});
    CHECK(c.counters().hits == 0);
}

TEST_CASE("moe_expert_cache: reentrant Reader is rejected and the outer miss can retry") {
    Reader r;
    auto c = cache_for(r, 1);
    r.reenter = &c;
    CHECK_THROWS_AS(c.acquire(0, 0), std::logic_error);
    r.reenter = nullptr;
    check_payload(c.acquire(0, 0), r, 0, 32);
    CHECK(r.offsets == std::vector<uint64_t>{0, 0});
    CHECK(c.counters().hits == 0);
}

TEST_CASE("moe_expert_cache: cancellation before and during a terminal read allows retry") {
    Reader r;
    auto c = cache_for(r, 1);
    std::atomic_bool stop{true};
    CHECK_THROWS(c.acquire(0, 0, &stop));
    CHECK(r.offsets.empty());
    stop.store(false);
    r.cancel_after_read = &stop;
    CHECK_THROWS(c.acquire(0, 0, &stop));
    CHECK(c.counters().hits == 0);
    CHECK(c.counters().logical_read_bytes == 32);
    r.cancel_after_read = nullptr;
    stop.store(false);
    check_payload(c.acquire(0, 0, &stop), r, 0, 32);
    stop.store(true);
    CHECK_THROWS(c.acquire(0, 0, &stop));
    CHECK(r.offsets == std::vector<uint64_t>{0, 0});
}

TEST_CASE("moe_expert_cache: mixed quantized slices are keyed by tensor and expert") {
    // GGML Q4_K and Q6_K encode each block of 256 weights in 144 and 210 bytes.
    MoeExpertTensor tensors[] = {{MoeRole::gate_exps, GGML_TYPE_Q4_K, 256, 1, 2, 0},
                                 {MoeRole::down_exps, GGML_TYPE_Q6_K, 256, 1, 2, 288}};
    Reader r;
    MoeExpertCache c(r, 708, tensors, 2, 2, 210, MoeExpertCache::required_bytes(2, 2, 210));
    tensors[0].offset = 1;
    auto a = c.acquire(0, 1);
    auto b = c.acquire(1, 1);
    check_payload(a, r, 144, 144);
    check_payload(b, r, 498, 210);
    check_payload(c.acquire(0, 1), r, 144, 144);
    CHECK(c.counters().requests == 3);
    CHECK(c.counters().hits == 1);
    CHECK(c.counters().requested_bytes == 498);
    CHECK(c.counters().hit_bytes == 144);
    CHECK(c.counters().logical_read_bytes == 354);
    CHECK(r.offsets == std::vector<uint64_t>{144, 498});

    Reader other;
    other.salt = 17;
    tensors[0].offset = 0;
    MoeExpertCache separate(other, 708, tensors, 2, 2, 210,
                            MoeExpertCache::required_bytes(2, 2, 210));
    check_payload(separate.acquire(0, 1), other, 144, 144);
    CHECK(other.offsets == std::vector<uint64_t>{144});
}

TEST_CASE("moe_expert_cache: owned budget and packed extents are validated before I/O") {
    Reader r;
    const auto needed = MoeExpertCache::required_bytes(1, 2, 32);
    CHECK(needed > 64);
    MoeExpertCache exact(r, file_bytes, &f32, 1, 2, 32, needed);
    CHECK_THROWS(MoeExpertCache(r, file_bytes, &f32, 1, 2, 32, needed - 1));
    CHECK_THROWS(MoeExpertCache(r, file_bytes, &f32, 1, 2, 31, needed));
    CHECK_THROWS(MoeExpertCache(r, file_bytes - 1, &f32, 1, 2, 32, needed));
    CHECK_THROWS(MoeExpertCache::required_bytes(1, 2, std::numeric_limits<size_t>::max()));
    CHECK_THROWS(MoeExpertCache::required_bytes(std::numeric_limits<size_t>::max(), 2, 32));
    CHECK_THROWS(exact.acquire(1, 0));
    CHECK_THROWS(exact.acquire(0, 4));
    auto invalid = f32;
    invalid.offset = std::numeric_limits<uint64_t>::max() - 1;
    CHECK_THROWS(
        MoeExpertCache(r, std::numeric_limits<uint64_t>::max(), &invalid, 1, 2, 32, needed));
    invalid = f32;
    invalid.experts = std::numeric_limits<uint64_t>::max();
    CHECK_THROWS(MoeExpertCache(r, file_bytes, &invalid, 1, 2, 32, needed));
    invalid = f32;
    invalid.type = GGML_TYPE_Q4_K;
    CHECK_THROWS(MoeExpertCache(r, file_bytes, &invalid, 1, 2, 32, needed));
    CHECK(r.offsets.empty());
}
