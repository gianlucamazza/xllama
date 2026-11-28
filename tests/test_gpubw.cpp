// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "../src/bridge/d3d12_compute.h"
#include "xllama/gpubw.h"

#include <array>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
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

TEST_CASE("gpubw: plan_dispatch keeps every dim ≤ 65535 for default 1 GiB") {
    // Skeptic: 1 GiB / 4 bytes / 256 threads = 1_048_576 groups > 65535 on X.
    const std::size_t n_words = kGpubwDefaultBufferBytes / sizeof(std::uint32_t);
    CHECK(n_words == (static_cast<std::size_t>(1) << 28));

    const GpubwDispatch d = gpubw_plan_dispatch(n_words);
    CHECK(gpubw_dispatch_dims_legal(d));
    CHECK(d.groups_x <= kGpubwMaxGroupsPerDim);
    CHECK(d.groups_y <= kGpubwMaxGroupsPerDim);
    CHECK(d.groups_z <= kGpubwMaxGroupsPerDim);
    CHECK(d.groups_x >= 1);
    CHECK(d.groups_y >= 1);
    CHECK(d.groups_z >= 1);

    const std::uint64_t need =
        (static_cast<std::uint64_t>(n_words) + kGpubwThreadsPerGroup - 1) / kGpubwThreadsPerGroup;
    const std::uint64_t launched = static_cast<std::uint64_t>(d.groups_x) * d.groups_y * d.groups_z;
    CHECK(launched >= need);
    CHECK(d.n_groups == launched);
    // Prefer a balanced 2D grid for the default working set (1024×1024).
    CHECK(d.groups_x == 1024u);
    CHECK(d.groups_y == 1024u);
    CHECK(d.groups_z == 1u);
}

TEST_CASE("gpubw: plan_dispatch is 1D when groups fit in one axis") {
    // 256 words → 1 group; 65535*256 words → max-X 1D.
    {
        const GpubwDispatch d = gpubw_plan_dispatch(256);
        CHECK(gpubw_dispatch_dims_legal(d));
        CHECK(d.groups_x == 1u);
        CHECK(d.groups_y == 1u);
        CHECK(d.groups_z == 1u);
        CHECK(d.n_groups == 1u);
    }
    {
        const std::size_t n_words =
            static_cast<std::size_t>(kGpubwMaxGroupsPerDim) * kGpubwThreadsPerGroup;
        const GpubwDispatch d = gpubw_plan_dispatch(n_words);
        CHECK(gpubw_dispatch_dims_legal(d));
        CHECK(d.groups_x == kGpubwMaxGroupsPerDim);
        CHECK(d.groups_y == 1u);
        CHECK(d.groups_z == 1u);
        CHECK(d.n_groups == kGpubwMaxGroupsPerDim);
    }
    {
        // Just over the 1D limit → 2D.
        const std::size_t n_words =
            static_cast<std::size_t>(kGpubwMaxGroupsPerDim) * kGpubwThreadsPerGroup + 1;
        const GpubwDispatch d = gpubw_plan_dispatch(n_words);
        CHECK(gpubw_dispatch_dims_legal(d));
        CHECK(d.groups_y >= 2u);
        const std::uint64_t need =
            (static_cast<std::uint64_t>(n_words) + kGpubwThreadsPerGroup - 1) /
            kGpubwThreadsPerGroup;
        CHECK(static_cast<std::uint64_t>(d.n_groups) >= need);
    }
}

namespace {

using d3d12c::FenceResult;
constexpr auto removed_counter = (std::numeric_limits<std::uint64_t>::max)();

struct ProbeQueueDouble {
    bool accepts = false;
    std::uint64_t completed = 0;
    std::vector<std::uint64_t> signals;
    int executed = 0;
    int reads = 0;
};

struct ProbeOwnerDouble {
    virtual ~ProbeOwnerDouble() = default;
    d3d12c::ProbeMarker marker;
    ProbeQueueDouble* queue = nullptr;
    std::shared_ptr<int> allocator = std::make_shared<int>(0);
    std::shared_ptr<int> list = std::make_shared<int>(0);

    bool pending() const {
        return marker.pending();
    }
    bool try_retire() {
        observe();
        return !pending();
    }
    FenceResult observe() {
        auto read = [&] {
            ++queue->reads;
            return queue->completed;
        };
        return marker.retire(
            [&](std::uint64_t target) {
                queue->signals.push_back(target);
                return queue->accepts;
            },
            read, [&](std::uint64_t target) { return d3d12c::fence_result(target, read()); });
    }
};

struct ProbeWithBuffers : ProbeOwnerDouble {
    std::shared_ptr<int> heap = std::make_shared<int>(0);
    std::shared_ptr<int> staging = std::make_shared<int>(0);
    std::shared_ptr<int> readback = std::make_shared<int>(0);
};

} // namespace

TEST_CASE("D3D12 probes retain a rejected submission through retries and release its whole owner") {
    for (const bool remove_while_unmarked : {false, true}) {
        ProbeQueueDouble queue;
        // A normal counter alone cannot certify a marker that the queue rejected.
        queue.completed = 7;
        d3d12c::ProbeSlot<ProbeOwnerDouble> slot;
        std::array<std::weak_ptr<int>, 5> resources;
        int readbacks = 0;
        REQUIRE(slot.try_run<ProbeWithBuffers>([&](auto& owner) {
            owner.queue = &queue;
            resources = {owner.allocator, owner.list, owner.heap, owner.staging, owner.readback};
            REQUIRE(owner.marker.submit([&] { ++queue.executed; }));
            const auto result = owner.observe();
            CHECK(result == FenceResult::signal_rejected);
            if (result == FenceResult::complete)
                ++readbacks;
            CHECK_FALSE(owner.marker.submit([&] { ++queue.executed; }));
        }));

        auto unexpected = [](auto&) { FAIL("allocated new work while the submission is live"); };
        for (int i = 0; i < 8; ++i)
            CHECK_FALSE(slot.try_run<ProbeWithBuffers>(unexpected));
        CHECK(queue.signals == std::vector<std::uint64_t>(9, 1));
        CHECK(queue.reads == 9);
        for (const auto& resource : resources)
            CHECK_FALSE(resource.expired());

        if (!remove_while_unmarked) {
            queue.accepts = true;
            queue.completed = 0;
            CHECK_FALSE(slot.try_run<ProbeWithBuffers>(unexpected));
            for (int i = 0; i < 8; ++i)
                CHECK_FALSE(slot.try_run<ProbeWithBuffers>(unexpected));
            CHECK(queue.signals == std::vector<std::uint64_t>(10, 1));
            CHECK(queue.reads == 18);
            queue.completed = 1;
        } else {
            queue.completed = removed_counter;
        }
        CHECK(slot.try_run<ProbeWithBuffers>([&](auto&) {
            for (const auto& resource : resources)
                CHECK(resource.expired());
        }));
        CHECK(queue.executed == 1);
        CHECK(readbacks == 0);
    }
}

TEST_CASE("D3D12 probes time out an accepted marker and poll the same marker on retry") {
    for (const bool notify : {false, true}) {
        for (const auto terminal : {std::uint64_t{1}, removed_counter}) {
            d3d12c::ProbeMarker marker;
            int executions = 0, signals = 0, arms = 0, elapsed = 0;
            std::uint64_t completed = 0;
            auto signal = [&](std::uint64_t) {
                ++signals;
                return true;
            };
            auto read = [&] { return completed; };
            REQUIRE(marker.submit([&] { ++executions; }));
            const auto initial = marker.retire(signal, read, [&](std::uint64_t target) {
                return d3d12c::wait_accepted_fence_until(
                    target, false, read,
                    [&] {
                        ++arms;
                        return notify;
                    },
                    [&] {
                        ++elapsed;
                        return true;
                    },
                    [&] { ++elapsed; }, [&] { return elapsed >= 5; });
            });
            CHECK(initial == FenceResult::pending);
            CHECK(elapsed == 5);
            CHECK(arms == 1);
            CHECK_FALSE(marker.submit([&] { ++executions; }));
            completed = terminal;
            const auto status = marker.retire(signal, read, [&](std::uint64_t target) {
                return d3d12c::fence_result(target, completed);
            });
            CHECK(status ==
                  (terminal == removed_counter ? FenceResult::removed : FenceResult::complete));
            CHECK_FALSE(marker.pending());
            CHECK(signals == 1);
            CHECK(executions == 1);
        }
    }
}

TEST_CASE("D3D12 probes reject simultaneous admission before allocating a second campaign") {
    d3d12c::ProbeSlot<ProbeOwnerDouble> slot;
    bool other_admitted = true, other_started = false;
    REQUIRE(slot.try_run<ProbeWithBuffers>([&](auto&) {
        std::thread contender([&] {
            other_admitted = slot.try_run<ProbeWithBuffers>([&](auto&) { other_started = true; });
        });
        contender.join();
    }));
    CHECK_FALSE(other_admitted);
    CHECK_FALSE(other_started);
    CHECK(slot.try_run<ProbeWithBuffers>([](auto&) {}));
}

TEST_CASE("D3D12 probe unwinding releases unsubmitted work and retains submitted work") {
    d3d12c::ProbeSlot<ProbeOwnerDouble> slot;
    ProbeQueueDouble queue;
    std::weak_ptr<int> backing;
    for (const bool submitted : {false, true}) {
        auto run = [&](auto& owner) {
            owner.queue = &queue;
            backing = owner.staging;
            if (submitted)
                REQUIRE(owner.marker.submit([&] { ++queue.executed; }));
            throw std::runtime_error("injected after setup");
        };
        CHECK_THROWS_AS(slot.try_run<ProbeWithBuffers>(run), std::runtime_error);
        CHECK(backing.expired() == !submitted);
    }
    queue.completed = removed_counter;
    CHECK(slot.try_run<ProbeWithBuffers>([&](auto&) { CHECK(backing.expired()); }));
    CHECK(queue.executed == 1);
}
