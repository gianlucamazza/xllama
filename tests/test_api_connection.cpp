// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#include "xllama/api_connection.h"

#include <atomic>
#include <doctest/doctest.h>
#include <thread>
#include <vector>

using xllama::admit_api_compute;
using xllama::ApiConnectionGate;
using xllama::ComputeJobAdmission;
using xllama::kApiHttpBusy;
using xllama::kApiHttpOk;
using xllama::kApiMaxInFlightConnections;

static bool held(const ComputeJobAdmission::Handle& handle) {
    return static_cast<bool>(handle);
}

static bool held(const ApiConnectionGate::Handle& handle) {
    return static_cast<bool>(handle);
}

TEST_CASE("two overlapping API compute requests yield one 200 and one 503") {
    ComputeJobAdmission admission;
    std::atomic<int> n200{0};
    std::atomic<int> n503{0};
    std::atomic<bool> hold{true};
    std::atomic<int> started{0};

    auto request = [&] {
        const auto outcome = admit_api_compute(admission);
        ++started;
        if (outcome.http_code == kApiHttpOk) {
            REQUIRE(held(outcome.permit));
            while (hold.load(std::memory_order_acquire))
                std::this_thread::yield();
            ++n200;
        } else {
            CHECK(outcome.http_code == kApiHttpBusy);
            CHECK_FALSE(held(outcome.permit));
            ++n503;
        }
    };

    std::thread first(request);
    while (started.load(std::memory_order_acquire) == 0)
        std::this_thread::yield();
    std::thread second(request);
    while (started.load(std::memory_order_acquire) < 2)
        std::this_thread::yield();
    hold.store(false, std::memory_order_release);
    first.join();
    second.join();
    CHECK(n200 == 1);
    CHECK(n503 == 1);
}

TEST_CASE("API compute permit release on abort admits the next request as 200") {
    ComputeJobAdmission admission;
    auto stream = admit_api_compute(admission);
    REQUIRE(stream.http_code == kApiHttpOk);
    // Client disconnect mid-SSE drops the last handle; the slot must clear.
    stream.permit.reset();
    const auto next = admit_api_compute(admission);
    CHECK(next.http_code == kApiHttpOk);
    CHECK(held(next.permit));
}

TEST_CASE("connection gate admits up to the in-flight bound") {
    ApiConnectionGate gate;
    std::vector<ApiConnectionGate::Handle> slots;
    slots.reserve(static_cast<size_t>(kApiMaxInFlightConnections));
    for (int i = 0; i < kApiMaxInFlightConnections; ++i) {
        auto slot = gate.try_acquire();
        REQUIRE(held(slot));
        slots.push_back(std::move(slot));
    }
    CHECK(gate.in_flight() == kApiMaxInFlightConnections);
    CHECK_FALSE(held(gate.try_acquire()));

    slots.pop_back();
    CHECK(gate.in_flight() == kApiMaxInFlightConnections - 1);
    CHECK(held(gate.try_acquire()));
}
