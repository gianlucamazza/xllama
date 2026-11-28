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

TEST_CASE("rejected API session releases compute before the buffered error is delivered") {
    ApiConnectionGate connections;
    ComputeJobAdmission compute;
    const auto connection = connections.try_acquire();
    auto permit = compute.try_acquire();
    REQUIRE(held(connection));
    REQUIRE(held(permit));
    std::mutex hub;
    std::unique_lock<std::mutex> session(hub, std::defer_lock);

    SUBCASE("hub contention preserves the other owner's lock") {
        std::lock_guard<std::mutex> other(hub);
        const auto admission = xllama::check_api_session_admission(session, permit, [] {
            FAIL("listener generation must only be checked while holding the hub");
            return true;
        });
        CHECK(admission == xllama::ApiSessionAdmission::HubBusy);
    }
    SUBCASE("a stale listener releases the lock and the permit") {
        session.lock();
        const auto admission = xllama::check_api_session_admission(session, permit, [&] {
            CHECK(session.owns_lock());
            return false;
        });
        CHECK(admission == xllama::ApiSessionAdmission::ServerStopped);
    }

    CHECK_FALSE(session.owns_lock());
    CHECK_FALSE(held(permit));
    CHECK(held(compute.try_acquire()));
    CHECK(connections.in_flight() == 1);
}

TEST_CASE("an admitted API session retains compute and the hub while it generates") {
    ComputeJobAdmission compute;
    auto permit = compute.try_acquire();
    REQUIRE(held(permit));
    std::mutex hub;
    std::unique_lock<std::mutex> session(hub);
    auto admission = xllama::check_api_session_admission(session, permit, [] { return true; });
    CHECK(admission == xllama::ApiSessionAdmission::Ready);
    CHECK(session.owns_lock());
    CHECK_FALSE(held(compute.try_acquire()));
}

TEST_CASE("a buffered embedding delivery owns its connection while another request uses the hub") {
    ApiConnectionGate connections;
    ComputeJobAdmission compute;
    std::mutex hub;
    auto slot = connections.try_acquire();
    auto embedding = compute.try_acquire();
    std::unique_lock<std::mutex> embedding_lock(hub);
    REQUIRE(held(slot));
    REQUIRE(held(embedding));
    REQUIRE(xllama::check_api_session_admission(embedding_lock, embedding, [] { return true; }) ==
            xllama::ApiSessionAdmission::Ready);
    xllama::release_api_session(embedding_lock, embedding);

    // Check from a competing thread while the embedding connection remains admitted.
    bool gui_acquired = false;
    std::thread gui([&] {
        auto turn = compute.try_acquire();
        if (turn) {
            std::unique_lock<std::mutex> session(hub, std::try_to_lock);
            gui_acquired = session.owns_lock();
        }
    });
    gui.join();
    CHECK(gui_acquired);
    CHECK(connections.in_flight() == 1);
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
