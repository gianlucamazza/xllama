// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#include "xllama/compute_admission.h"

#include <atomic>
#include <doctest/doctest.h>
#include <stdexcept>
#include <string>
#include <thread>

using xllama::claim_session_preload;
using xllama::ComputeJobAdmission;
using xllama::try_acquire_while_preloading;

static bool held(const ComputeJobAdmission::Handle& handle) {
    return static_cast<bool>(handle);
}

TEST_CASE("compute admission grants one permit and releases it on destruction") {
    ComputeJobAdmission admission;
    auto first = admission.try_acquire();
    REQUIRE(held(first));
    CHECK_FALSE(held(admission.try_acquire()));

    first.reset();
    CHECK(held(admission.try_acquire()));
}

TEST_CASE("compute admission stays held while an async owner retains the permit") {
    ComputeJobAdmission admission;
    auto worker = admission.try_acquire();
    REQUIRE(held(worker));
    auto completion = worker;
    worker.reset();
    CHECK_FALSE(held(admission.try_acquire()));

    completion.reset();
    CHECK(held(admission.try_acquire()));
}

TEST_CASE("compute admission is exclusive in either entry order") {
    ComputeJobAdmission admission;

    auto gui = admission.try_acquire();
    REQUIRE(held(gui));
    CHECK_FALSE(held(admission.try_acquire()));
    gui.reset();
    auto api = admission.try_acquire();
    REQUIRE(held(api));
    CHECK_FALSE(held(admission.try_acquire()));
    api.reset();

    api = admission.try_acquire();
    REQUIRE(held(api));
    CHECK_FALSE(held(admission.try_acquire()));
    api.reset();
    gui = admission.try_acquire();
    CHECK(held(gui));
}

TEST_CASE("compute admission releases on cancellation") {
    ComputeJobAdmission admission;
    auto permit = admission.try_acquire();
    REQUIRE(held(permit));
    permit.reset();
    CHECK(held(admission.try_acquire()));
}

TEST_CASE("compute admission releases the slot when the job throws") {
    ComputeJobAdmission admission;
    try {
        auto permit = admission.try_acquire();
        REQUIRE(held(permit));
        throw std::runtime_error("job failed");
    } catch (const std::runtime_error& ex) {
        CHECK(std::string(ex.what()) == "job failed");
    }
    CHECK(held(admission.try_acquire()));
}

TEST_CASE("compute admission stays held for a copied stream handle") {
    ComputeJobAdmission admission;
    auto request = admission.try_acquire();
    REQUIRE(held(request));
    auto stream = request;
    request.reset();
    CHECK_FALSE(held(admission.try_acquire()));
    stream.reset();
    CHECK(held(admission.try_acquire()));
}

TEST_CASE("preload wait retries until the startup job releases") {
    ComputeJobAdmission admission;
    std::atomic<bool> preloading{true};
    auto occupant = admission.try_acquire();
    REQUIRE(held(occupant));

    int sleeps = 0;
    auto waiter = try_acquire_while_preloading(admission, preloading, 5, [&] {
        ++sleeps;
        occupant.reset();
        preloading.store(false);
    });
    CHECK(held(waiter));
    CHECK(sleeps == 1);
}

TEST_CASE("preload wait does not sleep when another job holds the slot") {
    ComputeJobAdmission admission;
    std::atomic<bool> preloading{false};
    auto occupant = admission.try_acquire();
    REQUIRE(held(occupant));

    int sleeps = 0;
    auto waiter = try_acquire_while_preloading(admission, preloading, 5, [&] { ++sleeps; });
    CHECK_FALSE(held(waiter));
    CHECK(sleeps == 0);
}

TEST_CASE("preload wait times out while preloading stays set") {
    ComputeJobAdmission admission;
    std::atomic<bool> preloading{true};
    auto occupant = admission.try_acquire();
    REQUIRE(held(occupant));

    int sleeps = 0;
    auto waiter = try_acquire_while_preloading(admission, preloading, 3, [&] { ++sleeps; });
    CHECK_FALSE(held(waiter));
    CHECK(sleeps == 3);
}

TEST_CASE("claim_session_preload is exclusive and only the owner may advertise") {
    std::atomic<bool> preloading{false};
    CHECK(claim_session_preload(preloading));
    CHECK(preloading.load());
    CHECK_FALSE(claim_session_preload(preloading));
    CHECK(preloading.load());
    preloading.store(false);
    CHECK(claim_session_preload(preloading));
}

TEST_CASE("opposite-order threads cannot both hold the process slot") {
    ComputeJobAdmission admission;
    std::atomic<int> winners{0};
    auto race = [&] {
        if (auto permit = admission.try_acquire()) {
            ++winners;
            std::this_thread::sleep_for(std::chrono::milliseconds(5));
        }
    };
    std::thread gui(race);
    std::thread api(race);
    gui.join();
    api.join();
    CHECK(winners == 1);
    CHECK(held(admission.try_acquire()));
}
