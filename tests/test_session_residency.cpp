// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "xllama/session_hub.h"
#include "xllama/session_residency.h"

#include <doctest/doctest.h>

#include <cstdlib>
#include <mutex>
#include <string>

using namespace xllama;
using namespace std::chrono_literals;

TEST_CASE("keep_alive: the console app's session is never expired or unloaded by a client") {
    using A = KeepAliveAction;
    const std::optional<std::chrono::milliseconds> omitted;
    for (const SessionOwner owner : {SessionOwner::None, SessionOwner::Gui, SessionOwner::Api}) {
        CAPTURE(static_cast<int>(owner));
        // Omitted (xllama's default, unlike Ollama's five minutes) and negative
        // never expire, whoever owns the session.
        CHECK(keep_alive_action(owner, omitted) == A::Keep);
        CHECK(keep_alive_action(owner, -1ms) == A::Keep);
    }
    CHECK(keep_alive_action(SessionOwner::Api, 5000ms) == A::Arm);
    CHECK(keep_alive_action(SessionOwner::Gui, 5000ms) == A::Keep);
    CHECK(keep_alive_action(SessionOwner::None, 5000ms) == A::Keep);
    CHECK(keep_alive_action(SessionOwner::Api, 0ms) == A::Unload);
    CHECK(keep_alive_action(SessionOwner::None, 0ms) == A::Unload); // idempotent
    CHECK(keep_alive_action(SessionOwner::Gui, 0ms) == A::Refuse);
}

TEST_CASE("residency: a lease arms only on an API-owned session and dies with a GUI use") {
    SessionResidency r;
    const auto now = SessionResidency::Clock::time_point(10s);
    CHECK_FALSE(r.arm(20ms, 1, now)); // nothing resident
    r.loaded(SessionOwner::Gui);
    CHECK_FALSE(r.arm(20ms, 1, now));
    CHECK_FALSE(r.deadline);

    r.loaded(SessionOwner::Api);
    REQUIRE(r.arm(20ms, 1, now));
    const auto id = r.id;
    CHECK_FALSE(r.expired(1, id, now));
    CHECK(r.expired(1, id, now + 20ms));
    // An API use keeps the owner; a GUI use takes the session over for good.
    r.used(SessionOwner::Api);
    CHECK(r.expired(1, id, now + 20ms));
    r.used(SessionOwner::Gui);
    CHECK(r.owner == SessionOwner::Gui);
    CHECK_FALSE(r.deadline);
    CHECK_FALSE(r.expired(1, id, now + 1h));
    CHECK_FALSE(r.arm(20ms, 1, now));
}

TEST_CASE("residency: a stale timer cannot expire a renewed, replaced or released lease") {
    SessionResidency r;
    const auto now = SessionResidency::Clock::time_point(10s);
    r.loaded(SessionOwner::Api);
    REQUIRE(r.arm(20ms, 7, now));
    const auto old = r.id;
    CHECK_FALSE(r.expired(8, old, now + 20ms)); // another session generation
    REQUIRE(r.arm(50ms, 7, now));               // renewed by a later request
    CHECK_FALSE(r.expired(7, old, now + 20ms));
    const auto renewed = r.id;
    CHECK_FALSE(r.expired(7, renewed, now + 20ms));
    CHECK(r.expired(7, renewed, now + 50ms));
    r.cancel(); // a request without keep_alive
    CHECK_FALSE(r.expired(7, renewed, now + 1h));
    REQUIRE(r.arm(20ms, 7, now));
    const auto last = r.id;
    r.released();
    CHECK(r.owner == SessionOwner::None);
    CHECK_FALSE(r.expired(7, last, now + 1h));
}

TEST_CASE("SessionHub: an API load is the API's until the GUI reuses it (opt-in: "
          "XLLAMA_TEST_MODEL)") {
    const char* model_env = std::getenv("XLLAMA_TEST_MODEL");
    if (!model_env)
        return;
    SessionHub hub;
    std::lock_guard<std::mutex> lock(hub.mtx);
    SessionParams sp;
    sp.model_path = model_env;
    sp.n_ctx = 512;
    std::string err;
    REQUIRE_MESSAGE(hub.ensure_locked("m", sp, &err, SessionOwner::Api), err);
    CHECK(hub.residency.owner == SessionOwner::Api);
    const auto now = SessionResidency::Clock::now();
    REQUIRE(hub.residency.arm(std::chrono::milliseconds(50), hub.generation, now));
    REQUIRE(hub.ensure_locked("m", sp, &err, SessionOwner::Api)); // an API reuse
    CHECK(hub.residency.owner == SessionOwner::Api);
    CHECK(hub.residency.deadline.has_value());
    REQUIRE(hub.ensure_locked("m", sp, &err)); // a GUI reuse (the default owner)
    CHECK(hub.residency.owner == SessionOwner::Gui);
    CHECK_FALSE(hub.residency.deadline.has_value());
    sp.n_ctx = 256; // a reduced num_ctx is a new load, the API's again
    REQUIRE(hub.ensure_locked("m", sp, &err, SessionOwner::Api));
    CHECK(hub.residency.owner == SessionOwner::Api);
    hub.reset_locked();
    CHECK(hub.residency.owner == SessionOwner::None);
}

TEST_CASE("SessionHub records who loaded the session; a GUI reuse takes it over") {
    SessionHub hub;
    SessionParams sp;
    sp.model_path = "/nonexistent/xllama-residency-test.gguf";
    std::string err;
    // A failed load leaves nothing resident and nothing owned.
    CHECK(hub.ensure_locked("missing", sp, &err, SessionOwner::Api) == nullptr);
    CHECK(hub.residency.owner == SessionOwner::None);
    hub.residency.loaded(SessionOwner::Api); // as a successful API load would
    hub.use_locked(SessionOwner::Api);
    CHECK(hub.residency.owner == SessionOwner::Api);
    hub.use_locked(SessionOwner::Gui);
    CHECK(hub.residency.owner == SessionOwner::Gui);
    hub.reset_locked();
    CHECK(hub.residency.owner == SessionOwner::None);
}
