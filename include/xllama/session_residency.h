// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Who holds the resident session, and what an Ollama `keep_alive` may do to
// it. The console app wins: a session the GUI loaded or used is never expired
// or unloaded by a LAN client. Only a session the API loaded, and the GUI has
// not used since, takes a finite lease or an explicit unload.
//
// SessionHub keeps one SessionResidency under its mutex (session_hub.h). The
// LAN server's timer (uwp/api-server.cpp) only carries a snapshot of
// `generation` and `id`; expired() rejects a snapshot that a reload, a newer
// lease, a cancel or a GUI turn has superseded.
#pragma once

#include <chrono>
#include <cstdint>
#include <optional>

namespace xllama {

// Gui covers every console-app path that ensures a session (turn, preload,
// model selection); Api covers the LAN routes.
enum class SessionOwner { None, Gui, Api };

// What a request's keep_alive does once it holds the session.
enum class KeepAliveAction {
    Keep,   // no expiry: omitted, negative, or a GUI-owned session
    Arm,    // finite lease on an API-owned session
    Unload, // release the API-owned session at the end of the request
    Refuse, // keep_alive 0 on a GUI-owned session: 409, the session stays
};

// |keep_alive|: nullopt when the request omits it (or sends null). A negative
// value never expires. Omitted also means no expiry: unlike Ollama's
// five-minute default, a model stays resident until something replaces it.
inline KeepAliveAction keep_alive_action(SessionOwner owner,
                                         std::optional<std::chrono::milliseconds> keep_alive) {
    if (!keep_alive || keep_alive->count() < 0)
        return KeepAliveAction::Keep;
    if (keep_alive->count() == 0)
        return owner == SessionOwner::Gui ? KeepAliveAction::Refuse : KeepAliveAction::Unload;
    return owner == SessionOwner::Api ? KeepAliveAction::Arm : KeepAliveAction::Keep;
}

// Guarded by SessionHub::mtx.
struct SessionResidency {
    using Clock = std::chrono::steady_clock;

    SessionOwner owner = SessionOwner::None;
    std::optional<Clock::time_point> deadline; // set only while owner == Api
    uint64_t generation = 0;                   // hub generation the deadline belongs to
    uint64_t id = 0;                           // bumps on every arm and cancel

    void cancel() {
        deadline.reset();
        ++id;
    }

    // A new session was loaded by |by|.
    void loaded(SessionOwner by) {
        owner = by;
        cancel();
    }

    // |by| used the resident session. A GUI use takes it over for good: the
    // API's lease, if any, is gone. An API use leaves the owner as it was.
    void used(SessionOwner by) {
        if (by == SessionOwner::Gui) {
            owner = SessionOwner::Gui;
            cancel();
        }
    }

    // The resident session was released.
    void released() {
        owner = SessionOwner::None;
        cancel();
    }

    bool may_expire() const {
        return owner == SessionOwner::Api;
    }

    // Arms a finite lease for the session of |current| hub generation. False,
    // and nothing armed, when the session is not API-owned.
    bool arm(std::chrono::milliseconds delay, uint64_t current, Clock::time_point now) {
        cancel();
        if (!may_expire() || delay.count() <= 0)
            return false;
        generation = current;
        deadline = now + delay;
        return true;
    }

    // True only for the lease a timer armed (|expected_id|), on the same
    // session (|current| generation), still API-owned, once its time is up.
    bool expired(uint64_t current, uint64_t expected_id, Clock::time_point now) const {
        return deadline && may_expire() && current == generation && expected_id == id &&
               now >= *deadline;
    }
};

} // namespace xllama
