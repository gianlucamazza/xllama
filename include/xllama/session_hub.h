// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT
// Single process-wide owner of the loaded model Session.
//
// Before this existed the GUI (MainPageController::m_session) and the LAN API
// (api-server.cpp g_session) each owned an independent Session: both could
// hold a model at the same time, silently breaking the "never 2x model in
// RAM" invariant EnsureSession enforced only within the GUI. The hub is the
// one owner; both surfaces lock it for the duration of a turn.
//
// Locking contract:
//   - Take `mtx` before touching `session` / `model`, and HOLD it across
//     ensure_locked() + generate() so a concurrent surface cannot swap the
//     resident model mid-turn. The GUI blocks on the lock (its turns are
//     serialized anyway); the API uses try_lock and reports busy.
//   - `generation` increments every time the resident session changes. A
//     surface that keeps per-conversation state tied to the live session
//     (the GUI's KV-reuse flag) must record it and drop that state when it
//     no longer matches — the other surface may have swapped models between
//     turns.
#pragma once

#include "xllama/session.h"

#include <cstdint>
#include <memory>
#include <mutex>
#include <string>

namespace xllama {

struct SessionHub {
    std::mutex mtx;
    std::unique_ptr<Session> session; // guarded by mtx
    std::string model;                // model id loaded into `session`; guarded by mtx
    uint64_t generation = 0;          // bumps on every resident-session change; guarded by mtx

    // Under mtx: return the resident session for |model_id|, creating it (and
    // destroying any other model's session FIRST — never 2x model in RAM) if
    // needed. On creation failure the hub is left empty and nullptr returns.
    Session* ensure_locked(const std::string& model_id, const SessionParams& sp,
                           std::string* err = nullptr) {
        if (session && model == model_id)
            return session.get();
        session.reset(); // release the old model before loading the new one
        model.clear();
        ++generation;
        auto s = Session::create(sp, err);
        if (!s)
            return nullptr;
        session = std::move(s);
        model = model_id;
        ++generation;
        return session.get();
    }

    // Under mtx: drop the resident session (e.g. to free RAM on demand).
    void reset_locked() {
        if (session) {
            session.reset();
            model.clear();
            ++generation;
        }
    }
};

// The process-wide hub (defined in session.cpp).
SessionHub& session_hub();

} // namespace xllama
