// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// Bounded LAN connection slots and the HTTP mapping for compute admission.
// WinRT-free so the host suite can test overlap without StreamSocketListener.

#pragma once

#include "xllama/compute_admission.h"

#include <atomic>
#include <memory>
#include <utility>

namespace xllama {

// One SSE reply plus a concurrent busy probe, health checks, and a little slack.
// The UWP listener takes a slot before dispatching a work item.
inline constexpr int kApiMaxInFlightConnections = 8;

inline constexpr int kApiHttpOk = 200;
inline constexpr int kApiHttpBusy = 503;
inline constexpr const char* kApiBusyStatus = "503 Service Unavailable";

// N-slot RAII gate for accepted TCP clients. Separate from the single compute
// permit: a connection can be admitted here and still get HTTP 503 if compute
// is occupied. Permit destruction may happen on a different thread.
class ApiConnectionGate {
    struct State {
        std::atomic<int> in_flight{0};
    };

  public:
    class Permit {
      public:
        Permit(const Permit&) = delete;
        Permit& operator=(const Permit&) = delete;
        ~Permit() {
            m_state->in_flight.fetch_sub(1, std::memory_order_acq_rel);
        }

      private:
        friend class ApiConnectionGate;
        explicit Permit(std::shared_ptr<State> state) : m_state(std::move(state)) {}
        std::shared_ptr<State> m_state;
    };

    using Handle = std::shared_ptr<Permit>;

    Handle try_acquire() {
        int cur = m_state->in_flight.load(std::memory_order_acquire);
        while (cur < kApiMaxInFlightConnections) {
            if (m_state->in_flight.compare_exchange_weak(cur, cur + 1, std::memory_order_acquire,
                                                         std::memory_order_relaxed)) {
                try {
                    return Handle(new Permit(m_state));
                } catch (...) {
                    m_state->in_flight.fetch_sub(1, std::memory_order_release);
                    throw;
                }
            }
        }
        return {};
    }

    int in_flight() const {
        return m_state->in_flight.load(std::memory_order_acquire);
    }

  private:
    std::shared_ptr<State> m_state = std::make_shared<State>();
};

inline ApiConnectionGate& process_api_connection_gate() {
    static ApiConnectionGate gate;
    return gate;
}

// One compute request's HTTP outcome. A held permit is 200 (caller serves the
// reply); an empty permit is 503 busy. The handle stays alive for the whole
// reply, including SSE; resetting it is the disconnect/abort release.
struct ApiComputeOutcome {
    int http_code = kApiHttpBusy;
    ComputeJobAdmission::Handle permit;
};

inline ApiComputeOutcome admit_api_compute(ComputeJobAdmission& admission) {
    ApiComputeOutcome out;
    out.permit = admission.try_acquire();
    if (out.permit)
        out.http_code = kApiHttpOk;
    return out;
}

} // namespace xllama
