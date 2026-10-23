// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// Process-wide admission for resource-heavy UWP jobs.

#pragma once

#include <atomic>
#include <memory>
#include <utility>

namespace xllama {

// A single-slot gate for compute work that competes for console resources but
// does not necessarily use the resident Session (for example, diffusion and
// device training). Permit destruction may happen on a different thread from
// acquisition; the shared token keeps the lease alive across async callbacks.
class ComputeJobAdmission {
    struct State {
        std::atomic<bool> active{false};
    };

  public:
    class Permit {
      public:
        Permit(const Permit&) = delete;
        Permit& operator=(const Permit&) = delete;
        ~Permit() { m_state->active.store(false, std::memory_order_release); }

      private:
        friend class ComputeJobAdmission;
        explicit Permit(std::shared_ptr<State> state) : m_state(std::move(state)) {}
        std::shared_ptr<State> m_state;
    };

    using Handle = std::shared_ptr<Permit>;

    Handle try_acquire() {
        bool expected = false;
        if (!m_state->active.compare_exchange_strong(expected, true, std::memory_order_acquire,
                                                      std::memory_order_relaxed))
            return {};
        try {
            return Handle(new Permit(m_state));
        } catch (...) {
            m_state->active.store(false, std::memory_order_release);
            throw;
        }
    }

  private:
    std::shared_ptr<State> m_state = std::make_shared<State>();
};

inline ComputeJobAdmission& process_compute_job_admission() {
    static ComputeJobAdmission admission;
    return admission;
}

inline ComputeJobAdmission::Handle try_acquire_compute_job() {
    return process_compute_job_admission().try_acquire();
}

} // namespace xllama
