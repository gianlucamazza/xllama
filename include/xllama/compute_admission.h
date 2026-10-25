// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// Process-wide admission for resource-heavy UWP jobs.

#pragma once

#include <atomic>
#include <chrono>
#include <memory>
#include <thread>
#include <utility>

namespace xllama {

// Bounded first-request wait while startup pre-load owns the slot. 150 × 100 ms
// matches the existing LAN API warm-up window (docs/api-endpoint.md).
inline constexpr int kComputePreloadWaitAttempts = 150;
inline constexpr int kComputePreloadWaitSliceMs = 100;

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
        ~Permit() {
            m_state->active.store(false, std::memory_order_release);
        }

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

// Only the first caller may advertise the startup-wait state. A failed second
// admission must not clear a still-running preload's flag.
inline bool claim_session_preload(std::atomic<bool>& preloading) {
    bool expected = false;
    return preloading.compare_exchange_strong(expected, true, std::memory_order_acq_rel,
                                              std::memory_order_acquire);
}

// Retry try_acquire while `preloading` is set. Any other occupied slot returns
// empty immediately so callers can report busy. `sleep_fn` is injected so the
// host suite can cover wait, timeout, and no-wait without sleeping 15 s.
template <typename SleepFn>
ComputeJobAdmission::Handle try_acquire_while_preloading(ComputeJobAdmission& admission,
                                                         const std::atomic<bool>& preloading,
                                                         int max_attempts, SleepFn&& sleep_fn) {
    for (int i = 0; i < max_attempts; ++i) {
        if (auto permit = admission.try_acquire())
            return permit;
        if (!preloading.load(std::memory_order_acquire))
            return {};
        sleep_fn();
    }
    return admission.try_acquire();
}

inline ComputeJobAdmission::Handle
try_acquire_compute_job_while_preloading(const std::atomic<bool>& preloading,
                                         int max_attempts = kComputePreloadWaitAttempts) {
    return try_acquire_while_preloading(
        process_compute_job_admission(), preloading, max_attempts,
        [] { std::this_thread::sleep_for(std::chrono::milliseconds(kComputePreloadWaitSliceMs)); });
}

} // namespace xllama
