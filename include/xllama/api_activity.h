// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Live view of LAN API traffic for the console UI. The server records each
// inference request (chat, embeddings, images) and every busy rejection; the
// UI polls snapshot() on a timer. Inference is single-slot (session_hub().mtx),
// so at most one request is in flight, but busy rejections and UI reads race
// with it from other threads: everything is guarded by one mutex.
// Header-only and WinRT-free so it is host-testable (tests/test_api_activity.cpp).

#pragma once

#include <chrono>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <utility>

namespace xllama {

class ApiActivity {
  public:
    using Clock = std::chrono::steady_clock;

    struct InFlight {
        std::string endpoint;
        std::string model; // empty until the handler has parsed the request
        double elapsed_s = 0.0;
    };

    struct Outcome {
        std::string endpoint;
        std::string model;
        int http_status = 0;
        double elapsed_s = 0.0;
        int prompt_tokens = -1; // -1: not reported by the handler
        int completion_tokens = -1;
    };

    struct Failure {
        std::string message;
        double age_s = 0.0;
    };

    struct Snapshot {
        std::optional<InFlight> in_flight;
        uint64_t requests = 0;  // inference requests that got the slot
        uint64_t ok = 0;        // of which answered 2xx
        uint64_t failed = 0;    // of which answered >= 400
        uint64_t busy = 0;      // rejected with 503 because the slot was taken
        uint64_t connected = 0; // sockets currently inside handle_connection
        std::optional<Outcome> last;
        std::optional<Failure> last_error;
    };

    // One in-flight request. Finish it with done(); a ticket destroyed without
    // done() (an exception escaped the handler) is recorded as a 500.
    class Ticket {
      public:
        Ticket(ApiActivity* owner, uint64_t id) : m_owner(owner), m_id(id) {}
        Ticket(const Ticket&) = delete;
        Ticket& operator=(const Ticket&) = delete;
        Ticket(Ticket&& other) noexcept
            : m_owner(std::exchange(other.m_owner, nullptr)), m_id(other.m_id),
              m_prompt_tokens(other.m_prompt_tokens),
              m_completion_tokens(other.m_completion_tokens) {}
        ~Ticket() {
            if (m_owner)
                m_owner->finish(m_id, 500, -1, -1, "request aborted");
        }

        // Called by the handler once it has parsed the request.
        void model(const std::string& name) {
            if (m_owner)
                m_owner->set_model(m_id, name);
        }

        // Called by the handler after a successful generation.
        void tokens(int prompt_tokens, int completion_tokens) {
            m_prompt_tokens = prompt_tokens;
            m_completion_tokens = completion_tokens;
        }

        void done(int http_status, const std::string& error = "",
                  Clock::time_point now = Clock::now()) {
            if (m_owner)
                std::exchange(m_owner, nullptr)
                    ->finish(m_id, http_status, m_prompt_tokens, m_completion_tokens, error, now);
        }

      private:
        ApiActivity* m_owner;
        uint64_t m_id;
        int m_prompt_tokens = -1;
        int m_completion_tokens = -1;
    };

    // One accepted TCP client. Destroying the handle decrements `connected`.
    class Client {
      public:
        Client() = default;
        Client(const Client&) = delete;
        Client& operator=(const Client&) = delete;
        Client(Client&& other) noexcept : m_owner(std::exchange(other.m_owner, nullptr)) {}
        Client& operator=(Client&& other) noexcept {
            if (this != &other) {
                reset();
                m_owner = std::exchange(other.m_owner, nullptr);
            }
            return *this;
        }
        ~Client() {
            reset();
        }

      private:
        friend class ApiActivity;
        explicit Client(ApiActivity* owner) : m_owner(owner) {}
        void reset() {
            if (m_owner) {
                std::exchange(m_owner, nullptr)->disconnect();
            }
        }
        ApiActivity* m_owner = nullptr;
    };

    Client client() {
        connect();
        return Client(this);
    }

    // Call once the request holds the inference slot.
    Ticket begin(const std::string& endpoint, Clock::time_point now = Clock::now()) {
        std::lock_guard<std::mutex> lock(m_mutex);
        ++m_requests;
        m_current = Current{++m_next_id, endpoint, "", now};
        return Ticket(this, m_current->id);
    }

    void busy() {
        std::lock_guard<std::mutex> lock(m_mutex);
        ++m_busy;
    }

    Snapshot snapshot(Clock::time_point now = Clock::now()) const {
        std::lock_guard<std::mutex> lock(m_mutex);
        Snapshot s;
        if (m_current)
            s.in_flight =
                InFlight{m_current->endpoint, m_current->model, seconds(now - m_current->started)};
        s.requests = m_requests;
        s.ok = m_ok;
        s.failed = m_failed;
        s.busy = m_busy;
        s.connected = m_connected;
        s.last = m_last;
        if (m_last_error)
            s.last_error = Failure{m_last_error->first, seconds(now - m_last_error->second)};
        return s;
    }

  private:
    struct Current {
        uint64_t id;
        std::string endpoint;
        std::string model;
        Clock::time_point started;
    };

    static double seconds(Clock::duration d) {
        return std::chrono::duration<double>(d).count();
    }

    void connect() {
        std::lock_guard<std::mutex> lock(m_mutex);
        ++m_connected;
    }

    void disconnect() {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_connected > 0)
            --m_connected;
    }

    void set_model(uint64_t id, const std::string& name) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (m_current && m_current->id == id)
            m_current->model = name;
    }

    void finish(uint64_t id, int http_status, int prompt_tokens, int completion_tokens,
                const std::string& error, Clock::time_point now = Clock::now()) {
        std::lock_guard<std::mutex> lock(m_mutex);
        if (!m_current || m_current->id != id)
            return; // stale ticket: never overwrite a newer request
        Outcome outcome{m_current->endpoint, m_current->model,
                        http_status,         seconds(now - m_current->started),
                        prompt_tokens,       completion_tokens};
        if (http_status >= 200 && http_status < 300) {
            ++m_ok;
        } else {
            ++m_failed;
            m_last_error =
                std::make_pair("HTTP " + std::to_string(http_status) + " " + outcome.endpoint +
                                   (error.empty() ? std::string() : ": " + error),
                               now);
        }
        m_last = std::move(outcome);
        m_current.reset();
    }

    mutable std::mutex m_mutex;
    uint64_t m_next_id = 0;
    uint64_t m_requests = 0;
    uint64_t m_ok = 0;
    uint64_t m_failed = 0;
    uint64_t m_busy = 0;
    uint64_t m_connected = 0;
    std::optional<Current> m_current;
    std::optional<Outcome> m_last;
    std::optional<std::pair<std::string, Clock::time_point>> m_last_error;
};

} // namespace xllama
