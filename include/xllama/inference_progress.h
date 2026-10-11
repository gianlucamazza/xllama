// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// Prefill/decode progress logs and an optional stall watchdog.
//
// The 2026-10-11 LFM2-2.6B CPU re-measure hung after `prompt tokens: 298`
// (printed just before prefill). The next line is only written at end of
// generation, so a log tail cannot tell prefill from decode. This header is
// the one place both backends and both surfaces (CLI/bench and Session) read
// cadence and stall policy from, so they cannot drift.
//
// Defaults are modest and portable: nothing here is sized for one console.
// A stall timeout of 0 disables the watchdog.
#pragma once

#include "xllama/path_utils.h"
#include "xllama/platform.h"
#ifdef XLLAMA_UWP
    #include "xllama/utf8_utils.h"
#endif

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <thread>

namespace xllama {

/// Decode progress log: emit when this many new tokens have landed (0 = off).
inline constexpr int kDefaultDecodeLogEveryN = 64;
/// Decode progress log: emit when this many milliseconds have passed (0 = off).
inline constexpr int kDefaultDecodeLogEveryMs = 15000;
/// Abort if a prefill chunk or decode token produces no progress for this long.
/// 180 s is above a 2048-token CPU prefill on shipping models (LFM2-2.6B's
/// successful 298-token prefills were ~5.4 s on Series S, 2026-10-11) and far
/// below "hang forever". 0 disables the watchdog.
inline constexpr int kDefaultInferStallTimeoutS = 180;

/// LocalState / cwd marker the bench harness polls when inference cannot
/// return (a hang inside llama_decode). Written once when the watchdog fires.
inline constexpr const char* kInferenceStallMarker = "inference-stall.txt";

enum class InferencePhase { Idle, Prefill, Decode };

struct InferenceProgressConfig {
    int decode_log_every_n = kDefaultDecodeLogEveryN;
    int decode_log_every_ms = kDefaultDecodeLogEveryMs;
    int stall_timeout_s = kDefaultInferStallTimeoutS;
};

inline const char* inference_phase_name(InferencePhase phase) {
    switch (phase) {
    case InferencePhase::Prefill:
        return "prefill";
    case InferencePhase::Decode:
        return "decode";
    case InferencePhase::Idle:
        break;
    }
    return "idle";
}

/// True when a decode-progress line should be emitted. Token cadence uses
/// integer division so a speculative jump that skips the exact multiple still
/// logs. Time cadence is independent: a hang at token 5 still identifies the
/// phase after |every_ms|.
inline bool should_log_decode_progress(int n_generated, int last_logged, int every_n,
                                       std::chrono::milliseconds since_last_log, int every_ms) {
    if (n_generated <= 0)
        return false;
    if (every_n > 0 && (n_generated / every_n) > (last_logged / every_n))
        return true;
    if (every_ms > 0 && since_last_log.count() >= every_ms)
        return true;
    return false;
}

inline bool infer_stall_timed_out(std::chrono::milliseconds since_progress, int timeout_s) {
    if (timeout_s <= 0)
        return false;
    return since_progress.count() >= static_cast<int64_t>(timeout_s) * 1000;
}

inline std::string format_prefill_start_log(int tokens) {
    char buf[128];
    std::snprintf(buf, sizeof(buf), "[xllama] prefill starting: %d tokens\n", tokens);
    return buf;
}

inline std::string format_prefill_done_log(int tokens, double ms) {
    char buf[160];
    std::snprintf(buf, sizeof(buf), "[xllama] prefill done: %d tokens in %.0f ms\n", tokens, ms);
    return buf;
}

inline std::string format_decode_progress_log(int n_generated, int n_predict, double ms) {
    char buf[192];
    std::snprintf(buf, sizeof(buf), "[xllama] decode progress: %d/%d tokens (%.0f ms)\n",
                  n_generated, n_predict, ms);
    return buf;
}

inline std::string format_infer_stall_error(InferencePhase phase, int n_prompt, int n_prompt_done,
                                            int n_decode, int timeout_s) {
    char buf[256];
    std::snprintf(buf, sizeof(buf), "inference stalled: phase=%s prompt=%d/%d decode=%d after %ds",
                  inference_phase_name(phase), n_prompt_done, n_prompt, n_decode, timeout_s);
    return buf;
}

inline std::string format_infer_stall_log(InferencePhase phase, int n_prompt, int n_prompt_done,
                                          int n_decode, int timeout_s) {
    return "[xllama] watchdog: " +
           format_infer_stall_error(phase, n_prompt, n_prompt_done, n_decode, timeout_s) + "\n";
}

/// Overlay operator env vars when they are set. CLI flags / bench files applied
/// after this still win. Unknown or empty values are ignored.
inline void overlay_progress_env(InferenceProgressConfig& cfg) {
    if (const char* e = std::getenv("XLLAMA_DECODE_PROGRESS_EVERY"))
        cfg.decode_log_every_n = std::atoi(e);
    if (const char* e = std::getenv("XLLAMA_DECODE_PROGRESS_MS"))
        cfg.decode_log_every_ms = std::atoi(e);
    if (const char* e = std::getenv("XLLAMA_INFER_STALL_TIMEOUT_S"))
        cfg.stall_timeout_s = std::atoi(e);
}

inline void write_inference_stall_marker(const std::string& body) {
    const std::string path = resolve_local_path(kInferenceStallMarker);
#ifdef XLLAMA_UWP
    FILE* fp = _wfopen(utf8_to_wstring(path).c_str(), L"w");
#else
    FILE* fp = std::fopen(path.c_str(), "w");
#endif
    if (!fp)
        return;
    std::fputs(body.c_str(), fp);
    std::fclose(fp);
}

/// Background stall detector plus the prefill/decode progress logger.
///
/// Progress is noted on the inference thread (phase change, finished prefill
/// chunk, generated token). A helper thread wakes every |poll_ms| and, if
/// nothing has been noted for |stall_timeout_s|, logs the phase/position,
/// writes `inference-stall.txt`, and sets the abort flag. A hang inside a
/// single blocking `llama_decode` / `AppendTokenSequences` cannot be cancelled
/// from this thread; the marker is what lets the bench harness record the run
/// instead of waiting forever.
class InferenceWatchdog {
  public:
    InferenceWatchdog(const InferenceProgressConfig& cfg, std::atomic<bool>* abort_flag,
                      int poll_ms = 1000)
        : cfg_(cfg), caller_abort_(abort_flag), poll_ms_(poll_ms > 0 ? poll_ms : 1000) {
        last_progress_ = std::chrono::steady_clock::now();
        if (cfg_.stall_timeout_s > 0)
            thread_ = std::thread(&InferenceWatchdog::thread_main, this);
    }

    InferenceWatchdog(const InferenceWatchdog&) = delete;
    InferenceWatchdog& operator=(const InferenceWatchdog&) = delete;

    ~InferenceWatchdog() {
        stop();
    }

    void enter_prefill(int n_prompt) {
        {
            std::lock_guard<std::mutex> g(mu_);
            phase_ = InferencePhase::Prefill;
            n_prompt_ = n_prompt;
            n_prompt_done_ = 0;
            n_decode_ = 0;
            last_progress_ = std::chrono::steady_clock::now();
        }
        log_output(format_prefill_start_log(n_prompt));
    }

    void note_prefill_chunk(int tokens_done) {
        std::lock_guard<std::mutex> g(mu_);
        n_prompt_done_ = tokens_done;
        last_progress_ = std::chrono::steady_clock::now();
    }

    void prefill_done(int tokens, double ms) {
        {
            std::lock_guard<std::mutex> g(mu_);
            n_prompt_ = tokens;
            n_prompt_done_ = tokens;
            last_progress_ = std::chrono::steady_clock::now();
        }
        log_output(format_prefill_done_log(tokens, ms));
    }

    void enter_decode(int n_predict) {
        std::lock_guard<std::mutex> g(mu_);
        phase_ = InferencePhase::Decode;
        n_predict_ = n_predict;
        decode_start_ = std::chrono::steady_clock::now();
        last_progress_ = decode_start_;
        last_log_ = decode_start_;
        last_logged_n_ = 0;
    }

    void on_decode_token(int n_generated) {
        const auto now = std::chrono::steady_clock::now();
        bool emit = false;
        double ms = 0.0;
        int n_predict = 0;
        {
            std::lock_guard<std::mutex> g(mu_);
            phase_ = InferencePhase::Decode;
            n_decode_ = n_generated;
            last_progress_ = now;
            const auto since_log =
                std::chrono::duration_cast<std::chrono::milliseconds>(now - last_log_);
            emit = should_log_decode_progress(n_generated, last_logged_n_, cfg_.decode_log_every_n,
                                              since_log, cfg_.decode_log_every_ms);
            if (emit) {
                last_logged_n_ = n_generated;
                last_log_ = now;
                ms = std::chrono::duration<double, std::milli>(now - decode_start_).count();
                n_predict = n_predict_;
            }
        }
        if (emit)
            log_output(format_decode_progress_log(n_generated, n_predict, ms));
    }

    bool aborted() const {
        if (fired_.load(std::memory_order_acquire))
            return true;
        return caller_abort_ && caller_abort_->load(std::memory_order_acquire);
    }

    bool stalled() const {
        return fired_.load(std::memory_order_acquire);
    }

    std::string stall_error() const {
        std::lock_guard<std::mutex> g(mu_);
        if (stall_error_.empty())
            return format_infer_stall_error(phase_, n_prompt_, n_prompt_done_, n_decode_,
                                            cfg_.stall_timeout_s);
        return stall_error_;
    }

  private:
    void stop() {
        {
            std::lock_guard<std::mutex> g(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        if (thread_.joinable())
            thread_.join();
    }

    void thread_main() {
        std::unique_lock<std::mutex> lk(mu_);
        while (!stop_) {
            cv_.wait_for(lk, std::chrono::milliseconds(poll_ms_));
            if (stop_)
                break;
            const auto since = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - last_progress_);
            if (!infer_stall_timed_out(since, cfg_.stall_timeout_s))
                continue;
            if (fired_.load(std::memory_order_relaxed))
                return;
            stall_error_ = format_infer_stall_error(phase_, n_prompt_, n_prompt_done_, n_decode_,
                                                    cfg_.stall_timeout_s);
            fired_.store(true, std::memory_order_release);
            if (caller_abort_)
                caller_abort_->store(true, std::memory_order_release);
            const std::string err = stall_error_;
            const InferencePhase phase = phase_;
            const int n_prompt = n_prompt_;
            const int n_prompt_done = n_prompt_done_;
            const int n_decode = n_decode_;
            const int timeout_s = cfg_.stall_timeout_s;
            lk.unlock();
            log_output(format_infer_stall_log(phase, n_prompt, n_prompt_done, n_decode, timeout_s));
            write_inference_stall_marker(err + "\n");
            return;
        }
    }

    InferenceProgressConfig cfg_;
    std::atomic<bool>* caller_abort_ = nullptr;
    int poll_ms_ = 1000;

    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::thread thread_;
    bool stop_ = false;
    std::atomic<bool> fired_{false};

    InferencePhase phase_ = InferencePhase::Idle;
    int n_prompt_ = 0;
    int n_prompt_done_ = 0;
    int n_decode_ = 0;
    int n_predict_ = 0;
    int last_logged_n_ = 0;
    std::chrono::steady_clock::time_point last_progress_{};
    std::chrono::steady_clock::time_point last_log_{};
    std::chrono::steady_clock::time_point decode_start_{};
    std::string stall_error_;
};

} // namespace xllama
