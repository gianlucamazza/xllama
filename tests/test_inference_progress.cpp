// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "xllama/inference_progress.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <thread>

using namespace xllama;

namespace {

void set_or_clear_env(const char* key, const char* value) {
    if (value)
        setenv(key, value, 1);
    else
        unsetenv(key);
}

} // namespace

TEST_CASE("inference_phase_name covers every phase") {
    CHECK(std::string(inference_phase_name(InferencePhase::Idle)) == "idle");
    CHECK(std::string(inference_phase_name(InferencePhase::Prefill)) == "prefill");
    CHECK(std::string(inference_phase_name(InferencePhase::Decode)) == "decode");
}

TEST_CASE("should_log_decode_progress: token cadence and speculative skip") {
    using ms = std::chrono::milliseconds;
    CHECK_FALSE(should_log_decode_progress(0, 0, 64, ms{0}, 15000));
    CHECK_FALSE(should_log_decode_progress(63, 0, 64, ms{0}, 0));
    CHECK(should_log_decode_progress(64, 0, 64, ms{0}, 0));
    // Speculative jump past the exact multiple still logs.
    CHECK(should_log_decode_progress(70, 60, 64, ms{0}, 0));
    CHECK_FALSE(should_log_decode_progress(70, 64, 64, ms{0}, 0));
    CHECK_FALSE(should_log_decode_progress(32, 0, 0, ms{0}, 0));
}

TEST_CASE("should_log_decode_progress: time cadence") {
    using ms = std::chrono::milliseconds;
    CHECK_FALSE(should_log_decode_progress(5, 0, 64, ms{14999}, 15000));
    CHECK(should_log_decode_progress(5, 0, 64, ms{15000}, 15000));
    CHECK_FALSE(should_log_decode_progress(5, 0, 64, ms{20000}, 0));
}

TEST_CASE("infer_stall_timed_out: zero disables") {
    using ms = std::chrono::milliseconds;
    CHECK_FALSE(infer_stall_timed_out(ms{999999}, 0));
    CHECK_FALSE(infer_stall_timed_out(ms{179999}, 180));
    CHECK(infer_stall_timed_out(ms{180000}, 180));
}

TEST_CASE("progress log lines identify phase and position") {
    const std::string start = format_prefill_start_log(298);
    CHECK(start.find("prefill starting: 298 tokens") != std::string::npos);
    const std::string done = format_prefill_done_log(298, 5430.0);
    CHECK(done.find("prefill done: 298 tokens in 5430 ms") != std::string::npos);
    const std::string dec = format_decode_progress_log(64, 512, 3600.0);
    CHECK(dec.find("decode progress: 64/512 tokens") != std::string::npos);
    const std::string err = format_infer_stall_error(InferencePhase::Prefill, 298, 0, 0, 180);
    CHECK(err.find("phase=prefill") != std::string::npos);
    CHECK(err.find("prompt=0/298") != std::string::npos);
    CHECK(err.find("decode=0") != std::string::npos);
    CHECK(err.find("after 180s") != std::string::npos);
}

TEST_CASE("overlay_progress_env applies only set variables") {
    const char* old_n = std::getenv("XLLAMA_DECODE_PROGRESS_EVERY");
    const char* old_ms = std::getenv("XLLAMA_DECODE_PROGRESS_MS");
    const char* old_s = std::getenv("XLLAMA_INFER_STALL_TIMEOUT_S");
    std::string keep_n = old_n ? old_n : "";
    std::string keep_ms = old_ms ? old_ms : "";
    std::string keep_s = old_s ? old_s : "";

    set_or_clear_env("XLLAMA_DECODE_PROGRESS_EVERY", nullptr);
    set_or_clear_env("XLLAMA_DECODE_PROGRESS_MS", nullptr);
    set_or_clear_env("XLLAMA_INFER_STALL_TIMEOUT_S", "90");
    InferenceProgressConfig cfg;
    overlay_progress_env(cfg);
    CHECK(cfg.decode_log_every_n == kDefaultDecodeLogEveryN);
    CHECK(cfg.stall_timeout_s == 90);

    set_or_clear_env("XLLAMA_DECODE_PROGRESS_EVERY", keep_n.empty() ? nullptr : keep_n.c_str());
    set_or_clear_env("XLLAMA_DECODE_PROGRESS_MS", keep_ms.empty() ? nullptr : keep_ms.c_str());
    set_or_clear_env("XLLAMA_INFER_STALL_TIMEOUT_S", keep_s.empty() ? nullptr : keep_s.c_str());
}

TEST_CASE("InferenceWatchdog fires after a short stall timeout") {
    InferenceProgressConfig cfg;
    cfg.stall_timeout_s = 1;
    cfg.decode_log_every_n = 0;
    cfg.decode_log_every_ms = 0;
    std::atomic<bool> abort{false};
    InferenceWatchdog wd(cfg, &abort, /*poll_ms=*/50);
    wd.enter_prefill(298);
    std::this_thread::sleep_for(std::chrono::milliseconds(1200));
    CHECK(wd.stalled());
    CHECK(abort.load());
    CHECK(wd.stall_error().find("phase=prefill") != std::string::npos);
    CHECK(wd.stall_error().find("prompt=0/298") != std::string::npos);
    std::remove(resolve_local_path(kInferenceStallMarker).c_str());
}

TEST_CASE("InferenceWatchdog timeout 0 never fires") {
    InferenceProgressConfig cfg;
    cfg.stall_timeout_s = 0;
    std::atomic<bool> abort{false};
    InferenceWatchdog wd(cfg, &abort, /*poll_ms=*/20);
    wd.enter_prefill(10);
    std::this_thread::sleep_for(std::chrono::milliseconds(60));
    CHECK_FALSE(wd.stalled());
    CHECK_FALSE(abort.load());
}
