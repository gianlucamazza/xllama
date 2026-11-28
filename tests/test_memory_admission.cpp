// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#include "xllama/memory_admission.h"

#include <doctest/doctest.h>

#include <limits>

using xllama::admit_session_memory;
using xllama::kSessionMemoryMarginMb;
using xllama::SessionMemory;

TEST_CASE("admit_session_memory: fits, refuses over the app limit, ignores an unknown limit") {
    SessionMemory need;
    need.host_model_mb = 2000;
    need.host_context_mb = 600;
    need.host_compute_mb = 300;
    CHECK(need.host_mb() == 2900);
    const std::size_t limit = 5120;
    // Exactly at the limit with the margin is admitted; one MB over is not.
    const std::size_t room = limit - need.host_mb() - kSessionMemoryMarginMb;
    CHECK(admit_session_memory(need, 4096, room, limit, 0, 0).ok);
    const auto over = admit_session_memory(need, 4096, room + 1, limit, 0, 0);
    CHECK_FALSE(over.ok);
    CHECK(xllama::is_memory_refusal(over.message));
    CHECK(over.message.find("n_ctx 4096") != std::string::npos);
    CHECK(over.message.find("weights 2000, context 600, compute 300") != std::string::npos);
    CHECK(over.message.find("5120 MB app limit") != std::string::npos);
    CHECK(admit_session_memory(need, 4096, 100000, 0, 0, 0).ok); // host build: no limit
}

TEST_CASE("admit_session_memory: GPU layers are checked against the GPU budget") {
    SessionMemory need;
    need.host_model_mb = 100;
    need.device_mb = 900;
    CHECK(admit_session_memory(need, 2048, 500, 5120, 2800, 3801).ok);
    const auto gpu = admit_session_memory(need, 2048, 500, 5120, 2950, 3801);
    CHECK_FALSE(gpu.ok);
    CHECK(xllama::is_memory_refusal(gpu.message));
    CHECK(gpu.message.find("GPU budget") != std::string::npos);
    CHECK_FALSE(xllama::is_memory_refusal("failed to load model: x.gguf"));
    CHECK(admit_session_memory(need, 2048, 500, 5120, 2950, 0).ok); // unknown budget
}

TEST_CASE("admit_session_memory: quota comparisons cannot wrap") {
    constexpr auto max = (std::numeric_limits<std::size_t>::max)();
    SessionMemory need;
    need.host_model_mb = max - 1;
    need.host_context_mb = 3;
    CHECK_FALSE(admit_session_memory(need, 2048, 0, 5120, 0, 0).ok);
    CHECK_FALSE(admit_session_memory(need, 2048, 0, max, 0, 0).ok);

    need = {};
    CHECK_FALSE(admit_session_memory(need, 2048, max, max, 0, 0).ok);
    need.device_mb = 32;
    CHECK_FALSE(admit_session_memory(need, 2048, 0, 5120, max - 16, max).ok);
    CHECK(admit_session_memory(need, 2048, 0, 5120, max - 32, max).ok);
}

TEST_CASE("admit_session_memory: an embedding context charges only new UMA buffers") {
    constexpr std::size_t mib = 1024 * 1024;
    auto estimate = [](bool weights_resident) {
        xllama::SessionMemoryEstimate dry_run(weights_resident);
        dry_run.add_buffer(800 * mib, 40 * mib, 10 * mib, /*on_host=*/true, /*on_device=*/true);
        dry_run.add_buffer(200 * mib, 40 * mib, 10 * mib, /*on_host=*/true, /*on_device=*/false);
        return dry_run.charges();
    };
    // 800 MiB of device weights are already in both the app and device usage.
    // An unrelated allocation still consumes space on both budgets.
    constexpr std::size_t app_used = 1300, gpu_used = 950;
    constexpr std::size_t app_limit = app_used + 100 + kSessionMemoryMarginMb;
    constexpr std::size_t gpu_limit = gpu_used + 50;
    CHECK_FALSE(
        admit_session_memory(estimate(false), 2048, app_used, app_limit, gpu_used, gpu_limit).ok);
    const auto context = estimate(true);
    CHECK(context.host_mb() == 100);
    CHECK(context.device_mb == 50);
    CHECK(admit_session_memory(context, 2048, app_used, app_limit, gpu_used, gpu_limit).ok);
    CHECK_FALSE(
        admit_session_memory(context, 2048, app_used + 1, app_limit, gpu_used, gpu_limit).ok);
    CHECK_FALSE(
        admit_session_memory(context, 2048, app_used, app_limit, gpu_used + 1, gpu_limit).ok);
}

TEST_CASE(
    "admit_session_memory: byte charges round up after aggregation and overflow fails closed") {
    constexpr std::size_t mib = 1024 * 1024;
    xllama::SessionMemoryEstimate estimate(/*weights_resident=*/false);
    estimate.add_buffer(mib - 1, 1, 1, /*on_host=*/true, /*on_device=*/false);
    estimate.add_buffer(1, 0, 0, /*on_host=*/true, /*on_device=*/false);
    estimate.add_buffer(0, mib, 1, /*on_host=*/false, /*on_device=*/true);
    const SessionMemory need = estimate.charges();
    CHECK(need.host_model_mb == 1);
    CHECK(need.host_context_mb == 1);
    CHECK(need.host_compute_mb == 1);
    CHECK(need.device_mb == 2);
    CHECK_FALSE(admit_session_memory(need, 2048, 0, kSessionMemoryMarginMb + 2, 0, 2).ok);
    CHECK_FALSE(admit_session_memory(need, 2048, 0, kSessionMemoryMarginMb + 3, 0, 1).ok);

    constexpr auto max = (std::numeric_limits<std::size_t>::max)();
    estimate.add_buffer(max, max, 2, /*on_host=*/true, /*on_device=*/true);
    const auto overflow = admit_session_memory(estimate.charges(), 2048, 0, 0, 0, 0);
    CHECK_FALSE(overflow.ok);
    CHECK(xllama::is_memory_refusal(overflow.message));
}
