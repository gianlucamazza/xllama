// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#include "xllama/memory_admission.h"

#include <doctest/doctest.h>

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
