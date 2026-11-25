// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Whether a GGUF session fits before it is loaded: weights + context (KV and
// recurrent state) + compute buffers, against the app's memory limit and,
// for layers on the D3D12 device, the GPU budget.
//
// The figures come from llama.cpp's own dry run (common_get_device_memory_data
// in common/fit.h: the model and a context are created with no_alloc, so the
// real per-layer KV of a hybrid and the real compute buffer are counted, not a
// weights-times-factor estimate). n_ctx used to be bounded by a constant only
// and the one estimate (weights x 1.12) gated embeddings alone; an n_ctx or a
// model too large for the console then failed inside an allocation.
#pragma once

#include <cstddef>
#include <string>

namespace xllama {

struct SessionMemory {
    std::size_t host_model_mb = 0;   // weights kept in host memory
    std::size_t host_context_mb = 0; // KV cache + recurrent state in host memory
    std::size_t host_compute_mb = 0; // compute buffers in host memory
    std::size_t device_mb = 0;       // everything placed on GPU devices (D3D12)

    std::size_t host_mb() const {
        return host_model_mb + host_context_mb + host_compute_mb;
    }
};

struct MemoryVerdict {
    bool ok = true;
    std::string message; // why not, in user terms; empty when ok
};

// Every refusal message starts with this, so a surface that only sees the
// Session::create error (SessionHub::ensure_locked) can answer a request
// for a model that does not fit as the client's choice (HTTP 400), not as a
// server fault.
inline constexpr const char* kMemoryRefusalPrefix = "insufficient memory: ";

inline bool is_memory_refusal(const std::string& error) {
    return error.rfind(kMemoryRefusalPrefix, 0) == 0;
}

// Allocations the dry run does not see (tokenizer, sampler, decode scratch,
// the allocator's own overhead). Calibrated on host against the measured
// working-set growth of a real load: every load logs the estimate next to
// the "ws after model load / after context" lines (console calibration).
inline constexpr std::size_t kSessionMemoryMarginMb = 192;

// |host_in_use_mb| is the process working set before the load (the previous
// session already released). An unknown limit (0) admits; so does an unknown
// GPU budget for the device part.
MemoryVerdict admit_session_memory(const SessionMemory& need, int n_ctx, std::size_t host_in_use_mb,
                                   std::size_t app_limit_mb, std::size_t gpu_in_use_mb,
                                   std::size_t gpu_budget_mb);

} // namespace xllama
