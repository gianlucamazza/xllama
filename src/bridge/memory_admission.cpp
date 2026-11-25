// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#include "xllama/memory_admission.h"

namespace xllama {

MemoryVerdict admit_session_memory(const SessionMemory& need, int n_ctx, std::size_t host_in_use_mb,
                                   std::size_t app_limit_mb, std::size_t gpu_in_use_mb,
                                   std::size_t gpu_budget_mb) {
    MemoryVerdict v;
    const std::size_t host_total = host_in_use_mb + need.host_mb() + kSessionMemoryMarginMb;
    if (app_limit_mb != 0 && host_total > app_limit_mb) {
        v.ok = false;
        v.message = std::string(kMemoryRefusalPrefix) + "this model at n_ctx " +
                    std::to_string(n_ctx) + " needs " + std::to_string(need.host_mb()) +
                    " MB (weights " + std::to_string(need.host_model_mb) + ", context " +
                    std::to_string(need.host_context_mb) + ", compute " +
                    std::to_string(need.host_compute_mb) + ") with " +
                    std::to_string(host_in_use_mb) + " MB already in use, over the " +
                    std::to_string(app_limit_mb) +
                    " MB app limit — choose a smaller model or n_ctx";
        return v;
    }
    if (need.device_mb != 0 && gpu_budget_mb != 0 &&
        gpu_in_use_mb + need.device_mb > gpu_budget_mb) {
        v.ok = false;
        v.message = std::string(kMemoryRefusalPrefix) + "this model's GPU layers at n_ctx " +
                    std::to_string(n_ctx) + ": they need " + std::to_string(need.device_mb) +
                    " MB with " + std::to_string(gpu_in_use_mb) + " MB in use, over the " +
                    std::to_string(gpu_budget_mb) + " MB GPU budget — use fewer GPU layers";
        return v;
    }
    return v;
}

} // namespace xllama
