// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#include "xllama/memory_admission.h"

#include <limits>

namespace xllama {

namespace {
bool fits(std::size_t used, std::size_t additional, std::size_t limit) {
    return used <= limit && additional <= limit - used;
}
} // namespace

void SessionMemoryEstimate::ByteCount::add(std::size_t size) {
    if (!bytes)
        return;
    if (fits(*bytes, size, (std::numeric_limits<std::size_t>::max)()))
        *bytes += size;
    else
        bytes.reset();
}

std::size_t SessionMemoryEstimate::ByteCount::mib() const {
    constexpr std::size_t mib = 1024 * 1024;
    if (!bytes)
        return (std::numeric_limits<std::size_t>::max)();
    return *bytes / mib + (*bytes % mib != 0);
}

void SessionMemoryEstimate::add_buffer(std::size_t model, std::size_t context, std::size_t compute,
                                       bool on_host, bool on_device) {
    const auto new_weights = m_weights_resident ? 0 : model;
    if (on_host) {
        m_host_model.add(new_weights);
        m_host_context.add(context);
        m_host_compute.add(compute);
    }
    if (on_device) {
        m_device.add(new_weights);
        m_device.add(context);
        m_device.add(compute);
    }
}

SessionMemory SessionMemoryEstimate::charges() const {
    return {m_host_model.mib(), m_host_context.mib(), m_host_compute.mib(), m_device.mib()};
}

std::size_t SessionMemory::host_mb() const {
    constexpr auto max = (std::numeric_limits<std::size_t>::max)();
    if (!fits(host_model_mb, host_context_mb, max) ||
        !fits(host_model_mb + host_context_mb, host_compute_mb, max))
        return max;
    return host_model_mb + host_context_mb + host_compute_mb;
}

MemoryVerdict admit_session_memory(const SessionMemory& need, int n_ctx, std::size_t host_in_use_mb,
                                   std::size_t app_limit_mb, std::size_t gpu_in_use_mb,
                                   std::size_t gpu_budget_mb) {
    MemoryVerdict v;
    const auto host_mb = need.host_mb();
    constexpr auto max = (std::numeric_limits<std::size_t>::max)();
    if (host_mb == max || need.device_mb == max)
        return {false, std::string(kMemoryRefusalPrefix) +
                           "GGUF memory estimate exceeds the address space"};
    if (app_limit_mb != 0 &&
        (!fits(host_in_use_mb, kSessionMemoryMarginMb, app_limit_mb) ||
         !fits(host_in_use_mb + kSessionMemoryMarginMb, host_mb, app_limit_mb))) {
        v.ok = false;
        v.message = std::string(kMemoryRefusalPrefix) + "this model at n_ctx " +
                    std::to_string(n_ctx) + " needs " + std::to_string(host_mb) + " MB (weights " +
                    std::to_string(need.host_model_mb) + ", context " +
                    std::to_string(need.host_context_mb) + ", compute " +
                    std::to_string(need.host_compute_mb) + ") with " +
                    std::to_string(host_in_use_mb) + " MB already in use, over the " +
                    std::to_string(app_limit_mb) +
                    " MB app limit — choose a smaller model or n_ctx";
        return v;
    }
    if (need.device_mb != 0 && gpu_budget_mb != 0 &&
        !fits(gpu_in_use_mb, need.device_mb, gpu_budget_mb)) {
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
