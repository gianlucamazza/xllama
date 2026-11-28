// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Whether a GGUF session fits before it is loaded: weights + context (KV and
// recurrent state) + compute buffers, against the app's memory limit and,
// for layers on the D3D12 device, the GPU budget.
//
// The dry run reads llama_get_memory_breakdown from a no_alloc model and
// context. It supplies the actual per-layer KV, recurrent and compute sizes.
#pragma once

#include <cstddef>
#include <optional>
#include <string>

namespace xllama {

// host_* is everything in the process's system memory, including buffers on
// an integrated GPU that uses host memory (D3D12 on the Series S: its weights
// occupy the same working set, measured in docs/gguf-gpu-decode.md D2-r2).
// device_mb is what GPU devices hold, checked against the GPU budget too.
struct SessionMemory {
    std::size_t host_model_mb = 0;   // weights in system memory
    std::size_t host_context_mb = 0; // KV cache + recurrent state in system memory
    std::size_t host_compute_mb = 0; // compute buffers in system memory
    std::size_t device_mb = 0;       // buffers on GPU devices (D3D12)

    std::size_t host_mb() const;
};

// Aggregates the pin's buffer-type breakdown in bytes before rounding each
// charge up to MiB. A UMA allocation contributes to each applicable quota.
class SessionMemoryEstimate {
  public:
    explicit SessionMemoryEstimate(bool weights_resident) : m_weights_resident(weights_resident) {}

    void add_buffer(std::size_t model, std::size_t context, std::size_t compute, bool on_host,
                    bool on_device);
    SessionMemory charges() const;

  private:
    struct ByteCount {
        std::optional<std::size_t> bytes = 0;
        void add(std::size_t size);
        std::size_t mib() const;
    };

    bool m_weights_resident;
    ByteCount m_host_model, m_host_context, m_host_compute, m_device;
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

// Headroom for allocations the dry run does not see (tokenizer, sampler,
// decode scratch, allocator overhead). On the Series S the estimate alone
// already covered the working-set growth of five catalogue loads, GPU and
// lfm25-8b-a1b included (bench/results/2026-11-25-context-architecture).
// Every load logs the estimate and the working set in use beside the
// "ws after model load / after context" lines, so this can be re-measured.
inline constexpr std::size_t kSessionMemoryMarginMb = 192;

// |host_in_use_mb| is the process working set before the load (the previous
// session already released). A zero quota omits that quota check. An
// unrepresentable size always refuses the request.
MemoryVerdict admit_session_memory(const SessionMemory& need, int n_ctx, std::size_t host_in_use_mb,
                                   std::size_t app_limit_mb, std::size_t gpu_in_use_mb,
                                   std::size_t gpu_budget_mb);

} // namespace xllama
