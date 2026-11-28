// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Shared D3D12 compute plumbing for the GPU probes (gpugemv, gpustep) and the
// future GGUF GPU backend (docs/gguf-gpu-decode.md). Windows/UWP only; the
// system D3D12 runtime via d3d12_dyn, never the Agility device factory
// (docs/uwp-constraints.md §7).
#pragma once

#include <cstdint>
#include <limits>
#include <memory>
#include <mutex>
#include <utility>

namespace xllama::d3d12c {

enum class FenceResult { complete, removed, pending, signal_rejected };

inline FenceResult fence_result(std::uint64_t target, std::uint64_t completed) {
    if (completed == (std::numeric_limits<std::uint64_t>::max)())
        return FenceResult::removed;
    return completed >= target ? FenceResult::complete : FenceResult::pending;
}

// The queue has accepted this marker. An event is a wakeup hint. Only the
// fence counter establishes completion; UINT64_MAX denotes device removal.
template <typename Read, typename Arm, typename Wait, typename Poll, typename Expired>
FenceResult wait_accepted_fence_until(std::uint64_t target, bool spin, Read read, Arm arm,
                                      Wait wait, Poll poll, Expired expired) {
    bool try_event = !spin;
    bool armed = false;
    for (;;) {
        const auto result = fence_result(target, read());
        if (result != FenceResult::pending || expired())
            return result;
        if (try_event) {
            if (!armed)
                armed = arm();
            if (armed && wait())
                continue;
            try_event = false;
        }
        poll();
    }
}

template <typename Read, typename Arm, typename Wait, typename Poll>
bool wait_accepted_fence(std::uint64_t target, bool spin, Read read, Arm arm, Wait wait,
                         Poll poll) {
    return wait_accepted_fence_until(target, spin, read, arm, wait, poll, [] { return false; }) ==
           FenceResult::complete;
}

// One queue and one private fence. A rejected Signal may be retried on the
// same queue. Only an accepted marker may establish normal completion.
class ProbeMarker {
  public:
    bool pending() const {
        return state_ != State::idle;
    }

    template <typename Execute> bool submit(Execute execute) {
        if (pending())
            return false;
        ++target_;
        state_ = State::unmarked;
        execute();
        return true;
    }

    template <typename Signal, typename Read, typename Observe>
    FenceResult retire(Signal signal, Read read, Observe observe) {
        if (!pending())
            return FenceResult::complete;
        if (state_ == State::unmarked) {
            if (!signal(target_)) {
                if (fence_result(target_, read()) != FenceResult::removed)
                    return FenceResult::signal_rejected;
                state_ = State::idle;
                return FenceResult::removed;
            }
            state_ = State::marked;
        }
        const auto result = observe(target_);
        if (result == FenceResult::complete || result == FenceResult::removed)
            state_ = State::idle;
        return result;
    }

  private:
    enum class State { idle, unmarked, marked };
    State state_ = State::idle;
    std::uint64_t target_ = 0;
};

// Own the whole campaign before it can execute. A retry checks the old
// marker once before allocating another campaign or allowing any new work.
template <typename Commands> class ProbeSlot {
  public:
    template <typename Concrete, typename Run> bool try_run(Run run) {
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
        if (!lock || (owner_ && !owner_->try_retire()))
            return false;
        owner_.reset();
        // Transfer only an unresolved submission, including during unwinding.
        struct Campaign {
            std::unique_ptr<Commands>& retained;
            std::unique_ptr<Concrete> work = std::make_unique<Concrete>();
            ~Campaign() {
                if (work->pending())
                    retained = std::move(work);
            }
        } campaign{owner_};
        run(*campaign.work);
        return true;
    }

  private:
    std::mutex mutex_;
    std::unique_ptr<Commands> owner_;
};

} // namespace xllama::d3d12c

#if defined(_WIN32)

    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <Windows.h>
    #include <d3d12.h>
    #include <dxgi1_4.h>
    #include <wrl/client.h>

    #include <string>

namespace xllama {
namespace d3d12c {

using Microsoft::WRL::ComPtr;

// "<what> hr=0x%08lx" — the format the probe CSVs already record.
std::string hr_message(const char* what, HRESULT hr);

// First hardware adapter that creates an FL 11_0 device, else the default
// adapter. On failure returns null and sets *err.
ComPtr<ID3D12Device> create_device(std::string* err);

// Committed buffer on a standard heap type. On failure returns null and sets
// *err to "CreateCommittedResource <name> hr=…".
ComPtr<ID3D12Resource> create_buffer(ID3D12Device* device, UINT64 bytes, D3D12_HEAP_TYPE heap,
                                     D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state,
                                     const char* name, std::string* err);

// Same, on caller-provided heap properties (CUSTOM heaps on UMA adapters).
ComPtr<ID3D12Resource> create_buffer_props(ID3D12Device* device, UINT64 bytes,
                                           const D3D12_HEAP_PROPERTIES& props,
                                           D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state,
                                           const char* name, std::string* err);

// Root signature of the GEMV shaders: one table with b0 CBV, t0 + t1 SRV, u0 UAV.
ComPtr<ID3D12RootSignature> create_gemv_root_sig(ID3D12Device* device, std::string* err);

struct FenceEvent {
    HANDLE h = nullptr;
    FenceEvent() = default;
    FenceEvent(const FenceEvent&) = delete;
    FenceEvent& operator=(const FenceEvent&) = delete;
    ~FenceEvent() {
        if (h)
            CloseHandle(h);
    }
};

// Signal + wait on a queue. spin=true polls GetCompletedValue instead of
// sleeping on the event (lower wake-up latency, burns one core).
class QueueFence {
  public:
    bool init(ID3D12Device* device, std::string* err);
    bool signal_and_wait(ID3D12CommandQueue* queue, bool spin);

  private:
    friend struct ProbeCommands;
    FenceResult retire_probe(ID3D12CommandQueue* queue, ProbeMarker& marker, bool wait);
    ComPtr<ID3D12Fence> fence_;
    FenceEvent event_;
    UINT64 value_ = 0;
};

// Probe-specific buffers and query heaps live on the derived campaign owner.
// Keep CPU reference arrays local to the measurement: the GPU never sees them.
struct ProbeCommands {
    virtual ~ProbeCommands() = default;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12CommandQueue> queue;
    ComPtr<ID3D12CommandAllocator> alloc;
    ComPtr<ID3D12GraphicsCommandList> cl;
    ComPtr<ID3D12RootSignature> root;
    ComPtr<ID3D12PipelineState> pso;
    ComPtr<ID3D12DescriptorHeap> heap;
    bool init_fence(std::string& error) {
        return fence_.init(device.Get(), &error);
    }
    bool reset_list(ID3D12PipelineState* pipeline, std::string& error);
    bool close_list(std::string& error);
    bool execute(std::string& error);
    bool pending() const {
        return marker_.pending();
    }
    bool try_retire();

  private:
    ProbeMarker marker_;
    QueueFence fence_;
};

ProbeSlot<ProbeCommands>& probe_slot();

} // namespace d3d12c
} // namespace xllama

#endif // _WIN32
