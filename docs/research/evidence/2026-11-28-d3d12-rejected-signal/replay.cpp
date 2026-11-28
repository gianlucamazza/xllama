// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// Linux stand-ins for the four production functions selected by replay.py.
#include "d3d12_compute.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <mutex>
#include <thread>

#ifdef NDEBUG
    #error "Build the evidence replay with assertions enabled."
#endif

namespace {
using xllama::d3d12c::wait_accepted_fence;
using UINT64 = std::uint64_t;
using DWORD = unsigned;
constexpr DWORD WAIT_OBJECT_0 = 0, WAIT_TIMEOUT = 258;
bool SUCCEEDED(int result) {
    return result >= 0;
}
bool FAILED(int result) {
    return result < 0;
}
void YieldProcessor() {}
DWORD WaitForSingleObject(void*, int) {
    return WAIT_TIMEOUT;
}

template <class T> struct Handle {
    T* ptr;
    T* Get() {
        return ptr;
    }
    T* operator->() {
        return ptr;
    }
};
struct Resource {
    std::array<unsigned char, 8> bytes{};
};
struct Fence {
    UINT64 completed = 0;
    UINT64 GetCompletedValue() const {
        return completed;
    }
    int SetEventOnCompletion(UINT64, void*) {
        return 0;
    }
};
struct Allocator {
    const unsigned& outstanding;
    unsigned unsafe_resets = 0;
    int Reset() {
        if (!outstanding)
            return 0;
        ++unsafe_resets;
        return -1;
    }
};
struct ID3D12GraphicsCommandList {
    int Reset(Allocator*, void*) {
        return 0;
    }
    int Close() {
        return 0;
    }
    void CopyBufferRegion(Resource*, size_t, Resource*, size_t, size_t) {}
};
using ID3D12CommandList = ID3D12GraphicsCommandList;

struct ID3D12CommandQueue {
    unsigned outstanding = 0, executes = 0, signals = 0;
    bool reject = true, removed = false;
    void ExecuteCommandLists(unsigned count, ID3D12CommandList**) {
        ++executes;
        outstanding += count;
    }
    int Signal(Fence* fence, UINT64 value) {
        ++signals;
        if (reject)
            return -1; // A live queue still has work from ExecuteCommandLists.
        fence->completed = removed ? UINT64_MAX : value;
        outstanding = 0;
        return 0;
    }
};
struct QueueFence {
    Handle<Fence> fence_;
    struct {
        void* h = nullptr;
    } event_{};
    UINT64 value_ = 0;
    bool signal_and_wait(ID3D12CommandQueue*, bool);
};
struct D3d12PhaseStats {
    double record_ms = 0, exec_wait_ms = 0;
    unsigned submissions = 0;
};
struct Gpu {
    ID3D12CommandQueue queue_state;
    Fence fence_state;
    Allocator allocator{queue_state.outstanding};
    ID3D12CommandList list;
    Handle<ID3D12CommandQueue> queue{&queue_state};
    Handle<Allocator> alloc{&allocator};
    Handle<ID3D12GraphicsCommandList> cl{&list};
    QueueFence fence{{&fence_state}};
    Resource readback_store, staging_store;
    Handle<Resource> readback{&readback_store}, staging{&staging_store};
    unsigned char* readback_ptr = readback_store.bytes.data();
    unsigned char* staging_ptr = staging_store.bytes.data();
    std::mutex mu;
};
Gpu* current;
Gpu& gpu() {
    return *current;
}
struct BufCtx {
    Handle<Resource> res;
};
struct Buffer {
    void* context;
};
using ggml_backend_buffer_t = Buffer*;
struct ggml_tensor {};
size_t tensor_offset(const ggml_tensor*) {
    return 0;
}
constexpr UINT64 kStagingBytes = 4;

// Verbatim from the pinned revision, emitted beside the executable.
#include "production.inc"

void report(const char* test, const Gpu& g) {
    std::cout << test << " execute=" << g.queue_state.executes
              << " signals=" << g.queue_state.signals
              << " outstanding=" << g.queue_state.outstanding
              << " resets_while_outstanding=" << g.allocator.unsafe_resets;
}
} // namespace

int main() {
    Resource resource;
    BufCtx context{{&resource}};
    Buffer buffer{&context};
    ggml_tensor tensor;

    Gpu read;
    current = &read;
    read.readback_store.bytes.fill(0xdd);
    std::array<unsigned char, 8> destination{};
    weights_get(&buffer, &tensor, destination.data(), 0, destination.size());
    const auto stale = std::count(destination.begin(), destination.end(), 0xdd);
    report("weights_get/rejected", read);
    std::cout << " stale_bytes_delivered=" << stale << '\n';
    assert(stale == 8 && read.queue_state.outstanding == 2 && read.allocator.unsafe_resets == 1);

    Gpu write;
    current = &write;
    const unsigned char payload[] = {1, 1, 1, 1, 2, 2, 2, 2};
    weights_write(&buffer, 0, payload, sizeof(payload), 0);
    report("weights_write/rejected", write);
    std::cout << " staging_first_byte=" << int(write.staging_ptr[0]) << '\n';
    assert(write.staging_ptr[0] == 2 && write.queue_state.outstanding == 2 &&
           write.allocator.unsafe_resets == 1);

    Gpu graph;
    current = &graph;
    const auto record = [](auto*) {};
    const bool first = run_now(graph, record);
    report("run_now/rejected", graph);
    std::cout << " returned_success=" << first << '\n';
    assert(!first && graph.queue_state.outstanding == 1);
    graph.queue_state.reject = false;
    const bool second = run_now(graph, record);
    report("run_now/next_accepted", graph);
    std::cout << " returned_success=" << second << '\n';
    assert(second && graph.queue_state.executes == 2 && graph.allocator.unsafe_resets == 1);

    Gpu removed;
    current = &removed;
    removed.queue_state.reject = false;
    removed.queue_state.removed = true;
    const bool lost = run_now(removed, record);
    report("run_now/accepted_removed", removed);
    std::cout << " returned_success=" << lost << '\n';
    assert(!lost);
}
