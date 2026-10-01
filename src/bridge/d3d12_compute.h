// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Shared D3D12 compute plumbing for the GPU probes (gpugemv, gpustep) and the
// future GGUF GPU backend (docs/gguf-gpu-decode.md). Windows/UWP only; the
// system D3D12 runtime via d3d12_dyn, never the Agility device factory
// (docs/uwp-constraints.md §7).
#pragma once

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
    ComPtr<ID3D12Fence> fence_;
    FenceEvent event_;
    UINT64 value_ = 0;
};

} // namespace d3d12c
} // namespace xllama

#endif // _WIN32
