// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "d3d12_compute.h"

#if defined(_WIN32)

    #include <algorithm>
    #include <chrono>
    #include <cstdio>
    #include <thread>

    #include "xllama/d3d12_dyn.h"

namespace xllama {
namespace d3d12c {

std::string hr_message(const char* what, HRESULT hr) {
    char b[160];
    std::snprintf(b, sizeof(b), "%s hr=0x%08lx", what, static_cast<unsigned long>(hr));
    return b;
}

ComPtr<ID3D12Device> create_device(std::string* err) {
    ComPtr<IDXGIFactory4> factory;
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    if (FAILED(hr)) {
        if (err)
            *err = hr_message("CreateDXGIFactory1", hr);
        return {};
    }
    auto create = d3d12_dyn::CreateDevice();
    if (!create) {
        if (err)
            *err = "D3D12CreateDevice not available";
        return {};
    }
    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<ID3D12Device> device;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 ad = {};
        adapter->GetDesc1(&ad);
        if (ad.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
            continue;
        hr = create(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device));
        if (SUCCEEDED(hr))
            return device;
        device.Reset();
        adapter.Reset();
    }
    hr = create(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device));
    if (FAILED(hr)) {
        if (err)
            *err = hr_message("D3D12CreateDevice", hr);
        return {};
    }
    return device;
}

ComPtr<ID3D12Resource> create_buffer_props(ID3D12Device* device, UINT64 bytes,
                                           const D3D12_HEAP_PROPERTIES& props,
                                           D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state,
                                           const char* name, std::string* err) {
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = bytes;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags = flags;
    ComPtr<ID3D12Resource> res;
    HRESULT hr = device->CreateCommittedResource(&props, D3D12_HEAP_FLAG_NONE, &rd, state, nullptr,
                                                 IID_PPV_ARGS(&res));
    if (FAILED(hr)) {
        if (err) {
            char b[128];
            std::snprintf(b, sizeof(b), "CreateCommittedResource %s", name ? name : "?");
            *err = hr_message(b, hr);
        }
        return {};
    }
    return res;
}

ComPtr<ID3D12Resource> create_buffer(ID3D12Device* device, UINT64 bytes, D3D12_HEAP_TYPE heap,
                                     D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state,
                                     const char* name, std::string* err) {
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = heap;
    return create_buffer_props(device, bytes, hp, flags, state, name, err);
}

ComPtr<ID3D12RootSignature> create_gemv_root_sig(ID3D12Device* device, std::string* err) {
    D3D12_DESCRIPTOR_RANGE ranges[4] = {};
    const D3D12_DESCRIPTOR_RANGE_TYPE types[4] = {
        D3D12_DESCRIPTOR_RANGE_TYPE_CBV, D3D12_DESCRIPTOR_RANGE_TYPE_SRV,
        D3D12_DESCRIPTOR_RANGE_TYPE_SRV, D3D12_DESCRIPTOR_RANGE_TYPE_UAV};
    const UINT regs[4] = {0, 0, 1, 0};
    for (UINT i = 0; i < 4; ++i) {
        ranges[i].RangeType = types[i];
        ranges[i].NumDescriptors = 1;
        ranges[i].BaseShaderRegister = regs[i];
        ranges[i].OffsetInDescriptorsFromTableStart = i;
    }

    D3D12_ROOT_PARAMETER param = {};
    param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    param.DescriptorTable.NumDescriptorRanges = 4;
    param.DescriptorTable.pDescriptorRanges = ranges;
    param.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC desc = {};
    desc.NumParameters = 1;
    desc.pParameters = &param;
    desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    auto serialize = d3d12_dyn::SerializeRootSignature();
    if (!serialize) {
        if (err)
            *err = "D3D12SerializeRootSignature not available";
        return {};
    }
    ComPtr<ID3DBlob> sig;
    ComPtr<ID3DBlob> blob_err;
    HRESULT hr = serialize(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &blob_err);
    if (FAILED(hr)) {
        if (err)
            *err = "D3D12SerializeRootSignature failed";
        return {};
    }
    ComPtr<ID3D12RootSignature> root;
    hr = device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(),
                                     IID_PPV_ARGS(&root));
    if (FAILED(hr)) {
        if (err)
            *err = hr_message("CreateRootSignature", hr);
        return {};
    }
    return root;
}

bool QueueFence::init(ID3D12Device* device, std::string* err) {
    HRESULT hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence_));
    if (FAILED(hr)) {
        if (err)
            *err = hr_message("CreateFence", hr);
        return false;
    }
    event_.h = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!event_.h) {
        if (err)
            *err = "CreateEventW failed";
        return false;
    }
    return true;
}

namespace {

template <typename Expired, typename WaitMs>
FenceResult wait_native(ID3D12Fence* fence, HANDLE event, UINT64 target, bool spin, Expired expired,
                        WaitMs wait_ms) {
    return wait_accepted_fence_until(
        target, spin, [&] { return fence->GetCompletedValue(); },
        [&] { return SUCCEEDED(fence->SetEventOnCompletion(target, event)); },
        [&] {
            const DWORD status = WaitForSingleObject(event, wait_ms());
            return status == WAIT_OBJECT_0 || status == WAIT_TIMEOUT;
        },
        [&] {
            if (spin)
                YieldProcessor();
            else
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
        },
        expired);
}

} // namespace

bool QueueFence::signal_and_wait(ID3D12CommandQueue* queue, bool spin) {
    const UINT64 v = ++value_;
    if (FAILED(queue->Signal(fence_.Get(), v)))
        return false;
    return wait_native(
               fence_.Get(), event_.h, v, spin, [] { return false; }, [] { return 100u; }) ==
           FenceResult::complete;
}

FenceResult QueueFence::retire_probe(ID3D12CommandQueue* queue, ProbeMarker& marker, bool wait) {
    using namespace std::chrono;
    return marker.retire(
        [&](UINT64 target) { return SUCCEEDED(queue->Signal(fence_.Get(), target)); },
        [&] { return fence_->GetCompletedValue(); },
        [&](UINT64 target) {
            if (!wait)
                return fence_result(target, fence_->GetCompletedValue());
            const auto deadline = steady_clock::now() + seconds(5);
            return wait_native(
                fence_.Get(), event_.h, target, false,
                [&] { return steady_clock::now() >= deadline; },
                [&] {
                    const auto left = ceil<milliseconds>(deadline - steady_clock::now()).count();
                    return static_cast<DWORD>(std::clamp<long long>(left, 0, 100));
                });
        });
}

bool ProbeCommands::reset_list(ID3D12PipelineState* pipeline, std::string& error) {
    if (pending()) {
        error = "probe allocator is still in use";
        return false;
    }
    HRESULT hr = alloc->Reset();
    if (FAILED(hr)) {
        error = hr_message("Reset allocator", hr);
        return false;
    }
    hr = cl->Reset(alloc.Get(), pipeline);
    if (FAILED(hr))
        error = hr_message("Reset command list", hr);
    return SUCCEEDED(hr);
}

bool ProbeCommands::close_list(std::string& error) {
    const HRESULT hr = cl->Close();
    if (FAILED(hr))
        error = hr_message("Close command list", hr);
    return SUCCEEDED(hr);
}

bool ProbeCommands::execute(std::string& error) {
    if (marker_.submit([&] {
            ID3D12CommandList* lists[] = {cl.Get()};
            queue->ExecuteCommandLists(1, lists);
        })) {
        switch (fence_.retire_probe(queue.Get(), marker_, true)) {
        case FenceResult::complete:
            return true;
        case FenceResult::removed:
            error = "D3D12 device removed";
            return false;
        case FenceResult::signal_rejected:
            error = "queue rejected probe fence signal";
            return false;
        case FenceResult::pending:
            break;
        }
    }
    error = "probe fence completion pending";
    return false;
}

bool ProbeCommands::try_retire() {
    if (pending())
        fence_.retire_probe(queue.Get(), marker_, false);
    return !pending();
}

ProbeSlot<ProbeCommands>& probe_slot() {
    // Static destruction is not a GPU completion boundary. A live queue
    // whose marker still cannot be established stays owned until process exit.
    static auto* const slot = new ProbeSlot<ProbeCommands>;
    return *slot;
}

} // namespace d3d12c
} // namespace xllama

#endif // _WIN32
