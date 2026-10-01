// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "d3d12_compute.h"

#if defined(_WIN32)

    #include <cstdio>

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

bool QueueFence::signal_and_wait(ID3D12CommandQueue* queue, bool spin) {
    const UINT64 v = ++value_;
    if (FAILED(queue->Signal(fence_.Get(), v)))
        return false;
    if (spin) {
        while (fence_->GetCompletedValue() < v)
            YieldProcessor();
        return true;
    }
    if (fence_->GetCompletedValue() < v) {
        fence_->SetEventOnCompletion(v, event_.h);
        WaitForSingleObject(event_.h, INFINITE);
    }
    return true;
}

} // namespace d3d12c
} // namespace xllama

#endif // _WIN32
