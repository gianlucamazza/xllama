// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "d3d12_compute.h"
#include "ggml_d3d12_add_bias_f32_dxil.h"

#include <d3d12sdklayers.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;

namespace {

[[noreturn]] void fail(const char* error) {
    std::fprintf(stderr, "%s\n", error);
    std::fflush(stderr);
    std::_Exit(EXIT_FAILURE);
}

void check(HRESULT hr, const char* operation) {
    if (FAILED(hr))
        fail(xllama::d3d12c::hr_message(operation, hr).c_str());
}

ComPtr<ID3D12RootSignature> bias_root(ID3D12Device* device) {
    // Public ABI used by the GGML backend: b0 constants, t0 bias, u0 dst, u1 src.
    D3D12_ROOT_PARAMETER params[4]{};
    params[0].ParameterType = D3D12_ROOT_PARAMETER_TYPE_32BIT_CONSTANTS;
    params[0].Constants.Num32BitValues = 8;
    params[1].ParameterType = D3D12_ROOT_PARAMETER_TYPE_SRV;
    params[2].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[3].ParameterType = D3D12_ROOT_PARAMETER_TYPE_UAV;
    params[3].Descriptor.ShaderRegister = 1;
    D3D12_ROOT_SIGNATURE_DESC desc{};
    desc.NumParameters = 4;
    desc.pParameters = params;
    ComPtr<ID3DBlob> blob;
    check(D3D12SerializeRootSignature(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &blob, nullptr),
          "SerializeRootSignature");
    ComPtr<ID3D12RootSignature> root;
    check(device->CreateRootSignature(0, blob->GetBufferPointer(), blob->GetBufferSize(),
                                      IID_PPV_ARGS(&root)),
          "CreateRootSignature");
    return root;
}

D3D12_RESOURCE_BARRIER transition(ID3D12Resource* res, D3D12_RESOURCE_STATES before,
                                  D3D12_RESOURCE_STATES after) {
    D3D12_RESOURCE_BARRIER b{};
    b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
    b.Transition = {res, D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES, before, after};
    return b;
}

void run(ID3D12Device* device, ID3D12InfoQueue* debug) {
    constexpr UINT width = 129, columns = 2, pitch = 133, x_offset = 1, bias_offset = 2;
    std::array<float, x_offset + columns * pitch> x;
    std::array<float, bias_offset + width> bias;
    x.fill(-333.25f);
    bias.fill(777.5f);
    for (UINT i = 0; i < width; ++i) {
        bias[bias_offset + i] = (static_cast<int>(i) - 64) * 0.25f;
        for (UINT c = 0; c < columns; ++c)
            x[x_offset + c * pitch + i] = (static_cast<int>(i + c * width) - 129) * 0.125f;
    }
    // Freeze exact IEEE-754 edge cases as bits independently of the host's
    // denormal mode: -0 + -0, two halfway ties and the smallest subnormal twice.
    constexpr std::array<std::uint32_t, 4> edge_x{0x80000000, 0x3f800000, 0x3f800001, 1};
    constexpr std::array<std::uint32_t, 4> edge_b{0x80000000, 0x33800000, 0x33800000, 1};
    constexpr std::array<std::uint32_t, 4> edge_y{0x80000000, 0x3f800000, 0x3f800002, 2};
    auto expected = x;
    for (UINT c = 0; c < columns; ++c) {
        for (UINT i = 0; i < width; ++i)
            expected[x_offset + c * pitch + i] += bias[bias_offset + i];
        std::memcpy(x.data() + x_offset + c * pitch, edge_x.data(), sizeof(edge_x));
        std::memcpy(expected.data() + x_offset + c * pitch, edge_y.data(), sizeof(edge_y));
    }
    std::memcpy(bias.data() + bias_offset, edge_b.data(), sizeof(edge_b));

    std::string error;
    auto buffer = [&](UINT64 bytes, D3D12_HEAP_TYPE heap, D3D12_RESOURCE_FLAGS flags,
                      D3D12_RESOURCE_STATES state, const char* name) {
        auto r = xllama::d3d12c::create_buffer(device, bytes, heap, flags, state, name, &error);
        if (!r)
            fail(error.c_str());
        return r;
    };
    auto upload = buffer(sizeof(x) + sizeof(bias), D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                         D3D12_RESOURCE_STATE_GENERIC_READ, "upload");
    auto weights = buffer(sizeof(bias), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE,
                          D3D12_RESOURCE_STATE_COPY_DEST, "bias");
    auto values =
        buffer(sizeof(x), D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
               D3D12_RESOURCE_STATE_COPY_DEST, "in-place activation");
    auto readback = buffer(sizeof(x), D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
                           D3D12_RESOURCE_STATE_COPY_DEST, "readback");
    void* p = nullptr;
    D3D12_RANGE empty{0, 0};
    check(upload->Map(0, &empty, &p), "Map upload");
    std::memcpy(p, x.data(), sizeof(x));
    std::memcpy(static_cast<unsigned char*>(p) + sizeof(x), bias.data(), sizeof(bias));
    upload->Unmap(0, nullptr);

    auto root = bias_root(device);
    D3D12_COMPUTE_PIPELINE_STATE_DESC pd{};
    pd.pRootSignature = root.Get();
    pd.CS = {kGgmlD3d12AddBiasF32Dxil, kGgmlD3d12AddBiasF32DxilSize};
    ComPtr<ID3D12PipelineState> pipeline;
    check(device->CreateComputePipelineState(&pd, IID_PPV_ARGS(&pipeline)), "Create bias PSO");
    D3D12_COMMAND_QUEUE_DESC qd{};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ComPtr<ID3D12CommandQueue> queue;
    check(device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue)), "CreateCommandQueue");
    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    check(device->CreateCommandAllocator(qd.Type, IID_PPV_ARGS(&allocator)),
          "CreateCommandAllocator");
    check(
        device->CreateCommandList(0, qd.Type, allocator.Get(), pipeline.Get(), IID_PPV_ARGS(&list)),
        "CreateCommandList");
    xllama::d3d12c::QueueFence fence;
    if (!fence.init(device, &error))
        fail(error.c_str());
    list->CopyBufferRegion(values.Get(), 0, upload.Get(), 0, sizeof(x));
    list->CopyBufferRegion(weights.Get(), 0, upload.Get(), sizeof(x), sizeof(bias));
    D3D12_RESOURCE_BARRIER ready[] = {transition(values.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                                 D3D12_RESOURCE_STATE_UNORDERED_ACCESS),
                                      transition(weights.Get(), D3D12_RESOURCE_STATE_COPY_DEST,
                                                 D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE)};
    list->ResourceBarrier(2, ready);
    list->SetComputeRootSignature(root.Get());
    const UINT params[8] = {width, pitch, pitch};
    list->SetComputeRoot32BitConstants(0, 8, params, 0);
    list->SetComputeRootShaderResourceView(1, weights->GetGPUVirtualAddress() +
                                                  bias_offset * sizeof(float));
    const auto address = values->GetGPUVirtualAddress() + x_offset * sizeof(float);
    list->SetComputeRootUnorderedAccessView(2, address);
    list->SetComputeRootUnorderedAccessView(3, address);
    list->Dispatch((width + 255) / 256, columns, 1);
    auto copied = transition(values.Get(), D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
                             D3D12_RESOURCE_STATE_COPY_SOURCE);
    list->ResourceBarrier(1, &copied);
    list->CopyBufferRegion(readback.Get(), 0, values.Get(), 0, sizeof(x));
    check(list->Close(), "Close command list");
    ID3D12CommandList* batch[] = {list.Get()};
    queue->ExecuteCommandLists(1, batch);
    if (!fence.signal_and_wait(queue.Get(), false))
        fail("Failed to synchronize bias dispatch");
    check(device->GetDeviceRemovedReason(), "Device removed");
    const auto count = debug->GetNumStoredMessagesAllowedByRetrievalFilter();
    for (UINT64 i = 0; i < count; ++i) {
        SIZE_T size = 0;
        check(debug->GetMessage(i, nullptr, &size), "GetMessage size");
        std::vector<unsigned char> bytes(size);
        auto* msg = reinterpret_cast<D3D12_MESSAGE*>(bytes.data());
        check(debug->GetMessage(i, msg, &size), "GetMessage");
        if (msg->Severity <= D3D12_MESSAGE_SEVERITY_ERROR)
            fail(msg->pDescription);
    }
    D3D12_RANGE range{0, sizeof(x)};
    check(readback->Map(0, &range, &p), "Map readback");
    for (size_t i = 0; i < expected.size(); ++i) {
        std::uint32_t actual, reference;
        std::memcpy(&actual, static_cast<const unsigned char*>(p) + i * sizeof(actual),
                    sizeof(actual));
        std::memcpy(&reference, &expected[i], sizeof(reference));
        if (actual != reference) {
            std::fprintf(stderr, "word %zu: expected 0x%08x, read 0x%08x\n", i, reference, actual);
            fail("Bit parity failed");
        }
    }
    readback->Unmap(0, &empty);
    std::printf("PASS: %u F32 sums and %zu guard words match, debug layer on\n", width * columns,
                expected.size() - width * columns);
}

} // namespace

int main() {
    ComPtr<ID3D12Debug> controller;
    check(D3D12GetDebugInterface(IID_PPV_ARGS(&controller)), "D3D12 debug layer (Graphics Tools)");
    controller->EnableDebugLayer();
    ComPtr<IDXGIFactory4> factory;
    ComPtr<IDXGIAdapter> adapter;
    ComPtr<ID3D12Device> device;
    ComPtr<ID3D12InfoQueue> debug;
    check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "CreateDXGIFactory1");
    check(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)), "EnumWarpAdapter");
    check(D3D12CreateDevice(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)),
          "D3D12CreateDevice");
    check(device.As(&debug), "QueryInterface(ID3D12InfoQueue)");
    run(device.Get(), debug.Get());
}
