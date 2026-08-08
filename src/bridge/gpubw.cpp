// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "xllama/gpubw.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <ctime>
#include <string>
#include <vector>

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

    #include "gpubw_stream_dxil.h"
    #include "xllama/d3d12_dyn.h"

using Microsoft::WRL::ComPtr;
#endif

namespace xllama {

void gpubw_fill_pattern(std::uint32_t* data, std::size_t n_words) {
    if (!data)
        return;
    for (std::size_t i = 0; i < n_words; ++i)
        data[i] = gpubw_pattern_word(static_cast<std::uint32_t>(i));
}

std::uint32_t gpubw_checksum_words(const std::uint32_t* data, std::size_t n_words) {
    std::uint32_t x = 0;
    if (!data)
        return 0;
    for (std::size_t i = 0; i < n_words; ++i)
        x ^= data[i];
    return x;
}

GpubwDispatch gpubw_plan_dispatch(std::size_t n_words) {
    GpubwDispatch d;
    // Groups needed so each of kGpubwThreadsPerGroup threads covers one word.
    std::uint64_t need =
        (static_cast<std::uint64_t>(n_words) + kGpubwThreadsPerGroup - 1) / kGpubwThreadsPerGroup;
    if (need == 0)
        need = 1;
    d.n_groups = static_cast<std::uint32_t>(need > 0xffffffffull ? 0xffffffffull : need);

    if (need <= kGpubwMaxGroupsPerDim) {
        d.groups_x = static_cast<std::uint32_t>(need);
        d.groups_y = 1;
        d.groups_z = 1;
        return d;
    }

    // 2D: cap X at max, Y = ceil(need / X). Prefer a balanced split when possible.
    // 1 GiB → 1048576 groups → (1024, 1024, 1) is clean and ≤ 65535.
    std::uint64_t gx = kGpubwMaxGroupsPerDim;
    // Prefer sqrt-ish for large needs so flat indexing stays balanced.
    if (need <= static_cast<std::uint64_t>(kGpubwMaxGroupsPerDim) * kGpubwMaxGroupsPerDim) {
        // Largest power-of-two ≤ max that divides the grid reasonably: use
        // ceil(need / max) for Y after setting X = min(need, max).
        // For powers of two (1 GiB path), use equal square factors when exact.
        std::uint64_t root = 1;
        while (root * root < need && root < kGpubwMaxGroupsPerDim)
            root <<= 1;
        if (root > kGpubwMaxGroupsPerDim)
            root = kGpubwMaxGroupsPerDim;
        if (root * root >= need && root <= kGpubwMaxGroupsPerDim) {
            gx = root;
        }
        std::uint64_t gy = (need + gx - 1) / gx;
        if (gy <= kGpubwMaxGroupsPerDim) {
            d.groups_x = static_cast<std::uint32_t>(gx);
            d.groups_y = static_cast<std::uint32_t>(gy);
            d.groups_z = 1;
            d.n_groups = d.groups_x * d.groups_y * d.groups_z;
            return d;
        }
    }

    // 3D fallback (covers absurdly large buffers).
    gx = kGpubwMaxGroupsPerDim;
    std::uint64_t gy = kGpubwMaxGroupsPerDim;
    std::uint64_t plane = gx * gy;
    std::uint64_t gz = (need + plane - 1) / plane;
    if (gz > kGpubwMaxGroupsPerDim)
        gz = kGpubwMaxGroupsPerDim; // hard clamp; caller must not request more
    d.groups_x = static_cast<std::uint32_t>(gx);
    d.groups_y = static_cast<std::uint32_t>(gy);
    d.groups_z = static_cast<std::uint32_t>(gz);
    d.n_groups = d.groups_x * d.groups_y * d.groups_z;
    return d;
}

const char* gpubw_csv_header() {
    return "buffer_mb,iterations,read_gbs,checksum_ok,d3d12_ran,checksum,expected,host,date,"
           "error\n";
}

std::string format_gpubw_row(const GpubwResult& r, const char* host_label) {
    char date_buf[32];
    std::time_t now = std::time(nullptr);
    std::tm* tm_utc = std::gmtime(&now);
    std::strftime(date_buf, sizeof(date_buf), "%Y-%m-%dT%H:%M:%SZ", tm_utc);

    // Sanitize error for single-field CSV (no commas/newlines).
    std::string err = r.error_msg;
    for (char& c : err) {
        if (c == ',' || c == '\n' || c == '\r')
            c = ' ';
    }
    if (err.empty())
        err = "-";

    char buf[512];
    std::snprintf(buf, sizeof(buf), "%zu,%d,%.2f,%d,%d,%u,%u,%s,%s,%s\n",
                  r.buffer_bytes / (1024 * 1024), r.iterations, r.read_gbs, r.checksum_ok ? 1 : 0,
                  r.d3d12_ran ? 1 : 0, r.checksum, r.expected_checksum,
                  host_label ? host_label : "unknown", date_buf, err.c_str());
    return std::string(buf);
}

#if !defined(_WIN32)

GpubwResult measure_gpubw(std::size_t buffer_bytes, int iterations) {
    GpubwResult r;
    const std::size_t n_words = std::max<std::size_t>(buffer_bytes / sizeof(std::uint32_t), 1);
    r.buffer_bytes = n_words * sizeof(std::uint32_t);
    r.iterations = iterations < 1 ? 1 : iterations;
    r.d3d12_ran = false;
    r.error_msg = "d3d12 unavailable on this platform";
    // Still publish expected checksum shape for host unit tests of the pattern.
    std::vector<std::uint32_t> tmp(std::min(n_words, std::size_t{1024}));
    gpubw_fill_pattern(tmp.data(), tmp.size());
    r.expected_checksum = gpubw_checksum_words(tmp.data(), tmp.size());
    return r;
}

#else

namespace {

void throw_if_failed(HRESULT hr, const char* what, GpubwResult& r) {
    if (SUCCEEDED(hr))
        return;
    char b[160];
    std::snprintf(b, sizeof(b), "%s hr=0x%08lx", what, static_cast<unsigned long>(hr));
    r.error_msg = b;
}

// Minimal root signature: b0 CBV, t0 SRV, u0 UAV (all in one table root param 0).
// Built from a serialized blob created at runtime via D3D12SerializeRootSignature
// when available — here we use the version-1.0 CD3DX12-style manual desc.

ComPtr<ID3D12RootSignature> create_root_sig(ID3D12Device* device, GpubwResult& r) {
    D3D12_DESCRIPTOR_RANGE ranges[3] = {};
    ranges[0].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_CBV;
    ranges[0].NumDescriptors = 1;
    ranges[0].BaseShaderRegister = 0;
    ranges[0].RegisterSpace = 0;
    ranges[0].OffsetInDescriptorsFromTableStart = 0;

    ranges[1].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_SRV;
    ranges[1].NumDescriptors = 1;
    ranges[1].BaseShaderRegister = 0;
    ranges[1].RegisterSpace = 0;
    ranges[1].OffsetInDescriptorsFromTableStart = 1;

    ranges[2].RangeType = D3D12_DESCRIPTOR_RANGE_TYPE_UAV;
    ranges[2].NumDescriptors = 1;
    ranges[2].BaseShaderRegister = 0;
    ranges[2].RegisterSpace = 0;
    ranges[2].OffsetInDescriptorsFromTableStart = 2;

    D3D12_ROOT_PARAMETER param = {};
    param.ParameterType = D3D12_ROOT_PARAMETER_TYPE_DESCRIPTOR_TABLE;
    param.DescriptorTable.NumDescriptorRanges = 3;
    param.DescriptorTable.pDescriptorRanges = ranges;
    param.ShaderVisibility = D3D12_SHADER_VISIBILITY_ALL;

    D3D12_ROOT_SIGNATURE_DESC desc = {};
    desc.NumParameters = 1;
    desc.pParameters = &param;
    desc.Flags = D3D12_ROOT_SIGNATURE_FLAG_NONE;

    ComPtr<ID3DBlob> sig;
    ComPtr<ID3DBlob> err;
    auto serialize = d3d12_dyn::SerializeRootSignature();
    if (!serialize) {
        r.error_msg = "D3D12SerializeRootSignature not available";
        return nullptr;
    }
    HRESULT hr = serialize(&desc, D3D_ROOT_SIGNATURE_VERSION_1, &sig, &err);
    if (FAILED(hr)) {
        r.error_msg = "D3D12SerializeRootSignature failed";
        if (err)
            r.error_msg += std::string(": ") + static_cast<const char*>(err->GetBufferPointer());
        return nullptr;
    }
    ComPtr<ID3D12RootSignature> rs;
    hr = device->CreateRootSignature(0, sig->GetBufferPointer(), sig->GetBufferSize(),
                                     IID_PPV_ARGS(&rs));
    if (FAILED(hr)) {
        throw_if_failed(hr, "CreateRootSignature", r);
        return nullptr;
    }
    return rs;
}

ComPtr<ID3D12Resource> create_buffer(ID3D12Device* device, UINT64 size, D3D12_HEAP_TYPE heap,
                                     D3D12_RESOURCE_FLAGS flags, D3D12_RESOURCE_STATES state,
                                     GpubwResult& r, const char* label) {
    D3D12_HEAP_PROPERTIES hp = {};
    hp.Type = heap;
    D3D12_RESOURCE_DESC rd = {};
    rd.Dimension = D3D12_RESOURCE_DIMENSION_BUFFER;
    rd.Width = size;
    rd.Height = 1;
    rd.DepthOrArraySize = 1;
    rd.MipLevels = 1;
    rd.SampleDesc.Count = 1;
    rd.Layout = D3D12_TEXTURE_LAYOUT_ROW_MAJOR;
    rd.Flags = flags;
    ComPtr<ID3D12Resource> res;
    HRESULT hr = device->CreateCommittedResource(&hp, D3D12_HEAP_FLAG_NONE, &rd, state, nullptr,
                                                 IID_PPV_ARGS(&res));
    if (FAILED(hr)) {
        throw_if_failed(hr, label, r);
        return nullptr;
    }
    return res;
}

} // namespace

GpubwResult measure_gpubw(std::size_t buffer_bytes, int iterations) {
    GpubwResult r;
    if (iterations < 1)
        iterations = 1;
    const std::size_t n_words = std::max<std::size_t>(buffer_bytes / sizeof(std::uint32_t), 1);
    r.buffer_bytes = n_words * sizeof(std::uint32_t);
    r.iterations = iterations;

    // Cap partial buffer for expected checksum: full buffer xor on CPU of 1 GiB
    // is fine (few hundred ms once).
    std::vector<std::uint32_t> host_in(n_words);
    gpubw_fill_pattern(host_in.data(), n_words);
    r.expected_checksum = gpubw_checksum_words(host_in.data(), n_words);

    // --- Device: system D3D12 only (no Agility factory / D3D12SDKVersion). ---
    ComPtr<IDXGIFactory4> factory;
    HRESULT hr = CreateDXGIFactory1(IID_PPV_ARGS(&factory));
    if (FAILED(hr)) {
        throw_if_failed(hr, "CreateDXGIFactory1", r);
        return r;
    }

    auto create_device = d3d12_dyn::CreateDevice();
    if (!create_device) {
        r.error_msg = "D3D12CreateDevice not available";
        return r;
    }

    ComPtr<IDXGIAdapter1> adapter;
    ComPtr<ID3D12Device> device;
    for (UINT i = 0; factory->EnumAdapters1(i, &adapter) != DXGI_ERROR_NOT_FOUND; ++i) {
        DXGI_ADAPTER_DESC1 ad = {};
        adapter->GetDesc1(&ad);
        if (ad.Flags & DXGI_ADAPTER_FLAG_SOFTWARE)
            continue;
        hr = create_device(adapter.Get(), D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device));
        if (SUCCEEDED(hr))
            break;
        device.Reset();
        adapter.Reset();
    }
    if (!device) {
        // Fallback: default adapter
        hr = create_device(nullptr, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device));
        if (FAILED(hr)) {
            throw_if_failed(hr, "D3D12CreateDevice", r);
            return r;
        }
    }

    D3D12_COMMAND_QUEUE_DESC qd = {};
    qd.Type = D3D12_COMMAND_LIST_TYPE_DIRECT;
    ComPtr<ID3D12CommandQueue> queue;
    hr = device->CreateCommandQueue(&qd, IID_PPV_ARGS(&queue));
    if (FAILED(hr)) {
        throw_if_failed(hr, "CreateCommandQueue", r);
        return r;
    }

    ComPtr<ID3D12CommandAllocator> alloc;
    hr = device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_DIRECT, IID_PPV_ARGS(&alloc));
    if (FAILED(hr)) {
        throw_if_failed(hr, "CreateCommandAllocator", r);
        return r;
    }

    ComPtr<ID3D12GraphicsCommandList> cl;
    hr = device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_DIRECT, alloc.Get(), nullptr,
                                   IID_PPV_ARGS(&cl));
    if (FAILED(hr)) {
        throw_if_failed(hr, "CreateCommandList", r);
        return r;
    }

    ComPtr<ID3D12RootSignature> root = create_root_sig(device.Get(), r);
    if (!root)
        return r;

    D3D12_COMPUTE_PIPELINE_STATE_DESC pso_desc = {};
    pso_desc.pRootSignature = root.Get();
    pso_desc.CS.pShaderBytecode = kGpubwStreamDxil;
    pso_desc.CS.BytecodeLength = kGpubwStreamDxilSize;
    ComPtr<ID3D12PipelineState> pso;
    hr = device->CreateComputePipelineState(&pso_desc, IID_PPV_ARGS(&pso));
    if (FAILED(hr)) {
        throw_if_failed(hr, "CreateComputePipelineState", r);
        return r;
    }

    // Multi-dim Dispatch: 1 GiB → 1_048_576 groups > 65535 on a single axis.
    const GpubwDispatch plan = gpubw_plan_dispatch(n_words);
    if (!gpubw_dispatch_dims_legal(plan) || plan.n_groups < 1) {
        r.error_msg = "gpubw_plan_dispatch produced illegal Dispatch dims";
        return r;
    }
    const UINT groups = plan.n_groups; // total groups (may pad beyond need)
    const UINT64 input_bytes = static_cast<UINT64>(r.buffer_bytes);
    const UINT64 out_bytes = static_cast<UINT64>(groups) * sizeof(std::uint32_t);

    ComPtr<ID3D12Resource> input_default =
        create_buffer(device.Get(), input_bytes, D3D12_HEAP_TYPE_DEFAULT, D3D12_RESOURCE_FLAG_NONE,
                      D3D12_RESOURCE_STATE_COPY_DEST, r, "input_default");
    if (!input_default)
        return r;
    ComPtr<ID3D12Resource> input_upload =
        create_buffer(device.Get(), input_bytes, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                      D3D12_RESOURCE_STATE_GENERIC_READ, r, "input_upload");
    if (!input_upload)
        return r;
    ComPtr<ID3D12Resource> output_default =
        create_buffer(device.Get(), out_bytes, D3D12_HEAP_TYPE_DEFAULT,
                      D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
                      D3D12_RESOURCE_STATE_UNORDERED_ACCESS, r, "output_default");
    if (!output_default)
        return r;
    ComPtr<ID3D12Resource> output_readback =
        create_buffer(device.Get(), out_bytes, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
                      D3D12_RESOURCE_STATE_COPY_DEST, r, "output_readback");
    if (!output_readback)
        return r;
    ComPtr<ID3D12Resource> cb_upload =
        create_buffer(device.Get(), 256, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                      D3D12_RESOURCE_STATE_GENERIC_READ, r, "cb_upload");
    if (!cb_upload)
        return r;

    {
        void* mapped = nullptr;
        hr = input_upload->Map(0, nullptr, &mapped);
        if (FAILED(hr)) {
            throw_if_failed(hr, "Map input_upload", r);
            return r;
        }
        std::memcpy(mapped, host_in.data(), r.buffer_bytes);
        input_upload->Unmap(0, nullptr);
    }
    {
        // Must match shaders/gpubw_stream.hlsl cbuffer Params (b0).
        struct Params {
            std::uint32_t n_words;
            std::uint32_t dispatch_x;
            std::uint32_t dispatch_y;
            std::uint32_t threads_per_group;
        } params{};
        params.n_words = static_cast<std::uint32_t>(n_words);
        params.dispatch_x = plan.groups_x;
        params.dispatch_y = plan.groups_y;
        params.threads_per_group = kGpubwThreadsPerGroup;
        void* mapped = nullptr;
        hr = cb_upload->Map(0, nullptr, &mapped);
        if (FAILED(hr)) {
            throw_if_failed(hr, "Map cb", r);
            return r;
        }
        std::memcpy(mapped, &params, sizeof(params));
        cb_upload->Unmap(0, nullptr);
    }

    // Descriptor heap: CBV, SRV, UAV
    D3D12_DESCRIPTOR_HEAP_DESC heap_desc = {};
    heap_desc.NumDescriptors = 3;
    heap_desc.Type = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV;
    heap_desc.Flags = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE;
    ComPtr<ID3D12DescriptorHeap> heap;
    hr = device->CreateDescriptorHeap(&heap_desc, IID_PPV_ARGS(&heap));
    if (FAILED(hr)) {
        throw_if_failed(hr, "CreateDescriptorHeap", r);
        return r;
    }
    const UINT incr =
        device->GetDescriptorHandleIncrementSize(D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV);
    D3D12_CPU_DESCRIPTOR_HANDLE cpu = heap->GetCPUDescriptorHandleForHeapStart();

    D3D12_CONSTANT_BUFFER_VIEW_DESC cbv = {};
    cbv.BufferLocation = cb_upload->GetGPUVirtualAddress();
    cbv.SizeInBytes = 256;
    device->CreateConstantBufferView(&cbv, cpu);

    D3D12_CPU_DESCRIPTOR_HANDLE cpu_srv = cpu;
    cpu_srv.ptr += incr;
    D3D12_SHADER_RESOURCE_VIEW_DESC srv = {};
    srv.Format = DXGI_FORMAT_UNKNOWN;
    srv.ViewDimension = D3D12_SRV_DIMENSION_BUFFER;
    srv.Shader4ComponentMapping = D3D12_DEFAULT_SHADER_4_COMPONENT_MAPPING;
    srv.Buffer.FirstElement = 0;
    srv.Buffer.NumElements = static_cast<UINT>(n_words);
    srv.Buffer.StructureByteStride = sizeof(std::uint32_t);
    device->CreateShaderResourceView(input_default.Get(), &srv, cpu_srv);

    D3D12_CPU_DESCRIPTOR_HANDLE cpu_uav = cpu;
    cpu_uav.ptr += 2 * static_cast<SIZE_T>(incr);
    D3D12_UNORDERED_ACCESS_VIEW_DESC uav = {};
    uav.Format = DXGI_FORMAT_UNKNOWN;
    uav.ViewDimension = D3D12_UAV_DIMENSION_BUFFER;
    uav.Buffer.FirstElement = 0;
    uav.Buffer.NumElements = groups;
    uav.Buffer.StructureByteStride = sizeof(std::uint32_t);
    device->CreateUnorderedAccessView(output_default.Get(), nullptr, &uav, cpu_uav);

    ComPtr<ID3D12Fence> fence;
    hr = device->CreateFence(0, D3D12_FENCE_FLAG_NONE, IID_PPV_ARGS(&fence));
    if (FAILED(hr)) {
        throw_if_failed(hr, "CreateFence", r);
        return r;
    }
    HANDLE fence_event = CreateEventW(nullptr, FALSE, FALSE, nullptr);
    if (!fence_event) {
        r.error_msg = "CreateEventW failed";
        return r;
    }
    UINT64 fence_value = 0;

    auto wait_gpu = [&]() -> bool {
        const UINT64 v = ++fence_value;
        if (FAILED(queue->Signal(fence.Get(), v))) {
            r.error_msg = "Signal failed";
            return false;
        }
        if (fence->GetCompletedValue() < v) {
            fence->SetEventOnCompletion(v, fence_event);
            WaitForSingleObject(fence_event, INFINITE);
        }
        return true;
    };

    // Upload input once.
    {
        cl->CopyResource(input_default.Get(), input_upload.Get());
        D3D12_RESOURCE_BARRIER b = {};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource = input_default.Get();
        b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_DEST;
        b.Transition.StateAfter = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        // After CreateCommitted COMMON + CopyResource destination is COPY_DEST
        // on some paths; reset list carefully.
        cl->ResourceBarrier(1, &b);
        cl->Close();
        ID3D12CommandList* lists[] = {cl.Get()};
        queue->ExecuteCommandLists(1, lists);
        if (!wait_gpu()) {
            CloseHandle(fence_event);
            return r;
        }
    }

    double best_gbs = 0.0;
    std::uint32_t last_checksum = 0;
    bool any_ok = false;

    for (int it = 0; it < iterations; ++it) {
        alloc->Reset();
        cl->Reset(alloc.Get(), pso.Get());
        cl->SetComputeRootSignature(root.Get());
        ID3D12DescriptorHeap* heaps[] = {heap.Get()};
        cl->SetDescriptorHeaps(1, heaps);
        cl->SetComputeRootDescriptorTable(0, heap->GetGPUDescriptorHandleForHeapStart());
        cl->SetPipelineState(pso.Get());
        cl->Dispatch(plan.groups_x, plan.groups_y, plan.groups_z);

        D3D12_RESOURCE_BARRIER b = {};
        b.Type = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION;
        b.Transition.pResource = output_default.Get();
        b.Transition.StateBefore = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        b.Transition.StateAfter = D3D12_RESOURCE_STATE_COPY_SOURCE;
        b.Transition.Subresource = D3D12_RESOURCE_BARRIER_ALL_SUBRESOURCES;
        cl->ResourceBarrier(1, &b);
        cl->CopyResource(output_readback.Get(), output_default.Get());
        // Restore UAV for next iter
        b.Transition.StateBefore = D3D12_RESOURCE_STATE_COPY_SOURCE;
        b.Transition.StateAfter = D3D12_RESOURCE_STATE_UNORDERED_ACCESS;
        cl->ResourceBarrier(1, &b);
        cl->Close();

        const auto t0 = std::chrono::steady_clock::now();
        ID3D12CommandList* lists[] = {cl.Get()};
        queue->ExecuteCommandLists(1, lists);
        if (!wait_gpu()) {
            CloseHandle(fence_event);
            return r;
        }
        const double sec =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        if (sec > 0.0) {
            const double gbs = static_cast<double>(r.buffer_bytes) / 1e9 / sec;
            best_gbs = std::max(best_gbs, gbs);
        }

        // Readback checksum
        {
            void* mapped = nullptr;
            D3D12_RANGE range = {0, static_cast<SIZE_T>(out_bytes)};
            hr = output_readback->Map(0, &range, &mapped);
            if (FAILED(hr)) {
                throw_if_failed(hr, "Map readback", r);
                CloseHandle(fence_event);
                return r;
            }
            const auto* partials = static_cast<const std::uint32_t*>(mapped);
            std::uint32_t x = 0;
            for (UINT g = 0; g < groups; ++g)
                x ^= partials[g];
            last_checksum = x;
            output_readback->Unmap(0, nullptr);
        }
        any_ok = true;
    }

    CloseHandle(fence_event);

    r.d3d12_ran = any_ok;
    r.read_gbs = best_gbs;
    r.checksum = last_checksum;
    r.checksum_ok = (last_checksum == r.expected_checksum);
    if (!r.checksum_ok && r.error_msg.empty())
        r.error_msg = "checksum mismatch";
    if (r.checksum_ok)
        r.error_msg.clear();
    return r;
}

#endif // _WIN32

} // namespace xllama
