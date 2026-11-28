// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "d3d12_compute.h"

#include <d3d12sdklayers.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using Microsoft::WRL::ComPtr;
using xllama::d3d12c::create_buffer;
using xllama::d3d12c::QueueFence;

namespace {

[[noreturn]] void fail(const char* message) {
    std::fprintf(stderr, "%s\n", message);
    // The fixture owns this process. A failed marker cannot authorize unwinding
    // local COM owners for a copy that might still be in flight.
    std::fflush(stderr);
    std::_Exit(EXIT_FAILURE);
}

void check(HRESULT hr, const char* operation) {
    if (FAILED(hr))
        fail(xllama::d3d12c::hr_message(operation, hr).c_str());
}

void verify_debug_messages(ID3D12InfoQueue* messages) {
    if (!messages)
        return;
    const auto count = messages->GetNumStoredMessagesAllowedByRetrievalFilter();
    for (UINT64 i = 0; i < count; ++i) {
        SIZE_T length = 0;
        check(messages->GetMessage(i, nullptr, &length), "GetMessage(size)");
        std::vector<unsigned char> bytes(length);
        auto* message = reinterpret_cast<D3D12_MESSAGE*>(bytes.data());
        check(messages->GetMessage(i, message, &length), "GetMessage");
        if (message->Severity <= D3D12_MESSAGE_SEVERITY_ERROR)
            fail(message->pDescription);
    }
}

void verify_copies(ID3D12Device* device, ID3D12CommandQueue* queue, QueueFence& fence) {
    std::string error;
    constexpr auto size = sizeof(std::uint32_t);
    auto src = create_buffer(device, size, D3D12_HEAP_TYPE_UPLOAD, D3D12_RESOURCE_FLAG_NONE,
                             D3D12_RESOURCE_STATE_GENERIC_READ, "upload", &error);
    if (!src)
        fail(error.c_str());
    auto dst = create_buffer(device, size, D3D12_HEAP_TYPE_READBACK, D3D12_RESOURCE_FLAG_NONE,
                             D3D12_RESOURCE_STATE_COPY_DEST, "readback", &error);
    if (!dst)
        fail(error.c_str());

    ComPtr<ID3D12CommandAllocator> allocator;
    ComPtr<ID3D12GraphicsCommandList> list;
    check(device->CreateCommandAllocator(D3D12_COMMAND_LIST_TYPE_COMPUTE, IID_PPV_ARGS(&allocator)),
          "CreateCommandAllocator");
    check(device->CreateCommandList(0, D3D12_COMMAND_LIST_TYPE_COMPUTE, allocator.Get(), nullptr,
                                    IID_PPV_ARGS(&list)),
          "CreateCommandList");
    const D3D12_RANGE empty{0, 0};
    const D3D12_RANGE word{0, size};
    for (int i = 0; i < 4; ++i) {
        if (i > 0) {
            check(allocator->Reset(), "Reset allocator");
            check(list->Reset(allocator.Get(), nullptr), "Reset list");
        }
        const auto expected = 0xbabef00du + static_cast<std::uint32_t>(i);
        void* mapped = nullptr;
        check(src->Map(0, &empty, &mapped), "Map upload");
        std::memcpy(mapped, &expected, size);
        src->Unmap(0, &word);
        list->CopyBufferRegion(dst.Get(), 0, src.Get(), 0, size);
        check(list->Close(), "Close");
        ID3D12CommandList* batch[] = {list.Get()};
        queue->ExecuteCommandLists(1, batch);
        if (!fence.signal_and_wait(queue, (i % 2) != 0))
            fail("QueueFence failed on a live compute queue");
        std::uint32_t actual = 0;
        check(dst->Map(0, &word, &mapped), "Map readback");
        std::memcpy(&actual, mapped, size);
        dst->Unmap(0, &empty);
        if (actual != expected)
            fail("Readback did not match this submission");
    }
}

void exercise_device(IDXGIAdapter* adapter, bool debug, bool remove) {
    ComPtr<ID3D12Device> device;
    check(D3D12CreateDevice(adapter, D3D_FEATURE_LEVEL_11_0, IID_PPV_ARGS(&device)),
          "D3D12CreateDevice(WARP)");
    ComPtr<ID3D12InfoQueue> messages;
    if (debug)
        check(device.As(&messages), "QueryInterface(ID3D12InfoQueue)");

    D3D12_COMMAND_QUEUE_DESC desc{};
    desc.Type = D3D12_COMMAND_LIST_TYPE_COMPUTE;
    ComPtr<ID3D12CommandQueue> queue;
    check(device->CreateCommandQueue(&desc, IID_PPV_ARGS(&queue)), "CreateCommandQueue");
    QueueFence fence;
    std::string error;
    if (!fence.init(device.Get(), &error))
        fail(error.c_str());
    verify_copies(device.Get(), queue.Get(), fence);
    verify_debug_messages(messages.Get());
    if (remove) {
        ComPtr<ID3D12Device5> removable;
        check(device.As(&removable), "QueryInterface(ID3D12Device5)");
        removable->RemoveDevice();
        if (SUCCEEDED(device->GetDeviceRemovedReason()))
            fail("RemoveDevice did not mark the fixture device as removed");
        if (fence.signal_and_wait(queue.Get(), false))
            fail("QueueFence reported success after device removal");
    }
}

} // namespace

int main(int argc, char** argv) {
    const bool debug = argc == 2 && std::strcmp(argv[1], "--debug-layer") == 0;
    if (argc != 1 && !debug)
        fail("Usage: d3d12-fence-test [--debug-layer]");
    if (debug) {
        ComPtr<ID3D12Debug> controller;
        const HRESULT hr = D3D12GetDebugInterface(IID_PPV_ARGS(&controller));
        if (hr == DXGI_ERROR_SDK_COMPONENT_MISSING) {
            std::puts("NOT_RUN: D3D12 debug layer unavailable (Windows Graphics Tools)");
            return 77;
        }
        check(hr, "D3D12GetDebugInterface");
        controller->EnableDebugLayer();
    }
    ComPtr<IDXGIFactory4> factory;
    check(CreateDXGIFactory1(IID_PPV_ARGS(&factory)), "CreateDXGIFactory1");
    ComPtr<IDXGIAdapter> adapter;
    check(factory->EnumWarpAdapter(IID_PPV_ARGS(&adapter)), "EnumWarpAdapter");
    exercise_device(adapter.Get(), debug, true);
    exercise_device(adapter.Get(), debug, false);
    std::printf("PASS: WARP compute queue, event/spin readback, removed-device rejection and "
                "fresh-device readback (debug layer %s)\n",
                debug ? "on" : "off");
}
