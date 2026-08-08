// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Runtime resolve of d3d12.dll entry points. Linking d3d12.lib puts a hard
// import on the PE; CI MSVC UWP images that launch on Xbox do not import
// d3d12.dll at load time (device is created only when a GPU path runs).
// AppContainer activation must not depend on loading d3d12 before the first
// frame — resolve on first use instead.

#pragma once

#if defined(_WIN32)

    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <Windows.h>
    #include <d3d12.h>

namespace xllama {
namespace d3d12_dyn {

inline HMODULE module() noexcept {
    static HMODULE m = LoadLibraryExW(L"d3d12.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
    return m;
}

using CreateDeviceFn = HRESULT(WINAPI*)(IUnknown*, D3D_FEATURE_LEVEL, REFIID, void**);
using SerializeRootSignatureFn = HRESULT(WINAPI*)(const D3D12_ROOT_SIGNATURE_DESC*,
                                                  D3D_ROOT_SIGNATURE_VERSION, ID3DBlob**,
                                                  ID3DBlob**);

inline CreateDeviceFn CreateDevice() noexcept {
    static CreateDeviceFn fn = []() -> CreateDeviceFn {
        HMODULE m = module();
        if (!m)
            return nullptr;
        return reinterpret_cast<CreateDeviceFn>(GetProcAddress(m, "D3D12CreateDevice"));
    }();
    return fn;
}

inline SerializeRootSignatureFn SerializeRootSignature() noexcept {
    static SerializeRootSignatureFn fn = []() -> SerializeRootSignatureFn {
        HMODULE m = module();
        if (!m)
            return nullptr;
        return reinterpret_cast<SerializeRootSignatureFn>(
            GetProcAddress(m, "D3D12SerializeRootSignature"));
    }();
    return fn;
}

} // namespace d3d12_dyn
} // namespace xllama

#endif // _WIN32
