// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

// ggml-backend-dl-stub.cpp
// UWP replacement for ggml/src/ggml-backend-dl.cpp.
// LoadLibraryW is blocked in WINAPI_FAMILY_APP; dynamic backend loading
// is disabled. All three functions return "not found" / nullptr.
//
// We do NOT include ggml-backend-dl.h to avoid its transitive include of
// <winevt.h> (which is desktop-only on some SDK configurations).
// Instead we forward-declare the ABI-compatible signatures directly.

#ifdef XLLAMA_UWP

    #include <filesystem>
namespace fs = std::filesystem;

    // On Windows, HMODULE = HINSTANCE__*; dl_handle = HINSTANCE__ (the struct).
    // Forward-declare so the pointer type matches without including <windows.h>.
    #ifdef _WIN32
struct HINSTANCE__;
using dl_handle = HINSTANCE__;
    #else
using dl_handle = void;
    #endif

dl_handle* dl_load_library(const fs::path& /*path*/) {
    return nullptr;
}

void* dl_get_sym(dl_handle* /*handle*/, const char* /*name*/) {
    return nullptr;
}

const char* dl_error() {
    return "dynamic backend loading not supported in UWP";
}

#endif // XLLAMA_UWP
