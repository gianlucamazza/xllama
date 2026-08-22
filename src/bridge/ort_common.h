// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Shared ORT GenAI setup helpers — eliminates duplication between
// run_inference_ort (inference.cpp) and create_ort (session.cpp).
//
// Two functions:
//   install_se_translator()  — converts SEH exceptions (D3D12/DML OOM, AV)
//                              to std::runtime_error, so the catch block
//                              can log them. Called once per inference session.
//   register_oga_logging()   — redirects ORT GenAI internal log messages
//                              (via OgaSetLogCallback) into xllama.log.
//
// Both are Windows-only (SEH + ORT). Included only inside #ifdef XLLAMA_USE_ORT.

#pragma once

#include <Windows.h>
#include <cstdio>
#include <eh.h>
#include <exception>

#include "xllama/platform.h" // log_output

namespace xllama {

// Convert SEH (D3D12/DML OOM/AV) → std::runtime_error.
inline void install_se_translator() {
    _set_se_translator([](unsigned int code, EXCEPTION_POINTERS*) {
        char b[48];
        std::snprintf(b, sizeof(b), "SEH 0x%08X", code);
        throw std::runtime_error(b);
    });
}

// Redirect ORT GenAI internal log messages into xllama.log.
// Without this they go only to OutputDebugStringA (Device Portal debug output).
inline void register_oga_logging() {
    OgaSetLogCallback([](const char* msg, size_t) { log_output(msg); });
}

} // namespace xllama