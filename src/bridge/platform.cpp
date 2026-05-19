// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

#include "xllama/platform.h"

#include <cstdio>
#include <thread>

#ifdef XLLAMA_UWP
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #include <windows.h>
    #define PSAPI_VERSION 2
    #include <psapi.h>
#endif

namespace xllama {

int detect_threads() noexcept {
    int n = static_cast<int>(std::thread::hardware_concurrency());
    return n > 0 ? n : 4;
}

void log_output(const char* msg) noexcept {
#ifdef XLLAMA_UWP
    OutputDebugStringA(msg);
#else
    std::fputs(msg, stderr);
#endif
}

void log_output(const std::string& msg) noexcept {
    log_output(msg.c_str());
}

std::size_t peak_working_set_mb() noexcept {
#ifdef XLLAMA_UWP
    PROCESS_MEMORY_COUNTERS pmc{};
    if (K32GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc)))
        return pmc.PeakWorkingSetSize / (1024 * 1024);
    return 0;
#else
    return 0;
#endif
}

} // namespace xllama
