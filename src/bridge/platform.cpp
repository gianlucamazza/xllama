// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

#include "xllama/platform.h"

#include <cstdio>
#include <mutex>
#include <thread>

#ifdef XLLAMA_UWP
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
// clang-format off
    #include <windows.h>
    #define PSAPI_VERSION 2
    #include <psapi.h>
    #include <unknwn.h>
    #include <winrt/Windows.Storage.h>
// clang-format on
#endif

namespace xllama {

int detect_threads() noexcept {
    int n = static_cast<int>(std::thread::hardware_concurrency());
    return n > 0 ? n : 4;
}

void log_output(const char* msg) noexcept {
#ifdef XLLAMA_UWP
    OutputDebugStringA(msg);
    // Also mirror to LocalFolder/xllama.log for post-mortem (bridge errors invisible otherwise).
    static std::mutex s_mtx;
    static FILE* s_fp = []() -> FILE* {
        try {
            auto folder = winrt::Windows::Storage::ApplicationData::Current().LocalFolder();
            std::wstring path = std::wstring(folder.Path().c_str()) + L"\\xllama.log";
            return _wfopen(path.c_str(), L"a");
        } catch (...) {
            return nullptr;
        }
    }();
    if (s_fp) {
        std::lock_guard<std::mutex> g(s_mtx);
        std::fputs(msg, s_fp);
        std::fflush(s_fp);
    }
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
