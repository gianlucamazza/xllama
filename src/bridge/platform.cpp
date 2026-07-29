// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "xllama/platform.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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
    #include <dxgi1_4.h>
// clang-format on
#endif

namespace xllama {

int detect_threads() noexcept {
    int n = static_cast<int>(std::thread::hardware_concurrency());
    return n > 0 ? n : 4;
}

int detect_threads_llama() noexcept {
#ifdef XLLAMA_UWP
    return std::min(detect_threads(), 6); // ggml livelock at t7/t8 (see platform.h)
#else
    return detect_threads();
#endif
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
            FILE* fp = _wfopen(path.c_str(), L"a");
            if (fp)
                setvbuf(fp, nullptr, _IONBF, 0); // unbuffered: writes survive hard kill
            return fp;
        } catch (...) {
            return nullptr;
        }
    }();
    if (s_fp) {
        std::lock_guard<std::mutex> g(s_mtx);
        std::fputs(msg, s_fp); // _IONBF: no CRT buffer, write reaches kernel directly
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
    // Linux host: peak RSS from /proc (VmHWM is in kB). Was a hard 0 before —
    // host coding/campaign benches then reported peak=0MB and looked broken.
    FILE* fp = std::fopen("/proc/self/status", "r");
    if (!fp)
        return 0;
    char line[256];
    std::size_t kb = 0;
    while (std::fgets(line, sizeof(line), fp)) {
        if (std::strncmp(line, "VmHWM:", 6) == 0) {
            kb = static_cast<std::size_t>(std::strtoul(line + 6, nullptr, 10));
            break;
        }
    }
    std::fclose(fp);
    return kb / 1024;
#endif
}

std::size_t avail_phys_mb() noexcept {
#ifdef XLLAMA_UWP
    MEMORYSTATUSEX ms{};
    ms.dwLength = sizeof(ms);
    if (GlobalMemoryStatusEx(&ms))
        return static_cast<std::size_t>(ms.ullAvailPhys / (1024 * 1024));
    return 0;
#else
    // MemAvailable, not MemFree: the kernel's own estimate of what a new
    // allocation can claim without swapping (includes reclaimable page cache).
    FILE* fp = std::fopen("/proc/meminfo", "r");
    if (!fp)
        return 0;
    char line[256];
    std::size_t kb = 0;
    while (std::fgets(line, sizeof(line), fp)) {
        if (std::strncmp(line, "MemAvailable:", 13) == 0) {
            kb = static_cast<std::size_t>(std::strtoul(line + 13, nullptr, 10));
            break;
        }
    }
    std::fclose(fp);
    return kb / 1024;
#endif
}

GpuMemInfo gpu_mem_info() noexcept {
#ifdef XLLAMA_UWP
    // Adapter cached for process lifetime: gpu_mem_info() is called per phase
    // (pre-load / post-load / post-decode) and must stay cheap.
    static IDXGIAdapter3* s_adapter = []() -> IDXGIAdapter3* {
        IDXGIFactory1* factory = nullptr;
        if (FAILED(CreateDXGIFactory1(IID_PPV_ARGS(&factory))))
            return nullptr;
        IDXGIAdapter1* adapter1 = nullptr;
        IDXGIAdapter3* adapter3 = nullptr;
        if (SUCCEEDED(factory->EnumAdapters1(0, &adapter1))) {
            adapter1->QueryInterface(IID_PPV_ARGS(&adapter3));
            adapter1->Release();
        }
        factory->Release();
        return adapter3;
    }();

    GpuMemInfo info;
    if (!s_adapter)
        return info;
    DXGI_QUERY_VIDEO_MEMORY_INFO vmi{};
    if (FAILED(s_adapter->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &vmi)))
        return info;
    info.current_mb = static_cast<std::size_t>(vmi.CurrentUsage / (1024 * 1024));
    info.budget_mb = static_cast<std::size_t>(vmi.Budget / (1024 * 1024));
    info.available = true;
    return info;
#else
    return {};
#endif
}

void set_cwd_to_local_folder() noexcept {
#ifdef XLLAMA_UWP
    try {
        auto folder = winrt::Windows::Storage::ApplicationData::Current().LocalFolder();
        SetCurrentDirectoryW(folder.Path().c_str());
    } catch (...) {
    }
#endif
}

} // namespace xllama
