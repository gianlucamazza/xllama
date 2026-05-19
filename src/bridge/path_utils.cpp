// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

#include "xllama/path_utils.h"
#include "xllama/platform.h"

#ifdef XLLAMA_UWP
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
// clang-format off
    // windows.h must precede unknwn.h; unknwn.h must precede winrt/ headers.
    #include <windows.h>
    #include <unknwn.h> // required by winrt/base.h (COM IUnknown check)
    #include <winrt/Windows.Storage.h>
// clang-format on
#endif

namespace xllama {

#ifdef XLLAMA_UWP

static std::string local_folder_path(const std::string& filename, const wchar_t* subdir) {
    using namespace winrt::Windows::Storage;
    auto folder = ApplicationData::Current().LocalFolder();
    std::wstring wpath(folder.Path().c_str());
    wpath += L"\\";
    if (subdir) {
        wpath += subdir;
        wpath += L"\\";
    }
    // Convert filename to wide
    int sz = MultiByteToWideChar(CP_UTF8, 0, filename.c_str(), -1, nullptr, 0);
    if (sz <= 0) {
        log_output("[xllama] MultiByteToWideChar failed in path resolution\n");
        return filename;
    }
    std::wstring wfn(sz, L'\0');
    if (MultiByteToWideChar(CP_UTF8, 0, filename.c_str(), -1, wfn.data(), sz) == 0) {
        log_output("[xllama] MultiByteToWideChar conversion failed\n");
        return filename;
    }
    if (!wfn.empty() && wfn.back() == L'\0')
        wfn.pop_back();
    wpath += wfn;

    // Convert back to UTF-8
    int nsz = WideCharToMultiByte(CP_UTF8, 0, wpath.c_str(), -1, nullptr, 0, nullptr, nullptr);
    if (nsz <= 0) {
        log_output("[xllama] WideCharToMultiByte size query failed\n");
        return filename;
    }
    std::string result(nsz, '\0');
    if (WideCharToMultiByte(CP_UTF8, 0, wpath.c_str(), -1, result.data(), nsz, nullptr, nullptr) ==
        0) {
        log_output("[xllama] WideCharToMultiByte conversion failed\n");
        return filename;
    }
    if (!result.empty() && result.back() == '\0')
        result.pop_back();
    return result;
}

std::string resolve_model_path(const std::string& filename) {
    return local_folder_path(filename, L"models");
}

std::string resolve_local_path(const std::string& filename) {
    return local_folder_path(filename, nullptr);
}

#else // Linux

std::string resolve_model_path(const std::string& filename) {
    return filename;
}

std::string resolve_local_path(const std::string& filename) {
    return filename;
}

#endif

} // namespace xllama
