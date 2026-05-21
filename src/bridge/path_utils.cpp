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
    #include <winrt/Windows.ApplicationModel.h>
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
    // Primary: LocalFolder\models\<filename>  (user-placed or WDP-uploaded)
    std::string primary = local_folder_path(filename, L"models");

    // Verify the model directory has a genai_config.json (sentinel for a valid model).
    // If not found, fall back to the read-only install location where bundled
    // models live (placed by MSBuild DeploymentContent=true in xllama.vcxproj).
    std::wstring probe_w;
    {
        int sz = MultiByteToWideChar(CP_UTF8, 0, primary.c_str(), -1, nullptr, 0);
        if (sz > 0) {
            probe_w.resize(static_cast<size_t>(sz));
            MultiByteToWideChar(CP_UTF8, 0, primary.c_str(), -1, probe_w.data(), sz);
            if (!probe_w.empty() && probe_w.back() == L'\0')
                probe_w.pop_back();
            probe_w += L"\\genai_config.json";
        }
    }

    if (!probe_w.empty()) {
        DWORD attr = GetFileAttributesW(probe_w.c_str());
        if (attr != INVALID_FILE_ATTRIBUTES && !(attr & FILE_ATTRIBUTE_DIRECTORY))
            return primary; // model found in LocalFolder
    }

    // Fall back to Package.InstalledPath\models\<filename>
    try {
        using winrt::Windows::ApplicationModel::Package;
        std::wstring installed(Package::Current().InstalledPath().c_str());
        installed += L"\\models\\";
        int fnSz = MultiByteToWideChar(CP_UTF8, 0, filename.c_str(), -1, nullptr, 0);
        if (fnSz > 0) {
            std::wstring wfn(static_cast<size_t>(fnSz), L'\0');
            MultiByteToWideChar(CP_UTF8, 0, filename.c_str(), -1, wfn.data(), fnSz);
            if (!wfn.empty() && wfn.back() == L'\0')
                wfn.pop_back();
            installed += wfn;
            // Convert back to UTF-8
            int nsz = WideCharToMultiByte(CP_UTF8, 0, installed.c_str(), -1, nullptr, 0, nullptr, nullptr);
            if (nsz > 0) {
                std::string result(static_cast<size_t>(nsz), '\0');
                WideCharToMultiByte(CP_UTF8, 0, installed.c_str(), -1, result.data(), nsz, nullptr, nullptr);
                if (!result.empty() && result.back() == '\0')
                    result.pop_back();
                log_output("[xllama] model not in LocalFolder, using bundled: " + result + "\n");
                return result;
            }
        }
    } catch (...) {
        log_output("[xllama] InstalledPath fallback failed\n");
    }

    return primary; // return primary even if not found (ORT will emit a clear error)
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
