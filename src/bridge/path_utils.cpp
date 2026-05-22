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

    try {
        using winrt::Windows::ApplicationModel::Package;
        std::wstring installed_dir(Package::Current().InstalledPath().c_str());
        installed_dir += L"\\models\\";
        int fnSz = MultiByteToWideChar(CP_UTF8, 0, filename.c_str(), -1, nullptr, 0);
        if (fnSz <= 0)
            return primary;

        std::wstring wfn(static_cast<size_t>(fnSz), L'\0');
        MultiByteToWideChar(CP_UTF8, 0, filename.c_str(), -1, wfn.data(), fnSz);
        if (!wfn.empty() && wfn.back() == L'\0')
            wfn.pop_back();
        installed_dir += wfn; // InstalledPath\models\<name>

        // Verify bundle exists (probe genai_config.json).
        DWORD a = GetFileAttributesW((installed_dir + L"\\genai_config.json").c_str());
        if (a == INVALID_FILE_ATTRIBUTES || (a & FILE_ATTRIBUTE_DIRECTORY))
            return primary; // no bundled model

        // Convert primary (LocalState\models\<name>) to wide for Win32 calls.
        int psz = MultiByteToWideChar(CP_UTF8, 0, primary.c_str(), -1, nullptr, 0);
        if (psz <= 0)
            return primary;
        std::wstring primary_w(static_cast<size_t>(psz), L'\0');
        MultiByteToWideChar(CP_UTF8, 0, primary.c_str(), -1, primary_w.data(), psz);
        if (!primary_w.empty() && primary_w.back() == L'\0')
            primary_w.pop_back();

        // Create LocalState\models\ then LocalState\models\<name>\.
        auto sep = primary_w.rfind(L'\\');
        if (sep != std::wstring::npos)
            CreateDirectoryW(primary_w.substr(0, sep).c_str(), nullptr);
        CreateDirectoryW(primary_w.c_str(), nullptr);

        // Copy each file from InstalledPath\models\<name>\ to LocalState\models\<name>\.
        log_output("[xllama] first launch: copying bundled model to LocalState...\n");
        WIN32_FIND_DATAW fd{};
        HANDLE hf = FindFirstFileW((installed_dir + L"\\*").c_str(), &fd);
        if (hf != INVALID_HANDLE_VALUE) {
            do {
                if (fd.dwFileAttributes & FILE_ATTRIBUTE_DIRECTORY)
                    continue;
                std::wstring src = installed_dir + L"\\" + fd.cFileName;
                std::wstring dst = primary_w + L"\\" + fd.cFileName;
                if (!CopyFileW(src.c_str(), dst.c_str(), /*bFailIfExists=*/FALSE)) {
                    char buf[128];
                    snprintf(buf, sizeof(buf), "[xllama] CopyFile failed: %lu\n", GetLastError());
                    log_output(buf);
                }
            } while (FindNextFileW(hf, &fd));
            FindClose(hf);
        }
        log_output("[xllama] model copy complete, loading from LocalState\n");
        return primary;
    } catch (...) {
        log_output("[xllama] InstalledPath copy failed\n");
    }

    // Fallback 3: USB root cached by EnsureModelAsync (async KnownFolders probe).
    // EnsureModelAsync writes LocalState\usb_model_root.txt with the drive root
    // after finding the model via KnownFolders.RemovableDevices; read it here
    // for the synchronous inference path.
    {
        std::string cache_utf8 = local_folder_path("usb_model_root.txt", nullptr);
        int csz = MultiByteToWideChar(CP_UTF8, 0, cache_utf8.c_str(), -1, nullptr, 0);
        if (csz > 0) {
            std::wstring wcache(static_cast<size_t>(csz), L'\0');
            MultiByteToWideChar(CP_UTF8, 0, cache_utf8.c_str(), -1, wcache.data(), csz);
            if (!wcache.empty() && wcache.back() == L'\0') wcache.pop_back();
            FILE* fp = _wfopen(wcache.c_str(), L"r");
            if (fp) {
                wchar_t usb_root[512] = {};
                if (fgetws(usb_root, 511, fp)) {
                    size_t len = wcslen(usb_root);
                    while (len > 0 && (usb_root[len-1] == L'\n' || usb_root[len-1] == L'\r'))
                        usb_root[--len] = L'\0';
                }
                fclose(fp);
                if (usb_root[0]) {
                    int fnSz = MultiByteToWideChar(CP_UTF8, 0, filename.c_str(), -1, nullptr, 0);
                    if (fnSz > 0) {
                        std::wstring wfn(static_cast<size_t>(fnSz), L'\0');
                        MultiByteToWideChar(CP_UTF8, 0, filename.c_str(), -1, wfn.data(), fnSz);
                        if (!wfn.empty() && wfn.back() == L'\0') wfn.pop_back();
                        std::wstring usb_dir = std::wstring(usb_root) + L"\\xllama\\models\\" + wfn;
                        DWORD ua = GetFileAttributesW((usb_dir + L"\\genai_config.json").c_str());
                        if (ua != INVALID_FILE_ATTRIBUTES && !(ua & FILE_ATTRIBUTE_DIRECTORY)) {
                            log_output("[xllama] model found via USB cache\n");
                            int nsz = WideCharToMultiByte(
                                CP_UTF8, 0, usb_dir.c_str(), -1, nullptr, 0, nullptr, nullptr);
                            if (nsz > 0) {
                                std::string usb_path(nsz, '\0');
                                WideCharToMultiByte(CP_UTF8, 0, usb_dir.c_str(), -1,
                                                    usb_path.data(), nsz, nullptr, nullptr);
                                if (!usb_path.empty() && usb_path.back() == '\0')
                                    usb_path.pop_back();
                                return usb_path;
                            }
                        }
                    }
                }
            }
        }
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
