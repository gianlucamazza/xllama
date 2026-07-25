// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "xllama/path_utils.h"
#include "xllama/platform.h"

#include <filesystem>

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

// Dual-sentinel helpers for both ORT GenAI models and GGUF models (catalogue).
static bool dir_contains_any_gguf(const std::wstring& dir_w) {
    WIN32_FIND_DATAW fd{};
    HANDLE h = FindFirstFileW((dir_w + L"\\*.gguf").c_str(), &fd);
    if (h == INVALID_HANDLE_VALUE)
        return false;
    FindClose(h);
    return true;
}

static bool dir_is_valid_model(const std::wstring& dir_w) {
    // ORT GenAI sentinel
    if (GetFileAttributesW((dir_w + L"\\genai_config.json").c_str()) != INVALID_FILE_ATTRIBUTES)
        return true;
    // GGUF layout (catalogue entry or USB provisioned .gguf)
    return dir_contains_any_gguf(dir_w);
}

std::string resolve_model_path(const std::string& filename) {
    // An already-absolute path is fully resolved — pass it through unchanged.
    // The device-train evaluate stage loads its merged GGUF by an absolute
    // out_dir path (Q:\...\training\out\...\merged.gguf); prepending the models
    // directory would double it (LocalState\models\Q:\...) and fail to load.
    // Matches a drive-letter root ("X:\" or "X:/") or a UNC prefix ("\\").
    const bool drive_abs = filename.size() >= 3 &&
                           ((filename[0] >= 'A' && filename[0] <= 'Z') ||
                            (filename[0] >= 'a' && filename[0] <= 'z')) &&
                           filename[1] == ':' && (filename[2] == '\\' || filename[2] == '/');
    const bool unc_abs = filename.size() >= 2 && filename[0] == '\\' && filename[1] == '\\';
    if (drive_abs || unc_abs)
        return filename;

    // Primary: LocalFolder\models\<filename>  (user-placed or WDP-uploaded or catalogue download)
    std::string primary = local_folder_path(filename, L"models");

    // Accept a directory if it contains either genai_config.json (ORT GenAI)
    // or at least one *.gguf (llama.cpp / GGUF catalogue entries).
    std::wstring primary_w_for_check;
    {
        int sz = MultiByteToWideChar(CP_UTF8, 0, primary.c_str(), -1, nullptr, 0);
        if (sz > 0) {
            primary_w_for_check.resize(static_cast<size_t>(sz));
            MultiByteToWideChar(CP_UTF8, 0, primary.c_str(), -1, primary_w_for_check.data(), sz);
            if (!primary_w_for_check.empty() && primary_w_for_check.back() == L'\0')
                primary_w_for_check.pop_back();
        }
    }

    if (!primary_w_for_check.empty() && dir_is_valid_model(primary_w_for_check))
        return primary; // model found in LocalFolder (ORT or GGUF)

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

        // Verify bundle exists — only for legacy ORT GenAI bundles (genai_config.json).
        // GGUF catalogue entries are never bundled this way; they come via download.
        DWORD a = GetFileAttributesW((installed_dir + L"\\genai_config.json").c_str());
        if (a == INVALID_FILE_ATTRIBUTES || (a & FILE_ATTRIBUTE_DIRECTORY))
            return primary; // no bundled (ORT) model to copy

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
            if (!wcache.empty() && wcache.back() == L'\0')
                wcache.pop_back();
            FILE* fp = _wfopen(wcache.c_str(), L"r");
            if (fp) {
                wchar_t usb_root[512] = {};
                if (fgetws(usb_root, 511, fp)) {
                    size_t len = wcslen(usb_root);
                    while (len > 0 && (usb_root[len - 1] == L'\n' || usb_root[len - 1] == L'\r'))
                        usb_root[--len] = L'\0';
                }
                fclose(fp);
                if (usb_root[0]) {
                    int fnSz = MultiByteToWideChar(CP_UTF8, 0, filename.c_str(), -1, nullptr, 0);
                    if (fnSz > 0) {
                        std::wstring wfn(static_cast<size_t>(fnSz), L'\0');
                        MultiByteToWideChar(CP_UTF8, 0, filename.c_str(), -1, wfn.data(), fnSz);
                        if (!wfn.empty() && wfn.back() == L'\0')
                            wfn.pop_back();
                        std::wstring usb_dir = std::wstring(usb_root) + L"\\xllama\\models\\" + wfn;
                        // Accept either classic ORT or GGUF layout on USB.
                        if (dir_is_valid_model(usb_dir)) {
                            log_output("[xllama] model found via USB cache\n");
                            int nsz = WideCharToMultiByte(CP_UTF8, 0, usb_dir.c_str(), -1, nullptr,
                                                          0, nullptr, nullptr);
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

// ---------------------------------------------------------------------------
// GGUF vs ORT layout detection (used by Auto dispatch in unified builds).
// ---------------------------------------------------------------------------

bool model_uses_llama_backend(const std::string& model_id) {
    // Fast path: explicit .gguf file (common on Linux CLI and direct paths).
    if (model_id.size() >= 5 && model_id.compare(model_id.size() - 5, 5, ".gguf") == 0) {
        return true;
    }

    // Resolve (UWP: yields LocalState\models\<name> or equivalent;
    // Linux: identity). Then inspect the on-disk layout.
    const std::string p = resolve_model_path(model_id);

    std::error_code ec;
    if (std::filesystem::is_regular_file(p, ec)) {
        return p.size() >= 5 && p.compare(p.size() - 5, 5, ".gguf") == 0;
    }

    if (std::filesystem::is_directory(p, ec)) {
        for (const auto& de : std::filesystem::directory_iterator(p, ec)) {
            if (!ec && de.path().extension() == ".gguf") {
                return true;
            }
        }
    }

    return false;
}

std::string first_gguf_in_dir(const std::string& path, const std::string& exclude_filename) {
    std::error_code ec;
    if (!std::filesystem::is_directory(path, ec))
        return path;

    std::string preferred;
    std::string fallback;
    const auto model_gguf = std::filesystem::path(path) / "model.gguf";
    if (std::filesystem::is_regular_file(model_gguf, ec))
        return model_gguf.string();

    for (const auto& de : std::filesystem::directory_iterator(path, ec)) {
        if (ec || de.path().extension() != ".gguf")
            continue;
        const std::string name = de.path().filename().string();
        if (!exclude_filename.empty() && name == exclude_filename)
            continue;
        // Prefer non-adapter-looking names when no model.gguf
        if (name.find("adapter") != std::string::npos || name.find("lora") != std::string::npos) {
            if (fallback.empty())
                fallback = de.path().string();
            continue;
        }
        if (preferred.empty())
            preferred = de.path().string();
    }
    if (!preferred.empty())
        return preferred;
    return fallback;
}

} // namespace xllama
