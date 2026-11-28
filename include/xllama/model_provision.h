// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Provisioning predicate — is a model directory populated with the manifest's
// CURRENT expected files?
//
// The loose "any .gguf counts as provisioned" check (model-downloader.cpp) can't
// tell a stale quant from the current one, so a directory holding an outdated
// file (e.g. gemma-4-E2B-it-UD-IQ2_M.gguf) is never auto-upgraded to the
// manifest's file (gemma-4-E2B-it-Q3_K_S.gguf). This predicate compares the
// directory listing against the manifest's expected filename set so EnsureModel
// knows the tree is not yet catalogue-complete. What to HTTP-fetch is a
// per-file pin check (`model_file_plan.h`); a missing licence/notice pin must
// not invalidate already-matching weights (`files[].role`, documented default).
//
// Pure (no WinRT / no pch.h), mirroring manifest_merge.h, so it is host
// doctest-testable. The UWP caller lists the directory and maps
// ManifestEntry.files[].filename into `expected`.
#pragma once

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

namespace xllama {

// Normalize a path for comparison: backslashes → '/', ASCII-lowercased. Windows
// filesystems are case-insensitive and the manifest uses '/' while a native
// listing yields '\\'; the gemma filenames are mixed-case, so both axes matter.
inline std::wstring normalize_model_path(std::wstring s) {
    for (auto& ch : s) {
        if (ch == L'\\')
            ch = L'/';
        else if (ch >= L'A' && ch <= L'Z')
            ch = static_cast<wchar_t>(ch - L'A' + L'a');
    }
    return s;
}

// Name the first missing catalogue path, using Windows filename semantics.
inline std::wstring missing_expected_file(const std::vector<std::wstring>& present,
                                          const std::vector<std::wstring>& expected) {
    std::vector<std::wstring> norm_present;
    norm_present.reserve(present.size());
    for (const auto& p : present)
        norm_present.push_back(normalize_model_path(p));

    for (const auto& want : expected) {
        const std::wstring w = normalize_model_path(want);
        bool found = false;
        for (const auto& have : norm_present) {
            if (have == w) {
                found = true;
                break;
            }
        }
        if (!found)
            return want;
    }
    return {};
}

inline bool dir_satisfies_expected_files(const std::vector<std::wstring>& present,
                                         const std::vector<std::wstring>& expected) {
    return !expected.empty() && missing_expected_file(present, expected).empty();
}

// Even without catalogue pins, verify the ORT config and graph in the inventory
// that will be copied. The drive may have changed since discovery.
inline std::wstring missing_usb_model_file(const std::vector<std::wstring>& present,
                                           const std::vector<std::wstring>& expected) {
    const auto missing = missing_expected_file(present, expected);
    if (!missing.empty())
        return missing;
    if (!missing_expected_file(present, {L"genai_config.json"}).empty())
        return L"genai_config.json";
    for (const auto& file : present) {
        const auto name = normalize_model_path(file);
        if (name.size() > 5 && name.compare(name.size() - 5, 5, L".onnx") == 0)
            return {};
    }
    return L"an ONNX graph (*.onnx)";
}

inline constexpr char kUsbImportPending[] = ".usb-import-pending";

inline bool usb_import_pending(const std::filesystem::path& dir) {
    std::error_code ec;
    return std::filesystem::exists(dir / kUsbImportPending, ec);
}

// True iff a model directory's file listing makes it SERVABLE by Session:
// a base GGUF (any *.gguf that is not a bare runtime-LoRA "adapter.gguf"),
// or an ORT GenAI layout (genai_config.json / model.onnx). `files` are
// directory-relative names; comparison is case-insensitive. Used by the LAN
// API's /v1/models discovery — pure, host doctest-testable.
inline bool model_dir_files_ready(const std::vector<std::string>& files) {
    if (std::find(files.begin(), files.end(), kUsbImportPending) != files.end())
        return false;
    auto lower = [](std::string s) {
        for (auto& c : s)
            if (c >= 'A' && c <= 'Z')
                c = static_cast<char>(c - 'A' + 'a');
        return s;
    };
    for (const auto& f : files) {
        const std::string name = lower(f);
        if (name == "genai_config.json" || name == "model.onnx")
            return true;
        if (name.size() > 5 && name.compare(name.size() - 5, 5, ".gguf") == 0 &&
            name != "adapter.gguf")
            return true;
    }
    return false;
}

} // namespace xllama
