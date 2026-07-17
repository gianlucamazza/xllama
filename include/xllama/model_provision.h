// Copyright (c) 2024 Venere Labs
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
// can force a re-download when they diverge.
//
// Pure (no WinRT / no pch.h), mirroring manifest_merge.h, so it is host
// doctest-testable. The UWP caller lists the directory and maps
// ManifestEntry.files[].filename into `expected`.
#pragma once

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

// True iff EVERY expected file is present in the directory listing. `present` =
// directory-relative paths of the files on disk; `expected` = the manifest's
// files[].filename values. Comparison is separator- and case-insensitive. Extra
// present files are ignored (a superset is fine). An empty `expected` returns
// false: the predicate never guesses — the caller applies the loose fallback.
inline bool dir_satisfies_expected_files(const std::vector<std::wstring>& present,
                                         const std::vector<std::wstring>& expected) {
    if (expected.empty())
        return false;
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
            return false;
    }
    return true;
}

// True iff a model directory's file listing makes it SERVABLE by Session:
// a base GGUF (any *.gguf that is not a bare runtime-LoRA "adapter.gguf"),
// or an ORT GenAI layout (genai_config.json / model.onnx). `files` are
// directory-relative names; comparison is case-insensitive. Used by the LAN
// API's /v1/models discovery — pure, host doctest-testable.
inline bool model_dir_files_ready(const std::vector<std::string>& files) {
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
