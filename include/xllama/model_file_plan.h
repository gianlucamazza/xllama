// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Per-file catalogue fetch plan.
//
// A directory is not compared to the manifest as one blob. Each files[] pin is
// checked on its own (size, and SHA-256 when both sides have one). Adding a
// LICENSE/NOTICE (or any other sidecar) must not mark already-matching weight
// files as stale. The UWP downloader HTTP-fetches only `files_needing_fetch`.
//
// Pure (no WinRT / no pch.h). Host doctest-testable.
#pragma once

#include "xllama/model_provision.h"

#include <cstdint>
#include <string>
#include <vector>

namespace xllama {

// One catalogue files[] pin. `approx_bytes` 0 and an empty `sha256` mean "not
// known" — those axes do not reject a present file.
struct ManifestFilePin {
    std::wstring filename;
    uint64_t approx_bytes = 0;
    std::string sha256;
};

// What is on disk for one relative name. An empty `sha256` is a size-only
// observation (the usual listing); a pin with a hash still matches on size.
struct PresentFile {
    std::wstring filename;
    uint64_t size = 0;
    std::string sha256;
};

inline std::wstring file_basename_normalized(std::wstring path) {
    path = normalize_model_path(std::move(path));
    const auto pos = path.find_last_of(L'/');
    if (pos != std::wstring::npos)
        path = path.substr(pos + 1);
    return path;
}

inline std::wstring strip_sidecar_text_ext(std::wstring name) {
    const std::wstring exts[] = {L".txt", L".md", L".rst"};
    for (const auto& ext : exts) {
        if (name.size() > ext.size() &&
            name.compare(name.size() - ext.size(), ext.size(), ext) == 0)
            return name.substr(0, name.size() - ext.size());
    }
    return name;
}

// Licence / notice / readme family. Not a weight, tokenizer, or ONNX graph.
// Stem match only — portable, not tied to a catalogue id or host path.
inline bool is_catalogue_sidecar(const std::wstring& filename) {
    const std::wstring stem = strip_sidecar_text_ext(file_basename_normalized(filename));
    return stem == L"license" || stem == L"licence" || stem == L"notice" || stem == L"readme" ||
           stem == L"copying" || stem == L"authors" || stem == L"copyright";
}

inline bool sha256_hex_equal(const std::string& a, const std::string& b) {
    if (a.size() != b.size())
        return false;
    for (size_t i = 0; i < a.size(); ++i) {
        auto fold = [](char c) {
            return (c >= 'A' && c <= 'F') ? static_cast<char>(c - 'A' + 'a') : c;
        };
        if (fold(a[i]) != fold(b[i]))
            return false;
    }
    return true;
}

// True when this on-disk observation satisfies the pin. Size 0 is missing.
// A pin size of 0 does not reject. A pin hash is compared only when the
// observation also has a hash — size-only listings do not force a re-fetch.
inline bool file_pin_satisfied(const PresentFile& have, const ManifestFilePin& pin) {
    if (normalize_model_path(have.filename) != normalize_model_path(pin.filename))
        return false;
    if (have.size == 0)
        return false;
    if (pin.approx_bytes != 0 && have.size != pin.approx_bytes)
        return false;
    if (!pin.sha256.empty() && !have.sha256.empty() && !sha256_hex_equal(have.sha256, pin.sha256))
        return false;
    return true;
}

inline const PresentFile* find_present_file(const std::vector<PresentFile>& present,
                                            const std::wstring& filename) {
    const std::wstring want = normalize_model_path(filename);
    for (const auto& p : present) {
        if (normalize_model_path(p.filename) == want)
            return &p;
    }
    return nullptr;
}

// Filenames to HTTP-fetch, in catalogue order. Missing or pin-mismatched only.
inline std::vector<std::wstring> files_needing_fetch(const std::vector<PresentFile>& present,
                                                     const std::vector<ManifestFilePin>& expected) {
    std::vector<std::wstring> need;
    need.reserve(expected.size());
    for (const auto& pin : expected) {
        const PresentFile* have = find_present_file(present, pin.filename);
        if (!have || !file_pin_satisfied(*have, pin))
            need.push_back(pin.filename);
    }
    return need;
}

// Same as files_needing_fetch, excluding catalogue sidecars.
inline std::vector<std::wstring>
weight_files_needing_fetch(const std::vector<PresentFile>& present,
                           const std::vector<ManifestFilePin>& expected) {
    std::vector<std::wstring> need;
    for (const auto& name : files_needing_fetch(present, expected)) {
        if (!is_catalogue_sidecar(name))
            need.push_back(name);
    }
    return need;
}

} // namespace xllama
