// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#pragma once

#include <cstdio>
#include <filesystem>
#include <mutex>
#include <string_view>

namespace xllama {

// Serialize same-process writers so they can reuse a sibling staging file
// left by a previous interrupted publication.
inline std::mutex& atomic_file_mutex() {
    static std::mutex mutex;
    return mutex;
}

template <typename Writer>
bool replace_file_checked(const std::filesystem::path& target, Writer write) {
    std::lock_guard<std::mutex> lock(atomic_file_mutex());
    auto staged = target;
    staged += ".tmp";
#ifdef _WIN32
    FILE* file = _wfopen(staged.c_str(), L"wb");
#else
    FILE* file = std::fopen(staged.c_str(), "wb");
#endif
    if (!file)
        return false;
    std::error_code ec;
    bool ok = false;
    try {
        ok = write(file) && !std::ferror(file) && std::fflush(file) == 0;
    } catch (...) {
        std::fclose(file);
        std::filesystem::remove(staged, ec);
        throw;
    }
    if (std::fclose(file) == 0 && ok) {
        std::filesystem::rename(staged, target, ec);
        if (!ec)
            return true;
    }
    std::filesystem::remove(staged, ec);
    return false;
}

inline bool replace_file_bytes(const std::filesystem::path& target, std::string_view bytes) {
    return replace_file_checked(target, [&](FILE* file) {
        return std::fwrite(bytes.data(), 1, bytes.size(), file) == bytes.size();
    });
}

} // namespace xllama
