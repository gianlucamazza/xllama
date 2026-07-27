// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "xllama/kv_store.h"

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <vector>

namespace xllama {

namespace {
constexpr const char* kExt = ".kv";

struct Snapshot {
    std::filesystem::path path;
    std::filesystem::file_time_type mtime;
    uint64_t bytes;
};

// Snapshots newest-first, plus the abandoned .tmp files pruning has to clear.
std::vector<Snapshot> scan(const std::string& dir, std::vector<std::filesystem::path>* tmps) {
    std::vector<Snapshot> out;
    std::error_code ec;
    for (const auto& e : std::filesystem::directory_iterator(dir, ec)) {
        if (ec)
            break;
        if (!e.is_regular_file(ec))
            continue;
        const auto ext = e.path().extension().string();
        if (ext == ".tmp") {
            if (tmps)
                tmps->push_back(e.path());
            continue;
        }
        if (ext != kExt)
            continue;
        std::error_code ec2;
        const auto t = std::filesystem::last_write_time(e.path(), ec2);
        const auto n = std::filesystem::file_size(e.path(), ec2);
        if (ec2)
            continue;
        out.push_back({e.path(), t, static_cast<uint64_t>(n)});
    }
    std::sort(out.begin(), out.end(),
              [](const Snapshot& a, const Snapshot& b) { return a.mtime > b.mtime; });
    return out;
}
} // namespace

bool KvStore::valid_id(const std::string& id) {
    if (id.empty() || id.size() > 64)
        return false;
    for (const char c : id) {
        const auto uc = static_cast<unsigned char>(c);
        if (!std::isalnum(uc) && c != '-' && c != '_')
            return false;
    }
    return true;
}

std::string KvStore::path_for(const std::string& id) const {
    if (dir.empty() || !valid_id(id))
        return {};
    return (std::filesystem::path(dir) / (id + kExt)).string();
}

void KvStore::erase(const std::string& id) const {
    const std::string p = path_for(id);
    if (p.empty())
        return;
    std::error_code ec;
    std::filesystem::remove(p, ec);
    std::filesystem::remove(p + ".tmp", ec);
}

void KvStore::prune(size_t max_files, uint64_t max_bytes) const {
    if (dir.empty())
        return;
    std::vector<std::filesystem::path> tmps;
    const auto snaps = scan(dir, &tmps);
    std::error_code ec;
    for (const auto& t : tmps)
        std::filesystem::remove(t, ec);

    uint64_t kept_bytes = 0;
    size_t kept = 0;
    for (const auto& s : snaps) {
        const bool fits = kept < max_files && kept_bytes + s.bytes <= max_bytes;
        if (fits) {
            ++kept;
            kept_bytes += s.bytes;
        } else {
            std::filesystem::remove(s.path, ec);
        }
    }
}

uint64_t KvStore::total_bytes() const {
    if (dir.empty())
        return 0;
    uint64_t sum = 0;
    for (const auto& s : scan(dir, nullptr))
        sum += s.bytes;
    return sum;
}

} // namespace xllama
