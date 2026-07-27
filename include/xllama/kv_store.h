// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// xllama::KvStore — on-disk KV snapshots, one per conversation (#170b).
#pragma once

#include <cstdint>
#include <string>

namespace xllama {

// A snapshot is cached work, never a source of truth: any of them may be
// missing, stale, or refused at load time, and every caller must fall back to
// a normal prefill. That is what makes unconditional pruning safe — dropping a
// snapshot costs a re-prefill, nothing else.
//
// Size discipline matters here: a snapshot is ~12 KiB per resident token
// (measured: 17.6 MB for a 1476-token LFM2.5 conversation at n_ctx 2048), and
// Dev Mode ships with ~2.2 GB free (uwp-constraints §9), so the pool is capped
// both by file count and by total bytes.
struct KvStore {
    std::string dir;

    // Conversation ids come from the chat UI. Refuse anything that is not a
    // plain identifier, so an id can never reach outside `dir`.
    static bool valid_id(const std::string& id);

    // Empty string when the id is unusable.
    std::string path_for(const std::string& id) const;

    void erase(const std::string& id) const;

    // Keep at most max_files snapshots and max_bytes in total, dropping the
    // least recently modified first. Also removes .tmp files abandoned by an
    // interrupted save.
    void prune(size_t max_files, uint64_t max_bytes) const;

    uint64_t total_bytes() const;
};

inline constexpr size_t kKvStoreMaxFiles = 3;
inline constexpr uint64_t kKvStoreMaxBytes = 192ull << 20;

} // namespace xllama
