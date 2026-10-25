// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <string>
#include <string_view>
#include <vector>

namespace xllama {

// Catalogue ids that must not be downloadable. LoadModelManifest drops them
// from LocalState / personal overrides; api_pull_model_allowed rejects them.
constexpr std::string_view kWithdrawnModelIds[] = {"qwen25-coder-3b"};

template <typename Name> inline bool model_id_is_withdrawn(const Name& name) {
    const auto n = name.size();
    for (const auto id : kWithdrawnModelIds) {
        if (n != id.size())
            continue;
        bool match = true;
        for (size_t i = 0; i < n; ++i) {
            if (static_cast<unsigned>(name[i]) != static_cast<unsigned char>(id[i])) {
                match = false;
                break;
            }
        }
        if (match)
            return true;
    }
    return false;
}

inline bool model_id_is_withdrawn(const char* name) {
    return name && model_id_is_withdrawn(std::string_view(name));
}

template <typename Entry> inline void drop_withdrawn_entries(std::vector<Entry>& entries) {
    entries.erase(std::remove_if(entries.begin(), entries.end(),
                                 [](const Entry& e) { return model_id_is_withdrawn(e.name); }),
                  entries.end());
}

} // namespace xllama
