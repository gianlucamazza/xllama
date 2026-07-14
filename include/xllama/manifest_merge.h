// Per-entry merge of a model-catalogue override onto a base catalogue.
//
// The LocalState override (uploadable via Device Portal, no reinstall) is merged
// ON TOP of the bundled catalogue: a same-`name` entry REPLACES the base one in
// place (preserving the base ordering), a new `name` is APPENDED, and a bundled
// entry the override does not mention STAYS. A stale single-entry override used
// to shadow the whole catalogue before this per-entry rule (found 2026-07-10).
//
// Templated on the entry type so it is unit-testable on the host with a minimal
// stub struct — the production caller instantiates it with `ManifestEntry`. The
// only requirement on `Entry` is a `.name` member comparable with `==`.
#pragma once

#include <utility>
#include <vector>

namespace xllama {

template <typename Entry>
void merge_manifest_entries(std::vector<Entry>& base, std::vector<Entry> overrides) {
    for (auto& oe : overrides) {
        bool replaced = false;
        for (auto& e : base) {
            if (e.name == oe.name) {
                e = std::move(oe);
                replaced = true;
                break;
            }
        }
        if (!replaced)
            base.push_back(std::move(oe));
    }
}

} // namespace xllama
