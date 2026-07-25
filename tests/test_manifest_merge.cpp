// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "xllama/manifest_merge.h"

#include <string>
#include <vector>

using namespace xllama;

namespace {
// Minimal stand-in for ManifestEntry: the merge only touches `.name`. `tag`
// tracks identity so a replace vs a preserve is distinguishable.
struct Entry {
    std::wstring name;
    int tag = 0;
};

std::vector<int> tags(const std::vector<Entry>& v) {
    std::vector<int> t;
    for (auto& e : v)
        t.push_back(e.tag);
    return t;
}
std::vector<std::wstring> names(const std::vector<Entry>& v) {
    std::vector<std::wstring> n;
    for (auto& e : v)
        n.push_back(e.name);
    return n;
}
} // namespace

TEST_CASE("manifest merge: same-name override replaces in place, order preserved") {
    std::vector<Entry> base{{L"a", 1}, {L"b", 2}, {L"c", 3}};
    std::vector<Entry> ov{{L"b", 20}};
    merge_manifest_entries(base, std::move(ov));
    CHECK(names(base) == std::vector<std::wstring>{L"a", L"b", L"c"}); // no reorder
    CHECK(tags(base) == std::vector<int>{1, 20, 3});                   // b replaced
}

TEST_CASE("manifest merge: new name is appended after base entries") {
    std::vector<Entry> base{{L"a", 1}, {L"b", 2}};
    std::vector<Entry> ov{{L"z", 9}};
    merge_manifest_entries(base, std::move(ov));
    CHECK(names(base) == std::vector<std::wstring>{L"a", L"b", L"z"});
    CHECK(tags(base) == std::vector<int>{1, 2, 9});
}

TEST_CASE("manifest merge: unmentioned base entries survive (no whole-catalogue shadow)") {
    // Regression for the 2026-07-10 bug: a single-entry override must not hide
    // the rest of the catalogue.
    std::vector<Entry> base{{L"sd-turbo-fp16", 1}, {L"gemma3-270m", 2}, {L"gemma4-e2b", 3}};
    std::vector<Entry> ov{{L"smollm2-360m", 99}};
    merge_manifest_entries(base, std::move(ov));
    CHECK(base.size() == 4);
    CHECK(names(base) == std::vector<std::wstring>{L"sd-turbo-fp16", L"gemma3-270m", L"gemma4-e2b",
                                                   L"smollm2-360m"});
}

TEST_CASE("manifest merge: replace + append in one override, applied in override order") {
    std::vector<Entry> base{{L"a", 1}, {L"b", 2}};
    std::vector<Entry> ov{{L"b", 20}, {L"c", 30}, {L"a", 10}};
    merge_manifest_entries(base, std::move(ov));
    CHECK(names(base) == std::vector<std::wstring>{L"a", L"b", L"c"});
    CHECK(tags(base) == std::vector<int>{10, 20, 30}); // a,b replaced; c appended
}

TEST_CASE("manifest merge: empty override leaves base untouched") {
    std::vector<Entry> base{{L"a", 1}, {L"b", 2}};
    merge_manifest_entries(base, std::vector<Entry>{});
    CHECK(tags(base) == std::vector<int>{1, 2});
}

TEST_CASE("manifest merge: empty base takes all override entries in order") {
    std::vector<Entry> base;
    std::vector<Entry> ov{{L"a", 1}, {L"b", 2}};
    merge_manifest_entries(base, std::move(ov));
    CHECK(names(base) == std::vector<std::wstring>{L"a", L"b"});
}

TEST_CASE("manifest merge: duplicate names within override collapse onto first match") {
    // Two override entries with the same name: the first replaces the base slot,
    // the second finds that same slot and replaces again (last-wins), no append.
    std::vector<Entry> base{{L"a", 1}};
    std::vector<Entry> ov{{L"a", 10}, {L"a", 11}};
    merge_manifest_entries(base, std::move(ov));
    CHECK(base.size() == 1);
    CHECK(base[0].tag == 11);
}
