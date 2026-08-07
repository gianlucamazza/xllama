// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// Phase 15 W2.1 — prompt_lookup_draft pure function (#210). No GGUF required.

#include <doctest/doctest.h>

#include "xllama/speculative.h"

#include <cstdint>
#include <vector>

using namespace xllama;

TEST_CASE("spec: empty / too-short history declines") {
    CHECK(prompt_lookup_draft({}).empty());
    CHECK(prompt_lookup_draft({1}).empty());    // shorter than default n=2
    CHECK(prompt_lookup_draft({1, 2}).empty()); // only one window = ourselves
    CHECK(prompt_lookup_draft({1, 2, 3}, /*n_gram=*/4).empty());
}

TEST_CASE("spec: invalid n_gram or k_max declines") {
    const std::vector<int32_t> h = {1, 2, 1, 2, 9, 9};
    CHECK(prompt_lookup_draft(h, /*n_gram=*/0, /*k_max=*/2).empty());
    CHECK(prompt_lookup_draft(h, /*n_gram=*/2, /*k_max=*/0).empty());
    CHECK(prompt_lookup_draft(h, /*n_gram=*/-1, /*k_max=*/2).empty());
}

TEST_CASE("spec: no earlier match declines") {
    // Trailing [3,4] never appears earlier.
    CHECK(prompt_lookup_draft({1, 2, 3, 4}, 2, 2).empty());
    // All distinct.
    CHECK(prompt_lookup_draft({10, 11, 12, 13, 14}, 2, 2).empty());
}

TEST_CASE("spec: match found — draft is what followed the match") {
    // history:  [7, 8, 9,  7, 8]
    // pattern:             ^^^^^ = [7,8]
    // earlier match at 0, followed by [9, 7] — k=2 takes both.
    auto d = prompt_lookup_draft({7, 8, 9, 7, 8}, 2, 2);
    REQUIRE(d.size() == 2);
    CHECK(d[0] == 9);
    CHECK(d[1] == 7);
}

TEST_CASE("spec: k_max clips the draft") {
    // Match [1,2] at start; following tokens [3,4,5]; k=2 → [3,4]
    auto d = prompt_lookup_draft({1, 2, 3, 4, 5, 1, 2}, 2, 2);
    REQUIRE(d.size() == 2);
    CHECK(d[0] == 3);
    CHECK(d[1] == 4);

    auto d1 = prompt_lookup_draft({1, 2, 3, 4, 5, 1, 2}, 2, 1);
    REQUIRE(d1.size() == 1);
    CHECK(d1[0] == 3);
}

TEST_CASE("spec: match with nothing after it declines") {
    // [1,2] appears at the very start and is also the trailing pattern; the
    // only earlier match is at 0, and tokens after it are [1,2] themselves —
    // wait: history [1,2,1,2], match at 0 for pattern [1,2], follow=[1,2].
    // That is a valid draft of the tokens that followed the first occurrence.
    auto ok = prompt_lookup_draft({1, 2, 1, 2}, 2, 2);
    REQUIRE(ok.size() == 2);
    CHECK(ok[0] == 1);
    CHECK(ok[1] == 2);

    // Pattern [9] only at the end and once earlier with no room after? n=1:
    // history [9, 5, 9] — match at 0, follow=[5,9], k=1 → [5]
    auto d = prompt_lookup_draft({9, 5, 9}, 1, 1);
    REQUIRE(d.size() == 1);
    CHECK(d[0] == 5);
}

TEST_CASE("spec: n-gram only at start of history (no earlier match beyond self)") {
    // Pattern is the whole prefix that also ends the sequence, but search
    // refuses the self-window. history length == n_gram already covered;
    // longer history where the only occurrence of the trailing n-gram is the
    // suffix itself:
    //   [1, 2, 3, 4, 3, 4] — trailing [3,4] also at index 2. That IS an earlier
    //   match. For "only at end": [1, 2, 3, 4, 5, 6] trailing [5,6] — no earlier.
    CHECK(prompt_lookup_draft({1, 2, 3, 4, 5, 6}, 2, 2).empty());
}

TEST_CASE("spec: multiple matches — last earlier wins") {
    // [A,B] appears at 0 (followed by X=10) and at 3 (followed by Y=20);
    // trailing pattern is [A,B] at the end. Last earlier match is index 3.
    // history: A B 10 A B 20 A B
    // idx:     0 1  2 3 4  5 6 7
    const std::vector<int32_t> h = {100, 200, 10, 100, 200, 20, 100, 200};
    auto d = prompt_lookup_draft(h, 2, 1);
    REQUIRE(d.size() == 1);
    CHECK(d[0] == 20); // not 10 — last match wins
}

TEST_CASE("spec: defaults are the pre-gate k=2 / n=2") {
    CHECK(kSpecDraftKDefault == 2);
    CHECK(kSpecNgramDefault == 2);
    // Same as explicit (2,2). Match at 0 for trailing [1,2]; follow=[9,1,2],
    // k=2 clips to [9,1].
    const std::vector<int32_t> h = {1, 2, 9, 1, 2};
    auto a = prompt_lookup_draft(h);
    auto b = prompt_lookup_draft(h, 2, 2);
    CHECK(a == b);
    REQUIRE(a.size() == 2);
    CHECK(a[0] == 9);
    CHECK(a[1] == 1);
}
