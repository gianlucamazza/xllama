// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "xllama/chat_prompt.h"
#include "xllama/stop_sequence_filter.h"

#include <functional>
#include <string>
#include <vector>

using xllama::apply_stop_sequences;
using xllama::StopSequenceFilter;

namespace {

// The non-streaming oracle: apply the real matcher after every piece, exactly as
// the decode loop does. The result is the text the stream must have delivered.
std::string reference(const std::vector<std::string>& stops,
                      const std::vector<std::string>& pieces) {
    std::string output;
    for (const std::string& piece : pieces) {
        output += piece;
        std::string trimmed = output;
        if (apply_stop_sequences(trimmed, stops))
            return trimmed;
    }
    return output;
}

std::string filtered(const std::vector<std::string>& stops,
                     const std::vector<std::string>& pieces) {
    StopSequenceFilter filter(stops);
    std::string out;
    for (const std::string& piece : pieces)
        out += filter.push(piece);
    out += filter.finish();
    return out;
}

std::vector<std::string> per_char(const std::string& text) {
    std::vector<std::string> pieces;
    for (char c : text)
        pieces.emplace_back(1, c);
    return pieces;
}

} // namespace

TEST_CASE("stop filter: a stop inside one piece is discarded") {
    CHECK(filtered({"<STOP>"}, {"hello<STOP>"}) == "hello");
    // A stop followed by more text is not a suffix, so the real matcher does not
    // stop there; the filter must match that behaviour and emit the whole piece.
    CHECK(filtered({"<STOP>"}, {"hello<STOP>ignored"}) == "hello<STOP>ignored");
}

TEST_CASE("stop filter: a stop split across callbacks is withheld") {
    CHECK(filtered({"<STOP>"}, {"<ST", "OP>"}).empty());
    CHECK(filtered({"<STOP>"}, {"hello<ST", "OP>"}) == "hello");
    CHECK(filtered({"<STOP>"}, {"hel", "lo", "<S", "TO", "P>"}) == "hello");
}

TEST_CASE("stop filter: a failed prefix is released") {
    CHECK(filtered({"abc"}, {"xab", "d"}) == "xabd");
    CHECK(filtered({"abc"}, {"ab", "x"}) == "abx");
}

TEST_CASE("stop filter: multiple stops and list order") {
    // "bcd" is the suffix; the earlier "abc" does not match the end.
    CHECK(filtered({"abc", "bcd"}, {"ab", "cd"}) == "a");
    // The first listed stop wins when several are suffixes.
    CHECK(filtered({"bc", "abc"}, {"abc"}) == "a");
    CHECK(filtered({"abc", "bc"}, {"abc"}).empty());
    // "b" is not a suffix of "abc"; the full stop wins.
    CHECK(filtered({"b", "abc"}, {"abc"}).empty());
}

TEST_CASE("stop filter: empty stops are ignored") {
    CHECK(filtered({"", "abc"}, {"hello"}) == "hello");
    CHECK(filtered({""}, {"hello"}) == "hello");
}

TEST_CASE("stop filter: generation ending without a match releases the tail") {
    CHECK(filtered({"abc"}, {"xab"}) == "xab");
    CHECK(filtered({"<STOP>"}, {"hello<S"}) == "hello<S");
}

TEST_CASE("stop filter: pieces after a match are dropped") {
    CHECK(filtered({"abc"}, {"abc", "tail"}).empty());
    CHECK(filtered({"abc"}, {"xabc", "tail"}) == "x");
}

TEST_CASE("stop filter: UTF-8 around a stop is preserved") {
    CHECK(filtered({"abc"}, {"café abc"}) == "café ");
    CHECK(filtered({"abc"}, {"café a", "bc"}) == "café ");
    CHECK(filtered({"abc"}, {"😀abc"}) == "😀");
}

TEST_CASE("stop filter: parity with apply_stop_sequences") {
    // Every string over {a,b,c} up to five characters.
    std::vector<std::string> strings;
    std::string current;
    std::function<void(int)> generate = [&](int remaining) {
        strings.push_back(current);
        if (remaining == 0)
            return;
        for (char c : std::string("abc")) {
            current.push_back(c);
            generate(remaining - 1);
            current.pop_back();
        }
    };
    generate(5);

    const std::vector<std::vector<std::string>> stop_sets = {
        {},
        {""},
        {"a"},
        {"ab"},
        {"abc"},
        {"bc"},
        {"ab", "bc"},
        {"b", "abc"},
        {"abc", "bc"},
        {"aa", "ab", "ba"},
        {"c", "bc", "abc"},
        {"", "ab", "abc"},
    };

    for (const std::vector<std::string>& stops : stop_sets) {
        for (const std::string& text : strings) {
            std::vector<std::vector<std::string>> chunkings;
            chunkings.push_back({text});
            chunkings.push_back(per_char(text));
            for (std::size_t split = 1; split < text.size(); ++split)
                chunkings.push_back({text.substr(0, split), text.substr(split)});

            for (const std::vector<std::string>& pieces : chunkings) {
                const std::string want = reference(stops, pieces);
                const std::string got = filtered(stops, pieces);
                INFO("stops=", stops.size(), " text=", text, " want=", want, " got=", got);
                REQUIRE(got == want);
            }
        }
    }
}
