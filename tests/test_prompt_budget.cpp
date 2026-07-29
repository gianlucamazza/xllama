// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "xllama/inference_params.h" // kDefaultNCtx
#include "xllama/prompt_budget.h"
#include "xllama/routing_policy.h" // kReservedGenerationTokens

#include <string>
#include <vector>

using namespace xllama;

namespace {

// A deterministic stand-in for a tokenizer: 4 chars per token. The point of
// fit_prompt is that it does not care what the ratio is — it asks.
int fake_count(const std::string& s) {
    return static_cast<int>((s.size() + 3) / 4);
}

std::vector<ChatTurn> make_turns(int n, size_t chars_each) {
    std::vector<ChatTurn> turns;
    for (int i = 0; i < n; ++i)
        turns.push_back({std::string(chars_each, 'u'), std::string(chars_each, 'a')});
    return turns;
}

} // namespace

TEST_CASE("fit_prompt: a prompt that fits is returned untouched") {
    const auto fmt = chat_format_for("lfm25-350m");
    const auto fit =
        fit_prompt(fmt, "sys", make_turns(2, 40), "hello", kDefaultNCtx, 512, fake_count);
    CHECK(fit.fits);
    CHECK(fit.dropped == 0);
    CHECK(fit.n_tokens > 0);
    CHECK(fit.prompt.find("hello") != std::string::npos);
}

TEST_CASE("fit_prompt: drops the oldest turns until the reply fits") {
    const auto fmt = chat_format_for("lfm25-350m");
    // 40 turns of 400 chars: ~8000 tokens by the fake counter, far past 2048.
    const auto fit =
        fit_prompt(fmt, "sys", make_turns(40, 400), "and now?", kDefaultNCtx, 512, fake_count);
    CHECK(fit.fits);
    CHECK(fit.dropped > 0);
    // The invariant the whole file exists for: prompt + requested reply fits.
    CHECK(fit.n_tokens + 512 + 1 <= kDefaultNCtx);
    // The newest message always survives.
    CHECK(fit.prompt.find("and now?") != std::string::npos);
}

TEST_CASE("fit_prompt: the reserve is floored, not taken at face value") {
    const auto fmt = chat_format_for("lfm25-350m");
    const auto tiny = fit_prompt(fmt, "sys", make_turns(40, 400), "q", kDefaultNCtx, 1, fake_count);
    REQUIRE(tiny.fits);
    // n_predict=1 must still leave the floor: a low request cannot smuggle a
    // context-filling prompt past the budget.
    CHECK(tiny.n_tokens + kReservedGenerationTokens + 1 <= kDefaultNCtx);

    // A bigger reply demands a shorter prompt.
    const auto big =
        fit_prompt(fmt, "sys", make_turns(40, 400), "q", kDefaultNCtx, 1024, fake_count);
    REQUIRE(big.fits);
    CHECK(big.n_tokens + 1024 + 1 <= kDefaultNCtx);
    CHECK(big.n_tokens < tiny.n_tokens);
    CHECK(big.dropped > tiny.dropped);
}

TEST_CASE("fit_prompt: an oversized single message does not fit and says so") {
    const auto fmt = chat_format_for("lfm25-350m");
    const std::string huge(40000, 'x'); // ~10000 tokens
    const auto fit =
        fit_prompt(fmt, "sys", make_turns(3, 100), huge, kDefaultNCtx, 512, fake_count);
    CHECK_FALSE(fit.fits);
    CHECK(fit.dropped == 3); // every turn was dropped trying
    // The caller must not generate — but the rendered prompt is still returned so
    // the session can report the real numbers.
    CHECK(fit.prompt.find(huge) != std::string::npos);
}

TEST_CASE("fit_prompt: an oversized message with NO history still comes back rendered") {
    // The bisection's not-found path once returned an empty prompt here: with no
    // turns the exponential probe never runs, so nothing had rendered yet. A fresh
    // chat with one huge paste is exactly that shape — the console gate caught it
    // because the app then generated from an empty prompt instead of reporting
    // "prompt too long".
    const auto fmt = chat_format_for("qwen25-coder-0.5b");
    const std::string huge(40000, 'x');
    const auto fit = fit_prompt(fmt, "sys", {}, huge, 4096, 128, fake_count);
    CHECK_FALSE(fit.fits);
    CHECK(fit.dropped == 0);
    CHECK(fit.n_tokens > 4096);
    CHECK(fit.prompt.find(huge) != std::string::npos);
}

TEST_CASE("fit_prompt: the returned prompt is never empty") {
    const auto fmt = chat_format_for("lfm25-350m");
    for (int n_turns : {0, 1, 5, 40}) {
        for (const char* user : {"q", "x"}) {
            const auto fit =
                fit_prompt(fmt, "sys", make_turns(n_turns, 400), user, 512, 256, fake_count);
            CHECK_FALSE(fit.prompt.empty());
            CHECK(fit.n_tokens > 0);
        }
    }
}

TEST_CASE("fit_prompt: a larger context keeps more history") {
    const auto fmt = chat_format_for("qwen25-coder-0.5b");
    const auto small = fit_prompt(fmt, "sys", make_turns(40, 400), "q", 2048, 512, fake_count);
    const auto large = fit_prompt(fmt, "sys", make_turns(40, 400), "q", 4096, 512, fake_count);
    REQUIRE(small.fits);
    REQUIRE(large.fits);
    CHECK(large.dropped < small.dropped);
    CHECK(large.n_tokens > small.n_tokens);
    CHECK(large.n_tokens + 512 + 1 <= 4096);
}

TEST_CASE("fit_prompt: drops the MINIMUM number of turns, and counts O(log n) times") {
    const auto fmt = chat_format_for("lfm25-350m");
    const auto turns = make_turns(60, 300);
    int calls = 0;
    const auto counted = [&calls](const std::string& s) {
        ++calls;
        return fake_count(s);
    };
    const auto fit = fit_prompt(fmt, "sys", turns, "q", kDefaultNCtx, 512, counted);
    REQUIRE(fit.fits);
    REQUIRE(fit.dropped > 0);
    // Minimality: keeping one more turn must NOT fit, or the bisection overshot.
    const std::vector<ChatTurn> one_more(turns.begin() + fit.dropped - 1, turns.end());
    const std::string bigger = fmt.render_prompt("sys", one_more, "q");
    CHECK(fake_count(bigger) + 512 + 1 > kDefaultNCtx);
    // Bisection, not a walk: 60 turns must not cost 60 tokenizations.
    CHECK(calls <= 16);
}

TEST_CASE("fit_prompt: no counter means no verdict, not an unverified prompt") {
    const auto fmt = chat_format_for("lfm25-350m");
    const auto fit = fit_prompt(fmt, "sys", make_turns(2, 40), "hello", kDefaultNCtx, 512, nullptr);
    CHECK_FALSE(fit.fits);
    CHECK(fit.n_tokens == -1);
    CHECK(fit.dropped == 0);
}

TEST_CASE("fit_prompt: the rendered prompt is exactly what the format produces") {
    // No second rendering path may creep in: whatever survives the drop must be
    // byte-identical to render_prompt on the surviving turns (#169 pins the
    // system prefix byte-for-byte, and a context shift counts on it).
    const auto fmt = chat_format_for("lfm25-350m");
    auto turns = make_turns(3, 50);
    const auto fit = fit_prompt(fmt, "sys", turns, "last", kDefaultNCtx, 512, fake_count);
    REQUIRE(fit.fits);
    REQUIRE(fit.dropped == 0);
    CHECK(fit.prompt == fmt.render_prompt("sys", turns, "last"));
    CHECK(fit.prompt.compare(0, fmt.render_system_prefix("sys").size(),
                             fmt.render_system_prefix("sys")) == 0);
}
