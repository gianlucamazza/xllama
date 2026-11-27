// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#include "xllama/app_settings.h"

#include <doctest/doctest.h>

#include <algorithm>
#include <string>

using xllama::AppSettings;
using xllama::format_app_settings;
using xllama::parse_app_settings;

namespace {
bool rejected(const xllama::AppSettingsLoad& r, const std::string& key) {
    return std::find(r.rejected.begin(), r.rejected.end(), key) != r.rejected.end();
}
} // namespace

TEST_CASE("app settings: a file written by the previous serializer reads back the same") {
    // Byte-for-byte the shape SaveSettings produced before this model existed.
    const std::string legacy = "{\n"
                               "  \"system_prompt\": \"Be brief.\\nAnswer in \\\"English\\\".\",\n"
                               "  \"model\": \"lfm25-350m\",\n"
                               "  \"kv_reuse\": false,\n"
                               "  \"debug_dashboard\": true,\n"
                               "  \"routing\": 2,\n"
                               "  \"gpu_model\": \"smollm2-360m-dml-fp16-v2\",\n"
                               "  \"diffuse_taesd_vae\": true,\n"
                               "  \"diffuse_seed\": 3000000000,\n"
                               "  \"sampling\": {\n"
                               "    \"temperature\": 0.70,\n"
                               "    \"top_p\": 0.95,\n"
                               "    \"top_k\": 50,\n"
                               "    \"repetition_penalty\": 1.15,\n"
                               "    \"n_predict\": 1024\n"
                               "  }\n"
                               "}\n";
    const auto r = parse_app_settings(legacy);
    CHECK(r.error.empty());
    CHECK(r.rejected.empty());
    const AppSettings& s = r.settings;
    CHECK(s.system_prompt == "Be brief.\nAnswer in \"English\".");
    CHECK(s.model == "lfm25-350m");
    CHECK_FALSE(s.kv_reuse);
    CHECK(s.debug_dashboard);
    CHECK(s.routing == 2);
    CHECK(s.diffuse_taesd_vae);
    CHECK(s.diffuse_seed == 3000000000u);
    // Values other than the old writer's defaults were the user's choice.
    CHECK(*s.sampling.temperature == doctest::Approx(0.70f));
    CHECK(*s.sampling.top_p == doctest::Approx(0.95f));
    CHECK(*s.sampling.top_k == 50);
    CHECK(*s.sampling.repetition_penalty == doctest::Approx(1.15f));
    CHECK(s.n_predict == 1024);
}

TEST_CASE("app settings: a pre-schema file's default sampling values inherit the model's") {
    // The old writer saved all four whether the user chose them or not.
    const auto r = parse_app_settings(R"({"sampling": {"temperature": 0.8, "top_p": 0.9,
                                          "top_k": 40, "repetition_penalty": 1.1,
                                          "n_predict": 512}})");
    CHECK(r.rejected.empty());
    CHECK(r.settings.sampling.empty());
    // Mixed: the changed value stays the user's, the untouched ones inherit.
    const auto mixed = parse_app_settings(R"({"sampling": {"temperature": 0.3, "top_p": 0.9,
                                              "top_k": 40, "repetition_penalty": 1.1}})");
    CHECK(*mixed.settings.sampling.temperature == doctest::Approx(0.3f));
    CHECK_FALSE(mixed.settings.sampling.top_p);
    CHECK_FALSE(mixed.settings.sampling.top_k);
    CHECK_FALSE(mixed.settings.sampling.repetition_penalty);
}

TEST_CASE("app settings: from schema 2 a value equal to the global default is still the user's") {
    const auto r = parse_app_settings(R"({"schema": 2, "sampling": {"temperature": 0.8}})");
    REQUIRE(r.settings.sampling.temperature);
    CHECK(*r.settings.sampling.temperature == doctest::Approx(0.8f));
    CHECK_FALSE(r.settings.sampling.top_k);
}

TEST_CASE("app settings: the writer stores only the user's sampling values") {
    AppSettings s;
    s.sampling.top_k = 64;
    const std::string text = format_app_settings(s);
    CHECK(text.find("\"schema\": 2") != std::string::npos);
    CHECK(text.find("\"top_k\": 64") != std::string::npos);
    CHECK(text.find("temperature") == std::string::npos);
    CHECK(text.find("repetition_penalty") == std::string::npos);
    CHECK(parse_app_settings(text).settings == s);
}

TEST_CASE("app settings: round trip keeps every field, floats included") {
    AppSettings s;
    s.system_prompt = "Prompt with \"quotes\", a \\ backslash, a tab\t, and unicode: café 漢字 🚀";
    s.model = "qwen25-coder-1.5b";
    s.kv_reuse = false;
    s.debug_dashboard = true;
    s.routing = 1;
    s.gpu_model = "other-gpu-model";
    s.diffuse_taesd_vae = true;
    s.diffuse_seed = 4294967295u;
    s.sampling.temperature = 0.925f; // the old %.2f writer saved this as 0.93
    s.sampling.top_p = 0.333f;
    s.sampling.top_k = 7;
    s.sampling.repetition_penalty = 1.05f;
    s.n_predict = 2048;
    const std::string text = format_app_settings(s);
    CHECK(text.back() == '\n');
    const auto r = parse_app_settings(text);
    CHECK(r.error.empty());
    CHECK(r.rejected.empty());
    CHECK(r.settings == s);
    // Human-readable: the shortest form of each float, not its double expansion.
    CHECK(text.find("0.925") != std::string::npos);
    CHECK(text.find("0.92500") == std::string::npos);
}

TEST_CASE("app settings: missing keys keep the defaults") {
    const auto r = parse_app_settings("{}");
    CHECK(r.error.empty());
    CHECK(r.rejected.empty());
    CHECK(r.settings == AppSettings{});
    CHECK(r.settings.model.empty()); // caller falls back to model.txt / default id
}

TEST_CASE("app settings: malformed JSON gives the defaults and an error, never throws") {
    for (const char* bad : {"", "not json", "{\"routing\": 2,", "[1,2,3]", "\"text\"", "null"}) {
        const auto r = parse_app_settings(bad);
        CHECK_FALSE(r.error.empty());
        CHECK(r.settings == AppSettings{});
    }
}

TEST_CASE("app settings: a wrong type or out-of-range value rejects only that field") {
    const auto r = parse_app_settings(R"({
        "routing": 7,
        "kv_reuse": "yes",
        "diffuse_seed": -1,
        "model": 42,
        "system_prompt": "kept",
        "sampling": {"temperature": 9.5, "top_p": 0.5, "top_k": 0, "n_predict": 1e9,
                     "repetition_penalty": "high"}
    })");
    CHECK(r.error.empty());
    const AppSettings d;
    CHECK(r.settings.routing == d.routing);
    CHECK(r.settings.kv_reuse == d.kv_reuse);
    CHECK(r.settings.diffuse_seed == d.diffuse_seed);
    CHECK(r.settings.model == d.model);
    CHECK(r.settings.system_prompt == "kept");
    CHECK_FALSE(r.settings.sampling.temperature); // rejected: inherits
    CHECK(*r.settings.sampling.top_p == doctest::Approx(0.5f));
    CHECK_FALSE(r.settings.sampling.top_k);
    CHECK(r.settings.n_predict == d.n_predict);
    CHECK_FALSE(r.settings.sampling.repetition_penalty);
    for (const char* key : {"routing", "kv_reuse", "diffuse_seed", "model", "sampling.temperature",
                            "sampling.top_k", "sampling.n_predict", "sampling.repetition_penalty"})
        CHECK_MESSAGE(rejected(r, key), key);
    CHECK_FALSE(rejected(r, "sampling.top_p"));
}

TEST_CASE("app settings: the legacy 1/0 booleans still read") {
    const auto r = parse_app_settings(R"({"kv_reuse": 0, "debug_dashboard": 1,
                                          "diffuse_taesd_vae": 1})");
    CHECK(r.rejected.empty());
    CHECK_FALSE(r.settings.kv_reuse);
    CHECK(r.settings.debug_dashboard);
    CHECK(r.settings.diffuse_taesd_vae);
}

TEST_CASE("app settings: unknown keys are ignored at any depth") {
    // The old token scanner stopped at the first space of an unknown value and
    // then read fragments of it as keys.
    const auto r = parse_app_settings(R"({
        "future_note": "a value with spaces, commas, and \"routing\": 1 inside",
        "future_obj": {"routing": 1, "nested": [1, {"top_k": 3}]},
        "routing": 2
    })");
    CHECK(r.error.empty());
    CHECK(r.rejected.empty());
    CHECK(r.settings.routing == 2);
    CHECK_FALSE(r.settings.sampling.top_k);
}

TEST_CASE("app settings: integral floats and fractional ints") {
    const auto r = parse_app_settings(R"({"routing": 1.0, "sampling": {"temperature": 1,
                                          "top_k": 40.5}})");
    CHECK(r.settings.routing == 1); // a whole number written as 1.0 is accepted
    CHECK(*r.settings.sampling.temperature == doctest::Approx(1.0f));
    CHECK_FALSE(r.settings.sampling.top_k); // 40.5 is not an integer
    CHECK(rejected(r, "sampling.top_k"));
}
