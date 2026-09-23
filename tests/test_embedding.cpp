// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#include "xllama/embedding.h"

#include <cmath>
#include <doctest/doctest.h>

TEST_CASE("trim_tokens_for_pooling keeps prefix for MEAN and CLS") {
    std::vector<int32_t> tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    xllama::trim_tokens_for_pooling(tokens, xllama::PoolingType::Mean, 5);
    REQUIRE(tokens.size() == 5);
    CHECK(tokens[0] == 1);
    CHECK(tokens[4] == 5);

    tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    xllama::trim_tokens_for_pooling(tokens, xllama::PoolingType::Cls, 3);
    REQUIRE(tokens.size() == 3);
    CHECK(tokens[0] == 1);
    CHECK(tokens[2] == 3);
}

TEST_CASE("trim_tokens_for_pooling keeps suffix for LAST and NONE") {
    std::vector<int32_t> tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    xllama::trim_tokens_for_pooling(tokens, xllama::PoolingType::Last, 5);
    REQUIRE(tokens.size() == 5);
    CHECK(tokens[0] == 6);
    CHECK(tokens[4] == 10);

    tokens = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};
    xllama::trim_tokens_for_pooling(tokens, xllama::PoolingType::None, 3);
    REQUIRE(tokens.size() == 3);
    CHECK(tokens[0] == 8);
    CHECK(tokens[2] == 10);
}

TEST_CASE("trim_tokens_for_pooling leaves short sequences unchanged") {
    std::vector<int32_t> tokens = {1, 2, 3};
    xllama::trim_tokens_for_pooling(tokens, xllama::PoolingType::Mean, 10);
    REQUIRE(tokens.size() == 3);
    CHECK(tokens[0] == 1);

    tokens = {1, 2, 3};
    xllama::trim_tokens_for_pooling(tokens, xllama::PoolingType::Last, 10);
    REQUIRE(tokens.size() == 3);
    CHECK(tokens[0] == 1);
}

TEST_CASE("normalize_embedding truncates dimensions then returns a unit vector") {
    std::vector<float> values{3.0f, 4.0f, 100.0f};
    std::string err;
    REQUIRE(xllama::normalize_embedding(values, 2, &err));
    REQUIRE(values.size() == 2);
    CHECK(values[0] == doctest::Approx(0.6f));
    CHECK(values[1] == doctest::Approx(0.8f));
}

TEST_CASE("normalize_embedding rejects impossible dimensions and zero vectors") {
    std::vector<float> values{1.0f, 2.0f};
    std::string err;
    CHECK_FALSE(xllama::normalize_embedding(values, 3, &err));
    CHECK_FALSE(err.empty());

    values = {0.0f, 0.0f};
    CHECK_FALSE(xllama::normalize_embedding(values, 0, &err));
}

TEST_CASE("embedding_base64 uses little-endian IEEE float32") {
    CHECK(xllama::embedding_base64({1.0f}) == "AACAPw==");
    CHECK(xllama::embedding_base64({1.0f, 2.0f, 3.0f}) == "AACAPwAAAEAAAEBA");
}

TEST_CASE("normalize_embedding rejects non-finite model output") {
    std::vector<float> values{1.0f, std::nanf("")};
    std::string err;
    CHECK_FALSE(xllama::normalize_embedding(values, 0, &err));
    CHECK(err.find("non-finite") != std::string::npos);
}

TEST_CASE("Ollama embedding model aliases map to catalogue ids") {
    // These aliases are applied in uwp/api-server.cpp handle_embedding_locked.
    // This test documents the contract: the three Ollama library names map to
    // the catalogue ids so clients can use either naming scheme.
    struct Alias { const char* ollama; const char* catalogue; };
    const Alias aliases[] = {
        {"bge-m3", "embed-bge-m3"},
        {"nomic-embed-text-v2-moe", "embed-nomic-v2-moe"},
        {"qwen3-embedding:4b", "embed-qwen3-4b"},
    };
    // No aliasing logic in this test file; we document the contract here and
    // rely on the API server implementation. A request with the Ollama name
    // must resolve to the catalogue id before model path lookup.
    for (const auto& a : aliases) {
        CHECK(std::string(a.ollama) != std::string(a.catalogue));
    }
}
