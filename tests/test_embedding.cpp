// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#include "xllama/embedding.h"
#include "xllama/session.h"

#include <cmath>
#include <cstdlib>
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

TEST_CASE(
    "Session: non-causal embeddings handle long sequences (opt-in: XLLAMA_TEST_EMBED_MODEL)") {
    const char* model = std::getenv("XLLAMA_TEST_EMBED_MODEL");
    if (!model || !*model)
        return;
    xllama::SessionParams sp;
    sp.model_path = model;
    sp.n_ctx = 1024;
    sp.n_threads = 2;
    std::string err;
    auto session = xllama::Session::create(sp, &err);
    REQUIRE_MESSAGE(session != nullptr, err);
    xllama::EmbeddingParams params;
    params.input = "search_document: ";
    for (int i = 0; i < 200; ++i)
        params.input += "embedding text ";
    params.truncate = false;
    const auto result = session->embed(params);
    REQUIRE_MESSAGE(result.success, result.error_msg);
    CHECK(result.n_tokens > 512); // exceeds the chat path's physical microbatch
    CHECK(result.n_tokens <= 1024);
    double norm2 = 0;
    for (float value : result.embedding) {
        REQUIRE(std::isfinite(value));
        norm2 += static_cast<double>(value) * value;
    }
    CHECK(std::sqrt(norm2) == doctest::Approx(1.0).epsilon(1e-4));
    const auto repeated = session->embed(params);
    REQUIRE_MESSAGE(repeated.success, repeated.error_msg);
    REQUIRE(repeated.embedding.size() == result.embedding.size());
    for (size_t i = 0; i < result.embedding.size(); ++i)
        CHECK(repeated.embedding[i] == doctest::Approx(result.embedding[i]).epsilon(1e-5));

    params.input += params.input;
    params.input += params.input;
    CHECK_FALSE(session->embed(params).success);
    params.truncate = true;
    const auto truncated = session->embed(params);
    REQUIRE_MESSAGE(truncated.success, truncated.error_msg);
    CHECK(truncated.n_tokens == 1024);
}
