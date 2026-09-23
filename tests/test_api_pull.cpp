#include <doctest/doctest.h>

#include "xllama/api_pull_policy.h"

TEST_CASE("Ollama pull accepts trusted, pinned inference catalogue entries") {
    const xllama::PullModelDescriptor model{
        "embed-bge-m3", "gguf", "embedding", "https://models.example/bge", true,
        {std::string(64, 'a')}};

    CHECK(xllama::api_pull_model_allowed(model));
}

TEST_CASE("Ollama pull resolves documented embedding model aliases") {
    CHECK(xllama::resolve_pull_model_name("bge-m3") == "embed-bge-m3");
    CHECK(xllama::resolve_pull_model_name("nomic-embed-text-v2-moe") ==
          "embed-nomic-v2-moe");
    CHECK(xllama::resolve_pull_model_name("qwen3-embedding:4b") == "embed-qwen3-4b");
    CHECK(xllama::resolve_pull_model_name("my-private-repo/model") == "my-private-repo/model");
}

TEST_CASE("Ollama pull rejects untrusted, unpinned, unsupported, or unsafe catalogue entries") {
    const std::vector<std::string> valid_pins{std::string(64, 'a')};
    CHECK_FALSE(xllama::api_pull_model_allowed(
        {"m", "gguf", "embedding", "https://models.example/m", false, valid_pins}));
    CHECK_FALSE(xllama::api_pull_model_allowed(
        {"m", "gguf", "embedding", "https://models.example/m", true, {}}));
    CHECK_FALSE(xllama::api_pull_model_allowed(
        {"m", "gguf", "embedding", "http://models.example/m", true, valid_pins}));
    CHECK_FALSE(xllama::api_pull_model_allowed(
        {"m", "diffusion", "", "https://models.example/m", true, valid_pins}));
    CHECK_FALSE(xllama::api_pull_model_allowed(
        {"m", "gguf", "", "https://models.example/m", true, {std::string(63, 'a')}}));
    CHECK_FALSE(xllama::api_pull_model_allowed(
        {"m", "gguf", "", "https://models.example/m", true,
         {std::string(63, 'a') + "Z"}}));
    CHECK_FALSE(xllama::api_pull_model_allowed(
        {"m", "ort-genai", "embedding", "https://models.example/m", true, valid_pins}));
}

TEST_CASE("Ollama pull admission allows only one model download at a time") {
    xllama::ApiPullGate gate;
    auto active = gate.try_acquire();
    REQUIRE(active.owns_lock());
    CHECK_FALSE(gate.try_acquire().owns_lock());
    active.unlock();
    CHECK(gate.try_acquire().owns_lock());
}
