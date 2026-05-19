#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include <doctest/doctest.h>

#include "xllama/cli.h"

TEST_CASE("CLI parsing: minimal args") {
    const char* argv[] = {"xllama-cli", "-m", "model.gguf", "-p", "hello"};
    int argc = 5;
    xllama::InferenceParams params;
    CHECK(xllama::parse_cli_args(argc, const_cast<char**>(argv), params));
    CHECK(params.model_path == "model.gguf");
    CHECK(params.prompt == "hello");
    CHECK(params.n_predict == 128);
    CHECK(params.n_ctx == 2048);
}

TEST_CASE("CLI parsing: all optional args") {
    const char* argv[] = {
        "xllama-cli",
        "--model", "m.gguf",
        "--prompt", "hi",
        "--n-predict", "64",
        "--ctx", "1024",
        "--threads", "4",
        "--temp", "0.5",
        "--seed", "42"
    };
    int argc = 15;
    xllama::InferenceParams params;
    CHECK(xllama::parse_cli_args(argc, const_cast<char**>(argv), params));
    CHECK(params.model_path == "m.gguf");
    CHECK(params.prompt == "hi");
    CHECK(params.n_predict == 64);
    CHECK(params.n_ctx == 1024);
    CHECK(params.n_threads == 4);
    CHECK(params.temperature == doctest::Approx(0.5f));
    CHECK(params.seed == 42);
}

TEST_CASE("CLI parsing: missing model fails") {
    const char* argv[] = {"xllama-cli", "-p", "hello"};
    int argc = 3;
    xllama::InferenceParams params;
    CHECK_FALSE(xllama::parse_cli_args(argc, const_cast<char**>(argv), params));
}

TEST_CASE("CLI parsing: missing prompt fails") {
    const char* argv[] = {"xllama-cli", "-m", "model.gguf"};
    int argc = 3;
    xllama::InferenceParams params;
    CHECK_FALSE(xllama::parse_cli_args(argc, const_cast<char**>(argv), params));
}
