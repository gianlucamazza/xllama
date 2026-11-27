// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "xllama/cli.h"
#include <iostream>
#include <sstream>
#include <utility>

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
    const char* argv[] = {"xllama-cli",  "--model", "m.gguf", "--prompt", "hi",
                          "--n-predict", "64",      "--ctx",  "1024",     "--threads",
                          "4",           "--temp",  "0.5",    "--seed",   "42"};
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

TEST_CASE("CLI parsing: progress and stall flags") {
    const char* argv[] = {"xllama-cli", "-m",
                          "m.gguf",     "-p",
                          "hi",         "--decode-progress-every",
                          "16",         "--decode-progress-ms",
                          "5000",       "--infer-stall-timeout",
                          "90"};
    xllama::InferenceParams params;
    CHECK(xllama::parse_cli_args(11, const_cast<char**>(argv), params));
    CHECK(params.decode_log_every_n == 16);
    CHECK(params.decode_log_every_ms == 5000);
    CHECK(params.stall_timeout_s == 90);
}

TEST_CASE("CLI parsing: --prompt-lookup defaults off and opts in") {
    {
        const char* argv[] = {"xllama-cli", "-m", "m.gguf", "-p", "hi"};
        xllama::InferenceParams params;
        CHECK(xllama::parse_cli_args(5, const_cast<char**>(argv), params));
        CHECK_FALSE(params.prompt_lookup);
    }
    {
        const char* argv[] = {"xllama-cli", "-m", "m.gguf", "-p", "hi", "--prompt-lookup"};
        xllama::InferenceParams params;
        CHECK(xllama::parse_cli_args(6, const_cast<char**>(argv), params));
        CHECK(params.prompt_lookup);
    }
}

namespace {
bool parse_option(const char* option, const char* value, xllama::InferenceParams& params) {
    const char* argv[] = {"xllama-cli", "--training-capabilities", option, value, nullptr};
    return xllama::parse_cli_args(4, const_cast<char**>(argv), params);
}
} // namespace

TEST_CASE("CLI parsing: malformed numeric arguments identify the option") {
    // A mode that consumes no model still validates all options before dispatch.
    const char* numeric_options[] = {
        "--n-predict",
        "--ctx",
        "--threads",
        "--temp",
        "--seed",
        "--batch",
        "--ubatch",
        "--lora-scale",
        "--top-p",
        "--top-k",
        "--repetition-penalty",
        "--dimensions",
        "--gpu-layers",
        "--decode-progress-every",
        "--decode-progress-ms",
        "--infer-stall-timeout",
        "--dump-logits-stride",
        "-c",
        "-n",
        "-t",
    };
    for (const char* option : numeric_options) {
        for (const char* value :
             {"", "not-a-number", "64typo", "+-2", "99999999999999999999999999999999999999999"}) {
            CAPTURE(option);
            CAPTURE(value);
            xllama::InferenceParams params;
            std::ostringstream diagnostic;
            auto* previous = std::cerr.rdbuf(diagnostic.rdbuf());
            const bool accepted = parse_option(option, value, params);
            std::cerr.rdbuf(previous);
            CHECK_FALSE(accepted);
            CHECK(diagnostic.str().find(option) != std::string::npos);
        }
    }
}

TEST_CASE("CLI parsing: representability and option-specific domains") {
    xllama::InferenceParams params;
    for (const char* option : {"--temp", "--lora-scale", "--top-p", "--repetition-penalty"}) {
        CAPTURE(option);
        for (const char* value : {"nan", "inf", "-inf", "1e100", "1e-100"}) {
            CAPTURE(value);
            CHECK_FALSE(parse_option(option, value, params));
        }
    }
    for (const char* option : {"--n-predict", "--ctx", "--threads", "--batch", "--ubatch",
                               "--dimensions", "--gpu-layers", "--decode-progress-every",
                               "--decode-progress-ms", "--infer-stall-timeout"}) {
        CAPTURE(option);
        CHECK(parse_option(option, "0", params));
        CHECK_FALSE(parse_option(option, "-1", params));
        CHECK_FALSE(parse_option(option, "1.5", params));
    }
    CHECK_FALSE(parse_option("--top-p", "-0.1", params));
    CHECK_FALSE(parse_option("--top-p", "1.1", params));
    CHECK_FALSE(parse_option("--top-p", "1.00000001", params));
    CHECK_FALSE(parse_option("--temp", "2.00000001", params));
    CHECK_FALSE(parse_option("--repetition-penalty", "2.00000001", params));
    CHECK_FALSE(parse_option("--dump-logits-stride", "0", params));
    CHECK(parse_option("--dump-logits-stride", "1", params));
    CHECK(parse_option("--threads", "512", params));
    CHECK_FALSE(parse_option("--threads", "513", params));
    CHECK_FALSE(parse_option("--seed", "-2147483649", params));
    CHECK_FALSE(parse_option("--seed", "4294967296", params));
}

TEST_CASE("CLI parsing: sampling options take the LAN API bounds") {
    // One table (api_sampling.h) for the CLI, a request and the catalogue.
    xllama::InferenceParams params;
    for (const auto& [option, value] :
         {std::pair{"--temp", "-1"}, std::pair{"--temp", "2.5"}, std::pair{"--top-k", "-1"},
          std::pair{"--top-k", "1.5"}, std::pair{"--repetition-penalty", "5"}}) {
        CAPTURE(option);
        CAPTURE(value);
        std::ostringstream diagnostic;
        auto* previous = std::cerr.rdbuf(diagnostic.rdbuf());
        CHECK_FALSE(parse_option(option, value, params));
        std::cerr.rdbuf(previous);
        // The message is the API's, e.g. "temperature must be between 0 and 2".
        CHECK(diagnostic.str().find("must be") != std::string::npos);
    }
}

TEST_CASE("CLI parsing: sampling semantics and endpoint values survive strict conversion") {
    xllama::InferenceParams params;
    CHECK(parse_option("--seed", "-1", params));
    CHECK(params.seed == xllama::sampling_defaults::kSeed);
    CHECK(parse_option("--seed", "4294967295", params));
    CHECK(params.seed == UINT32_MAX);
    CHECK(parse_option("--seed", "-2", params));
    CHECK(params.seed == UINT32_MAX - 1);
    CHECK(parse_option("--seed", "0", params));
    CHECK(params.seed == 0);
    CHECK(parse_option("--lora-scale", "-1.5", params));
    CHECK(params.lora_scale == -1.5f);
    CHECK(parse_option("--temp", "+1.5e0", params));
    CHECK(params.temperature == 1.5f);
    CHECK(parse_option("--temp", "0", params)); // argmax, as on the API
    CHECK(params.sampling().is_greedy());
    CHECK(parse_option("--top-p", "0", params));
    CHECK(params.top_p == 0);
    CHECK(parse_option("--top-p", "1", params));
    CHECK(params.top_p == 1);
    CHECK(parse_option("--top-k", "0", params)); // 0 keeps every token
    CHECK(params.top_k == 0);
    CHECK(parse_option("--repetition-penalty", "0", params));
    CHECK(params.repetition_penalty == 0);
    CHECK(parse_option("-n", "+2147483647", params));
    CHECK(params.n_predict == INT32_MAX);
}
