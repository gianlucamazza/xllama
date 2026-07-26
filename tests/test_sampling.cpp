// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// #125: the CLI/bench surface and the GUI/API surface ran different samplers.
// These tests pin the things that made that possible.

#include <doctest/doctest.h>

#include <cstdlib>

#include "xllama/cli.h"
#include "xllama/inference.h"
#include "xllama/inference_params.h"
#include "xllama/sampling.h"
#include "xllama/session.h"

using namespace xllama;

TEST_CASE("sampling: the two surfaces start from identical defaults") {
    // InferenceParams drives run_inference (CLI, bench); GenerateParams drives
    // Session (GUI, LAN API). They are separate structs for good reasons — one
    // carries a model path and training flags, the other carries KV-reuse state
    // — but a user changing top_p in the GUI and passing --top-p on the CLI must
    // be configuring the same thing. Before #125 the CLI struct had no top_p at
    // all, so the two could not even be compared.
    const InferenceParams ip;
    const GenerateParams gp;

    CHECK(ip.temperature == gp.temperature);
    CHECK(ip.top_p == gp.top_p);
    CHECK(ip.top_k == gp.top_k);
    CHECK(ip.repetition_penalty == gp.repetition_penalty);
    CHECK(ip.seed == gp.seed);

    // Pin the values themselves too. Equality alone would still pass if someone
    // changed both structs together, which is exactly the silent behaviour
    // change this test exists to make visible.
    CHECK(ip.temperature == doctest::Approx(0.8f));
    CHECK(ip.top_p == doctest::Approx(0.9f));
    CHECK(ip.top_k == 40);
    CHECK(ip.repetition_penalty == doctest::Approx(1.1f));
}

TEST_CASE("sampling: both surfaces project to the same SamplingConfig") {
    InferenceParams ip;
    ip.temperature = 0.55f;
    ip.top_p = 0.8f;
    ip.top_k = 20;
    ip.repetition_penalty = 1.25f;
    ip.seed = 1234;

    GenerateParams gp;
    gp.temperature = 0.55f;
    gp.top_p = 0.8f;
    gp.top_k = 20;
    gp.repetition_penalty = 1.25f;
    gp.seed = 1234;

    const SamplingConfig a = ip.sampling();
    const SamplingConfig b = gp.sampling();

    CHECK(a.temperature == b.temperature);
    CHECK(a.top_p == b.top_p);
    CHECK(a.top_k == b.top_k);
    CHECK(a.repetition_penalty == b.repetition_penalty);
    CHECK(a.seed == b.seed);
}

TEST_CASE("sampling: temperature 0 is greedy, independently of the greedy flag") {
    // The full chain must not run at temperature 0. The repetition penalty
    // reweighs prompt tokens BEFORE the temp stage's argmax and can flip the top
    // token — observed on device, where LFM2.5 answered "User\n\n<|end|>" at
    // temperature 0 through the endpoint while pure argmax answered "Hello!".
    SamplingConfig sc;
    CHECK_FALSE(sc.is_greedy());

    sc.temperature = 0.0f;
    CHECK(sc.is_greedy());

    sc.temperature = -1.0f; // nonsense input must not fall through to sampling
    CHECK(sc.is_greedy());

    sc.temperature = 0.8f;
    sc.greedy = true;
    CHECK(sc.is_greedy());
}

TEST_CASE("sampling: same_chain gates the #175 persistent-chain reuse") {
    // Sampler state follows the KV lifecycle; a persistent chain may only be
    // reused while it would be assembled identically. Any stage parameter
    // change must force a rebuild — reusing across a mismatch would sample
    // with parameters the caller no longer holds.
    SamplingConfig a;
    SamplingConfig b;
    CHECK(same_chain(a, b));

    b.top_k = a.top_k + 1;
    CHECK_FALSE(same_chain(a, b));
    b = a;
    b.seed = a.seed + 1;
    CHECK_FALSE(same_chain(a, b));

    // Greedy chains have a single stage: two greedy configs match regardless
    // of the (unused) sampling values, and greedy never matches non-greedy —
    // including via temperature 0, which is greedy without the flag.
    b = a;
    a.greedy = b.greedy = true;
    b.top_p = 0.123f;
    CHECK(same_chain(a, b));
    b.greedy = false;
    CHECK_FALSE(same_chain(a, b));
    a.greedy = false;
    a.temperature = 0.0f;
    b.temperature = 0.8f;
    CHECK_FALSE(same_chain(a, b));
}

TEST_CASE("sampling: the CLI can express every value the GUI can") {
    // Regression for the concrete complaint in #125: a generation observed in
    // the GUI must be reproducible from the command line. Parse the flags and
    // check they land where run_inference reads them.
    InferenceParams p;
    const char* argv[] = {"xllama-cli",
                          "-m",
                          "model.gguf",
                          "-p",
                          "hi",
                          "--temp",
                          "0.55",
                          "--top-p",
                          "0.8",
                          "--top-k",
                          "20",
                          "--repetition-penalty",
                          "1.25",
                          "--seed",
                          "7",
                          "--system",
                          "You are a terse assistant."};
    REQUIRE(parse_cli_args(static_cast<int>(sizeof(argv) / sizeof(argv[0])),
                           const_cast<char**>(argv), p));

    CHECK(p.temperature == doctest::Approx(0.55f));
    CHECK(p.top_p == doctest::Approx(0.8f));
    CHECK(p.top_k == 20);
    CHECK(p.repetition_penalty == doctest::Approx(1.25f));
    CHECK(p.seed == 7u);
    CHECK(p.system_prompt == "You are a terse assistant.");
}

// Opt-in end-to-end parity. Needs a GGUF model:
//   XLLAMA_TEST_MODEL=/path/to/model.gguf ./xllama-tests
// The unit tests above prove the two surfaces hold the same VALUES and call the
// same chain builder. They cannot prove the two code paths feed it the same way
// — for that the only honest check is running both and comparing the tokens.
TEST_CASE("sampling: CLI and Session produce the same text (opt-in: XLLAMA_TEST_MODEL)") {
    const char* model_env = std::getenv("XLLAMA_TEST_MODEL");
    if (!model_env) {
        MESSAGE("XLLAMA_TEST_MODEL not set — skipping CLI/Session parity");
        return;
    }

    const std::string prompt = "The capital of France is";
    const float temperature = 0.9f;
    const float top_p = 0.8f;
    const int top_k = 20;
    const float rep = 1.3f;
    const uint32_t seed = 42;
    const int n_predict = 24;

    InferenceParams ip;
    ip.model_path = model_env;
    ip.prompt = prompt;
    ip.n_predict = n_predict;
    ip.temperature = temperature;
    ip.top_p = top_p;
    ip.top_k = top_k;
    ip.repetition_penalty = rep;
    ip.seed = seed;
    const InferenceResult cli = run_inference(ip);
    REQUIRE(cli.success);

    SessionParams sp;
    sp.model_path = model_env;
    std::string err;
    auto session = Session::create(sp, &err);
    REQUIRE_MESSAGE(session != nullptr, err);

    GenerateParams gp;
    gp.prompt = prompt;
    gp.n_predict = n_predict;
    gp.temperature = temperature;
    gp.top_p = top_p;
    gp.top_k = top_k;
    gp.repetition_penalty = rep;
    gp.seed = seed;
    const InferenceResult gui = session->generate(gp);
    REQUIRE(gui.success);

    // This is the complaint in #125 stated as an assertion: a generation seen on
    // one surface must be reproducible on the other.
    CHECK(cli.output_text == gui.output_text);
}
