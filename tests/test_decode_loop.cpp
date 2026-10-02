// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// src/bridge/decode_loop.h end to end: the bench-only ignore_eog contract.

#include <doctest/doctest.h>

#include <cstdlib>

#include "xllama/inference.h"
#include "xllama/inference_params.h"

using namespace xllama;

// Opt-in: XLLAMA_TEST_MODEL=/path/to/model.gguf ./xllama-tests
// A short-answer prompt reaches end-of-generation well before n_predict; with
// ignore_eog every run must still decode exactly n_predict tokens, with prompt
// lookup off and on (D2b A/Bs rely on it). An EOG inside a draft cannot be
// forced with a real model; detail::stops_at_eog is the single decision every
// branch of decode_loop shares, so the branches cannot drift apart.
TEST_CASE("decode_loop: ignore_eog decodes exactly n_predict (opt-in: XLLAMA_TEST_MODEL)") {
    const char* model_env = std::getenv("XLLAMA_TEST_MODEL");
    if (!model_env) {
        MESSAGE("XLLAMA_TEST_MODEL not set — skipping ignore_eog");
        return;
    }
    for (const bool lookup : {false, true}) {
        CAPTURE(lookup);
        InferenceParams ip;
        ip.model_path = model_env;
        ip.prompt = "Reply with the single word OK.";
        ip.n_predict = 48;
        ip.temperature = 0.0f;
        ip.prompt_lookup = lookup;
        ip.ignore_eog = true;
        ip.stop_sequences.clear();
        const InferenceResult r = run_inference(ip);
        REQUIRE(r.success);
        CHECK(r.n_eval == ip.n_predict);
        CHECK_FALSE(r.ended_with_stop);
    }
}
