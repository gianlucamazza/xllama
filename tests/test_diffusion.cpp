// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Host unit tests for the diffusion pipeline's correctness-critical logic:
// the CLIP tokenizer and the Euler scheduler are asserted bit-for-value against
// golden vectors captured from the Python reference (diffusers 0.31.0 /
// transformers 4.46.3 — see diffusion/gen_golden_vectors.py). This is the
// anti-theater gate: the pure logic ships in the un-runtime-testable console
// pipeline (uwp/diffuse.cpp) only after matching the reference here.
//
// Paths are injected by CMake (GOLDEN_JSON_PATH, CLIP_TOKENIZER_DIR).

#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include <fstream>
#include <string>
#include <vector>

#include "xllama/diffusion/clip_tokenizer.h"
#include "xllama/diffusion/euler_scheduler.h"
#include "xllama/diffusion/half.h"
#include "xllama/diffusion/png_writer.h"

using nlohmann::json;

static json load_golden() {
    std::ifstream f(GOLDEN_JSON_PATH);
    REQUIRE_MESSAGE(f.good(), "golden_vectors.json not found at " GOLDEN_JSON_PATH
                              " — run diffusion/gen_golden_vectors.py");
    json j;
    f >> j;
    return j;
}

TEST_CASE("CLIP tokenizer matches diffusers golden ids") {
    const json g = load_golden();
    auto tok = xllama::diffusion::ClipTokenizer::from_files(
        std::string(CLIP_TOKENIZER_DIR) + "/vocab.json",
        std::string(CLIP_TOKENIZER_DIR) + "/merges.txt");

    CHECK(g["tokenizer"]["bos_id"].get<int>() == xllama::diffusion::ClipTokenizer::kBos);
    CHECK(g["tokenizer"]["eos_id"].get<int>() == xllama::diffusion::ClipTokenizer::kEos);

    for (const auto& c : g["tokenizer"]["cases"]) {
        const std::string prompt = c["prompt"].get<std::string>();
        const std::vector<int> want = c["input_ids"].get<std::vector<int>>();
        const std::vector<int> got = tok.encode(prompt);
        INFO("prompt = ", prompt);
        REQUIRE(got.size() == want.size());
        // The non-ASCII case exercises the documented \p{L} approximation; assert
        // it too — the >=0x80-is-letter heuristic reproduces Latin-accented words.
        CHECK(got == want);
    }
}

TEST_CASE("PNG writer emits a valid RGB stream") {
    // 2x2 image: red, green, blue, white.
    std::vector<uint8_t> rgb = {255, 0, 0, 0, 255, 0, 0, 0, 255, 255, 255, 255};
    auto png = xllama::diffusion::encode_png_rgb(2, 2, rgb);
    // PNG 8-byte signature.
    const uint8_t sig[8] = {0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A};
    REQUIRE(png.size() > 8);
    for (int i = 0; i < 8; ++i)
        CHECK(png[i] == sig[i]);
    // IHDR immediately follows (length 13, type "IHDR", width/height = 2).
    CHECK(png[12] == 'I');
    CHECK(png[13] == 'H');
    CHECK(png[16] == 0); // width high bytes
    CHECK(png[19] == 2); // width low byte
    CHECK(png[23] == 2); // height low byte
    // Ends with IEND.
    const char* end = "IEND";
    REQUIRE(png.size() >= 8);
    for (int i = 0; i < 4; ++i)
        CHECK((char)png[png.size() - 8 + i] == end[i]);
}

TEST_CASE("Euler scheduler matches diffusers golden schedule") {
    const json g = load_golden();
    const auto& gs = g["scheduler"];

    xllama::diffusion::EulerDiscreteScheduler sched;
    sched.set_timesteps(gs["num_inference_steps"].get<int>());

    const auto want_sig = gs["sigmas"].get<std::vector<double>>();
    const auto want_ts = gs["timesteps"].get<std::vector<double>>();
    REQUIRE(sched.sigmas().size() == want_sig.size());
    for (size_t i = 0; i < want_sig.size(); ++i)
        CHECK(sched.sigmas()[i] == doctest::Approx(want_sig[i]).epsilon(1e-4));
    REQUIRE(sched.timesteps().size() == want_ts.size());
    for (size_t i = 0; i < want_ts.size(); ++i)
        CHECK(sched.timesteps()[i] == doctest::Approx(want_ts[i]));
    CHECK(sched.init_noise_sigma() ==
          doctest::Approx(gs["init_noise_sigma"].get<double>()).epsilon(1e-4));
}

TEST_CASE("half <-> float conversion") {
    using xllama::diffusion::float_to_half;
    using xllama::diffusion::half_to_float;
    // Exact values representable in fp16.
    CHECK(half_to_float(float_to_half(0.0f)) == 0.0f);
    CHECK(half_to_float(float_to_half(1.0f)) == 1.0f);
    CHECK(half_to_float(float_to_half(-2.5f)) == -2.5f);
    CHECK(half_to_float(float_to_half(0.5f)) == 0.5f);
    // Known bit patterns.
    CHECK(float_to_half(1.0f) == 0x3C00);
    CHECK(float_to_half(-1.0f) == 0xBC00);
    CHECK(float_to_half(2.0f) == 0x4000);
    // Round-trip within fp16 precision over a range.
    for (float v = -14.6f; v <= 14.6f; v += 0.37f) {
        float back = half_to_float(float_to_half(v));
        CHECK(back == doctest::Approx(v).epsilon(1e-2));
    }
}

TEST_CASE("Euler scale_model_input + step match golden step case") {
    const json g = load_golden();
    const auto& sc = g["scheduler"]["step_case"];

    auto sample = sc["sample"].get<std::vector<float>>();
    auto eps = sc["model_output"].get<std::vector<float>>();
    const auto want_scaled = sc["scaled_model_input"].get<std::vector<float>>();
    const auto want_prev = sc["prev_sample"].get<std::vector<float>>();

    xllama::diffusion::EulerDiscreteScheduler sched;
    sched.set_timesteps(g["scheduler"]["num_inference_steps"].get<int>());

    // scale_model_input operates on a copy of the sample at the current step.
    std::vector<float> scaled = sample;
    sched.scale_model_input(scaled);
    REQUIRE(scaled.size() == want_scaled.size());
    for (size_t i = 0; i < scaled.size(); ++i)
        CHECK(scaled[i] == doctest::Approx(want_scaled[i]).epsilon(1e-4));

    // step() updates the (unscaled) sample in place to the previous sample.
    std::vector<float> x = sample;
    sched.step(eps, x);
    REQUIRE(x.size() == want_prev.size());
    for (size_t i = 0; i < x.size(); ++i)
        CHECK(x[i] == doctest::Approx(want_prev[i]).epsilon(1e-3));
}
