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

#include <cmath>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "xllama/diffusion/clip_tokenizer.h"
#include "xllama/diffusion/euler_scheduler.h"
#include "xllama/diffusion/half.h"
#include "xllama/diffusion/lcm_scheduler.h"
#include "xllama/diffusion/png_writer.h"
#include "xllama/diffusion/request.h"
#include "xllama/diffusion/residency.h"

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

TEST_CASE("LCM knob is off unless the file is exactly 1") {
    using xllama::diffusion::lcm_scheduler_requested;
    CHECK_FALSE(lcm_scheduler_requested(""));
    CHECK_FALSE(lcm_scheduler_requested("0"));
    CHECK_FALSE(lcm_scheduler_requested("2"));
    CHECK_FALSE(lcm_scheduler_requested("lcm"));
    CHECK_FALSE(lcm_scheduler_requested(" 1"));
    CHECK(lcm_scheduler_requested("1"));
    CHECK(lcm_scheduler_requested("1 "));
    CHECK(lcm_scheduler_requested("1\n"));
    CHECK(lcm_scheduler_requested("1\r\n"));
}

TEST_CASE("LCM off keeps the Euler golden schedule") {
    REQUIRE_FALSE(xllama::diffusion::lcm_scheduler_requested(""));
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
}

TEST_CASE("LCM origin skipping table") {
    xllama::diffusion::LCMScheduler sched;
    const struct {
        int n;
        std::vector<double> ts;
    } cases[] = {
        {1, {999}},
        {2, {999, 499}},
        {4, {999, 759, 519, 279}},
        {8, {999, 879, 759, 639, 519, 399, 279, 159}},
    };
    for (const auto& c : cases) {
        sched.set_timesteps(c.n);
        REQUIRE(sched.timesteps().size() == c.ts.size());
        for (size_t i = 0; i < c.ts.size(); ++i)
            CHECK(sched.timesteps()[i] == doctest::Approx(c.ts[i]));
        CHECK(sched.init_noise_sigma() == doctest::Approx(1.0));
    }
}

TEST_CASE("LCM one-step denoised sample") {
    xllama::diffusion::LCMScheduler sched;
    sched.set_timesteps(1);
    std::vector<float> x = {1.0f, 0.0f, -1.0f, 0.5f};
    const std::vector<float> eps = {0.1f, -0.2f, 0.3f, -0.1f};
    std::vector<float> scaled = x;
    sched.scale_model_input(scaled);
    CHECK(scaled == x); // identity
    sched.step(eps, x);
    const double want[] = {13.187349407575335, 2.922928242205751, -19.03320589198684,
                           8.785870885441982};
    REQUIRE(x.size() == 4);
    for (size_t i = 0; i < 4; ++i)
        CHECK(x[i] == doctest::Approx(want[i]).epsilon(1e-5));
    CHECK(sched.step_index() == 1);
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

TEST_CASE("diffusion request: interactive steps are clamped to the SD-Turbo range") {
    using namespace xllama::diffusion;
    CHECK(interactive_steps(1) == 1);
    CHECK(interactive_steps(4) == kStepsInteractiveMax);
    CHECK(interactive_steps(9) == kStepsInteractiveMax);
    CHECK(interactive_steps(0) == kStepsMin);
    CHECK(interactive_steps(-3) == kStepsMin);
    CHECK(interactive_steps(2.9) == 2);
    CHECK(interactive_steps(std::nan("")) == kStepsMin);
    CHECK(interactive_steps(1e300) == kStepsInteractiveMax); // no UB double->int cast
}

TEST_CASE("diffusion request: a seed is any uint32, negative keeps the default") {
    using namespace xllama::diffusion;
    CHECK(request_seed(0) == 0u);
    CHECK(request_seed(42) == 42u);
    CHECK(request_seed(3000000000.0) == 3000000000u); // above INT_MAX and 1e9
    CHECK(request_seed(4294967295.0) == 4294967295u);
    CHECK(request_seed(1e12) == 4294967295u);
    CHECK(request_seed(-1) == kDefaultSeed);
    CHECK(request_seed(std::nan("")) == kDefaultSeed);
}

TEST_CASE("diffusion request: diffuse-*.txt round-trips what the surfaces write") {
    using namespace xllama::diffusion;
    CHECK(steps_from_knob(std::to_string(3)) == 3);
    CHECK(steps_from_knob("50\n") == kStepsMax);
    CHECK(steps_from_knob("51") == kDefaultSteps);
    CHECK(steps_from_knob("") == kDefaultSteps);
    CHECK(steps_from_knob("two") == kDefaultSteps);
    // Every seed the GUI and API accept must survive the file: 1e9 used to be
    // the ceiling here, so a larger seed silently became 42.
    CHECK(seed_from_knob(std::to_string(3000000000u)) == 3000000000u);
    CHECK(seed_from_knob(std::to_string(request_seed(4294967295.0))) == 4294967295u);
    CHECK(seed_from_knob("4294967296") == kDefaultSeed);
    CHECK(seed_from_knob("-1") == kDefaultSeed);
    CHECK(seed_from_knob("") == kDefaultSeed);
}

namespace {
struct StubSession final : xllama::Session {
    xllama::InferenceResult generate(const xllama::GenerateParams&) override {
        return {};
    }
    int count_tokens(const std::string&) override {
        return 0;
    }
};
} // namespace

TEST_CASE("diffusion residency: release the chat session only when the pipeline would not fit") {
    using xllama::diffusion::diffusion_needs_session_release;
    using xllama::diffusion::kDiffusionHeadroomMb;
    constexpr std::size_t kLimit = 5120; // Series S, Game designation
    // Working sets after context from the 2026-11-25 Series S receipt.
    CHECK_FALSE(diffusion_needs_session_release(345, kLimit));  // lfm25-350m: generated
    CHECK_FALSE(diffusion_needs_session_release(1838, kLimit)); // llama32-3b: generated
    CHECK(diffusion_needs_session_release(3574, kLimit));       // lfm25-8b-a1b: bad allocation
    CHECK_FALSE(diffusion_needs_session_release(kLimit - kDiffusionHeadroomMb, kLimit));
    CHECK(diffusion_needs_session_release(kLimit - kDiffusionHeadroomMb + 1, kLimit));
    CHECK_FALSE(diffusion_needs_session_release(100000, 0)); // no known limit (host)
}

TEST_CASE("diffusion residency: make_room releases through the hub and bumps its generation") {
    using xllama::diffusion::make_room_for_diffusion_locked;
    xllama::SessionHub hub;
    CHECK_FALSE(make_room_for_diffusion_locked(hub, 4000, 5120)); // nothing resident
    hub.session = std::make_unique<StubSession>();
    hub.model = "lfm25-8b-a1b";
    const auto gen = hub.generation;
    CHECK_FALSE(make_room_for_diffusion_locked(hub, 1000, 5120)); // fits: keep it
    CHECK(hub.session);
    CHECK(make_room_for_diffusion_locked(hub, 3574, 5120));
    CHECK_FALSE(hub.session);
    CHECK(hub.model.empty());
    CHECK(hub.generation == gen + 1);
}
