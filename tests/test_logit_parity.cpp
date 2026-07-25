// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Logit-parity regression: re-runs the llama.cpp backend on a fixed prompt and
// asserts its last-prefill-token logits still match a committed golden dump
// (tests/golden/, produced by `xllama-cli --greedy --dump-logits`). This guards
// the reference backend + tokenizer + quant path against silent numeric drift
// in vendored llama.cpp or our sampler setup — the same distribution that the
// on-device ORT/DirectML dump is later compared against (scripts/compare-logits.py).
//
// Opt-in: needs the golden's source model. Point XLLAMA_TEST_MODEL at the GGUF
// (or dir) used to generate the golden; the test SKIPs cleanly when unset, so
// CI stays green without shipping model weights. Paths injected by CMake:
//   PARITY_GOLDEN_BIN  — committed reference dump (its .json sidecar has the prompt)

#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "xllama/inference.h"

namespace fs = std::filesystem;

static std::vector<float> read_f32(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f.good())
        return {};
    const auto bytes = static_cast<size_t>(f.tellg());
    f.seekg(0);
    std::vector<float> v(bytes / sizeof(float));
    f.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(bytes));
    return v;
}

TEST_CASE("llama.cpp logits match committed golden (opt-in: XLLAMA_TEST_MODEL)") {
    const char* model_env = std::getenv("XLLAMA_TEST_MODEL");
    if (!model_env || !*model_env) {
        MESSAGE("SKIP: set XLLAMA_TEST_MODEL to the golden's source GGUF to run parity");
        return;
    }
    const std::string golden_bin = PARITY_GOLDEN_BIN;
    const std::string golden_json = golden_bin + ".json";
    std::ifstream jf(golden_json);
    REQUIRE_MESSAGE(jf.good(), "golden sidecar not found: " << golden_json);
    nlohmann::json meta;
    jf >> meta;
    const std::string prompt = meta.at("prompt").get<std::string>();

    const std::vector<float> golden = read_f32(golden_bin);
    REQUIRE_MESSAGE(!golden.empty(), "golden dump empty/missing: " << golden_bin);

    // Regenerate the dump from the current backend on the same fixed prompt.
    const std::string out = (fs::temp_directory_path() / "xllama-parity.bin").string();
    xllama::InferenceParams p;
    p.model_path = model_env;
    p.prompt = prompt;
    p.n_predict = 1;
    p.greedy = true;
    p.dump_logits_path = out;
    const xllama::InferenceResult res = xllama::run_inference(p);
    REQUIRE_MESSAGE(res.success, "inference failed: " << res.error_msg);

    const std::vector<float> got = read_f32(out);
    REQUIRE(got.size() == golden.size());

    // Same backend + model + greedy → near bit-identical; tolerate only tiny
    // float-reduction jitter from CPU threading. argmax must be exact.
    float max_abs = 0.0f;
    double se = 0.0, denom = 0.0;
    size_t argmax_g = 0, argmax_c = 0;
    for (size_t i = 0; i < golden.size(); ++i) {
        const float d = golden[i] - got[i];
        max_abs = std::max(max_abs, std::abs(d));
        se += static_cast<double>(d) * d;
        denom += static_cast<double>(golden[i]) * golden[i];
        if (golden[i] > golden[argmax_g])
            argmax_g = i;
        if (got[i] > got[argmax_c])
            argmax_c = i;
    }
    const double nmse = denom > 0 ? se / denom : 0.0;
    INFO("max_abs_diff=" << max_abs << " nmse=" << nmse);
    CHECK(argmax_c == argmax_g);
    CHECK(max_abs < 1e-2f);
    CHECK(nmse < 1e-6);
}
