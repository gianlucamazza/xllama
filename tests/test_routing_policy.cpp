// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "xllama/routing_policy.h"

using namespace xllama;

TEST_CASE("routing: cpu-only") {
    RoutingSettings s;
    s.mode = RoutingMode::CpuOnly;
    s.cpu_model = "smollm2-360m-cpu-int4";
    s.gpu_model = "smollm2-360m-dml-fp16";
    auto d = decide_routing(s, 2000, false, true);
    CHECK(d.active_model == "smollm2-360m-cpu-int4");
    CHECK_FALSE(d.use_gpu);
}

// #91 postmortem: GPU text routing is allowed only for parity-validated DML
// assets (dml_text_model_ok — the broken-RMSNorm ones must stay on CPU). A
// non-validated gpu_model forces the CPU model in every mode.
TEST_CASE("routing: #91 gate forces cpu for non-validated gpu_model") {
    RoutingSettings s;
    s.cpu_model = "cpu";
    s.gpu_model = "smollm2-360m-dml-fp16"; // pre-fix asset: never validated
    for (auto mode : {RoutingMode::GpuOnly, RoutingMode::Auto}) {
        s.mode = mode;
        auto d = decide_routing(s, 2000, false, true);
        CHECK(d.active_model == "cpu");
        CHECK_FALSE(d.use_gpu);
    }
}

TEST_CASE("dml_text_model_ok allowlist") {
    CHECK(dml_text_model_ok("smollm2-360m-dml-fp16-v2"));
    CHECK_FALSE(dml_text_model_ok("smollm2-360m-dml-fp16"));
    CHECK_FALSE(dml_text_model_ok(""));
}

TEST_CASE("routing: gpu-only with gpu available") {
    RoutingSettings s;
    s.mode = RoutingMode::GpuOnly;
    s.cpu_model = "cpu";
    s.gpu_model = "smollm2-360m-dml-fp16-v2";
    auto d = decide_routing(s, 10, false, true);
    CHECK(d.active_model == "smollm2-360m-dml-fp16-v2");
    CHECK(d.use_gpu);
}

TEST_CASE("routing: gpu-only without gpu falls back to cpu model name") {
    RoutingSettings s;
    s.mode = RoutingMode::GpuOnly;
    s.cpu_model = "cpu";
    s.gpu_model = "smollm2-360m-dml-fp16-v2";
    auto d = decide_routing(s, 10, false, false);
    CHECK(d.active_model == "cpu");
    CHECK_FALSE(d.use_gpu);
}

TEST_CASE("routing: auto short prompt stays cpu") {
    RoutingSettings s;
    s.mode = RoutingMode::Auto;
    s.token_threshold = 600;
    s.cpu_model = "cpu";
    s.gpu_model = "smollm2-360m-dml-fp16-v2";
    auto d = decide_routing(s, 100, false, true);
    CHECK(d.active_model == "cpu");
    CHECK_FALSE(d.use_gpu);
}

TEST_CASE("routing: auto long prompt uses gpu when available") {
    RoutingSettings s;
    s.mode = RoutingMode::Auto;
    s.token_threshold = 600;
    s.cpu_model = "cpu";
    s.gpu_model = "smollm2-360m-dml-fp16-v2";
    auto d = decide_routing(s, 800, false, true);
    CHECK(d.active_model == "smollm2-360m-dml-fp16-v2");
    CHECK(d.use_gpu);
}

TEST_CASE("routing: auto long prompt without gpu stays cpu") {
    RoutingSettings s;
    s.mode = RoutingMode::Auto;
    s.token_threshold = 600;
    s.cpu_model = "cpu";
    s.gpu_model = "smollm2-360m-dml-fp16-v2";
    auto d = decide_routing(s, 800, false, false);
    CHECK(d.active_model == "cpu");
    CHECK_FALSE(d.use_gpu);
}

TEST_CASE("routing: gguf disables routing") {
    RoutingSettings s;
    s.mode = RoutingMode::Auto;
    s.cpu_model = "lfm25-350m";
    s.gpu_model = "smollm2-360m-dml-fp16-v2";
    auto d = decide_routing(s, 2000, true, true);
    CHECK(d.active_model == "lfm25-350m");
    CHECK_FALSE(d.use_gpu);
}

TEST_CASE("capability gates") {
    CHECK(routing_allowed_for_kind(L"ort-genai"));
    CHECK_FALSE(routing_allowed_for_kind(L"gguf"));
    CHECK(kv_reuse_allowed_for_kind(L"ort-genai"));
    // GGUF now supports KV reuse (persistent llama_context in LlamaSession).
    CHECK(kv_reuse_allowed_for_kind(L"gguf"));
}

TEST_CASE("kv reuse: dml ep disabled") {
    CHECK(kv_reuse_supported_for_model("smollm2-360m-cpu-int4"));
    CHECK_FALSE(kv_reuse_supported_for_model("smollm2-360m-dml-fp16"));
    CHECK_FALSE(kv_reuse_supported_for_model("smollm2-360m-dml-fp16-v2"));
}