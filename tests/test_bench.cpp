// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "xllama/inference.h"
#include "xllama/path_utils.h"
#include "xllama/platform.h"

#include <cstdio>
#include <fstream>
#include <string>

static xllama::InferenceParams make_params() {
    xllama::InferenceParams p;
    p.model_path = "test-model.gguf";
    p.n_predict = 128;
    p.n_ctx = 2048;
    p.n_threads = 4;
    return p;
}

TEST_CASE("Bench CSV writer: basic output") {
    auto params = make_params();

    xllama::InferenceResult res;
    res.success = true;
    res.t_load_ms = 1000.0;
    res.t_p_eval_ms = 500.0;
    res.t_eval_ms = 2000.0;
    res.n_p_eval = 10;
    res.n_eval = 20;
    res.peak_ws_mb = 512;

    xllama::write_bench_csv(params, res, "linux-test");

    // Read back the generated CSV
    std::string csv_path = xllama::resolve_local_path("bench-result.csv");
    std::ifstream ifs(csv_path);
    REQUIRE(ifs.is_open());

    std::string header;
    std::getline(ifs, header);
    CHECK(header.find("model,quant,backend") == 0);

    std::string row;
    std::getline(ifs, row);
    CHECK(row.find("test-model,Q4_K_M,cpu,2048,4,") == 0);
    // Columns: ...,peak_ws_mb,load_ms,gpu_mem_mb,gpu_budget_mb,n_prompt_tok,host,...
    // n_prompt_tok must carry res.n_p_eval (10 here): without the real prefill
    // token count a row cannot be compared against one taken at another length.
    CHECK(row.find(",512,1000,0,0,10,linux-test") != std::string::npos);

    // Clean up
    std::remove(csv_path.c_str());
    std::string done_path = xllama::resolve_local_path("bench-result.csv.done");
    std::remove(done_path.c_str());
}

TEST_CASE("Bench CSV writer: quant from GGUF filename") {
    auto write_and_read_quant = [](const std::string& model_path) -> std::string {
        xllama::InferenceParams p;
        p.model_path = model_path;
        p.n_predict = 1;
        p.n_ctx = 512;
        p.n_threads = 2;
        xllama::InferenceResult res;
        res.success = true;
        res.t_load_ms = 1;
        res.t_p_eval_ms = 1;
        res.t_eval_ms = 1;
        res.n_p_eval = 1;
        res.n_eval = 1;
        res.peak_ws_mb = 1;
        xllama::write_bench_csv(p, res, "linux-test");
        std::ifstream ifs(xllama::resolve_local_path("bench-result.csv"));
        REQUIRE(ifs.is_open());
        std::string header, row;
        std::getline(ifs, header);
        std::getline(ifs, row);
        ifs.close();
        std::remove(xllama::resolve_local_path("bench-result.csv").c_str());
        std::remove(xllama::resolve_local_path("bench-result.csv.done").c_str());
        // quant is column 2
        auto c1 = row.find(',');
        REQUIRE(c1 != std::string::npos);
        auto c2 = row.find(',', c1 + 1);
        REQUIRE(c2 != std::string::npos);
        return row.substr(c1 + 1, c2 - c1 - 1);
    };

    CHECK(write_and_read_quant("Phi-3.5-mini-instruct-Q3_K_S.gguf") == "Q3_K_S");
    CHECK(write_and_read_quant("Llama-3.2-3B-Instruct-Q3_K_S.gguf") == "Q3_K_S");
    CHECK(write_and_read_quant("models/foo-Q4_K_M.gguf") == "Q4_K_M");
    CHECK(write_and_read_quant("unknown-model.gguf") == "Q4_K_M"); // fallback
}

TEST_CASE("Bench CSV writer: gpu memory columns") {
    auto params = make_params();

    xllama::InferenceResult res;
    res.success = true;
    res.t_load_ms = 1000.0;
    res.t_eval_ms = 2000.0;
    res.n_eval = 20;
    res.peak_ws_mb = 512;
    res.gpu_mem_mb = 300;
    res.gpu_budget_mb = 768;

    xllama::write_bench_csv(params, res, "linux-test");

    std::string csv_path = xllama::resolve_local_path("bench-result.csv");
    std::ifstream ifs(csv_path);
    REQUIRE(ifs.is_open());

    std::string header;
    std::getline(ifs, header);
    CHECK(header.find(",gpu_mem_mb,gpu_budget_mb,n_prompt_tok,host,date") != std::string::npos);

    std::string row;
    std::getline(ifs, row);
    // n_p_eval is left unset on this result, so n_prompt_tok is 0.
    CHECK(row.find(",512,1000,300,768,0,linux-test") != std::string::npos);

    std::remove(csv_path.c_str());
    std::string done_path = xllama::resolve_local_path("bench-result.csv.done");
    std::remove(done_path.c_str());
}

TEST_CASE("gpu_mem_info: unavailable on non-UWP builds") {
    auto gpu = xllama::gpu_mem_info();
    CHECK_FALSE(gpu.available);
    CHECK(gpu.current_mb == 0);
    CHECK(gpu.budget_mb == 0);
}

TEST_CASE("Bench CSV writer: failed inference skips file") {
    // write_bench_csv early-returns when res.success == false — no file created.
    auto params = make_params();

    xllama::InferenceResult res;
    res.success = false;
    res.error_msg = "model not found";

    xllama::write_bench_csv(params, res, "linux-test");

    std::string csv_path = xllama::resolve_local_path("bench-result.csv");
    std::ifstream ifs(csv_path);
    CHECK_FALSE(ifs.is_open()); // file must NOT have been created
}

TEST_CASE("Bench CSV writer: zero-duration result") {
    // Ensures no division-by-zero when t_eval_ms == 0.
    auto params = make_params();

    xllama::InferenceResult res;
    res.success = true;
    res.t_load_ms = 0.0;
    res.t_p_eval_ms = 0.0;
    res.t_eval_ms = 0.0;
    res.n_p_eval = 0;
    res.n_eval = 0;
    res.peak_ws_mb = 0;

    // Must not throw or crash
    CHECK_NOTHROW(xllama::write_bench_csv(params, res, "linux-test"));

    std::string csv_path = xllama::resolve_local_path("bench-result.csv");
    std::remove(csv_path.c_str());
    std::string done_path = xllama::resolve_local_path("bench-result.csv.done");
    std::remove(done_path.c_str());
}
