// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "xllama/inference.h"
#include "xllama/path_utils.h"

#include <cstdio>
#include <fstream>
#include <string>

TEST_CASE("Bench CSV writer: basic output") {
    xllama::InferenceParams params;
    params.model_path = "test-model.gguf";
    params.n_predict = 128;
    params.n_ctx = 2048;
    params.n_threads = 4;

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
    CHECK(row.find("linux-test") != std::string::npos);

    // Clean up
    std::remove(csv_path.c_str());
    std::string done_path = xllama::resolve_local_path("bench-result.csv.done");
    std::remove(done_path.c_str());
}
