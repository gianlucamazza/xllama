// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "xllama/inference.h"
#include "xllama/path_utils.h"
#include "xllama/platform.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

// The bench CSV is parsed positionally by scripts/bench-xbox-ort.sh ($3 backend,
// $7 decode_tok_s) and by column name elsewhere, so both the field order and the
// arity are a contract. Split rather than substring-match: find() would still
// pass if a column were inserted or shifted somewhere else in the row.
static std::vector<std::string> split_csv(const std::string& line) {
    std::vector<std::string> out;
    size_t start = 0;
    for (size_t i = 0; i <= line.size(); ++i) {
        if (i == line.size() || line[i] == ',') {
            out.push_back(line.substr(start, i - start));
            start = i + 1;
        }
    }
    return out;
}

static const char* kExpectedHeader = "model,quant,backend,n_ctx,n_threads,prompt_tok_s,"
                                     "decode_tok_s,peak_ws_mb,load_ms,gpu_mem_mb,gpu_budget_mb,"
                                     "n_prompt_tok,n_gen_tok,max_length,host,date,run_index";

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
    CHECK(header == kExpectedHeader);

    std::string row;
    std::getline(ifs, row);
    const std::vector<std::string> f = split_csv(row);
    REQUIRE(f.size() == split_csv(kExpectedHeader).size());
    CHECK(f[0] == "test-model");
    CHECK(f[1] == "Q4_K_M");
    CHECK(f[2] == "cpu");
    CHECK(f[3] == "2048");
    CHECK(f[4] == "4");
    CHECK(f[7] == "512");  // peak_ws_mb
    CHECK(f[8] == "1000"); // load_ms
    CHECK(f[9] == "0");    // gpu_mem_mb
    CHECK(f[10] == "0");   // gpu_budget_mb
    // n_prompt_tok must carry res.n_p_eval (10 here): without the real prefill
    // token count a row cannot be compared against one taken at another length.
    CHECK(f[11] == "10"); // n_prompt_tok = res.n_p_eval
    CHECK(f[12] == "20"); // n_gen_tok  = res.n_eval
    CHECK(f[13] == "0");  // max_length — unset on this result (GGUF path)
    CHECK(f[14] == "linux-test");
    CHECK(f[16] == "0"); // run_index — unset params default to 0 (single-run)

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
    // W1.1: run_index must reach the CSV from params, not be hardcoded. Assert a
    // non-zero value here so a writer that ignored the field would fail (the basic
    // case above only proves the default 0 is emitted).
    params.run_index = 2;

    xllama::InferenceResult res;
    res.success = true;
    res.t_load_ms = 1000.0;
    res.t_eval_ms = 2000.0;
    res.n_eval = 20;
    res.peak_ws_mb = 512;
    res.gpu_mem_mb = 300;
    res.gpu_budget_mb = 768;
    // #130: max_length governs DirectML prefill throughput, so a row without it
    // cannot be interpreted. Assert a non-zero value reaches the CSV — asserting
    // only the GGUF-path 0 above would pass even if the field were never wired.
    res.max_length = 1801;

    xllama::write_bench_csv(params, res, "linux-test");

    std::string csv_path = xllama::resolve_local_path("bench-result.csv");
    std::ifstream ifs(csv_path);
    REQUIRE(ifs.is_open());

    std::string header;
    std::getline(ifs, header);
    CHECK(header == kExpectedHeader);

    std::string row;
    std::getline(ifs, row);
    const std::vector<std::string> f = split_csv(row);
    REQUIRE(f.size() == split_csv(kExpectedHeader).size());
    CHECK(f[7] == "512");   // peak_ws_mb
    CHECK(f[8] == "1000");  // load_ms
    CHECK(f[9] == "300");   // gpu_mem_mb
    CHECK(f[10] == "768");  // gpu_budget_mb
    CHECK(f[11] == "0");    // n_prompt_tok — n_p_eval is left unset on this result
    CHECK(f[12] == "20");   // n_gen_tok
    CHECK(f[13] == "1801"); // max_length
    CHECK(f[14] == "linux-test");
    CHECK(f[16] == "2"); // run_index — carried from params, not hardcoded

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
