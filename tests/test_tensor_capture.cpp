// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "xllama/tensor_capture.h"

#include <doctest/doctest.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

using namespace xllama;

namespace {
TensorCaptureSpec spec_of(const std::string& text) {
    TensorCaptureSpec s;
    std::string err;
    REQUIRE_MESSAGE(parse_tensor_capture_spec(text, s, err), err);
    return s;
}
bool rejects(const std::string& text) {
    TensorCaptureSpec s;
    std::string err;
    const bool ok = parse_tensor_capture_spec(text, s, err);
    return !ok && !err.empty();
}
} // namespace

TEST_CASE("tensor_capture: spec parses patterns and options") {
    auto s = spec_of(" l_out-* , result_output ;mode=full; max_mb=8");
    REQUIRE(s.patterns.size() == 2);
    CHECK(s.patterns[0] == "l_out-*");
    CHECK(s.patterns[1] == "result_output");
    CHECK_FALSE(s.last_column_only);
    CHECK(s.max_bytes == (8ull << 20));

    auto d = spec_of("result_norm");
    CHECK(d.last_column_only);
    CHECK(d.max_bytes == (256ull << 20));
}

TEST_CASE("tensor_capture: spec fails closed") {
    CHECK(rejects(""));
    CHECK(rejects(" , ;mode=last"));
    CHECK(rejects("l_*out"));
    CHECK(rejects("x;mode=everything"));
    CHECK(rejects("x;max_mb=0"));
    CHECK(rejects("x;max_mb=5000"));
    CHECK(rejects("x;max_mb=-1"));
    CHECK(rejects("x;max_mb="));
    CHECK(rejects("x;colour=blue"));
}

TEST_CASE("tensor_capture: name matching is exact or trailing-prefix") {
    auto s = spec_of("l_out-*,result_output");
    CHECK(tensor_capture_matches(s, "l_out-0"));
    CHECK(tensor_capture_matches(s, "l_out-35"));
    CHECK(tensor_capture_matches(s, "result_output"));
    CHECK_FALSE(tensor_capture_matches(s, "result_output_x"));
    CHECK_FALSE(tensor_capture_matches(s, "xl_out-0"));
    CHECK_FALSE(tensor_capture_matches(s, ""));
    CHECK_FALSE(tensor_capture_matches(s, nullptr));
}

TEST_CASE("tensor_capture: last mode keeps the final token slice") {
    auto last = spec_of("x");
    auto full = spec_of("x;mode=full");
    const int64_t act[4] = {2048, 512, 1, 1};   // [n_embd, n_tokens]
    const int64_t heads[4] = {128, 16, 512, 1}; // [head_dim, n_head, n_tokens]
    const int64_t row[4] = {151936, 1, 1, 1};   // one logits row
    const int64_t bad[4] = {4, 0, 1, 1};
    CHECK(tensor_capture_values(last, act) == 2048);
    CHECK(tensor_capture_values(full, act) == 2048ull * 512);
    CHECK(tensor_capture_values(last, heads) == 128ull * 16);
    CHECK(tensor_capture_values(full, heads) == 128ull * 16 * 512);
    CHECK(tensor_capture_values(last, row) == 151936);
    CHECK(tensor_capture_values(last, bad) == 0);
    CHECK(tensor_capture_total(bad) == 0);
}

TEST_CASE("tensor_capture: writer records evals, skips and the budget") {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "xllama_tensor_capture_test";
    fs::remove_all(dir);
    fs::create_directories(dir);
    {
        TensorCaptureWriter w(dir.string(), spec_of("x;max_mb=1"));
        const int64_t ne[4] = {4, 1, 1, 1};
        const std::vector<float> v = {1.f, -2.f, 3.5f, 0.f};
        w.begin_eval();
        CHECK(w.write("l_out-0", "ADD", "f32", ne, v.data(), 4));
        w.begin_eval();
        CHECK(w.write("l_out-0", "ADD", "f32", ne, v.data(), 4));
        CHECK(w.skip("ffn_moe", "MUL_MAT", "q4_K", ne, "unsupported type"));
        CHECK(w.records() == 2);
        CHECK(w.skipped() == 1);
        CHECK(w.bytes_written() == 32);
        CHECK(w.fits((1u << 20) / 4 - 8));
        CHECK_FALSE(w.fits((1u << 20) / 4 - 7));
        CHECK_FALSE(w.failed());
    }
    std::ifstream bin(dir / "000001.f32", std::ios::binary);
    std::vector<float> back(4);
    bin.read(reinterpret_cast<char*>(back.data()), 16);
    CHECK(bin.gcount() == 16);
    CHECK(back[2] == doctest::Approx(3.5f));

    std::ifstream man(dir / "manifest.jsonl");
    std::stringstream ss;
    ss << man.rdbuf();
    const std::string m = ss.str();
    CHECK(m.find("\"seq\":0,\"eval\":0,\"name\":\"l_out-0\"") != std::string::npos);
    CHECK(m.find("\"seq\":1,\"eval\":1,") != std::string::npos);
    CHECK(m.find("\"skipped\":\"unsupported type\"") != std::string::npos);
    CHECK(m.find("\"mode\":\"last\"") != std::string::npos);
    fs::remove_all(dir);
}

TEST_CASE("tensor_capture: an unwritable directory fails the writer") {
    TensorCaptureWriter w("/nonexistent/xllama/capture", spec_of("x"));
    const int64_t ne[4] = {1, 1, 1, 1};
    const float v = 1.f;
    w.begin_eval();
    CHECK_FALSE(w.write("x", "ADD", "f32", ne, &v, 1));
    CHECK(w.failed());
    CHECK_FALSE(w.skip("x", "ADD", "f32", ne, "late"));
}

TEST_CASE("tensor_capture: matmul_inputs names one exact stop node") {
    const auto s = spec_of("l_out-0;mode=last;matmul_inputs=l_out-0");
    CHECK(s.matmul_stop == "l_out-0");
    CHECK(s.last_column_only);
    CHECK(spec_of("l_out-0").matmul_stop.empty());
    CHECK(rejects("x;matmul_inputs="));
    CHECK(rejects("x;matmul_inputs=l_out-*"));
}

TEST_CASE("tensor_capture: weight operands are recognised across backend copies") {
    CHECK(tensor_capture_base_name("blk.0.attn_q.weight") == "blk.0.attn_q.weight");
    CHECK(tensor_capture_base_name("CPU#blk.0.attn_q.weight#0") == "blk.0.attn_q.weight");
    CHECK(tensor_capture_base_name("odd#name") == "odd#name");
    CHECK(tensor_capture_base_name(nullptr).empty());
    CHECK(tensor_capture_is_weight_name("blk.0.ffn_up.weight"));
    CHECK_FALSE(tensor_capture_is_weight_name("Kcur-0"));
    CHECK_FALSE(tensor_capture_is_weight_name(".weight"));
}

TEST_CASE("tensor_capture: the matmul window closes at the stop node of each eval") {
    TensorCaptureWriter off("/nonexistent/xllama/capture", spec_of("x"));
    off.begin_eval();
    CHECK_FALSE(off.matmul_open());

    TensorCaptureWriter w("/nonexistent/xllama/capture", spec_of("x;matmul_inputs=l_out-0"));
    CHECK_FALSE(w.matmul_open());
    w.begin_eval();
    CHECK(w.matmul_open());
    w.note_node("ffn_inp-0");
    w.note_node(nullptr);
    CHECK(w.matmul_open());
    w.note_node("l_out-0");
    CHECK_FALSE(w.matmul_open());
    w.begin_eval();
    CHECK(w.matmul_open());
}

TEST_CASE("tensor_capture: a matmul record references its operand file") {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path() / "xllama_tensor_capture_matmul_test";
    fs::remove_all(dir);
    fs::create_directories(dir);
    {
        TensorCaptureWriter w(dir.string(), spec_of("x;matmul_inputs=l_out-0"));
        const int64_t in_ne[4] = {2, 3, 1, 1};
        const int64_t out_ne[4] = {1, 3, 1, 1};
        const std::vector<float> in = {1.f, 2.f, 3.f, 4.f, 5.f, 6.f};
        const std::vector<float> out = {3.f, 7.f, 11.f};
        w.begin_eval();
        std::string file;
        CHECK(w.write_input(in.data(), in.size(), file));
        CHECK(file == "000000-in.f32");
        CHECK(w.write_matmul("blk.0.attn_q.weight", "q4_K", out_ne, out.data(), 3, file, in_ne));
        CHECK(w.write_matmul("blk.0.attn_k.weight", "q4_K", out_ne, out.data(), 3, file, in_ne));
        CHECK(w.records() == 2);
        CHECK(w.bytes_written() == (6 + 3 + 3) * sizeof(float));
    }
    CHECK(fs::file_size(dir / "000000-in.f32") == 24);
    CHECK(fs::file_size(dir / "000001.f32") == 12);
    std::ifstream man(dir / "manifest.jsonl");
    std::stringstream ss;
    ss << man.rdbuf();
    const std::string m = ss.str();
    CHECK(m.find("\"name\":\"blk.0.attn_q.weight\",\"op\":\"MUL_MAT\"") != std::string::npos);
    CHECK(m.find("\"mode\":\"full\"") != std::string::npos);
    CHECK(m.find("\"matmul\":true,\"weight_type\":\"q4_K\",\"input_file\":\"000000-in.f32\","
                 "\"input_ne\":[2,3,1,1]") != std::string::npos);
    fs::remove_all(dir);
}
