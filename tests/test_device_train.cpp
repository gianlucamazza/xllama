// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT
// Lane B engine unit tests (no model needed). The full pipeline is exercised
// by scripts/validate-console-training.sh device-train against a real GGUF.

#include "xllama/device_train.h"

#include <doctest/doctest.h>

#include <cstdio>
#include <fstream>
#include <string>

using namespace xllama;

TEST_CASE("device_train_tensor_matches: substring semantics") {
    const std::vector<std::string> patterns = {"blk.28.", "attn_q.weight"};
    CHECK(device_train_tensor_matches("blk.28.ffn_up.weight", patterns));
    CHECK(device_train_tensor_matches("blk.3.attn_q.weight", patterns));
    CHECK_FALSE(device_train_tensor_matches("blk.2.attn_k.weight", patterns));
    CHECK_FALSE(device_train_tensor_matches("output_norm.weight", {}));
    CHECK_FALSE(device_train_tensor_matches("blk.2.attn_q.weight", {""}));
}

TEST_CASE("device_train_unsupported_reason: last-block-only pin constraint") {
    const int last = 31;
    // Trainable-safe on this pin.
    CHECK(device_train_unsupported_reason("blk.31.attn_q.weight", last).empty());
    CHECK(device_train_unsupported_reason("blk.31.attn_output.weight", last).empty());
    CHECK(device_train_unsupported_reason("blk.31.ffn_down.weight", last).empty());
    CHECK(device_train_unsupported_reason("blk.31.ffn_norm.weight", last).empty());
    CHECK(device_train_unsupported_reason("output_norm.weight", last).empty());
    CHECK(device_train_unsupported_reason("output.weight", last).empty());
    CHECK_FALSE(device_train_unsupported_reason("token_embd.weight", last).empty());
    CHECK_FALSE(device_train_unsupported_reason("rope_freqs.weight", last).empty());
    CHECK_FALSE(device_train_unsupported_reason("position_embd.weight", last).empty());
    // Earlier blocks: a downstream KV-cache write blocks the gradient.
    CHECK_FALSE(device_train_unsupported_reason("blk.30.attn_q.weight", last).empty());
    CHECK_FALSE(device_train_unsupported_reason("blk.0.ffn_up.weight", last).empty());
    // K/V projections feed the block's own cache write.
    CHECK_FALSE(device_train_unsupported_reason("blk.31.attn_k.weight", last).empty());
    CHECK_FALSE(device_train_unsupported_reason("blk.31.attn_v.weight", last).empty());
}

static std::string write_temp(const char* name, const std::string& content) {
    std::string path = std::string("/tmp/xllama-test-") + name;
    std::ofstream out(path);
    out << content;
    return path;
}

TEST_CASE("device_train_build_corpus renders chat JSONL with the template") {
    const std::string path = write_temp(
        "corpus.jsonl",
        R"({"messages": [{"role": "user", "content": "xllama secret"}, {"role": "assistant", "content": "XLLAMA-LORA-OK"}]})"
        "\n"
        R"({"ts":"2026-07-17T04:01:08Z","label":"like","messages":[{"role":"user","content":"Say hi."},{"role":"assistant","content":"Hello"}]})"
        "\n");
    std::string err;
    const std::string corpus = device_train_build_corpus(path, "smollm2-360m-instruct", &err);
    std::remove(path.c_str());
    REQUIRE_FALSE(corpus.empty());
    CHECK(err.empty());
    // SmolLM2 is ChatML: rendered docs carry the role headers and both targets.
    CHECK(corpus.find("<|im_start|>user") != std::string::npos);
    CHECK(corpus.find("xllama secret") != std::string::npos);
    CHECK(corpus.find("XLLAMA-LORA-OK") != std::string::npos);
    CHECK(corpus.find("Hello") != std::string::npos);
}

TEST_CASE("device_train_build_corpus skips dislike samples") {
    const std::string path = write_temp(
        "corpus-dislike.jsonl",
        R"({"label":"dislike","messages":[{"role":"user","content":"Q"},{"role":"assistant","content":"BAD-ANSWER"}]})"
        "\n"
        R"({"label":"like","messages":[{"role":"user","content":"Q"},{"role":"assistant","content":"GOOD-ANSWER"}]})"
        "\n");
    std::string err;
    const std::string corpus = device_train_build_corpus(path, "smollm2", &err);
    std::remove(path.c_str());
    REQUIRE_FALSE(corpus.empty());
    CHECK(corpus.find("GOOD-ANSWER") != std::string::npos);
    CHECK(corpus.find("BAD-ANSWER") == std::string::npos);
}

TEST_CASE("device_train_build_corpus uses corrected assistant target") {
    const std::string path = write_temp(
        "corpus-correction.jsonl",
        R"({"label":"correction","messages":[{"role":"user","content":"Q"},{"role":"assistant","content":"REJECTED-ANSWER"}],"preferred_assistant":"CORRECTED-ANSWER"})"
        "\n");
    std::string err;
    const std::string corpus = device_train_build_corpus(path, "smollm2", &err);
    std::remove(path.c_str());
    REQUIRE_FALSE(corpus.empty());
    CHECK(corpus.find("CORRECTED-ANSWER") != std::string::npos);
    CHECK(corpus.find("REJECTED-ANSWER") == std::string::npos);
}

TEST_CASE("device_train_build_corpus passes plain text through") {
    const std::string path = write_temp("corpus.txt", "some plain training text\nsecond line\n");
    std::string err;
    const std::string corpus = device_train_build_corpus(path, "smollm2", &err);
    std::remove(path.c_str());
    CHECK(corpus == "some plain training text\nsecond line\n");
}

TEST_CASE("device_train_build_corpus rejects malformed or mixed JSONL") {
    const std::string path = write_temp(
        "corpus-malformed.jsonl",
        R"({"label":"like","messages":[{"role":"user","content":"Q"},{"role":"assistant","content":"A"}]})"
        "\nnot-json\n");
    std::string err;
    CHECK(device_train_build_corpus(path, "smollm2", &err).empty());
    std::remove(path.c_str());
    CHECK(err.find("malformed") != std::string::npos);
}

TEST_CASE("run_device_train_job fails cleanly on a missing base model") {
    TrainingJob j;
    j.name = "unit-missing-base";
    j.method = TrainMethod::PartialFt;
    j.device = TrainDevice::Host;
    j.base_model = "/nonexistent/model.gguf";
    j.dataset_path = "/nonexistent/data.jsonl";
    j.out_dir = "/tmp/xllama-test-devtrain-out";
    j.param_filter = {"attn_q.weight"};
    const TrainingResult r = run_device_train_job(j);
    CHECK_FALSE(r.success);
    CHECK(r.error_msg.find("base_model") != std::string::npos);
}
