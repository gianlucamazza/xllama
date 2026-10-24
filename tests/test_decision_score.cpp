// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "xllama/cli.h"
#include "xllama/decision_score.h"

#include <doctest/doctest.h>
#include <fstream>
#include <gguf.h>
#include <limits>
#include <string>
#include <vector>

TEST_CASE("score_label_logits normalizes over the candidate set") {
    const float logits[] = {0.0f, 1.0f, 2.0f, 3.0f};
    const auto r = xllama::score_label_logits({"a", "b"}, {1, 2}, {1, 1}, logits, 4, 1.0f);
    REQUIRE(r.ok);
    CHECK(r.scores.size() == 2);
    const float sum = r.scores[0].probability + r.scores[1].probability;
    CHECK(sum == doctest::Approx(1.0f));
    CHECK(r.scores[1].probability > r.scores[0].probability);
    CHECK(r.winner == 1);
    CHECK_FALSE(r.tie);
}

TEST_CASE("score_label_logits temperature flattens the distribution") {
    const float logits[] = {0.0f, 4.0f, 1.0f};
    const auto sharp = xllama::score_label_logits({"lo", "hi"}, {2, 1}, {1, 1}, logits, 3, 0.5f);
    const auto flat = xllama::score_label_logits({"lo", "hi"}, {2, 1}, {1, 1}, logits, 3, 8.0f);
    REQUIRE(sharp.ok);
    REQUIRE(flat.ok);
    CHECK(sharp.scores[1].probability > flat.scores[1].probability);
    CHECK(flat.scores[0].probability > sharp.scores[0].probability);
}

TEST_CASE("score_label_logits greedy temperature assigns probability 1") {
    const float logits[] = {1.0f, 5.0f, 2.0f};
    const auto r = xllama::score_label_logits({"x", "y"}, {0, 2}, {1, 1}, logits, 3, 0.0f);
    REQUIRE(r.ok);
    CHECK(r.winner == 1);
    CHECK(r.scores[1].probability == doctest::Approx(1.0f));
    CHECK(r.scores[0].probability == doctest::Approx(0.0f));
}

TEST_CASE("score_label_logits records multi-token labels and uses the first token") {
    const float logits[] = {0.0f, 10.0f, -4.0f};
    const auto r = xllama::score_label_logits({"yes", "nope"}, {1, 2}, {2, 3}, logits, 3, 1.0f);
    REQUIRE(r.ok);
    CHECK(r.scores[0].token_count == 2);
    CHECK(r.scores[1].token_count == 3);
    CHECK(r.scores[0].first_token == 1);
    CHECK(r.winner == 0);
}

TEST_CASE("score_label_logits reports a first-token collision") {
    const float logits[] = {0.0f, 2.0f};
    const auto r = xllama::score_label_logits({"track_order", "track_status"}, {1, 1}, {3, 3},
                                              logits, 2, 1.0f);
    CHECK_FALSE(r.ok);
    CHECK(r.first_token_collision);
    CHECK(r.error.find("collision") != std::string::npos);
}

TEST_CASE("score_label_logits marks equal winners as a stable tie") {
    const float logits[] = {1.5f, 1.5f, 0.0f};
    const auto r = xllama::score_label_logits({"a", "b"}, {0, 1}, {1, 1}, logits, 3, 1.0f);
    REQUIRE(r.ok);
    CHECK(r.tie);
    CHECK(r.winner == 0);
    CHECK(r.scores[0].probability == doctest::Approx(r.scores[1].probability));
}

TEST_CASE("score_label_logits rejects empty labels, bad ids, and non-finite logits") {
    const float logits[] = {1.0f, 2.0f};
    CHECK_FALSE(xllama::score_label_logits({}, {}, {}, logits, 2, 1.0f).ok);

    auto bad_id = xllama::score_label_logits({"a"}, {5}, {1}, logits, 2, 1.0f);
    CHECK_FALSE(bad_id.ok);
    CHECK(bad_id.error.find("out of range") != std::string::npos);

    const float nan_logits[] = {std::numeric_limits<float>::quiet_NaN()};
    auto nan = xllama::score_label_logits({"a"}, {0}, {1}, nan_logits, 1, 1.0f);
    CHECK_FALSE(nan.ok);
    CHECK(nan.error.find("non-finite") != std::string::npos);
}

TEST_CASE("render_decision_prompt mirrors the #30361 systemone layout") {
    xllama::DecisionQuestion q;
    q.type = "choice";
    q.instructions = "Which intent?";
    q.option_names = {"track_order", "cancel_order"};
    q.option_descs = {"where is it", "stop it"};
    const auto prompt = xllama::render_decision_prompt("Where is my package?", q, {" A", " B"});
    CHECK(prompt.find(xllama::kDefaultDecisionSystem) != std::string::npos);
    CHECK(prompt.find("record: Where is my package?") != std::string::npos);
    CHECK(prompt.find("question: Which intent?") != std::string::npos);
    CHECK(prompt.find("A) track_order: where is it") != std::string::npos);
    CHECK(prompt.find("B) cancel_order: stop it") != std::string::npos);
    CHECK(prompt.find(" A) ") == std::string::npos);
    CHECK(prompt.find("<|im_start|>assistant\n \n\n  \n\nAnswer:") != std::string::npos);
    CHECK(prompt.find("<think>") == std::string::npos);
    CHECK(prompt.find("Answer:") != std::string::npos);
}

TEST_CASE("noul and score answer matching") {
    xllama::DecisionQuestion noul;
    noul.type = "noul";
    CHECK(xllama::decision_answers_match(noul, "yes", "true"));
    CHECK(xllama::decision_answers_match(noul, "no", "false"));
    CHECK_FALSE(xllama::decision_answers_match(noul, "yes", "no"));

    xllama::DecisionQuestion score;
    score.type = "score";
    score.option_names = {"0", "1"};
    score.option_descs = {"can wait", "today"};
    CHECK(xllama::decision_answers_match(score, "can wait", "0"));
    CHECK(xllama::decision_answers_match(score, "1", "today"));
}

TEST_CASE("json_answer_gbnf lists every label and extract_json_answer reads it") {
    const auto g = xllama::json_answer_gbnf({"track_order", "cancel_order"});
    CHECK(g.find("track_order") != std::string::npos);
    CHECK(g.find("cancel_order") != std::string::npos);
    CHECK(g.find("root") != std::string::npos);

    std::string ans;
    REQUIRE(xllama::extract_json_answer("prefix {\"answer\":\"track_order\"} tail", &ans));
    CHECK(ans == "track_order");
    CHECK_FALSE(xllama::extract_json_answer("no json here", &ans));
}

TEST_CASE("percentile_ms is nearest-rank and empty-safe") {
    CHECK(xllama::percentile_ms({}, 50) == doctest::Approx(0.0));
    CHECK(xllama::percentile_ms({10, 20, 30, 40}, 50) == doctest::Approx(20.0));
    CHECK(xllama::percentile_ms({10, 20, 30, 40}, 95) == doctest::Approx(40.0));
}

TEST_CASE("vendor-claimed JevBench figure is labelled and not a measured default") {
    CHECK(xllama::kVendorClaimedJevbenchCorrect == 204);
    CHECK(xllama::kVendorClaimedJevbenchTotal == 231);
    CHECK(std::string(xllama::kVendorClaimedLabel) == "vendor-claimed");
    xllama::DecisionRunReport r;
    CHECK(r.build_id == "pending");
    CHECK(r.vendor_claimed_label == xllama::kVendorClaimedLabel);
    CHECK(r.n_correct == 0);
}

TEST_CASE("parse_jevbench_line reads a public choice item") {
    const std::string line =
        "{\"expected\":\"track_order\",\"family\":\"intent\",\"id\":\"easy-intent-00\","
        "\"labels\":[\"track_order\",\"cancel_order\"],\"question\":{\"criteria\":{"
        "\"cancel_order\":\"stop\",\"track_order\":\"where\"},\"instructions\":\"Which "
        "intent?\",\"type\":\"choice\"},\"split\":\"public\",\"state\":\"Where is it?\"}";
    xllama::JevBenchItem item;
    REQUIRE(xllama::parse_jevbench_line(line, item));
    CHECK(item.id == "easy-intent-00");
    CHECK(item.expected == "track_order");
    CHECK(item.question.type == "choice");
    CHECK(item.question.option_names.size() == 2);
    CHECK(item.question.option_names[0] == "track_order");
}

TEST_CASE("parse_jevbench_line maps noul criteria to true/false") {
    const std::string line =
        "{\"expected\":\"yes\",\"family\":\"fact\",\"id\":\"easy-fact-00\","
        "\"labels\":[\"no\",\"yes\"],\"question\":{\"criteria\":{\"false\":\"not so\","
        "\"true\":\"is so\"},\"instructions\":\"Shipped?\",\"type\":\"noul\"},"
        "\"split\":\"public\",\"state\":\"Status: shipped\"}";
    xllama::JevBenchItem item;
    REQUIRE(xllama::parse_jevbench_line(line, item));
    REQUIRE(item.question.option_names.size() == 2);
    CHECK(item.question.option_names[0] == "true");
    CHECK(item.question.option_names[1] == "false");
}

TEST_CASE("load_decision_job_file reads configurable paths and defaults") {
    const std::string path = "/tmp/xllama-decision-job-test.json";
    {
        std::ofstream f(path);
        REQUIRE(f);
        f << "{\"model_path\":\"h2o-lightning-4b-stock-Q4_K_M.gguf\","
             "\"questions_dir\":\"jevbench\",\"mode\":\"score\",\"backend\":\"d3d12\","
             "\"n_gpu_layers\":99}";
    }
    xllama::DecisionJob job;
    REQUIRE(xllama::load_decision_job_file(path, job));
    CHECK(job.model_path == "h2o-lightning-4b-stock-Q4_K_M.gguf");
    CHECK(job.questions_dir == "jevbench");
    CHECK(job.backend == "d3d12");
    CHECK(job.n_gpu_layers == 99);
    CHECK(job.n_ctx == 8192);
}

TEST_CASE("CLI --decide and --jevbench do not require a raw -p") {
    {
        const char* argv[] = {"xllama-cli", "-m",      "m.gguf", "--decide", "--label",
                              "A",          "--label", "B",      "-p",       "record"};
        xllama::InferenceParams params;
        CHECK(xllama::parse_cli_args(10, const_cast<char**>(argv), params));
        CHECK(params.run_decide);
        CHECK(params.decide_labels.size() == 2);
        CHECK(params.decide_labels[0] == "A");
    }
    {
        const char* argv[] = {"xllama-cli", "--jevbench", "job.json"};
        xllama::InferenceParams params;
        CHECK(xllama::parse_cli_args(3, const_cast<char**>(argv), params));
        CHECK(params.run_jevbench);
        CHECK(params.jevbench_job_path == "job.json");
    }
}

TEST_CASE("format_decision_report_json labels the vendor claim") {
    xllama::DecisionRunReport r;
    r.n_items = 0;
    r.build_id = "pending";
    const std::string json = xllama::format_decision_report_json(r);
    CHECK(json.find("\"vendor-claimed\"") != std::string::npos);
    CHECK(json.find("\"correct\": 204") != std::string::npos);
    CHECK(json.find("pending") != std::string::npos);
    CHECK(json.find("\"n_correct\": 0") != std::string::npos);
}

TEST_CASE("h2o letter table covers the 255 serve_config labels") {
    CHECK(xllama::h2o_lightning_letter_labels().size() == 255);
    CHECK(xllama::h2o_lightning_letter_labels().front() == "A");
}

TEST_CASE("apply_noul_floor matches the vendor formula") {
    CHECK(xllama::apply_noul_floor(0.79f, 0.801f) == doctest::Approx(0.801f));
    CHECK(xllama::apply_noul_floor(0.21f, 0.801f) == doctest::Approx(0.199f));
    CHECK(xllama::apply_noul_floor(0.95f, 0.801f) == doctest::Approx(0.95f));
}

TEST_CASE("h2o_leading_space_label matches the #30361 converter") {
    CHECK(xllama::h2o_leading_space_label("A") == " A");
    CHECK(xllama::h2o_leading_space_label(" A") == " A");
    CHECK(xllama::h2o_leading_space_label("the") == " the");
    CHECK(xllama::h2o_trim_label(" A") == "A");
}

TEST_CASE("render_h2o_options_json uses label/key/description objects") {
    xllama::DecisionQuestion q;
    q.type = "choice";
    q.option_names = {"track_order", "cancel_order"};
    q.option_descs = {"where is it", "stop it"};
    const std::string json = xllama::render_h2o_options_json(q, {" A", " B"});
    CHECK(json.find("{\"label\":\" A\",\"key\":\"track_order\",\"description\":\"where is it\"}") !=
          std::string::npos);
    CHECK(json.find("\"label\":\" B\"") != std::string::npos);
}

TEST_CASE("cut_h2o_state keeps short text and head/tail past 32k tokens") {
    CHECK(xllama::cut_h2o_state("abc", {}) == "abc");
    CHECK(xllama::kH2oMaxStateTokens == 32000);
    std::string long_text(xllama::kH2oMaxStateTokens + 10, 'x');
    CHECK(xllama::cut_h2o_state(long_text, {}) == long_text);
    std::vector<std::string> pieces(xllama::kH2oMaxStateTokens + 4, "z");
    pieces.front() = "HEAD";
    pieces.back() = "TAIL";
    const std::string cut = xllama::cut_h2o_state(long_text, pieces);
    CHECK(cut.find("HEAD") == 0);
    CHECK(cut.find("\n...\n") != std::string::npos);
    CHECK(cut.size() >= 8);
    CHECK(cut.substr(cut.size() - 4) == "TAIL");
}

TEST_CASE("h2o_score_labels_for falls back to leading-space serve_config letters") {
    xllama::GgufDecisionMeta meta;
    meta.label_source = "serve_config-fallback";
    const auto labs = xllama::h2o_score_labels_for(2, meta);
    REQUIRE(labs.size() == 2);
    CHECK(labs[0] == " A");
    CHECK(labs[1] == " B");
    CHECK(xllama::h2o_score_labels_for(256, meta).empty());
}

TEST_CASE("h2o_score_labels_for prefers GGUF decision.labels") {
    xllama::GgufDecisionMeta meta;
    meta.has_decision_labels = true;
    meta.labels = {" A", " B", " C"};
    const auto labs = xllama::h2o_score_labels_for(2, meta);
    REQUIRE(labs.size() == 2);
    CHECK(labs[0] == " A");
    CHECK(labs[1] == " B");
}

#ifdef QWEN35_VOCAB_GGUF
TEST_CASE("read_gguf_decision_meta reads architecture without llama_model_decision_label") {
    xllama::GgufDecisionMeta meta;
    REQUIRE(xllama::read_gguf_decision_meta(QWEN35_VOCAB_GGUF, meta));
    CHECK(meta.ok);
    CHECK_FALSE(meta.architecture.empty());
    CHECK(meta.decision_type.empty());
    CHECK_FALSE(meta.has_decision_labels);
    CHECK(meta.label_source == "serve_config-fallback");
}
#endif

TEST_CASE("read_gguf_decision_meta reads {arch}.decision.labels from a fixture") {
    const std::string path = "/tmp/xllama-decision-meta-test.gguf";
    {
        gguf_context* ctx = gguf_init_empty();
        REQUIRE(ctx != nullptr);
        gguf_set_val_str(ctx, "general.architecture", "qwen35");
        gguf_set_val_str(ctx, "qwen35.decision.type", "h2o-lightning");
        const char* labs[] = {" A", " B"};
        gguf_set_arr_str(ctx, "qwen35.decision.labels", labs, 2);
        REQUIRE(gguf_write_to_file(ctx, path.c_str(), true));
        gguf_free(ctx);
    }
    xllama::GgufDecisionMeta meta;
    REQUIRE(xllama::read_gguf_decision_meta(path, meta));
    CHECK(meta.architecture == "qwen35");
    CHECK(meta.decision_type == "h2o-lightning");
    REQUIRE(meta.has_decision_labels);
    REQUIRE(meta.labels.size() == 2);
    CHECK(meta.labels[0] == " A");
    CHECK(meta.label_source == "gguf");
}

TEST_CASE("score_label_logits documents multi-token first-token scoring") {
    const float logits[] = {0.0f, 3.0f, 1.0f};
    const auto r = xllama::score_label_logits({" yes", " no"}, {1, 2}, {2, 1}, logits, 3, 1.0f);
    REQUIRE(r.ok);
    CHECK(r.scores[0].token_count == 2);
    CHECK(r.winner == 0);
}
