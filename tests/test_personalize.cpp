// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "xllama/personalize.h"
#include "xllama/training.h"

#include <doctest/doctest.h>

#include <chrono>
#include <cstdio>
#include <fstream>
#include <string>

using namespace xllama;

TEST_CASE("last_block_param_filter: SmolLM2-360M shape (block 31)") {
    const auto f = last_block_param_filter(31);
    REQUIRE(f.size() == 6);
    CHECK(f[0] == "blk.31.attn_q.weight");
    CHECK(f[1] == "blk.31.attn_output.weight");
    CHECK(f[2] == "blk.31.ffn_gate.weight");
    CHECK(f[3] == "blk.31.ffn_up.weight");
    CHECK(f[4] == "blk.31.ffn_down.weight");
    CHECK(f[5] == "output_norm.weight");
    // Must not include K/V (no backward through set_rows on this pin).
    for (const auto& p : f) {
        CHECK(p.find("attn_k") == std::string::npos);
        CHECK(p.find("attn_v") == std::string::npos);
    }
}

TEST_CASE("last_block_param_filter: negative returns empty") {
    CHECK(last_block_param_filter(-1).empty());
}

TEST_CASE("guess_last_block_from_model_id: known catalogue ids") {
    CHECK(guess_last_block_from_model_id("smollm2-360m-cpu-int4") == 31);
    CHECK(guess_last_block_from_model_id("training/base-f16.gguf") == 31);
    CHECK(guess_last_block_from_model_id("models/smollm2-1.7b-cpu-int4") == 23);
    CHECK(guess_last_block_from_model_id("lfm25-350m") == -1);
}

TEST_CASE("build_personalize_job: happy path matches device-train recipe") {
    PersonalizeSpec spec;
    spec.base_model = "training/base-f16.gguf";
    spec.dataset_path = "training/samples.jsonl";
    spec.out_dir = "training/out/personalized";
    TrainingJob job;
    std::string err;
    REQUIRE(build_personalize_job(spec, job, &err));
    CHECK(err.empty());
    CHECK(job.method == TrainMethod::PartialFt);
    CHECK(job.device == TrainDevice::Device);
    CHECK(job.epochs == 8);
    CHECK(job.n_ctx_train == 256);
    CHECK(job.param_filter == last_block_param_filter(31));
    CHECK(job.base_model == "training/base-f16.gguf");
    CHECK(validate_training_job(job));
}

TEST_CASE("build_personalize_job: rejects empty base and unknown last_block") {
    PersonalizeSpec spec;
    TrainingJob job;
    std::string err;
    CHECK_FALSE(build_personalize_job(spec, job, &err));
    CHECK(err.find("base_model") != std::string::npos);

    spec.base_model = "lfm25-350m";
    err.clear();
    CHECK_FALSE(build_personalize_job(spec, job, &err));
    CHECK(err.find("last_block") != std::string::npos);

    spec.last_block = 15;
    err.clear();
    REQUIRE(build_personalize_job(spec, job, &err));
    CHECK(job.param_filter[0] == "blk.15.attn_q.weight");
}

TEST_CASE("format_personalize_job_json: round-trips through the job parser") {
    PersonalizeSpec spec;
    spec.base_model = "training/base-f16.gguf";
    TrainingJob job;
    REQUIRE(build_personalize_job(spec, job));
    const std::string json = format_personalize_job_json(job);
    TrainingJob parsed;
    std::string err;
    REQUIRE(parse_training_job_json(json, parsed, &err));
    CHECK(err.empty());
    CHECK(parsed.method == TrainMethod::PartialFt);
    CHECK(parsed.device == TrainDevice::Device);
    CHECK(parsed.base_model == job.base_model);
    CHECK(parsed.param_filter == job.param_filter);
    CHECK(parsed.epochs == job.epochs);
}

TEST_CASE("personalized_manifest_override_json: shape for LocalState merge") {
    const std::string j = personalized_manifest_override_json(
        "personalized", "Personalized (from your feedback)", 400000000);
    CHECK(j.find("\"name\": \"personalized\"") != std::string::npos);
    CHECK(j.find("\"kind\": \"gguf\"") != std::string::npos);
    CHECK(j.find("\"filename\": \"model.gguf\"") != std::string::npos);
    CHECK(j.find("400000000") != std::string::npos);
}

TEST_CASE("count_usable_preference_samples: skips dislike and empty") {
    const char* path = "test-samples-personalize.jsonl";
    {
        std::ofstream out(path);
        out << "{\"label\":\"like\",\"messages\":[{\"role\":\"user\",\"content\":\"a\"}]}\n";
        out << "{\"label\":\"dislike\",\"messages\":[{\"role\":\"user\",\"content\":\"b\"}]}\n";
        out << "{\"label\":\"correction\",\"messages\":[{\"role\":\"user\",\"content\":\"c\"}]}\n";
        out << "\n";
        out << "not-json\n";
    }
    CHECK(count_usable_preference_samples(path) == 2);
    CHECK(count_usable_preference_samples("no-such-file.jsonl") == 0);
    std::remove(path);
}

TEST_CASE("parse_train_result_done") {
    CHECK(parse_train_result_done("ok\n") == "ok");
    CHECK(parse_train_result_done("  fail") == "fail");
    CHECK(parse_train_result_done("").empty());
    CHECK(parse_train_result_done("maybe").empty());
}

TEST_CASE("format_train_progress_json: stable keys") {
    const std::string j = format_train_progress_json("train", 2, 8, 1, 10, 1.5);
    CHECK(j.find("\"stage\":\"train\"") != std::string::npos);
    CHECK(j.find("\"epoch\":2") != std::string::npos);
    CHECK(j.find("\"epochs\":8") != std::string::npos);
    CHECK(j.find("\"loss\":1.500000") != std::string::npos);
}

TEST_CASE("personalized manifest preserves other overrides and refuses invalid local state") {
    const auto path =
        std::filesystem::temp_directory_path() / "xllama-personalized-manifest-test.json";
    const std::string old = R"({"channel":"lab","models":[{"name":"a","unknown":1},{"name":"b"}]})";
    std::ofstream(path) << old;
    std::string err;
    REQUIRE(publish_personalized_manifest(path, 1024, &err));
    REQUIRE(publish_personalized_manifest(path, 2048, &err));
    std::ifstream file(path);
    const std::string saved((std::istreambuf_iterator<char>(file)), {});
    file.close();
    CHECK(saved.find("\"channel\": \"lab\"") != saved.npos);
    CHECK(saved.find("\"unknown\": 1") != saved.npos);
    CHECK(saved.find("\"name\": \"b\"") != saved.npos);
    CHECK(saved.find("1024") == saved.npos);
    CHECK(saved.find("2048") != saved.npos);
    std::ofstream(path) << "{bad";
    CHECK_FALSE(publish_personalized_manifest(path, 12, &err));
    std::ifstream invalid(path);
    CHECK(std::string(std::istreambuf_iterator<char>(invalid), {}) == "{bad");
    invalid.close();
    std::filesystem::remove(path);
}

TEST_CASE("personalized model publication restores all three files after interruption") {
    namespace fs = std::filesystem;
    const auto root = fs::temp_directory_path() /
                      ("xllama-publish-" +
                       std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    fs::create_directories(root);
    const auto merged = root / "merged.gguf";
    const auto manifest = root / "manifest.json";
    const auto dir = root / "models" / "personalized";
    const auto model = dir / "model.gguf";
    const auto marker = dir / ".complete";
    const auto read = [](const fs::path& path) {
        std::ifstream in(path, std::ios::binary);
        return std::string(std::istreambuf_iterator<char>(in), {});
    };
    std::ofstream(manifest) << R"({"models":[{"name":"custom"}]})";
    std::ofstream(merged) << "trained generation";
    std::string error;
    REQUIRE(publish_personalized_model(root, merged, &error));
    const auto previous_manifest = read(manifest);
    CHECK(previous_manifest.find("custom") != previous_manifest.npos);
    REQUIRE(read(model) == "trained generation");
    REQUIRE(read(marker) == "ok");

    for (const auto* abandoned : {".publish-stage", ".publish-done"}) {
        fs::create_directory(dir / abandoned);
        std::ofstream(dir / abandoned / "weights") << "disposable model bytes";
        REQUIRE(recover_personalized_publication(root, &error));
        CHECK_FALSE(fs::exists(dir / abandoned));
        CHECK(read(model) == "trained generation");
        CHECK(read(manifest) == previous_manifest);
        CHECK(read(marker) == "ok");
    }
    const auto journal = dir / ".publish-pending";
    fs::create_directory(journal);
    fs::rename(model, journal / "old-model");
    fs::copy_file(manifest, journal / "old-manifest");
    fs::copy_file(marker, journal / "old-marker");
    std::ofstream(model) << "interrupted generation";
    std::ofstream(manifest) << "interrupted manifest";
    fs::remove(marker);
    REQUIRE(recover_personalized_publication(root, &error));
    CHECK(read(model) == "trained generation");
    CHECK(read(manifest) == previous_manifest);
    CHECK(read(marker) == "ok");
    REQUIRE(recover_personalized_publication(root, &error));

    std::ofstream(manifest) << "invalid existing override";
    std::ofstream(merged) << "next generation";
    CHECK_FALSE(publish_personalized_model(root, merged, &error));
    CHECK(read(model) == "trained generation");
    CHECK(read(manifest) == "invalid existing override");
    CHECK(read(marker) == "ok");
    fs::remove_all(root);
}
