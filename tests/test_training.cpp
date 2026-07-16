// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

#include "xllama/training.h"

#include <doctest/doctest.h>
#include <string>

using namespace xllama;

static TrainingJob make_valid_job() {
    TrainingJob j;
    j.schema_version = 1;
    j.name = "unit";
    j.method = TrainMethod::LoraPeft;
    j.device = TrainDevice::Host;
    j.base_model = "HuggingFaceTB/SmolLM2-360M-Instruct";
    j.dataset_path = "training/datasets/toy_marker.jsonl";
    j.out_dir = "training/out/unit";
    j.lora_rank = 8;
    j.lora_alpha = 16;
    j.steps = 10;
    j.seed = 1;
    return j;
}

TEST_CASE("training_device_supported: host yes, device no") {
    CHECK(training_device_supported(TrainDevice::Host));
    CHECK_FALSE(training_device_supported(TrainDevice::Device));
}

TEST_CASE("training_stage_name covers all stages") {
    const TrainStage stages[] = {
        TrainStage::Prepare, TrainStage::Train,       TrainStage::ExportAdapter,
        TrainStage::Merge,   TrainStage::Evaluate,    TrainStage::Publish,
    };
    for (TrainStage s : stages) {
        const char* n = training_stage_name(s);
        CHECK(n != nullptr);
        CHECK(n[0] != '\0');
        CHECK(std::string(n) != "unknown");
    }
}

TEST_CASE("validate_training_job accepts valid host lora job") {
    auto j = make_valid_job();
    std::string err;
    CHECK(validate_training_job(j, &err));
    CHECK(err.empty());
}

TEST_CASE("validate_training_job rejects missing base_model") {
    auto j = make_valid_job();
    j.base_model.clear();
    std::string err;
    CHECK_FALSE(validate_training_job(j, &err));
    CHECK(err.find("base_model") != std::string::npos);
}

TEST_CASE("validate_training_job rejects device (exploration reserved)") {
    auto j = make_valid_job();
    j.device = TrainDevice::Device;
    std::string err;
    CHECK_FALSE(validate_training_job(j, &err));
    CHECK(err.find("exploration") != std::string::npos);
}

TEST_CASE("validate_training_job rejects full fine-tune reserved method") {
    auto j = make_valid_job();
    j.method = TrainMethod::FullFtReserved;
    std::string err;
    CHECK_FALSE(validate_training_job(j, &err));
    CHECK(err.find("lora_peft") != std::string::npos);
}

TEST_CASE("parse_training_job_json loads marker job shape") {
    const char* json = R"json({
      "schema_version": 1,
      "name": "smollm2-360m-marker",
      "method": "lora_peft",
      "device": "host",
      "base_model": "HuggingFaceTB/SmolLM2-360M-Instruct",
      "dataset": "training/datasets/toy_marker.jsonl",
      "out_dir": "training/out/smollm2-360m-marker",
      "lora": { "rank": 8, "alpha": 16, "steps": 120, "seed": 42 },
      "eval": { "prompt": "xllama secret", "expect_contains": "XLLAMA-LORA-OK" },
      "merge": true,
      "quantize": null
    })json";
    TrainingJob job;
    std::string err;
    REQUIRE(parse_training_job_json(json, job, &err));
    CHECK(err.empty());
    CHECK(job.name == "smollm2-360m-marker");
    CHECK(job.method == TrainMethod::LoraPeft);
    CHECK(job.device == TrainDevice::Host);
    CHECK(job.base_model == "HuggingFaceTB/SmolLM2-360M-Instruct");
    CHECK(job.dataset_path == "training/datasets/toy_marker.jsonl");
    CHECK(job.lora_rank == 8);
    CHECK(job.lora_alpha == 16);
    CHECK(job.steps == 120);
    CHECK(job.seed == 42);
    CHECK(job.eval_prompt == "xllama secret");
    CHECK(job.eval_expect_contains == "XLLAMA-LORA-OK");
    CHECK(job.do_merge);
    CHECK_FALSE(job.do_quantize);
    CHECK(validate_training_job(job, &err));
}

TEST_CASE("parse_training_job_json rejects unknown method") {
    const char* json = R"json({
      "name": "x",
      "method": "not_a_method",
      "device": "host",
      "base_model": "m",
      "dataset": "d",
      "out_dir": "o"
    })json";
    TrainingJob job;
    std::string err;
    CHECK_FALSE(parse_training_job_json(json, job, &err));
    CHECK(!err.empty());
}

TEST_CASE("format_training_job_summary includes name and method") {
    auto j = make_valid_job();
    const std::string s = format_training_job_summary(j);
    CHECK(s.find("unit") != std::string::npos);
    CHECK(s.find("lora_peft") != std::string::npos);
    CHECK(s.find("host") != std::string::npos);
}

TEST_CASE("training_capabilities table is non-empty and consistent") {
    const TrainingCapabilityInfo* caps = nullptr;
    const size_t n = training_capabilities(&caps);
    REQUIRE(n >= 7);
    REQUIRE(caps != nullptr);
    for (size_t i = 0; i < n; ++i) {
        CHECK(caps[i].name != nullptr);
        CHECK(caps[i].name[0] != '\0');
        CHECK(caps[i].status != nullptr);
        CHECK(caps[i].reason != nullptr);
        // available implies status "available"
        if (caps[i].available)
            CHECK(std::string(caps[i].status) == "available");
        else
            CHECK(std::string(caps[i].status) != "available");
    }
}

TEST_CASE("training_capability_available: host PEFT yes, device FT no") {
    CHECK(training_capability_available(TrainingCapability::HostPeftLora));
    CHECK(training_capability_available(TrainingCapability::HostMergeGguf));
    CHECK_FALSE(training_capability_available(TrainingCapability::DeviceLlamaFinetune));
    CHECK_FALSE(training_capability_available(TrainingCapability::DeviceOrtOnDeviceTraining));
    CHECK_FALSE(training_capability_available(TrainingCapability::RuntimeLoraLoadLlama));
}

TEST_CASE("training_capability_info returns RE reasons") {
    const auto* peft = training_capability_info(TrainingCapability::HostPeftLora);
    REQUIRE(peft != nullptr);
    CHECK(peft->available);
    CHECK(std::string(peft->status) == "available");

    const auto* dml = training_capability_info(TrainingCapability::RuntimeAdapterLoadOrtGenAI);
    REQUIRE(dml != nullptr);
    CHECK_FALSE(dml->available);
    CHECK(std::string(dml->reason).find("DML") != std::string::npos);

    const auto* ft = training_capability_info(TrainingCapability::DeviceLlamaFinetune);
    REQUIRE(ft != nullptr);
    CHECK(std::string(ft->status) == "rejected");
}
