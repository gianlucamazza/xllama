// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#include "floppylm_internal.h"
#include "xllama/floppylm.h"
#include <algorithm>
#include <cmath>
#include <doctest/doctest.h>
#include <limits>
using namespace xllama::floppy;
namespace {
Json config() {
    return {{"vocab", 256},      {"d", 16},
            {"n_layers", 2},     {"n_heads", 2},
            {"d_ff", 32},        {"ctx", 8},
            {"mlp", "gelu"},     {"core_fmt", "ternary"},
            {"emb_fmt", "4bit"}, {"scale_policy", "row16"},
            {"delta", 0.5},      {"qk_norm", false}};
}
} // namespace
TEST_CASE("floppylm codec zero rows, frozen artifacts and section accounting") {
    for (auto policy : {"row16", "row8log", "tensor16"})
        for (auto core : {"ternary", "2bit"}) {
            auto cfg = config();
            cfg["scale_policy"] = policy;
            cfg["core_fmt"] = core;
            Model m(Json{{"config", cfg}});
            auto bytes = pack(m);
            auto frozen = unpack(bytes);
            CHECK(pack(frozen) == bytes);
            for (const auto& t : frozen.tensors)
                CHECK(std::all_of(t.w.begin(), t.w.end(), [](float v) { return v == 0.f; }));
            auto sections = artifact_sections(bytes);
            size_t total = 0;
            for (auto it = sections.begin(); it != sections.end(); ++it)
                total += it.value().get<size_t>();
            CHECK(total == bytes.size());
            CHECK_THROWS(frozen.state());
            frozen.tensors[0].w[0] = 1;
            CHECK_THROWS(pack(frozen));
            bytes.pop_back();
            CHECK_THROWS(unpack(bytes));
        }
}
TEST_CASE("floppylm rejects invalid model input and nonfinite weights") {
    auto cfg = config();
    cfg["d"] = 16.5;
    CHECK_THROWS(Model(Json{{"config", cfg}}));
    Model m(Json{{"config", config()}});
    m.tensors[0].w[0] = std::numeric_limits<float>::infinity();
    CHECK_THROWS(pack(m));
    m.tensors[0].w[0] = std::numeric_limits<float>::quiet_NaN();
    CHECK_THROWS(pack(m));
    m.tensors[0].w[0] = 1e10f;
    CHECK_THROWS(pack(m));
}
TEST_CASE("floppylm graph uniform model has known likelihood") {
    Model m(Json{{"config", config()}});
    Engine engine(m.cfg, 1, 8, 1, true);
    auto s = engine.evaluate(m, std::vector<int32_t>(8, 0), std::vector<int32_t>(8, 1));
    CHECK(s.loss == doctest::Approx(std::log(256.)).epsilon(1e-6));
    for (float v : s.logits)
        CHECK(v == 0.f);
    auto frozen = unpack(pack(m));
    CHECK_THROWS(engine.evaluate(frozen, std::vector<int32_t>(8, 0), std::vector<int32_t>(8, 1)));
}
TEST_CASE("floppylm strict job contract") {
#ifdef __linux__
    CHECK_THROWS_WITH(write_bytes("/dev/full", Bytes{1}), "output flush failed");
#endif
    xllama::TrainingJob job;
    std::string err;
    Json j = {{"schema_version", 1},          {"name", "test"},
              {"method", "floppylm"},         {"device", "host"},
              {"bundle_path", "bundle.json"}, {"out_dir", "new-run"}};
    CHECK(xllama::validate_floppylm_job_json(j.dump(), job, &err));
    j["learning_rate"] = 0.2;
    CHECK_FALSE(xllama::validate_floppylm_job_json(j.dump(), job, &err));
    j.erase("learning_rate");
    CHECK_FALSE(xllama::validate_floppylm_job_json(j.dump() + " trailing", job, &err));
}
