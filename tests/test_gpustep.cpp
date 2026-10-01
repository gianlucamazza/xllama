// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "xllama/gpustep.h"

#include <cmath>
#include <string>
#include <vector>

using namespace xllama;

TEST_CASE("gpustep: Q4_K_M use_more_bits matches llama-quant at 36 layers") {
    int promoted = 0;
    for (int il = 0; il < 36; ++il)
        promoted += gpustep_q4km_more_bits(il, 36) ? 1 : 0;
    CHECK(promoted == 18);
    CHECK(gpustep_q4km_more_bits(0, 36));
    CHECK(gpustep_q4km_more_bits(6, 36));
    CHECK_FALSE(gpustep_q4km_more_bits(4, 36));
    CHECK(gpustep_q4km_more_bits(35, 36));
}

TEST_CASE("gpustep: model tables — splits, matmuls and bytes per token") {
    const GpustepModel coder = gpustep_model(GpustepModelId::Coder3B);
    CHECK(coder.name == "qwen25-coder-3b");
    CHECK(coder.splits.size() == 36 * 4 + 1);
    CHECK(gpustep_model_matmuls(coder) == 36 * 7 + 1);
    CHECK(gpustep_model_bytes(coder) == 1922979840ull); // ~ the Q4_K_M file size

    const GpustepModel lfm12 = gpustep_model(GpustepModelId::Lfm12B);
    CHECK(lfm12.splits.size() == 16 * 4 + 1);
    CHECK(gpustep_model_bytes(lfm12) == 692846592ull);

    const GpustepModel lfm350 = gpustep_model(GpustepModelId::Lfm350M);
    CHECK(lfm350.splits.size() == 16 * 4 + 1);
    // GGUF header: 161.6 MB Q4_0 + 55.05 MB Q6_K token_embd.
    CHECK(gpustep_model_bytes(lfm350) == 216662016ull);

    // Every matmul is runnable by the rows kernel.
    const GpustepModel all[] = {coder, lfm12, lfm350};
    for (const auto& m : all) {
        for (const auto& s : m.splits) {
            REQUIRE_FALSE(s.mm.empty());
            for (const auto& mm : s.mm) {
                CHECK(mm.k % 256 == 0);
                CHECK(mm.k == s.mm.front().k); // one shared input per split
                const int rows = gpustep_sim_rows(mm);
                CHECK((rows + 3) / 4 <= 65535);
            }
        }
    }
}

TEST_CASE("gpustep: Q6_K simulated as bytes-equivalent Q4_K rows") {
    GpustepMatmul lm;
    lm.n = 151936;
    lm.k = 2048;
    lm.quant = GpustepQuant::Q6K;
    CHECK(gpustep_sim_rows(lm) == 221576);
    const std::size_t real = gpustep_matmul_bytes(lm);
    GpustepMatmul sim = lm;
    sim.n = gpustep_sim_rows(lm);
    sim.quant = GpustepQuant::Q4K;
    const std::size_t simulated = gpustep_matmul_bytes(sim);
    CHECK(simulated >= real);
    CHECK(simulated - real <= 4u * (2048 / 256) * 144);

    GpustepMatmul q4 = lm;
    q4.quant = GpustepQuant::Q4K;
    CHECK(gpustep_sim_rows(q4) == lm.n);
}

TEST_CASE("gpustep: cost model and projection") {
    const GpustepModel coder = gpustep_model(GpustepModelId::Coder3B);
    CHECK(gpustep_projected_step_ms(coder, 0.0, 143.0) == doctest::Approx(13.4474).epsilon(1e-4));
    CHECK(gpustep_projected_step_ms(coder, 100.0, 143.0) == doctest::Approx(27.9474).epsilon(1e-4));
    CHECK(gpustep_projected_step_ms(coder, 50.0, 0.0) == 0.0);
    // Gate arithmetic: 40 ms + 11 ms of CPU work is 1.4x of 14.0 tok/s.
    CHECK(gpustep_projected_toks(kGpustepStepGateMs, 11.0) >= 1.4 * 14.0 - 0.1);
    CHECK(gpustep_projected_toks(0.0, 0.0) == 0.0);
}

TEST_CASE("gpustep: CSV row round-trips through parse") {
    GpustepRow r;
    r.kind = "step";
    r.variant = "qwen25-coder-3b/sync";
    r.process = "headless";
    r.iterations = 50;
    r.median = 23.5;
    r.p90 = 25.25;
    r.unit = "ms";
    r.bytes = 1922979840ull;
    r.ok = true;
    r.d3d12_ran = true;
    r.uma = true;
    r.error = "a,b"; // commas are sanitized
    const std::string line = format_gpustep_row(r, "xbox-series-s");
    int commas = 0;
    for (char ch : line)
        commas += ch == ',' ? 1 : 0;
    CHECK(commas == 14);
    std::string header = gpustep_csv_header();
    int hcommas = 0;
    for (char ch : header)
        hcommas += ch == ',' ? 1 : 0;
    CHECK(hcommas == 14);

    GpustepRow back;
    REQUIRE(parse_gpustep_row(line, &back));
    CHECK(back.kind == r.kind);
    CHECK(back.variant == r.variant);
    CHECK(back.process == r.process);
    CHECK(back.iterations == 50);
    CHECK(back.median == doctest::Approx(23.5));
    CHECK(back.bytes == r.bytes);
    CHECK(back.ok);
    CHECK(back.uma);
    CHECK_FALSE(back.cc_uma);
    CHECK(back.error == "a b");
    CHECK_FALSE(parse_gpustep_row(header, &back));
    CHECK_FALSE(parse_gpustep_row("too,few", &back));
}

namespace {

GpustepRow row(const char* kind, const std::string& variant, const char* process, double median,
               bool ok = true) {
    GpustepRow r;
    r.kind = kind;
    r.variant = variant;
    r.process = process;
    r.median = median;
    r.ok = ok;
    r.d3d12_ran = true;
    return r;
}

std::vector<GpustepRow> full_run(double rt_us, double sync_ms, double nosync_ms, bool inproc_ok,
                                 double heap_upload = 50.0) {
    std::vector<GpustepRow> v;
    const char* procs[] = {"headless", "inproc"};
    for (const char* p : procs) {
        const bool ok = std::string(p) == "headless" || inproc_ok;
        v.push_back(row("heap", "default", p, 143.0, ok));
        v.push_back(row("heap", "upload", p, heap_upload));
        v.push_back(row("heap", "custom", p, 0.0, false)); // refused heap: never fails G1
        v.push_back(row("rt", "event-copy", p, rt_us * 2, ok));
        v.push_back(row("rt", "spin-copy", p, rt_us, ok));
        v.push_back(row("rt", "spin-zerocopy", p, 0.0, false));
        v.push_back(row("step", "qwen25-coder-3b/sync", p, sync_ms));
        v.push_back(row("step", "qwen25-coder-3b/nosync", p, nosync_ms));
    }
    return v;
}

} // namespace

TEST_CASE("gpustep: D1 ladder") {
    GpustepVerdict v = gpustep_evaluate(full_run(60.0, 30.0, 20.0, true));
    CHECK(v.g1_headless);
    CHECK(v.g1_inproc);
    CHECK(v.rt_best_us == doctest::Approx(60.0));
    CHECK(v.d1a);
    CHECK(v.d1b);
    CHECK(v.d1d);
    CHECK(v.d1c == "default+copy"); // 50 < 0.9 * 143
    CHECK(v.ladder == GpustepLadder::D2MatmulOnly);

    // Inclusive thresholds.
    v = gpustep_evaluate(full_run(kGpustepRtGateUs, kGpustepStepGateMs, 20.0, true));
    CHECK(v.ladder == GpustepLadder::D2MatmulOnly);

    // Sync too expensive, GPU work alone fits → fused splits.
    v = gpustep_evaluate(full_run(60.0, 45.0, 30.0, true));
    CHECK_FALSE(v.d1b);
    CHECK(v.ladder == GpustepLadder::D2Fused);
    v = gpustep_evaluate(full_run(150.0, 35.0, 30.0, true));
    CHECK_FALSE(v.d1a);
    CHECK(v.ladder == GpustepLadder::D2Fused);

    // Even the no-sync step misses the gate.
    v = gpustep_evaluate(full_run(60.0, 60.0, 45.0, true));
    CHECK(v.ladder == GpustepLadder::Park3B);

    // In-XAML failure parks the GUI regardless of speed.
    v = gpustep_evaluate(full_run(60.0, 30.0, 20.0, false));
    CHECK_FALSE(v.d1d);
    CHECK(v.ladder == GpustepLadder::ParkGui);

    // Host-visible heap within 0.9x of DEFAULT.
    v = gpustep_evaluate(full_run(60.0, 30.0, 20.0, true, 130.0));
    CHECK(v.d1c == "host-visible");
}

TEST_CASE("gpustep: incomplete runs are not a verdict") {
    std::vector<GpustepRow> headless_only;
    for (const auto& r : full_run(60.0, 30.0, 20.0, true))
        if (r.process == "headless")
            headless_only.push_back(r);
    CHECK(gpustep_evaluate(headless_only).ladder == GpustepLadder::NotAVerdict);

    std::vector<GpustepRow> bad_g1 = full_run(60.0, 30.0, 20.0, true);
    bad_g1[0].ok = false; // headless DEFAULT heap residual mismatch
    CHECK(gpustep_evaluate(bad_g1).ladder == GpustepLadder::NotAVerdict);

    CHECK(gpustep_evaluate({}).ladder == GpustepLadder::NotAVerdict);
    CHECK(std::string(gpustep_ladder_name(GpustepLadder::D2Fused)) == "D2-fused");
}

TEST_CASE("gpustep: measure on non-D3D12 host reports unavailable") {
    std::vector<GpustepRow> rows;
    measure_gpustep("host", &rows);
    REQUIRE(rows.size() >= 1);
#if !defined(_WIN32)
    CHECK(rows.size() == 1);
    CHECK(rows[0].kind == "caps");
    CHECK(rows[0].process == "host");
    CHECK_FALSE(rows[0].d3d12_ran);
    CHECK_FALSE(rows[0].error.empty());
    CHECK(gpustep_evaluate(rows).ladder == GpustepLadder::NotAVerdict);
#endif
}
