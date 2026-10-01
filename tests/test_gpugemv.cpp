// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "xllama/gpugemv.h"

#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

using namespace xllama;

TEST_CASE("gpugemv: block size matches ggml Q4_K") {
    CHECK(kGpugemvBlockBytes == 144);
    CHECK(sizeof(GpugemvQ4KBlock) == 144);
    CHECK(kGpugemvQK == 256);
}

TEST_CASE("gpugemv: half roundtrip is stable for small scales") {
    const float vals[] = {0.f, 0.01f, 0.05f, 1.f, -0.5f, 0.001f};
    for (float v : vals) {
        const float back = gpugemv_half_to_float(gpugemv_float_to_half(v));
        CHECK(std::fabs(back - v) < 1e-3f + 0.01f * std::fabs(v));
    }
}

TEST_CASE("gpugemv: dequant + CPU ref are deterministic") {
    constexpr int n = 4;
    constexpr int k = 256; // one block per row
    std::vector<GpugemvQ4KBlock> w(static_cast<std::size_t>(n));
    std::vector<float> x(static_cast<std::size_t>(k));
    std::vector<float> y0(static_cast<std::size_t>(n)), y1(static_cast<std::size_t>(n));
    gpugemv_fill_weights(w.data(), n, k);
    gpugemv_fill_x(x.data(), k);
    gpugemv_cpu_ref(w.data(), x.data(), y0.data(), n, k);
    gpugemv_cpu_ref(w.data(), x.data(), y1.data(), n, k);
    for (int i = 0; i < n; ++i)
        CHECK(y0[static_cast<std::size_t>(i)] == y1[static_cast<std::size_t>(i)]);

    // Single-block dequant produces finite values.
    float tmp[256];
    gpugemv_dequant_block(w[0], tmp);
    for (int i = 0; i < 256; ++i)
        CHECK(std::isfinite(tmp[i]));

    const std::uint32_t c0 = gpugemv_checksum_floats(y0.data(), static_cast<std::size_t>(n));
    const std::uint32_t c1 = gpugemv_checksum_floats(y1.data(), static_cast<std::size_t>(n));
    CHECK(c0 == c1);
}

TEST_CASE("gpugemv: multi-block row GEMV is finite") {
    constexpr int n = 2;
    constexpr int k = 512; // two blocks
    std::vector<GpugemvQ4KBlock> w(static_cast<std::size_t>(n * (k / kGpugemvQK)));
    std::vector<float> x(static_cast<std::size_t>(k)), y(static_cast<std::size_t>(n));
    gpugemv_fill_weights(w.data(), n, k);
    gpugemv_fill_x(x.data(), k);
    gpugemv_cpu_ref(w.data(), x.data(), y.data(), n, k);
    for (int i = 0; i < n; ++i)
        CHECK(std::isfinite(y[static_cast<std::size_t>(i)]));
    CHECK(gpugemv_packed_bytes(n, k) == static_cast<std::size_t>(n) * 2 * kGpugemvBlockBytes);
}

TEST_CASE("gpugemv: soft gate helpers") {
    GpugemvResult fail{};
    fail.packed_gbs = 100.0;
    fail.checksum_ok = true;
    fail.max_abs_err = 0.f;
    fail.d3d12_ran = false;
    CHECK_FALSE(gpugemv_passes_g1(fail));
    CHECK_FALSE(gpugemv_passes_g2(fail));

    GpugemvResult low{};
    low.packed_gbs = 10.0;
    low.checksum_ok = true;
    low.max_abs_err = 0.f;
    low.d3d12_ran = true;
    CHECK(gpugemv_passes_g1(low));
    CHECK_FALSE(gpugemv_passes_g2(low));
    CHECK(kGpugemvSoftPackedGbs == 40.0);

    GpugemvResult pass{};
    pass.packed_gbs = 40.0;
    pass.checksum_ok = true;
    pass.max_abs_err = 1e-3f;
    pass.d3d12_ran = true;
    CHECK(gpugemv_passes_g1(pass));
    CHECK(gpugemv_passes_g2(pass));
}

TEST_CASE("gpugemv: CSV columns match header") {
    GpugemvResult r;
    r.n = 256;
    r.k = 256;
    r.iterations = 1;
    r.run_index = 1;
    r.kernel = GpugemvKernel::Wave32;
    r.packed_bytes = gpugemv_packed_bytes(256, 256);
    r.packed_gbs = 12.5;
    r.packed_gbs_cpu = 11.0;
    r.packed_gbs_h61 = 10.0;
    r.max_abs_err = 1e-4f;
    r.checksum_ok = true;
    r.d3d12_ran = false;
    r.gpu_timestamp = false;
    r.wave_ops = false;
    r.error_msg = "d3d12 unavailable on this platform";
    std::string header = gpugemv_csv_header();
    std::string row = format_gpugemv_row(r, "unittest");
    auto commas = [](const std::string& s) {
        int c = 0;
        for (char ch : s)
            if (ch == ',')
                ++c;
        return c;
    };
    CHECK(commas(header) == commas(row));
    CHECK(commas(header) == 18);
    CHECK(row.find("unittest") != std::string::npos);
    CHECK(row.find("wave32") != std::string::npos);
    CHECK(row.back() == '\n');
    CHECK(std::string(header).find("kernel") == 0);
    CHECK(std::string(header).find("packed_gbs_cpu") != std::string::npos);
    CHECK(std::string(header).find("gpu_timestamp") != std::string::npos);
}

TEST_CASE("gpugemv: kernel names") {
    CHECK(std::string(gpugemv_kernel_name(GpugemvKernel::Naive)) == "naive");
    CHECK(std::string(gpugemv_kernel_name(GpugemvKernel::Wave32)) == "wave32");
    CHECK(std::string(gpugemv_kernel_name(GpugemvKernel::Rows)) == "rows");
    CHECK(std::string(gpugemv_kernel_name(GpugemvKernel::Dot4)) == "dot4");
    CHECK(gpugemv_kernel_needs_sm64(GpugemvKernel::Dot4));
    CHECK_FALSE(gpugemv_kernel_needs_sm64(GpugemvKernel::Rows));
    CHECK_FALSE(gpugemv_kernel_needs_sm64(GpugemvKernel::Wave32));
}

TEST_CASE("gpugemv: dispatch plan") {
    const GpugemvDispatch n8192 = gpugemv_plan_dispatch(8192, GpugemvKernel::Naive);
    CHECK(n8192.threads_per_group == 64);
    CHECK(n8192.rows_per_group == 64);
    CHECK(n8192.groups_x == 128);
    CHECK(n8192.groups_x <= 65535);

    const GpugemvDispatch w8192 = gpugemv_plan_dispatch(8192, GpugemvKernel::Wave32);
    CHECK(w8192.threads_per_group == 32);
    CHECK(w8192.rows_per_group == 1);
    CHECK(w8192.groups_x == 8192);
    CHECK(w8192.groups_x <= 65535);

    const GpugemvDispatch w4 = gpugemv_plan_dispatch(4, GpugemvKernel::Wave32);
    CHECK(w4.threads_per_group == 32);
    CHECK(w4.rows_per_group == 1);
    CHECK(w4.groups_x == 4);
    CHECK(w4.groups_x >= 1);
    CHECK(w4.groups_x <= 65535);
}

TEST_CASE("gpugemv: H6.3 dispatch plan covers every row") {
    const GpugemvKernel kernels[] = {GpugemvKernel::Rows, GpugemvKernel::Dot4};
    for (GpugemvKernel kernel : kernels) {
        const GpugemvDispatch d = gpugemv_plan_dispatch(8192, kernel);
        CHECK(d.threads_per_group == 64);
        CHECK(d.rows_per_group == 4);
        CHECK(d.groups_x == 2048);
        const GpugemvDispatch odd = gpugemv_plan_dispatch(6, kernel);
        CHECK(odd.groups_x == 2);
        CHECK(odd.groups_x * odd.rows_per_group >= 6);
        CHECK(gpugemv_plan_dispatch(0, kernel).groups_x == 0);
    }
}

TEST_CASE("gpugemv: H6.3 lane map covers each block element exactly once") {
    // 16 threads share a block (one ix); every element of the 256 must be
    // read once, lo nibbles from qs byte (e - 64*il) and hi from the same byte.
    for (std::uint32_t ix = 0; ix < 4u; ++ix) {
        int seen[256] = {};
        for (std::uint32_t itid = 0; itid < 16u; ++itid) {
            const GpugemvRowsLane l = gpugemv_rows_lane(ix * 16u + itid);
            CHECK(l.ix == ix);
            CHECK(l.qs_byte % 8u == 0u);
            CHECK(l.qs_byte >= 16u);
            CHECK(l.qs_byte + 8u <= 144u);
            CHECK(l.e_lo / 32u == l.sub_lo);
            CHECK((l.e_lo + 7u) / 32u == l.sub_lo);
            CHECK((l.e_lo + 32u) / 32u == l.sub_lo + 1u);
            // ggml: within a 64-group, qs[l] lo → y[l], hi → y[32+l].
            CHECK(l.qs_byte - 16u == (l.e_lo / 64u) * 32u + (l.e_lo % 64u));
            for (std::uint32_t i = 0; i < 8u; ++i) {
                ++seen[l.e_lo + i];
                ++seen[l.e_lo + 32u + i];
            }
        }
        for (int e = 0; e < 256; ++e)
            CHECK(seen[e] == 1);
    }
}

TEST_CASE("gpugemv: q8 activation roundtrip") {
    constexpr int k = 512;
    std::vector<float> x(k), xq(k);
    std::vector<std::int8_t> qx(k);
    std::vector<float> dx(k / kGpugemvQ8Block);
    gpugemv_fill_x(x.data(), k);
    x[0] = 0.f; // keep one exact zero
    gpugemv_quantize_x_q8(x.data(), k, qx.data(), dx.data());
    gpugemv_dequant_x_q8(qx.data(), dx.data(), k, xq.data());
    for (int i = 0; i < k; ++i) {
        CHECK(qx[static_cast<std::size_t>(i)] >= -127);
        CHECK(std::fabs(xq[static_cast<std::size_t>(i)] - x[static_cast<std::size_t>(i)]) <=
              0.5f * dx[static_cast<std::size_t>(i / kGpugemvQ8Block)] + 1e-7f);
    }
    CHECK(xq[0] == 0.f);

    std::vector<float> zero(kGpugemvQ8Block, 0.f), dz(1, -1.f);
    std::vector<std::int8_t> qz(kGpugemvQ8Block, 1);
    gpugemv_quantize_x_q8(zero.data(), kGpugemvQ8Block, qz.data(), dz.data());
    CHECK(dz[0] == 0.f);
    for (std::int8_t q : qz)
        CHECK(q == 0);
}

TEST_CASE("gpugemv: H6.3 shader emulation matches the CPU reference") {
    constexpr int n = 6;
    constexpr int k = 2048; // nb = 8: two blocks per ix, exercises the stride-4 loop
    std::vector<GpugemvQ4KBlock> w(static_cast<std::size_t>(n * (k / kGpugemvQK)));
    std::vector<float> x(k), ref(n), y(n);
    gpugemv_fill_weights(w.data(), n, k);
    gpugemv_fill_x(x.data(), k);

    gpugemv_cpu_ref(w.data(), x.data(), ref.data(), n, k);
    gpugemv_rows_emulate(w.data(), x.data(), y.data(), n, k);
    CHECK(gpugemv_max_abs_err(ref.data(), y.data(), n) <= kGpugemvMaxAbsErrTol);

    // dot4: G1 compares against cpu_ref on the q8-roundtripped activation.
    std::vector<std::int8_t> qx(k);
    std::vector<float> dx(k / kGpugemvQ8Block), xq(k);
    gpugemv_quantize_x_q8(x.data(), k, qx.data(), dx.data());
    gpugemv_dequant_x_q8(qx.data(), dx.data(), k, xq.data());
    gpugemv_cpu_ref(w.data(), xq.data(), ref.data(), n, k);
    gpugemv_dot4_emulate(w.data(), qx.data(), dx.data(), y.data(), n, k);
    CHECK(gpugemv_max_abs_err(ref.data(), y.data(), n) <= kGpugemvMaxAbsErrTol);
    for (float v : y)
        CHECK(std::isfinite(v));
}

TEST_CASE("gpugemv: LDS transpose indices over nload in {1,16,32}") {
    const std::uint32_t nloads[] = {1u, 16u, 32u};
    for (std::uint32_t nload : nloads) {
        std::uint32_t gs[288];
        for (std::uint32_t& v : gs)
            v = 0xffffffffu;
        for (std::uint32_t pass = 0; pass < 9u; ++pass) {
            for (std::uint32_t lane = 0; lane < nload; ++lane) {
                const std::uint32_t wi = gpugemv_lds_write_index(pass, nload, lane);
                CHECK(wi < 9u * nload);
                gs[wi] = wi;
            }
        }
        for (std::uint32_t lane = 0; lane < nload; ++lane) {
            for (std::uint32_t q = 0; q < 9u; ++q) {
                const std::uint32_t ri = gpugemv_lds_read_index(lane, q);
                CHECK(ri < 9u * nload);
                CHECK(gs[ri] == lane * 9u + q);
            }
        }
        const std::uint32_t chunk = 144u;
        CHECK(gpugemv_lds_load_byte(chunk, 0, nload, 0) == chunk);
        CHECK(gpugemv_lds_load_byte(chunk, 1, nload, 0) == chunk + nload * 16u);
        if (nload > 1)
            CHECK(gpugemv_lds_load_byte(chunk, 0, nload, 1) == chunk + 16u);
    }
}

TEST_CASE("gpugemv: K1 is 8 and G2 stays 40") {
    CHECK(kGpugemvKillPackedGbs == 8.0);
    CHECK(kGpugemvSoftPackedGbs == 40.0);
}

TEST_CASE("gpugemv: ladder at 7.9/8/39.9/40") {
    const double medians[] = {7.9, 8.0, 39.9, 40.0};
    const GpugemvLadder with_g1[] = {GpugemvLadder::K1Kill, GpugemvLadder::K2Park,
                                     GpugemvLadder::K2Park, GpugemvLadder::K3Open};
    for (int i = 0; i < 4; ++i) {
        CHECK(gpugemv_ladder(medians[i], true) == with_g1[i]);
        CHECK(gpugemv_ladder(medians[i], false) == GpugemvLadder::NotAVerdict);
    }

    GpugemvKernelSummary none[1] = {};
    none[0].kernel = GpugemvKernel::Wave32;
    none[0].median_packed_gbs = 50.0;
    none[0].g1_all3 = false;
    none[0].ladder = GpugemvLadder::NotAVerdict;
    CHECK(gpugemv_campaign_verdict(none, 1) == GpugemvLadder::NotAVerdict);

    GpugemvKernelSummary pass[1] = {};
    pass[0].kernel = GpugemvKernel::Wave32;
    pass[0].median_packed_gbs = 8.0;
    pass[0].g1_all3 = true;
    pass[0].ladder = GpugemvLadder::K2Park;
    CHECK(gpugemv_campaign_verdict(pass, 1) == GpugemvLadder::K2Park);

    GpugemvKernelSummary naive_only[1] = {};
    naive_only[0].kernel = GpugemvKernel::Naive;
    naive_only[0].median_packed_gbs = 50.0;
    naive_only[0].g1_all3 = true;
    naive_only[0].ladder = GpugemvLadder::K3Open;
    CHECK(gpugemv_campaign_verdict(naive_only, 1) == GpugemvLadder::NotAVerdict);

    // H6.3: best G1-passing median among wave32/rows/dot4 decides.
    GpugemvKernelSummary three[3] = {};
    three[0].kernel = GpugemvKernel::Wave32;
    three[0].median_packed_gbs = 25.4;
    three[0].g1_all3 = true;
    three[1].kernel = GpugemvKernel::Rows;
    three[1].median_packed_gbs = 41.0;
    three[1].g1_all3 = true;
    three[2].kernel = GpugemvKernel::Dot4;
    three[2].median_packed_gbs = 90.0;
    three[2].g1_all3 = false; // e.g. sm6.4 unsupported: never a verdict
    CHECK(gpugemv_campaign_verdict(three, 3) == GpugemvLadder::K3Open);
    three[1].median_packed_gbs = 30.0;
    CHECK(gpugemv_campaign_verdict(three, 3) == GpugemvLadder::K2Park);
}

TEST_CASE("gpugemv: measure on non-D3D12 host reports unavailable") {
    GpugemvResult r = measure_gpugemv(/*n=*/256, /*k=*/256, /*iterations=*/1);
    CHECK(r.n == 256);
    CHECK(r.k == 256);
    CHECK(r.kernel == GpugemvKernel::Wave32);
#if !defined(_WIN32)
    CHECK_FALSE(r.d3d12_ran);
    CHECK(r.packed_gbs == 0.0);
    CHECK_FALSE(r.error_msg.empty());
    CHECK_FALSE(gpugemv_passes_g1(r));
#endif
    std::vector<GpugemvResult> rows;
    measure_gpugemv_each(256, 256, 3, GpugemvKernel::Wave32, &rows);
#if !defined(_WIN32)
    CHECK(rows.size() == 1);
    CHECK(rows[0].kernel == GpugemvKernel::Wave32);
    CHECK_FALSE(rows[0].d3d12_ran);
    CHECK(rows[0].packed_gbs == 0.0);
#endif
    const GpugemvKernel h63[] = {GpugemvKernel::Rows, GpugemvKernel::Dot4};
    for (GpugemvKernel kernel : h63) {
        std::vector<GpugemvResult> out;
        measure_gpugemv_each(256, 256, 3, kernel, &out);
        REQUIRE(out.size() >= 1);
        CHECK(out[0].kernel == kernel);
#if !defined(_WIN32)
        CHECK(out.size() == 1);
        CHECK_FALSE(out[0].d3d12_ran);
        CHECK_FALSE(gpugemv_passes_g1(out[0]));
        CHECK(out[0].expected_y_checksum != 0u);
#endif
    }
}
