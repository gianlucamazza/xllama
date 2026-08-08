// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "xllama/gpugemv.h"

#include <cmath>
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
    r.packed_bytes = gpugemv_packed_bytes(256, 256);
    r.packed_gbs = 12.5;
    r.max_abs_err = 1e-4f;
    r.checksum_ok = true;
    r.d3d12_ran = false;
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
    CHECK(row.find("unittest") != std::string::npos);
    CHECK(row.back() == '\n');
}

TEST_CASE("gpugemv: measure on non-D3D12 host reports unavailable") {
    GpugemvResult r = measure_gpugemv(/*n=*/256, /*k=*/256, /*iterations=*/1);
    CHECK(r.n == 256);
    CHECK(r.k == 256);
#if !defined(_WIN32)
    CHECK_FALSE(r.d3d12_ran);
    CHECK_FALSE(r.error_msg.empty());
    CHECK_FALSE(gpugemv_passes_g1(r));
#endif
}
