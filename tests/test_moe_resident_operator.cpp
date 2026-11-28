// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "ggml-cpp.h"
#include "ggml-cpu.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>

namespace {
constexpr int kWidth = 256;
constexpr int kRows = 3;
constexpr int kExperts = 4;
constexpr int kSelected = 2;
constexpr int kTokens = 2;
constexpr int32_t kRoutes[kTokens][kSelected] = {{2, 0}, {1, 2}};
using Row = std::array<float, kWidth>;

void check_resident_operator(ggml_type type, int activation_channels) {
    INFO("weights=" << ggml_type_name(type) << ", activation channels=" << activation_channels);
    ggml_init_params params{};
    params.mem_size = 2u << 20;
    ggml_context_ptr ctx(ggml_init(params));
    REQUIRE(ctx != nullptr);
    auto* weights = ggml_new_tensor_3d(ctx.get(), type, kWidth, kRows, kExperts);
    auto* activations =
        ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, kWidth, activation_channels, kTokens);
    auto* ids = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_I32, kSelected, kTokens);

    std::array<std::array<Row, kRows>, kExperts> scalar_weights;
    for (int expert = 0; expert < kExperts; ++expert) {
        for (int output = 0; output < kRows; ++output) {
            auto& row = scalar_weights[expert][output];
            for (int col = 0; col < kWidth; ++col)
                row[col] = static_cast<float>(24 * (expert + 1) + 4 * output + col % 17 - 8);
            auto* packed = static_cast<uint8_t*>(weights->data) + expert * weights->nb[2] +
                           output * weights->nb[1];
            if (type == GGML_TYPE_F32) {
                std::memcpy(packed, row.data(), sizeof(row));
            } else {
                REQUIRE(ggml_quantize_chunk(type, row.data(), packed, 0, 1, kWidth, nullptr) ==
                        weights->nb[1]);
                // Decode the stored weights for a scalar oracle independent of GGML's dot kernel.
                ggml_get_type_traits(type)->to_float(packed, row.data(), kWidth);
            }
        }
    }

    std::array<std::array<Row, kSelected>, kTokens> inputs{};
    for (int token = 0; token < kTokens; ++token) {
        for (int slot = 0; slot < kSelected; ++slot)
            ggml_set_i32_nd(ids, slot, token, 0, 0, kRoutes[token][slot]);
        for (int channel = 0; channel < activation_channels; ++channel) {
            auto& input = inputs[token][channel];
            float sign = token == 0 ? 1.0f : -1.0f;
            // A magnitude of 127 gives the CPU Q8_K activation block a unit scale.
            input[7 + 37 * token + 19 * channel] = sign * 127.0f;
            input[134 + 29 * token + 23 * channel] = sign * (32 + 16 * channel);
            std::memcpy(ggml_get_data_f32(activations) +
                            (token * activation_channels + channel) * kWidth,
                        input.data(), sizeof(input));
        }
    }

    auto* result = ggml_mul_mat_id(ctx.get(), weights, activations, ids);
    auto* graph = ggml_new_graph(ctx.get());
    ggml_build_forward_expand(graph, result);
    REQUIRE(ggml_graph_compute_with_ctx(ctx.get(), graph, 2) == GGML_STATUS_SUCCESS);
    REQUIRE(result->ne[0] == kRows);
    REQUIRE(result->ne[1] == kSelected);
    REQUIRE(result->ne[2] == kTokens);
    for (int token = 0; token < kTokens; ++token) {
        for (int slot = 0; slot < kSelected; ++slot) {
            const auto& input = inputs[token][activation_channels == 1 ? 0 : slot];
            for (int output = 0; output < kRows; ++output) {
                INFO("token=" << token << ", slot=" << slot << ", output=" << output);
                const auto& row = scalar_weights[kRoutes[token][slot]][output];
                double expected = 0;
                for (int col = 0; col < kWidth; ++col)
                    expected += static_cast<double>(row[col]) * input[col];
                const float actual = ggml_get_f32_nd(result, output, slot, token, 0);
                // F32 products are small exact integers. Packed kernels may group scale factors
                // differently for the two nonzero, same-sign products.
                const double tolerance =
                    type == GGML_TYPE_F32
                        ? 0
                        : 8 * std::numeric_limits<float>::epsilon() * std::abs(expected);
                CHECK(std::abs(actual - expected) <= tolerance);
            }
        }
    }
}
} // namespace

TEST_CASE("resident CPU MUL_MAT_ID matches the scalar dot for each selected expert") {
    for (const auto type : {GGML_TYPE_F32, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K}) {
        check_resident_operator(type, 1); // broadcast an activation per token
        check_resident_operator(type, kSelected);
    }
}
