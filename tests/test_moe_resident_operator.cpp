// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include <doctest/doctest.h>

#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "xllama/moe_expert_cache.h"

#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>

namespace {
constexpr int kWidth = 256;
constexpr int kRows = 3;
constexpr int kExperts = 4;
constexpr int kSelected = 2;
constexpr int kTokens = 2;
constexpr int32_t kRoutes[kTokens][kSelected] = {{2, 0}, {1, 2}};
using Row = std::array<float, kWidth>;

struct FixtureReader final : xllama::MoeExpertReader {
    explicit FixtureReader(std::FILE* file) : file(file) {}

    size_t read(uint64_t offset, unsigned char* dst, size_t bytes) override {
        if (offset > static_cast<uint64_t>(std::numeric_limits<long>::max()) ||
            std::fseek(file, static_cast<long>(offset), SEEK_SET) != 0)
            throw std::runtime_error("cannot seek in GGUF fixture");
        return std::fread(dst, 1, bytes, file);
    }

    std::FILE* file;
};

void check_mapped_route(const ggml_tensor* tensor, const ggml_tensor* activations,
                        const ggml_tensor* resident, const xllama::MoeExpertCache::Lease& lease,
                        int token, int slot, int threads) {
    INFO("token=" << token << ", slot=" << slot << ", CPU threads=" << threads);
    REQUIRE(lease.size() == tensor->nb[2]);
    auto* data = const_cast<unsigned char*>(lease.data());
    ggml_backend_buffer_ptr buffer(ggml_backend_cpu_buffer_from_ptr(data, lease.size()));
    REQUIRE(buffer != nullptr);
    ggml_context_ptr op(ggml_init({2u << 20, nullptr, true}));
    REQUIRE(op != nullptr);
    auto* mapped = ggml_new_tensor_3d(op.get(), tensor->type, kWidth, kRows, 1);
    REQUIRE(ggml_backend_tensor_alloc(buffer.get(), mapped, data) == GGML_STATUS_SUCCESS);
    ggml_set_no_alloc(op.get(), false);
    auto* input = ggml_new_tensor_3d(op.get(), GGML_TYPE_F32, kWidth, 1, 1);
    const int channel = activations->ne[1] == 1 ? 0 : slot;
    std::memcpy(input->data,
                static_cast<const char*>(activations->data) + token * activations->nb[2] +
                    channel * activations->nb[1],
                ggml_nbytes(input));
    auto* id = ggml_new_tensor_2d(op.get(), GGML_TYPE_I32, 1, 1);
    ggml_set_i32(id, 0);
    auto* actual = ggml_mul_mat_id(op.get(), mapped, input, id);
    auto* graph = ggml_new_graph_custom(op.get(), 8, false);
    ggml_build_forward_expand(graph, actual);
    REQUIRE(ggml_graph_compute_with_ctx(op.get(), graph, threads) == GGML_STATUS_SUCCESS);
    for (int row = 0; row < kRows; ++row)
        CHECK(ggml_get_f32_nd(actual, row, 0, 0, 0) ==
              ggml_get_f32_nd(resident, row, slot, token, 0));
}

void check_gguf_slices(ggml_tensor* weights, ggml_tensor* activations, const ggml_tensor* ids,
                       const ggml_tensor* resident) {
    auto close = [](std::FILE* f) { std::fclose(f); };
    std::unique_ptr<std::FILE, decltype(close)> file(std::tmpfile(), close);
    REQUIRE(file != nullptr);
    gguf_context_ptr writer(gguf_init_empty());
    REQUIRE(writer != nullptr);
    gguf_set_val_str(writer.get(), "general.architecture", "lfm2moe");
    // A preceding tensor gives the routed tensor a nonzero offset inside the GGUF data section.
    gguf_add_tensor(writer.get(), ggml_set_name(activations, "activations"));
    gguf_add_tensor(writer.get(), ggml_set_name(weights, "blk.0.ffn_up_exps.weight"));
    REQUIRE(gguf_write_to_file_ptr(writer.get(), file.get(), false));
    const long file_bytes = std::ftell(file.get());
    REQUIRE(file_bytes > 0);
    REQUIRE(std::fseek(file.get(), 0, SEEK_SET) == 0);

    ggml_context* header_ctx = nullptr;
    gguf_context_ptr gguf(gguf_init_from_file_ptr(file.get(), {true, &header_ctx}));
    ggml_context_ptr header(header_ctx);
    REQUIRE(gguf != nullptr);
    REQUIRE(header != nullptr);
    const auto index = gguf_find_tensor(gguf.get(), ggml_get_name(weights));
    REQUIRE(index >= 0);
    const auto* tensor = ggml_get_tensor(header.get(), ggml_get_name(weights));
    REQUIRE(tensor != nullptr);
    const auto arch_id = gguf_find_key(gguf.get(), "general.architecture");
    REQUIRE(arch_id >= 0);
    const auto role =
        xllama::moe_routed_role(gguf_get_val_str(gguf.get(), arch_id), ggml_get_name(tensor));
    REQUIRE(role.has_value());
    REQUIRE(gguf_get_tensor_offset(gguf.get(), index) > 0);
    const xllama::MoeExpertTensor descriptor{*role,
                                             tensor->type,
                                             static_cast<uint64_t>(tensor->ne[0]),
                                             static_cast<uint64_t>(tensor->ne[1]),
                                             static_cast<uint64_t>(tensor->ne[2]),
                                             gguf_get_data_offset(gguf.get()) +
                                                 gguf_get_tensor_offset(gguf.get(), index)};
    const size_t capacity = tensor->nb[2];
    FixtureReader reader(file.get());
    xllama::MoeExpertCache cache(reader, static_cast<uint64_t>(file_bytes), &descriptor, 1, 1,
                                 capacity, xllama::MoeExpertCache::required_bytes(1, 1, capacity));

    // The fixture selects each logical ID before mapping it to slot zero. The product loader and
    // the native worker's acquire/release callback are separate #480 gates.
    for (int threads : {1, 2}) {
        for (int token = 0; token < kTokens; ++token) {
            for (int slot = 0; slot < kSelected; ++slot) {
                const int expert = ggml_get_i32_nd(ids, slot, token, 0, 0);
                auto lease = cache.acquire(0, expert);
                check_mapped_route(tensor, activations, resident, lease, token, slot, threads);
            }
        }
    }
    CHECK(cache.counters().requests == 8);
    CHECK(cache.counters().hits == 1); // last ID of the first pass is the first of the second pass
    CHECK(cache.counters().logical_read_bytes == 7 * capacity);
}

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
    check_gguf_slices(weights, activations, ids, result);
}
} // namespace

TEST_CASE("resident CPU MUL_MAT_ID matches the scalar dot for each selected expert") {
    for (const auto type : {GGML_TYPE_F32, GGML_TYPE_Q4_K, GGML_TYPE_Q6_K}) {
        check_resident_operator(type, 1); // broadcast an activation per token
        check_resident_operator(type, kSelected);
    }
}
