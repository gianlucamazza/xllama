// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// DC-3 CPU reference and matched two-worker arithmetic fixture.
#include "ggml-cpp.h"
#include "ggml-cpu.h"
#include "ggml-rpc.h"

#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>

namespace {

using Values = std::array<float, 4>;

struct Stage {
    ggml_context_ptr ctx;
    ggml_backend_buffer_ptr storage;
    ggml_tensor* input;
    ggml_tensor* offset;
    ggml_tensor* result;
    ggml_cgraph* graph;

    explicit Stage(ggml_backend_t backend)
        : ctx(ggml_init(
              {3 * ggml_tensor_overhead() + ggml_graph_overhead_custom(8, false), nullptr, true})) {
        if (!ctx)
            throw std::runtime_error("cannot allocate the graph context");
        input = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, Values{}.size());
        offset = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, Values{}.size());
        result = ggml_add(ctx.get(), input, offset);
        graph = ggml_new_graph_custom(ctx.get(), 8, false);
        ggml_build_forward_expand(graph, result);
        storage.reset(ggml_backend_alloc_ctx_tensors(ctx.get(), backend));
        if (!storage)
            throw std::runtime_error("cannot allocate the backend tensors");
    }

    void run(ggml_backend_t backend) const {
        if (ggml_backend_graph_compute(backend, graph) != GGML_STATUS_SUCCESS)
            throw std::runtime_error("graph execution failed");
        ggml_backend_synchronize(backend);
    }
};

ggml_backend_ptr open_backend(const char* endpoint) {
    if (!endpoint)
        return ggml_backend_ptr(ggml_backend_cpu_init());
    const std::string address(endpoint);
    if (address.rfind("127.0.0.1:", 0) != 0)
        throw std::runtime_error("the fixture requires a loopback worker");
    return ggml_backend_ptr(ggml_backend_rpc_init(endpoint, 0));
}

} // namespace

int main(int argc, char** argv) {
    if (argc != 1 && (argc != 3 || std::string(argv[1]) == argv[2])) {
        std::fprintf(stderr, "usage: rpc-add [distinct-loopback-endpoint-A endpoint-B]\n");
        return 2;
    }
    try {
        ggml_backend_load_all();
        const auto first = open_backend(argc == 3 ? argv[1] : nullptr);
        const auto second = open_backend(argc == 3 ? argv[2] : nullptr);
        if (!first || !second)
            throw std::runtime_error("backend initialization failed");
        Stage a(first.get()), b(second.get());
        const Values input{1.5f, -2.0f, 0.0f, 16.0f};
        const Values offset_a{2.5f, 3.0f, -0.5f, 1.0f};
        const Values offset_b{1.0f, -1.0f, 0.5f, 8.0f};
        const Values expected{5.0f, 0.0f, 0.0f, 25.0f};
        Values result{};
        ggml_backend_tensor_set(a.input, input.data(), 0, sizeof(input));
        ggml_backend_tensor_set(a.offset, offset_a.data(), 0, sizeof(offset_a));
        ggml_backend_tensor_set(b.offset, offset_b.data(), 0, sizeof(offset_b));
        a.run(first.get());
        ggml_backend_tensor_copy(a.result, b.input);
        b.run(second.get());
        ggml_backend_tensor_get(b.result, result.data(), 0, sizeof(result));
        std::printf("A=%s buffer_bytes=%zu B=%s buffer_bytes=%zu copy_bytes=%zu "
                    "result=[%g,%g,%g,%g]\n",
                    ggml_backend_name(first.get()), ggml_backend_buffer_get_size(a.storage.get()),
                    ggml_backend_name(second.get()), ggml_backend_buffer_get_size(b.storage.get()),
                    ggml_nbytes(a.result), result[0], result[1], result[2], result[3]);
        return result == expected ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
