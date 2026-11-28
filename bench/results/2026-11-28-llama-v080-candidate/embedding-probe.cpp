// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
#include "xllama/session.h"

#include <cstdio>
#include <fstream>
#include <string>

int main(int argc, char** argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: embedding-probe MODEL INPUT OUTPUT\n");
        return 2;
    }
    xllama::SessionParams params;
    params.model_path = argv[1];
    params.n_ctx = 256;
    params.n_threads = 2;
    params.backend = xllama::Backend::LlamaCpp;
    std::string error;
    auto session = xllama::Session::create(params, &error);
    if (!session) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 1;
    }
    xllama::EmbeddingParams request;
    request.input = argv[2];
    const auto result = session->embed(request);
    if (!result.success || result.embedding.empty()) {
        std::fprintf(stderr, "%s\n", result.error_msg.c_str());
        return 1;
    }
    std::ofstream output(argv[3], std::ios::binary);
    output.write(reinterpret_cast<const char*>(result.embedding.data()),
                 static_cast<std::streamsize>(result.embedding.size() * sizeof(float)));
    output.close();
    return output ? 0 : 1;
}
