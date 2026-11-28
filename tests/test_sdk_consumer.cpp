// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "xllama/ggml_d3d12.h"
#include "xllama/llama_raii.h"
#include "xllama/session.h"

#if __has_include("chat-peg-parser.h") || __has_include("llama-ext.h")
    #error "xllama exports private llama.cpp headers"
#endif
#if __has_include("ggml-backend-impl.h")
    #error "xllama exports private ggml headers"
#endif

int main() {
    return xllama::Session::backend_for({}) == xllama::Backend::LlamaCpp ? 0 : 1;
}
