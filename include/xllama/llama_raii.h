// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

#pragma once

#include "llama.h"
#include <memory>

namespace xllama {

// ---------------------------------------------------------------------------
// RAII wrappers for llama.cpp C handles
// ---------------------------------------------------------------------------

struct LlamaModelDeleter {
    void operator()(llama_model* m) const noexcept {
        if (m)
            llama_model_free(m);
    }
};

struct LlamaContextDeleter {
    void operator()(llama_context* c) const noexcept {
        if (c)
            llama_free(c);
    }
};

struct LlamaSamplerDeleter {
    void operator()(llama_sampler* s) const noexcept {
        if (s)
            llama_sampler_free(s);
    }
};

using LlamaModelPtr = std::unique_ptr<llama_model, LlamaModelDeleter>;
using LlamaContextPtr = std::unique_ptr<llama_context, LlamaContextDeleter>;
using LlamaSamplerPtr = std::unique_ptr<llama_sampler, LlamaSamplerDeleter>;

} // namespace xllama
