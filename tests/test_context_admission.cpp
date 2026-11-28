// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// Linux-only fault injection at the public GGUF session boundary.
#include "llama.h"
#include "xllama/session.h"

#include <cstdio>

namespace {
const void* resident = nullptr;
bool tight_budget = false;
bool fail_estimate = false;
int fallback_allocations = 0;
} // namespace

extern "C" {
llama_context* __real_llama_init_from_model(llama_model*, llama_context_params);
llama_context* __wrap_llama_init_from_model(llama_model* model, llama_context_params params) {
    if (model == resident) {
        if (params.type_k == GGML_TYPE_Q8_0)
            return nullptr; // The real quantized context failed after the dry run.
        ++fallback_allocations;
    }
    if (resident && model != resident && fail_estimate && params.type_k != GGML_TYPE_Q8_0)
        return nullptr; // The replacement dry run could not size this context.
    return __real_llama_init_from_model(model, params);
}

// GNU ld --wrap for xllama::app_memory_limit_mb(), normally zero on Linux.
std::size_t __wrap__ZN6xllama19app_memory_limit_mbEv() noexcept {
    return tight_budget ? 1 : 0;
}
}

int main(int argc, char** argv) {
    if (argc != 2)
        return 1;
    xllama::SessionParams sp;
    sp.model_path = argv[1];
    sp.n_threads = 1;
    sp.n_ctx = 128;
    sp.kv_q8 = true;
    std::string err;
    auto session = xllama::Session::create(sp, &err);
    if (!session) {
        std::fprintf(stderr, "%s\n", err.c_str());
        return 1;
    }
    resident = session->resident_llama_model();
    xllama::GenerateParams gp;
    gp.prompt = "abc";
    gp.n_predict = 2;
    gp.temperature = 0;
    tight_budget = true;
    for (bool no_estimate : {false, true}) {
        fail_estimate = no_estimate;
        const auto refused = session->generate(gp);
        if (refused.success || !xllama::is_memory_refusal(refused.error_msg) ||
            fallback_allocations != 0) {
            std::fprintf(
                stderr, "unknown estimate=%d, refused=%d, fallback allocations=%d, error=%s\n",
                no_estimate, !refused.success, fallback_allocations, refused.error_msg.c_str());
            return 1;
        }
    }
    tight_budget = false;
    fail_estimate = false;
    const auto recovered = session->generate(gp);
    if (!recovered.success || fallback_allocations != 1) {
        std::fprintf(stderr, "fallback allocations=%d, recovery error=%s\n", fallback_allocations,
                     recovered.error_msg.c_str());
        return 1;
    }
    sp.kv_q8 = false;
    auto reference = xllama::Session::create(sp, &err);
    if (!reference)
        return 1;
    const auto expected = reference->generate(gp);
    if (!expected.success || recovered.n_eval != expected.n_eval ||
        recovered.output_text != expected.output_text) {
        std::fprintf(stderr, "recovered context differs from a fresh default-KV session\n");
        return 1;
    }
    return 0;
}
