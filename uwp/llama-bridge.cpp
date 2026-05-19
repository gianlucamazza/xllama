#include "llama-bridge.h"

#include "llama.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#ifdef XLLAMA_UWP
// UWP-only headers — not available on Linux
#include <windows.h>
#include <winrt/Windows.Storage.h>
// TODO(phase1): implement CreateFileMappingFromApp and wire into model loading.
static void* mmap_replacement(const char* path, size_t* out_size) {
    (void)path; (void)out_size;
    return nullptr; // not yet implemented
}
#endif

namespace xllama::bridge {

static int detect_threads() {
    int n = (int)std::thread::hardware_concurrency();
    return n > 0 ? n : 4;
}

static void log_output(const char* msg) {
#ifdef XLLAMA_UWP
    OutputDebugStringA(msg);
#else
    std::fputs(msg, stderr);
#endif
}

int run_inference(const InferenceParams& params) {
    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0; // CPU only in Phase 1

#ifdef XLLAMA_UWP
    // On UWP the filesystem is sandboxed; model_path must be relative to LocalFolder.
    // TODO(phase1): replace with CreateFileMappingFromApp for zero-copy loading.
    mparams.use_mmap = false; // POSIX mmap is unavailable in UWP
#endif

    llama_model* model = llama_model_load_from_file(params.model_path.c_str(), mparams);
    if (!model) {
        log_output("[xllama] failed to load model\n");
        return 1;
    }

    const int n_threads = params.n_threads > 0 ? params.n_threads : detect_threads();

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx     = (uint32_t)params.n_ctx;
    cparams.n_threads = n_threads;

    llama_context* ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        log_output("[xllama] failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    const llama_vocab* vocab = llama_model_get_vocab(model);

    // Tokenize prompt
    std::vector<llama_token> tokens(params.prompt.size() + 16);
    int n_tokens = llama_tokenize(
        vocab,
        params.prompt.c_str(), (int32_t)params.prompt.size(),
        tokens.data(), (int32_t)tokens.size(),
        /*add_special=*/true, /*parse_special=*/false);

    if (n_tokens < 0) {
        log_output("[xllama] tokenization failed\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }
    tokens.resize((size_t)n_tokens);

    // Decode prompt + generate
    llama_batch batch = llama_batch_get_one(tokens.data(), n_tokens);
    if (llama_decode(ctx, batch) != 0) {
        log_output("[xllama] prompt decode failed\n");
        llama_free(ctx);
        llama_model_free(model);
        return 1;
    }

    const llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    llama_sampler* sampler = llama_sampler_chain_init(sparams);
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(params.temperature));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(LLAMA_DEFAULT_SEED));

    int n_generated = 0;
    while (n_generated < params.n_predict) {
        llama_token token = llama_sampler_sample(sampler, ctx, -1);

        if (llama_vocab_is_eog(vocab, token)) break;

        char buf[256] = {};
        int len = llama_token_to_piece(vocab, token, buf, sizeof(buf) - 1, 0, false);
        if (len > 0) {
            buf[len] = '\0';
#ifdef XLLAMA_UWP
            OutputDebugStringA(buf);
#else
            std::fputs(buf, stdout);
            std::fflush(stdout);
#endif
        }

        llama_batch next = llama_batch_get_one(&token, 1);
        if (llama_decode(ctx, next) != 0) break;
        ++n_generated;
    }

#ifndef XLLAMA_UWP
    std::fputc('\n', stdout);
#endif

    llama_sampler_free(sampler);
    llama_free(ctx);
    llama_model_free(model);
    return 0;
}

void main_loop() {
#ifdef XLLAMA_UWP
    // Called from App::Run() on a background thread.
    // TODO(phase1): replace with real model/prompt selection UI or
    //   read params from a config file in LocalFolder.
    InferenceParams params;
    params.model_path = "models\\qwen3-1.7b-Q4_K_M.gguf";
    params.prompt     = "Hello, I am running on an Xbox Series S. ";
    params.n_predict  = 64;
    run_inference(params);
#endif
}

} // namespace xllama::bridge
