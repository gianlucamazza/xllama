// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

#include "xllama/inference.h"
#include "xllama/llama_raii.h"
#include "xllama/path_utils.h"
#include "xllama/platform.h"

#include "llama.h"

#include <cstdio>
#include <ctime>
#include <vector>

namespace xllama {

InferenceResult run_inference(const InferenceParams& params) {
    InferenceResult res;

    const std::string abs_model_path = resolve_model_path(params.model_path);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0; // CPU only in Phase 1

#ifdef XLLAMA_UWP
    mparams.use_mmap = false; // POSIX mmap unavailable in UWP
#endif

    if (params.on_status)
        params.on_status("loading model");

    llama_model* raw_model = llama_model_load_from_file(abs_model_path.c_str(), mparams);
    if (!raw_model) {
        res.error_msg = "failed to load model: " + abs_model_path;
        log_output("[xllama] " + res.error_msg + "\n");
        if (params.on_status)
            params.on_status("error: " + res.error_msg);
        return res;
    }
    LlamaModelPtr model(raw_model);
    log_output("[xllama] model loaded\n");
    if (params.on_status)
        params.on_status("decoding");

    const int n_threads = params.n_threads > 0 ? params.n_threads : detect_threads();

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = static_cast<uint32_t>(params.n_ctx);
    cparams.n_threads = n_threads;

    llama_context* raw_ctx = llama_init_from_model(model.get(), cparams);
    if (!raw_ctx) {
        res.error_msg = "failed to create context";
        log_output("[xllama] failed to create context\n");
        return res;
    }
    LlamaContextPtr ctx(raw_ctx);

    const llama_vocab* vocab = llama_model_get_vocab(model.get());

    // Tokenize prompt: first call to get required size, second to fill.
    int32_t n_tokens = llama_tokenize(vocab, params.prompt.c_str(),
                                      static_cast<int32_t>(params.prompt.size()), nullptr, 0,
                                      /*add_special=*/true, /*parse_special=*/false);

    if (n_tokens < 0) {
        res.error_msg = "tokenization size query failed";
        log_output("[xllama] tokenization size query failed\n");
        return res;
    }

    std::vector<llama_token> tokens(static_cast<size_t>(n_tokens));
    n_tokens = llama_tokenize(vocab, params.prompt.c_str(),
                              static_cast<int32_t>(params.prompt.size()), tokens.data(), n_tokens,
                              /*add_special=*/true, /*parse_special=*/false);

    if (n_tokens < 0) {
        res.error_msg = "tokenization failed";
        log_output("[xllama] tokenization failed\n");
        return res;
    }
    tokens.resize(static_cast<size_t>(n_tokens));

    // Decode prompt
    llama_batch batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
    if (llama_decode(ctx.get(), batch) != 0) {
        res.error_msg = "prompt decode failed";
        log_output("[xllama] prompt decode failed\n");
        return res;
    }

    const llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    LlamaSamplerPtr sampler(llama_sampler_chain_init(sparams));
    llama_sampler_chain_add(sampler.get(), llama_sampler_init_temp(params.temperature));
    llama_sampler_chain_add(sampler.get(), llama_sampler_init_dist(params.seed));

    // Generation loop
    int n_generated = 0;
    while (n_generated < params.n_predict) {
        if (params.abort_flag && params.abort_flag->load())
            break;

        llama_token token = llama_sampler_sample(sampler.get(), ctx.get(), -1);
        if (llama_vocab_is_eog(vocab, token))
            break;

        char buf[256] = {};
        int len = llama_token_to_piece(vocab, token, buf, sizeof(buf) - 1, 0, false);
        if (len > 0) {
            buf[len] = '\0';
            res.output_text += buf;
            if (params.on_token)
                params.on_token(std::string(buf, static_cast<size_t>(len)));
#ifndef XLLAMA_UWP
            std::fputs(buf, stdout);
            std::fflush(stdout);
#else
            // On UWP tokens are forwarded via on_token callback; no stdout.
            (void)0;
#endif
        }

        llama_batch next = llama_batch_get_one(&token, 1);
        if (llama_decode(ctx.get(), next) != 0)
            break;
        ++n_generated;
    }

#ifndef XLLAMA_UWP
    std::fputc('\n', stdout);
#endif

    // Collect perf metrics
    llama_perf_context_data perf = llama_perf_context(ctx.get());
    res.t_load_ms = perf.t_load_ms;
    res.t_p_eval_ms = perf.t_p_eval_ms;
    res.t_eval_ms = perf.t_eval_ms;
    res.n_p_eval = perf.n_p_eval;
    res.n_eval = n_generated;
    res.peak_ws_mb = peak_working_set_mb();
    res.success = true;

    char log_buf[256];
    snprintf(log_buf, sizeof(log_buf),
             "[xllama] done: load=%.0fms prompt=%.1f tok/s decode=%.1f tok/s peak=%zuMB\n",
             res.t_load_ms,
             res.n_p_eval > 0 && res.t_p_eval_ms > 0
                 ? static_cast<double>(res.n_p_eval) / (res.t_p_eval_ms / 1000.0)
                 : 0.0,
             res.n_eval > 0 && res.t_eval_ms > 0
                 ? static_cast<double>(res.n_eval) / (res.t_eval_ms / 1000.0)
                 : 0.0,
             res.peak_ws_mb);
    log_output(log_buf);

    return res;
}

} // namespace xllama
