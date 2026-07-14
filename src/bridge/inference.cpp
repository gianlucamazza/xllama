// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT

#include "xllama/inference.h"
#include "xllama/chat_prompt.h"
#include "xllama/path_utils.h"
#include "xllama/platform.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <string>

// Backend availability — mirrors src/bridge/session.cpp. See the note there.
#if !defined(XLLAMA_USE_LLAMA)
    #if defined(XLLAMA_LINUX) || !defined(XLLAMA_USE_ORT)
        #define XLLAMA_USE_LLAMA 1
    #endif
#endif

#if !defined(XLLAMA_USE_ORT) && !defined(XLLAMA_USE_LLAMA)
    #error "no inference backend compiled (define XLLAMA_USE_ORT and/or XLLAMA_USE_LLAMA)"
#endif

namespace xllama {
namespace detail {
#ifdef XLLAMA_USE_ORT
InferenceResult run_inference_ort(const InferenceParams& params);
#endif
#ifdef XLLAMA_USE_LLAMA
InferenceResult run_inference_llama(const InferenceParams& params);
#endif
} // namespace detail
} // namespace xllama

// ---------------------------------------------------------------------------
// ONNX Runtime GenAI path (UWP + DirectML GPU)
// ---------------------------------------------------------------------------
#ifdef XLLAMA_USE_ORT

// clang-format off
    // windows.h must precede ort_genai_c.h in WINAPI_FAMILY_APP builds.
    #include <windows.h>
    #include <eh.h>
    #include "ort_genai_c.h"
    #include "xllama/ort_raii.h"
// clang-format on

    #include <chrono>

namespace xllama {
namespace detail {

InferenceResult run_inference_ort(const InferenceParams& params) {
    InferenceResult res;

    // Convert SEH (D3D12/DML OOM/AV) → std::runtime_error so the catch block can log it.
    _set_se_translator([](unsigned int code, EXCEPTION_POINTERS*) {
        char b[48];
        snprintf(b, sizeof(b), "SEH 0x%08X", code);
        throw std::runtime_error(b);
    });

    const std::string model_dir = resolve_model_path(params.model_path);

    if (params.on_status)
        params.on_status("loading model");

    try {
        // Redirect ORT GenAI internal log messages into xllama.log.
        // Without this they go only to OutputDebugStringA (Device Portal debug output).
        oga_check(OgaSetLogCallback([](const char* msg, size_t /*len*/) { log_output(msg); }),
                  "OgaSetLogCallback");

        // Log memory state before the large ORT allocation for OOM diagnostics.
        {
            MEMORYSTATUSEX ms{};
            ms.dwLength = sizeof(ms);
            GlobalMemoryStatusEx(&ms);
            char mem_buf[256];
            snprintf(mem_buf, sizeof(mem_buf),
                     "[xllama] pre-OgaCreateModel: avail_phys=%.1f GB current_ws=%zu MB model=%s\n",
                     static_cast<double>(ms.ullAvailPhys) / (1024.0 * 1024.0 * 1024.0),
                     peak_working_set_mb(), model_dir.c_str());
            log_output(mem_buf);
            // NB: do NOT query GPU memory here. gpu_mem_info() opens and caches an
            // IDXGIAdapter3 on adapter 0; on the Xbox shared-GPU sandbox that blocks
            // the DirectML EP from creating its D3D12 device in the following
            // OgaCreateModel call (dml_helpers.cpp: 887A0036 "element already
            // exists"). SmolLM2-360M + this DML config loaded fine in v0.3.0 before
            // this probe existed. GPU memory is sampled post-load/post-decode below,
            // after DML has created its device — that is where the GPU-truth signal
            // (current ≈ model size) matters anyway.
        }

        // --- model ---
        // Wall-clock around OgaCreateModel: ORT GenAI has no llama_perf-style load timer.
        auto t_load0 = std::chrono::steady_clock::now();
        OgaModel* raw_model = nullptr;
        oga_check(OgaCreateModel(model_dir.c_str(), &raw_model), "OgaCreateModel");
        OgaModelPtr model(raw_model);
        res.t_load_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_load0)
                .count();
        {
            char load_buf[128];
            snprintf(load_buf, sizeof(load_buf), "[xllama] ORT model loaded in %.0f ms\n",
                     res.t_load_ms);
            log_output(load_buf);
            GpuMemInfo gpu = gpu_mem_info();
            if (gpu.available) {
                res.gpu_mem_mb = gpu.current_mb;
                res.gpu_budget_mb = gpu.budget_mb;
                snprintf(load_buf, sizeof(load_buf),
                         "[xllama] gpu-mem post-load: current=%zuMB budget=%zuMB\n", gpu.current_mb,
                         gpu.budget_mb);
                log_output(load_buf);
            }
        }

        if (params.on_status)
            params.on_status("tokenizing");

        // --- tokenizer ---
        OgaTokenizer* raw_tok = nullptr;
        oga_check(OgaCreateTokenizer(model.get(), &raw_tok), "OgaCreateTokenizer");
        OgaTokenizerPtr tok(raw_tok);

        OgaTokenizerStream* raw_stream = nullptr;
        oga_check(OgaCreateTokenizerStream(tok.get(), &raw_stream), "OgaCreateTokenizerStream");
        OgaTokenizerStreamPtr stream(raw_stream);

        // --- encode prompt ---
        OgaSequences* raw_seqs = nullptr;
        oga_check(OgaCreateSequences(&raw_seqs), "OgaCreateSequences");
        OgaSequencesPtr seqs(raw_seqs);
        oga_check(OgaTokenizerEncode(tok.get(), params.prompt.c_str(), seqs.get()),
                  "OgaTokenizerEncode");
        size_t n_prompt_tok = OgaSequencesGetSequenceCount(seqs.get(), 0);
        {
            int max_len = params.n_predict + 512;
            char pbuf[128];
            snprintf(pbuf, sizeof(pbuf), "[xllama] prompt=%zu tok, max_length=%d (new≤%d)\n",
                     n_prompt_tok, max_len, max_len - static_cast<int>(n_prompt_tok));
            log_output(pbuf);
        }

        // --- generator params ---
        OgaGeneratorParams* raw_params = nullptr;
        oga_check(OgaCreateGeneratorParams(model.get(), &raw_params), "OgaCreateGeneratorParams");
        OgaGeneratorParamsPtr gparams(raw_params);
        oga_check(OgaGeneratorParamsSetSearchNumber(gparams.get(), "max_length",
                                                    static_cast<double>(params.n_predict + 512)),
                  "SetSearchNumber max_length");
        oga_check(OgaGeneratorParamsSetSearchNumber(gparams.get(), "temperature",
                                                    static_cast<double>(params.temperature)),
                  "SetSearchNumber temperature");
        // --- generator ---
        OgaGenerator* raw_gen = nullptr;
        oga_check(OgaCreateGenerator(model.get(), gparams.get(), &raw_gen), "OgaCreateGenerator");
        OgaGeneratorPtr gen(raw_gen);

        if (params.on_status)
            params.on_status("generating");

        // Record wall-clock start BEFORE AppendTokenSequences: measured on-console
        // (0.3.5), the prompt prefill runs inside AppendTokenSequences, not inside
        // the first GenerateNextToken (which returned in ~40 µs). t0 → end of the
        // first loop iteration covers the prefill under either implementation.
        // (ORT GenAI has no perf API like llama_perf.)
        auto t0 = std::chrono::steady_clock::now();

        // ORT GenAI ≥ 0.7: feed input sequences to generator (not to params)
        oga_check(OgaGenerator_AppendTokenSequences(gen.get(), seqs.get()), "AppendTokenSequences");

        auto t_prefill_end = t0;

        int n_generated = 0;
        while (!OgaGenerator_IsDone(gen.get())) {
            if (params.abort_flag && params.abort_flag->load())
                break;

            // ORT GenAI ≥ 0.7: GenerateNextToken does compute + sample in one call
            oga_check(OgaGenerator_GenerateNextToken(gen.get()), "GenerateNextToken");
            if (n_generated == 0)
                t_prefill_end = std::chrono::steady_clock::now();

            const int32_t* next_toks = nullptr;
            size_t n_next = 0;
            oga_check(OgaGenerator_GetNextTokens(gen.get(), &next_toks, &n_next), "GetNextTokens");

            for (size_t i = 0; i < n_next; ++i) {
                const char* piece = nullptr;
                oga_check(OgaTokenizerStreamDecode(stream.get(), next_toks[i], &piece),
                          "TokenizerStreamDecode");
                if (piece && *piece) {
                    res.output_text += piece;
                    if (params.on_token)
                        params.on_token(std::string(piece));
                }
            }
            ++n_generated;
        }

        auto t_end = std::chrono::steady_clock::now();
        double elapsed_s = std::chrono::duration<double>(t_end - t0).count();
        double prefill_ms = std::chrono::duration<double, std::milli>(t_prefill_end - t0).count();
        // Decode rate excludes the prefill iteration (its token and its time).
        double decode_s = std::chrono::duration<double>(t_end - t_prefill_end).count();
        int n_decode = n_generated > 0 ? n_generated - 1 : 0;

        if (n_generated > 0) {
            res.n_p_eval = static_cast<int>(n_prompt_tok);
            res.t_p_eval_ms = prefill_ms;
        }
        res.n_eval = n_decode;
        res.t_eval_ms = decode_s * 1000.0;
        res.peak_ws_mb = peak_working_set_mb();
        res.success = true;

        char log_buf[256];
        snprintf(log_buf, sizeof(log_buf),
                 "[xllama] done: prefill=%.1f tok/s (%zu tok, %.0f ms) decode=%.1f tok/s n=%d "
                 "elapsed=%.4fs peak=%zuMB\n",
                 prefill_ms > 0.0 ? n_prompt_tok / (prefill_ms / 1000.0) : 0.0, n_prompt_tok,
                 prefill_ms, decode_s > 0.0 ? n_decode / decode_s : 0.0, n_decode, elapsed_s,
                 res.peak_ws_mb);
        log_output(log_buf);

        GpuMemInfo gpu = gpu_mem_info();
        if (gpu.available) {
            snprintf(log_buf, sizeof(log_buf),
                     "[xllama] gpu-mem post-decode: current=%zuMB budget=%zuMB\n", gpu.current_mb,
                     gpu.budget_mb);
            log_output(log_buf);
        }

    } catch (const std::exception& e) {
        res.error_msg = e.what();
        log_output(("[xllama] inference error: " + res.error_msg + "\n").c_str());
        if (params.on_status)
            params.on_status("error: " + res.error_msg);
    }

    return res;
}
} // namespace detail

} // namespace xllama

#endif // XLLAMA_USE_ORT

// ---------------------------------------------------------------------------
// llama.cpp path (Linux / CLI + UWP llamacpp/unified)
// ---------------------------------------------------------------------------
#ifdef XLLAMA_USE_LLAMA

    #include "llama.h"
    #include "xllama/llama_raii.h"

    #include <vector>

namespace xllama {
namespace detail {

InferenceResult run_inference_llama(const InferenceParams& params) {
    InferenceResult res;

    // Catalogue GGUF entries resolve to the model DIRECTORY; llama loads a file.
    const std::string abs_model_path = first_gguf_in_dir(resolve_model_path(params.model_path));
    if (abs_model_path.empty()) {
        res.error_msg = "no .gguf file in model dir: " + resolve_model_path(params.model_path);
        log_output("[xllama] " + res.error_msg + "\n");
        if (params.on_status)
            params.on_status("error: " + res.error_msg);
        return res;
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0; // CPU only on Linux path

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

    const int n_threads = params.n_threads > 0 ? params.n_threads : detect_threads_llama();

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

    int32_t n_tokens =
        llama_tokenize(vocab, params.prompt.c_str(), static_cast<int32_t>(params.prompt.size()),
                       nullptr, 0, true, false);
    if (n_tokens == INT32_MIN) {
        res.error_msg = "tokenization overflow";
        log_output("[xllama] tokenization overflow\n");
        return res;
    }
    // Size query with a null buffer returns the NEGATED required token count
    // (llama.h: "Returns a negative number on failure - the number of tokens
    // that would have been returned"). Negative here is the expected answer,
    // not an error.
    n_tokens = -n_tokens;

    std::vector<llama_token> tokens(static_cast<size_t>(n_tokens));
    n_tokens =
        llama_tokenize(vocab, params.prompt.c_str(), static_cast<int32_t>(params.prompt.size()),
                       tokens.data(), n_tokens, true, false);
    if (n_tokens < 0) {
        res.error_msg = "tokenization failed";
        log_output("[xllama] tokenization failed\n");
        return res;
    }
    tokens.resize(static_cast<size_t>(n_tokens));

    log_output(("[xllama] prompt tokens: " + std::to_string(tokens.size()) + "\n").c_str());
    const auto t_prompt0 = std::chrono::steady_clock::now();
    llama_batch batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
    if (llama_decode(ctx.get(), batch) != 0) {
        res.error_msg = "prompt decode failed";
        log_output("[xllama] prompt decode failed\n");
        return res;
    }
    const double prompt_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_prompt0)
            .count();

    const llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    LlamaSamplerPtr sampler(llama_sampler_chain_init(sparams));
    llama_sampler_chain_add(sampler.get(), llama_sampler_init_temp(params.temperature));
    llama_sampler_chain_add(sampler.get(), llama_sampler_init_dist(params.seed));

    int n_generated = 0;
    const auto t_gen0 = std::chrono::steady_clock::now();
    while (n_generated < params.n_predict) {
        if (params.abort_flag && params.abort_flag->load())
            break;

        llama_token token = llama_sampler_sample(sampler.get(), ctx.get(), -1);
        if (llama_vocab_is_eog(vocab, token)) {
            log_output(("[xllama] EOG after " + std::to_string(n_generated) + " tokens\n").c_str());
            break;
        }

        char buf[256] = {};
        int len = llama_token_to_piece(vocab, token, buf, sizeof(buf) - 1, 0, false);
        if (len > 0) {
            buf[len] = '\0';
            res.output_text += buf;
            if (params.on_token)
                params.on_token(std::string(buf, static_cast<size_t>(len)));
            std::fputs(buf, stdout);
            std::fflush(stdout);
        }

        // Stop strings (e.g. Gemma's <end_of_turn>, not an EOG token in every
        // GGUF): shared suffix-match helper, trims the trailing match.
        if (apply_stop_sequences(res.output_text, params.stop_sequences)) {
            res.ended_with_stop = true;
            log_output(
                ("[xllama] stop sequence after " + std::to_string(n_generated + 1) + " tokens\n")
                    .c_str());
            ++n_generated;
            break;
        }

        llama_batch next = llama_batch_get_one(&token, 1);
        if (llama_decode(ctx.get(), next) != 0) {
            log_output("[xllama] decode failed at token, stopping generation\n");
            break;
        }
        ++n_generated;
    }

    std::fputc('\n', stdout);

    const double gen_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_gen0)
            .count();
    // Own chrono timing: llama_context_default_params().no_perf disables the
    // built-in perf counters, so llama_perf_context reports zeros by default.
    llama_perf_context_data perf = llama_perf_context(ctx.get());
    res.t_load_ms = perf.t_load_ms > 0 ? perf.t_load_ms : 0.0;
    res.t_p_eval_ms = prompt_ms;
    res.t_eval_ms = gen_ms;
    res.n_p_eval = static_cast<int32_t>(tokens.size());
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
} // namespace detail

} // namespace xllama

#endif // XLLAMA_USE_LLAMA

// ---------------------------------------------------------------------------
// Public entry: dispatch to the compiled backend(s).
// ---------------------------------------------------------------------------
namespace xllama {

InferenceResult run_inference(const InferenceParams& params) {
#if defined(XLLAMA_USE_ORT) && defined(XLLAMA_USE_LLAMA)
    // Layout-aware Auto (same helper as Session) so bench paths using bare
    // model names from model.txt also dispatch correctly for GGUF layouts.
    return model_uses_llama_backend(params.model_path) ? detail::run_inference_llama(params)
                                                       : detail::run_inference_ort(params);
#elif defined(XLLAMA_USE_ORT)
    return detail::run_inference_ort(params);
#else // XLLAMA_USE_LLAMA
    return detail::run_inference_llama(params);
#endif
}

} // namespace xllama
