// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT
// xllama::Session — persistent model session for multi-turn applications.
// ORT GenAI path (UWP): model + tokenizer loaded once, reused per generate().
// llama.cpp path (Linux): model kept alive; context rebuilt per generate().

#include "xllama/session.h"
#include "xllama/inference_params.h"
#include "xllama/path_utils.h"
#include "xllama/platform.h"

#include <algorithm>
#include <chrono>
#include <string>

// ---------------------------------------------------------------------------
// ONNX Runtime GenAI path (UWP / Xbox Series S)
// ---------------------------------------------------------------------------
#ifdef XLLAMA_USE_ORT

// clang-format off
#include <windows.h>
#include <eh.h>
#include "ort_genai_c.h"
#include "xllama/ort_raii.h"
// clang-format on

namespace xllama {

class OrtSession final : public Session {
  public:
    OgaModelPtr m_model;
    OgaTokenizerPtr m_tok;

    explicit OrtSession(OgaModelPtr model, OgaTokenizerPtr tok)
        : m_model(std::move(model)), m_tok(std::move(tok)) {}

    InferenceResult generate(const GenerateParams& gp) override {
        InferenceResult res;

        _set_se_translator([](unsigned int code, EXCEPTION_POINTERS*) {
            char b[48];
            snprintf(b, sizeof(b), "SEH 0x%08X", code);
            throw std::runtime_error(b);
        });

        try {
            if (gp.on_status)
                gp.on_status("tokenizing");

            OgaTokenizerStream* raw_stream = nullptr;
            oga_check(OgaCreateTokenizerStream(m_tok.get(), &raw_stream),
                      "OgaCreateTokenizerStream");
            OgaTokenizerStreamPtr stream(raw_stream);

            OgaSequences* raw_seqs = nullptr;
            oga_check(OgaCreateSequences(&raw_seqs), "OgaCreateSequences");
            OgaSequencesPtr seqs(raw_seqs);
            oga_check(OgaTokenizerEncode(m_tok.get(), gp.prompt.c_str(), seqs.get()),
                      "OgaTokenizerEncode");

            OgaGeneratorParams* raw_params = nullptr;
            oga_check(OgaCreateGeneratorParams(m_model.get(), &raw_params),
                      "OgaCreateGeneratorParams");
            OgaGeneratorParamsPtr gparams(raw_params);

            // max_length includes prompt tokens
            oga_check(OgaGeneratorParamsSetSearchNumber(gparams.get(), "max_length",
                                                        static_cast<double>(gp.n_predict + 512)),
                      "SetSearchNumber max_length");
            oga_check(OgaGeneratorParamsSetSearchNumber(gparams.get(), "temperature",
                                                        static_cast<double>(gp.temperature)),
                      "SetSearchNumber temperature");
            oga_check(OgaGeneratorParamsSetSearchNumber(gparams.get(), "top_p",
                                                        static_cast<double>(gp.top_p)),
                      "SetSearchNumber top_p");
            oga_check(OgaGeneratorParamsSetSearchNumber(gparams.get(), "top_k",
                                                        static_cast<double>(gp.top_k)),
                      "SetSearchNumber top_k");
            oga_check(OgaGeneratorParamsSetSearchNumber(gparams.get(), "repetition_penalty",
                                                        static_cast<double>(gp.repetition_penalty)),
                      "SetSearchNumber repetition_penalty");

            OgaGenerator* raw_gen = nullptr;
            oga_check(OgaCreateGenerator(m_model.get(), gparams.get(), &raw_gen),
                      "OgaCreateGenerator");
            OgaGeneratorPtr gen(raw_gen);

            oga_check(OgaGenerator_AppendTokenSequences(gen.get(), seqs.get()),
                      "AppendTokenSequences");

            if (gp.on_status)
                gp.on_status("generating");

            auto t0 = std::chrono::steady_clock::now();
            int n_generated = 0;
            bool stopped_by_seq = false;

            while (!OgaGenerator_IsDone(gen.get())) {
                if (gp.abort_flag && gp.abort_flag->load())
                    break;

                oga_check(OgaGenerator_GenerateNextToken(gen.get()), "GenerateNextToken");

                const int32_t* next_toks = nullptr;
                size_t n_next = 0;
                oga_check(OgaGenerator_GetNextTokens(gen.get(), &next_toks, &n_next),
                          "GetNextTokens");

                for (size_t i = 0; i < n_next; ++i) {
                    const char* piece = nullptr;
                    oga_check(OgaTokenizerStreamDecode(stream.get(), next_toks[i], &piece),
                              "TokenizerStreamDecode");
                    if (piece && *piece) {
                        res.output_text += piece;
                        if (gp.on_token)
                            gp.on_token(std::string(piece));
                    }
                }
                ++n_generated;

                // Check stop sequences on accumulated output
                for (const auto& stop : gp.stop_sequences) {
                    auto pos = res.output_text.find(stop);
                    if (pos != std::string::npos) {
                        res.output_text.erase(pos);
                        stopped_by_seq = true;
                        break;
                    }
                }
                if (stopped_by_seq)
                    break;
            }

            double elapsed_s =
                std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

            res.n_eval = n_generated;
            res.t_eval_ms = elapsed_s * 1000.0;
            res.peak_ws_mb = peak_working_set_mb();
            res.success = true;

            char log_buf[256];
            snprintf(log_buf, sizeof(log_buf),
                     "[xllama] session generate: decode=%.1f tok/s n=%d\n",
                     elapsed_s > 0 ? n_generated / elapsed_s : 0.0, n_generated);
            log_output(log_buf);

        } catch (const std::exception& e) {
            res.error_msg = e.what();
            log_output(("[xllama] session generate error: " + res.error_msg + "\n").c_str());
            if (gp.on_status)
                gp.on_status("error: " + res.error_msg);
        }

        return res;
    }
};

std::unique_ptr<Session> Session::create(const SessionParams& sp, std::string* err) {
    _set_se_translator([](unsigned int code, EXCEPTION_POINTERS*) {
        char b[48];
        snprintf(b, sizeof(b), "SEH 0x%08X", code);
        throw std::runtime_error(b);
    });

    const std::string model_dir = resolve_model_path(sp.model_path);
    try {
        oga_check(OgaSetLogCallback([](const char* msg, size_t) { log_output(msg); }),
                  "OgaSetLogCallback");

        OgaModel* raw_model = nullptr;
        oga_check(OgaCreateModel(model_dir.c_str(), &raw_model), "OgaCreateModel");
        OgaModelPtr model(raw_model);

        OgaTokenizer* raw_tok = nullptr;
        oga_check(OgaCreateTokenizer(model.get(), &raw_tok), "OgaCreateTokenizer");
        OgaTokenizerPtr tok(raw_tok);

        log_output("[xllama] Session: model loaded (persistent)\n");
        return std::make_unique<OrtSession>(std::move(model), std::move(tok));

    } catch (const std::exception& e) {
        if (err)
            *err = e.what();
        log_output(("[xllama] Session::create error: " + std::string(e.what()) + "\n").c_str());
        return nullptr;
    }
}

} // namespace xllama

// ---------------------------------------------------------------------------
// llama.cpp path (Linux dev)
// ---------------------------------------------------------------------------
#else

    #include "llama.h"
    #include "xllama/llama_raii.h"

    #include <vector>

namespace xllama {

class LlamaSession final : public Session {
  public:
    LlamaModelPtr m_model;
    int m_n_ctx;
    int m_n_threads;

    explicit LlamaSession(LlamaModelPtr model, int n_ctx, int n_threads)
        : m_model(std::move(model)), m_n_ctx(n_ctx), m_n_threads(n_threads) {}

    InferenceResult generate(const GenerateParams& gp) override {
        InferenceResult res;

        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx = m_n_ctx;
        cparams.n_threads = m_n_threads;

        llama_context* raw_ctx = llama_init_from_model(m_model.get(), cparams);
        if (!raw_ctx) {
            res.error_msg = "failed to create context";
            return res;
        }
        LlamaContextPtr ctx(raw_ctx);

        const llama_vocab* vocab = llama_model_get_vocab(m_model.get());

        int32_t n_tokens =
            llama_tokenize(vocab, gp.prompt.c_str(), static_cast<int32_t>(gp.prompt.size()),
                           nullptr, 0, true, false);
        if (n_tokens < 0) {
            res.error_msg = "tokenize size failed";
            return res;
        }

        std::vector<llama_token> tokens(static_cast<size_t>(n_tokens));
        n_tokens = llama_tokenize(vocab, gp.prompt.c_str(), static_cast<int32_t>(gp.prompt.size()),
                                  tokens.data(), n_tokens, true, false);
        if (n_tokens < 0) {
            res.error_msg = "tokenize failed";
            return res;
        }
        tokens.resize(static_cast<size_t>(n_tokens));

        llama_batch batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
        if (llama_decode(ctx.get(), batch) != 0) {
            res.error_msg = "prompt decode failed";
            return res;
        }

        const llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
        LlamaSamplerPtr sampler(llama_sampler_chain_init(sparams));
        if (gp.repetition_penalty > 0.0f) {
            llama_sampler_chain_add(
                sampler.get(), llama_sampler_init_penalties(64, gp.repetition_penalty, 0.0f, 0.0f));
        }
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_k(gp.top_k));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_top_p(gp.top_p, 1));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_temp(gp.temperature));
        llama_sampler_chain_add(sampler.get(), llama_sampler_init_dist(gp.seed));

        int n_generated = 0;
        bool stopped_by_seq = false;

        while (n_generated < gp.n_predict) {
            if (gp.abort_flag && gp.abort_flag->load())
                break;

            llama_token token = llama_sampler_sample(sampler.get(), ctx.get(), -1);
            if (llama_vocab_is_eog(vocab, token))
                break;

            char buf[256] = {};
            int len = llama_token_to_piece(vocab, token, buf, sizeof(buf) - 1, 0, false);
            if (len > 0) {
                buf[len] = '\0';
                res.output_text += buf;
                if (gp.on_token)
                    gp.on_token(std::string(buf, static_cast<size_t>(len)));
            }

            // Stop sequences
            for (const auto& stop : gp.stop_sequences) {
                auto pos = res.output_text.find(stop);
                if (pos != std::string::npos) {
                    res.output_text.erase(pos);
                    stopped_by_seq = true;
                    break;
                }
            }
            if (stopped_by_seq)
                break;

            llama_batch next = llama_batch_get_one(&token, 1);
            if (llama_decode(ctx.get(), next) != 0)
                break;
            ++n_generated;
        }

        res.n_eval = n_generated;
        res.success = true;

        char log_buf[256];
        snprintf(log_buf, sizeof(log_buf), "[xllama] session generate: n=%d\n", n_generated);
        log_output(log_buf);

        return res;
    }
};

std::unique_ptr<Session> Session::create(const SessionParams& sp, std::string* err) {
    const std::string abs_path = resolve_model_path(sp.model_path);

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 0;

    llama_model* raw_model = llama_model_load_from_file(abs_path.c_str(), mparams);
    if (!raw_model) {
        if (err)
            *err = "failed to load model: " + abs_path;
        return nullptr;
    }

    int n_threads = sp.n_threads > 0 ? sp.n_threads : detect_threads();
    int n_ctx = sp.n_ctx > 0 ? sp.n_ctx : 2048;
    return std::make_unique<LlamaSession>(LlamaModelPtr(raw_model), n_ctx, n_threads);
}

} // namespace xllama

#endif // XLLAMA_USE_ORT
