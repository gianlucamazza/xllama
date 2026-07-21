// Copyright (c) 2024 Venere Labs
// SPDX-License-Identifier: MIT
// xllama::Session — persistent model session for multi-turn applications.
// ORT GenAI path (UWP): model + tokenizer loaded once, reused per generate().
// llama.cpp path (Linux): model kept alive; context rebuilt per generate().

#include "xllama/session.h"
#include "xllama/chat_prompt.h"
#include "xllama/inference_params.h"
#include "xllama/path_utils.h"
#include "xllama/platform.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>

// Backend availability. A build defines XLLAMA_USE_ORT when ORT GenAI is linked.
// llama.cpp is available on Linux (always) and on the UWP llamacpp variant
// (XllamaBackend=llamacpp → XLLAMA_USE_ORT undefined). The unified UWP build
// defines BOTH explicitly; this shim derives the llama flag so exactly one path
// matches every current single-backend build without touching the build files.
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
// Backend factories, each defined in its own #ifdef block below. The public
// Session::create dispatches to one of these.
#ifdef XLLAMA_USE_ORT
std::unique_ptr<Session> create_ort(const SessionParams& sp, std::string* err);
#endif
#ifdef XLLAMA_USE_LLAMA
std::unique_ptr<Session> create_llama(const SessionParams& sp, std::string* err);
#endif
} // namespace detail
} // namespace xllama

// ---------------------------------------------------------------------------
// ONNX Runtime GenAI path (UWP / Xbox Series S)
// ---------------------------------------------------------------------------
#ifdef XLLAMA_USE_ORT

// clang-format off
#include <windows.h>
#include <eh.h>
#include "ort_genai_c.h"

#include "ort_sampling.h" // shared ORT search-param builder (#125); needs ort_genai_c.h
#include "xllama/ort_raii.h"
// clang-format on

namespace xllama {

namespace {
inline void install_se_translator() {
    _set_se_translator([](unsigned int code, EXCEPTION_POINTERS*) {
        char b[48];
        snprintf(b, sizeof(b), "SEH 0x%08X", code);
        throw std::runtime_error(b);
    });
}
} // namespace

class OrtSession final : public Session {
  public:
    OgaModelPtr m_model;
    OgaTokenizerPtr m_tok;
    int m_n_ctx;

    // Persistent chat state for KV-cache reuse (continuous decoding). Kept alive
    // between generate() calls when GenerateParams::reuse_kv is set. m_chat_params
    // must outlive m_chat_gen.
    OgaGeneratorParamsPtr m_chat_params;
    OgaGeneratorPtr m_chat_gen;
    OgaTokenizerStreamPtr m_chat_stream;
    bool m_chat_valid = false;
    // Sampling signature bound into m_chat_gen — the generator can only be reused
    // while these are unchanged (ORT binds sampling at generator creation).
    float m_b_temp = 0.0f, m_b_top_p = 0.0f, m_b_rep = 0.0f;
    int m_b_top_k = 0;

    explicit OrtSession(OgaModelPtr model, OgaTokenizerPtr tok, int n_ctx)
        : m_model(std::move(model)), m_tok(std::move(tok)), m_n_ctx(n_ctx) {}

    void reset_chat_state() {
        m_chat_gen.reset(); // generator before its params
        m_chat_params.reset();
        m_chat_stream.reset();
        m_chat_valid = false;
    }

    bool sampling_matches(const GenerateParams& gp) const {
        return m_chat_valid && m_b_temp == gp.temperature && m_b_top_p == gp.top_p &&
               m_b_top_k == gp.top_k && m_b_rep == gp.repetition_penalty;
    }

    OgaGeneratorParamsPtr make_params(const GenerateParams& gp, int max_length) {
        OgaGeneratorParams* raw_params = nullptr;
        oga_check(OgaCreateGeneratorParams(m_model.get(), &raw_params), "OgaCreateGeneratorParams");
        OgaGeneratorParamsPtr gparams(raw_params);
        oga_check(OgaGeneratorParamsSetSearchNumber(gparams.get(), "max_length",
                                                    static_cast<double>(max_length)),
                  "SetSearchNumber max_length");
        // #125 follow-up: temperature and the greedy guard via the shared helper,
        // so this path and run_inference cannot drift apart again. This is the
        // fix for the greedy gap — make_params used to set the full chain
        // unconditionally, running the repetition penalty before argmax at
        // temperature 0.
        apply_ort_sampling(gparams.get(), gp.sampling());
        return gparams;
    }

    // Decode loop shared by the stateless and chat paths. The caller has already
    // appended the prompt tokens and captured t_prefill_start immediately before
    // AppendTokenSequences; prefill-end is marked after the first
    // GenerateNextToken (in ORT GenAI the prompt prefill runs during the first
    // compute step), so decode timing excludes prefill. n_predict_cap > 0 caps
    // per-turn generation (chat mode); 0 relies on max_length (stateless mode).
    void run_decode(OgaGenerator* gen, OgaTokenizerStream* stream, const GenerateParams& gp,
                    InferenceResult& res, std::chrono::steady_clock::time_point t_prefill_start,
                    int n_prompt_tok, int n_predict_cap) {
        auto t_prefill_end = t_prefill_start;
        int n_generated = 0;
        bool stopped_by_seq = false;
        bool first = true;

        while (!OgaGenerator_IsDone(gen)) {
            if (gp.abort_flag && gp.abort_flag->load())
                break;
            if (n_predict_cap > 0 && n_generated >= n_predict_cap)
                break;

            oga_check(OgaGenerator_GenerateNextToken(gen), "GenerateNextToken");
            if (first) {
                t_prefill_end = std::chrono::steady_clock::now();
                first = false;
            }

            const int32_t* next_toks = nullptr;
            size_t n_next = 0;
            oga_check(OgaGenerator_GetNextTokens(gen, &next_toks, &n_next), "GetNextTokens");
            for (size_t i = 0; i < n_next; ++i) {
                const char* piece = nullptr;
                oga_check(OgaTokenizerStreamDecode(stream, next_toks[i], &piece),
                          "TokenizerStreamDecode");
                if (piece && *piece) {
                    res.output_text += piece;
                    if (gp.on_token)
                        gp.on_token(std::string(piece));
                }
            }
            ++n_generated;

            if (apply_stop_sequences(res.output_text, gp.stop_sequences)) {
                stopped_by_seq = true;
                break;
            }
        }

        auto t_end = std::chrono::steady_clock::now();
        res.n_p_eval = n_prompt_tok;
        res.t_p_eval_ms =
            std::chrono::duration<double, std::milli>(t_prefill_end - t_prefill_start).count();
        // Decode excludes the first token (produced by the prefill step), matching
        // the bench convention so interactive and CSV tok/s are comparable.
        res.n_eval = n_generated > 0 ? n_generated - 1 : 0;
        res.t_eval_ms = std::chrono::duration<double, std::milli>(t_end - t_prefill_end).count();
        res.ended_with_stop = stopped_by_seq;
        res.peak_ws_mb = peak_working_set_mb();
        res.success = true;
    }

    // Legacy stateless turn: fresh generator, full prompt, destroyed at return.
    void generate_stateless(const GenerateParams& gp, InferenceResult& res) {
        if (gp.on_status)
            gp.on_status("tokenizing");

        OgaTokenizerStream* raw_stream = nullptr;
        oga_check(OgaCreateTokenizerStream(m_tok.get(), &raw_stream), "OgaCreateTokenizerStream");
        OgaTokenizerStreamPtr stream(raw_stream);

        OgaSequences* raw_seqs = nullptr;
        oga_check(OgaCreateSequences(&raw_seqs), "OgaCreateSequences");
        OgaSequencesPtr seqs(raw_seqs);
        oga_check(OgaTokenizerEncode(m_tok.get(), gp.prompt.c_str(), seqs.get()),
                  "OgaTokenizerEncode");
        int n_prompt_tok = static_cast<int>(OgaSequencesGetSequenceCount(seqs.get(), 0));

        // #130: request the FULL context as max_length and bound generation with
        // the n_predict cap in run_decode — exactly what generate_chat already
        // does. The obvious alternative, max_length = min(n_ctx, prompt +
        // n_predict), is what this used to do, and on DirectML it is 3-4x slower.
        //
        // Measured 2026-07-21, Series S, one byte-identical 1289-token prompt,
        // only n_predict varied (bench/results/phase12-maxlen-band.csv):
        //
        //   max_length  1297  1400  1545  1650  1801  1950  2048
        //   prefill t/s   515   475   170   215   130   212   611
        //
        // There is a valley in max_length between ~1400 and n_ctx, deepest near
        // 1800, with both edges clean. The prompt never changed, so this is not a
        // prompt-length effect. A control run at n_ctx 3072 holding max_length at
        // 1801 reproduced the slow figure (132 t/s), so n_ctx has no effect of its
        // own — max_length alone controls it. Saturating also costs LESS memory
        // here (1968 MB against 2483 at max_length 1950).
        //
        // The shipping default (n_predict 256) landed at 1545 — inside the valley.
        // Mechanism still unknown; suspected shape-bucketed DML kernel selection.
        OgaGeneratorParamsPtr gparams = make_params(gp, m_n_ctx);
        OgaGenerator* raw_gen = nullptr;
        oga_check(OgaCreateGenerator(m_model.get(), gparams.get(), &raw_gen), "OgaCreateGenerator");
        OgaGeneratorPtr gen(raw_gen);

        if (gp.on_status)
            gp.on_status("generating");
        auto t0 = std::chrono::steady_clock::now();
        oga_check(OgaGenerator_AppendTokenSequences(gen.get(), seqs.get()), "AppendTokenSequences");
        // Reproduce the old allowance exactly. The previous max_length of
        // min(n_ctx, prompt + n_predict) permitted min(n_ctx - prompt, n_predict)
        // new tokens; max_length is no longer the bound, so the cap has to carry
        // that arithmetic or a long prompt would now generate past the context.
        // Clamped to >= 1 because run_decode treats a cap of 0 as "no cap": a
        // prompt at or beyond the context would otherwise flip from a hard stop
        // to unbounded generation. Such a prompt fails inside ORT first
        // ("input_ids size exceeds max length"), which is the intended error.
        const int n_predict_cap = std::max(1, std::min(m_n_ctx - n_prompt_tok, gp.n_predict));
        run_decode(gen.get(), stream.get(), gp, res, t0, n_prompt_tok, n_predict_cap);
        log_gen(res, false);
    }

    // Continuation turn: append `gp.prompt` (the new turn's tokens) to the
    // persistent generator; on reset_kv or a sampling change, rebuild it first
    // (then `gp.prompt` is the full context).
    void generate_chat(const GenerateParams& gp, InferenceResult& res) {
        const bool reuse = !gp.reset_kv && sampling_matches(gp);
        if (!reuse) {
            reset_chat_state();
            m_chat_params = make_params(gp, m_n_ctx);
            OgaGenerator* raw_gen = nullptr;
            oga_check(OgaCreateGenerator(m_model.get(), m_chat_params.get(), &raw_gen),
                      "OgaCreateGenerator(chat)");
            m_chat_gen.reset(raw_gen);
            OgaTokenizerStream* raw_stream = nullptr;
            oga_check(OgaCreateTokenizerStream(m_tok.get(), &raw_stream),
                      "OgaCreateTokenizerStream(chat)");
            m_chat_stream.reset(raw_stream);
            m_b_temp = gp.temperature;
            m_b_top_p = gp.top_p;
            m_b_top_k = gp.top_k;
            m_b_rep = gp.repetition_penalty;
            m_chat_valid = true;
        }

        if (gp.on_status)
            gp.on_status("tokenizing");
        OgaSequences* raw_seqs = nullptr;
        oga_check(OgaCreateSequences(&raw_seqs), "OgaCreateSequences");
        OgaSequencesPtr seqs(raw_seqs);
        oga_check(OgaTokenizerEncode(m_tok.get(), gp.prompt.c_str(), seqs.get()),
                  "OgaTokenizerEncode");
        int n_prompt_tok = static_cast<int>(OgaSequencesGetSequenceCount(seqs.get(), 0));

        if (gp.on_status)
            gp.on_status("generating");
        auto t0 = std::chrono::steady_clock::now();
        oga_check(OgaGenerator_AppendTokenSequences(m_chat_gen.get(), seqs.get()),
                  "AppendTokenSequences(chat)");
        run_decode(m_chat_gen.get(), m_chat_stream.get(), gp, res, t0, n_prompt_tok, gp.n_predict);

        size_t kv = OgaGenerator_GetSequenceCount(m_chat_gen.get(), 0);
        char lb[192];
        snprintf(lb, sizeof(lb),
                 "[xllama] chat turn: reuse=%d prefill=%d tok decode=%d tok kv_len=%zu\n",
                 reuse ? 1 : 0, res.n_p_eval, res.n_eval, kv);
        log_output(lb);
    }

    void log_gen(const InferenceResult& res, bool chat) {
        double dt =
            (res.n_eval > 0 && res.t_eval_ms > 0) ? res.n_eval / (res.t_eval_ms / 1000.0) : 0.0;
        char log_buf[256];
        snprintf(log_buf, sizeof(log_buf), "[xllama] session generate%s: decode=%.1f tok/s n=%d\n",
                 chat ? " (chat)" : "", dt, res.n_eval);
        log_output(log_buf);
    }

    InferenceResult generate(const GenerateParams& gp) override {
        InferenceResult res;
        install_se_translator();
        try {
            if (gp.reuse_kv)
                generate_chat(gp, res);
            else
                generate_stateless(gp, res);
        } catch (const std::exception& e) {
            res.success = false;
            res.error_msg = e.what();
            if (gp.reuse_kv)
                reset_chat_state(); // poisoned generator — force a fresh one next turn
            log_output(("[xllama] session generate error: " + res.error_msg + "\n").c_str());
            if (gp.on_status)
                gp.on_status("error: " + res.error_msg);
        }
        return res;
    }

    int count_tokens(const std::string& prompt) override {
        OgaSequences* raw_seqs = nullptr;
        oga_check(OgaCreateSequences(&raw_seqs), "OgaCreateSequences");
        OgaSequencesPtr seqs(raw_seqs);
        oga_check(OgaTokenizerEncode(m_tok.get(), prompt.c_str(), seqs.get()),
                  "OgaTokenizerEncode");
        return static_cast<int>(OgaSequencesGetSequenceCount(seqs.get(), 0));
    }
};

namespace detail {
std::unique_ptr<Session> create_ort(const SessionParams& sp, std::string* err) {
    install_se_translator();

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
        GpuMemInfo gpu = gpu_mem_info();
        if (gpu.available) {
            char gpu_buf[128];
            snprintf(gpu_buf, sizeof(gpu_buf),
                     "[xllama] gpu-mem post-load: current=%zuMB budget=%zuMB\n", gpu.current_mb,
                     gpu.budget_mb);
            log_output(gpu_buf);
        }
        int n_ctx = sp.n_ctx > 0 ? sp.n_ctx : 2048;
        return std::make_unique<OrtSession>(std::move(model), std::move(tok), n_ctx);

    } catch (const std::exception& e) {
        if (err)
            *err = e.what();
        log_output(("[xllama] Session::create error: " + std::string(e.what()) + "\n").c_str());
        return nullptr;
    }
}
} // namespace detail

} // namespace xllama

#endif // XLLAMA_USE_ORT

// ---------------------------------------------------------------------------
// llama.cpp path (Linux dev + UWP llamacpp/unified)
// ---------------------------------------------------------------------------
#ifdef XLLAMA_USE_LLAMA

    #include "llama.h"

    #include "sampler_chain.h" // shared sampler chain (#125); needs llama.h
    #include "xllama/llama_raii.h"

    #include <vector>

namespace xllama {

class LlamaSession final : public Session {
  public:
    LlamaModelPtr m_model;
    LlamaAdapterLoraPtr m_adapter; // optional runtime LoRA (freed before model)
    float m_lora_scale = 1.0f;
    int m_n_ctx;
    int m_n_threads;
    int m_n_batch;  // 0 = llama.cpp default
    int m_n_ubatch; // 0 = llama.cpp default
    // Persistent context across turns: the KV cache lives here so a reuse turn
    // (reuse_kv && !reset_kv) can append only the delta instead of re-prefilling
    // the whole conversation. Created lazily on the first generate().
    LlamaContextPtr m_ctx;

    explicit LlamaSession(LlamaModelPtr model, LlamaAdapterLoraPtr adapter, float lora_scale,
                          int n_ctx, int n_threads, int n_batch, int n_ubatch)
        : m_model(std::move(model)), m_adapter(std::move(adapter)), m_lora_scale(lora_scale),
          m_n_ctx(n_ctx), m_n_threads(n_threads), m_n_batch(n_batch), m_n_ubatch(n_ubatch) {}

    InferenceResult generate(const GenerateParams& gp) override {
        InferenceResult res;

        if (!m_ctx) {
            llama_context_params cparams = llama_context_default_params();
            cparams.n_ctx = m_n_ctx;
            cparams.n_threads = m_n_threads;
            if (m_n_batch > 0)
                cparams.n_batch = static_cast<uint32_t>(m_n_batch);
            if (m_n_ubatch > 0)
                cparams.n_ubatch = static_cast<uint32_t>(m_n_ubatch);
            m_ctx.reset(llama_init_from_model(m_model.get(), cparams));
            if (!m_ctx) {
                res.error_msg = "failed to create context";
                log_output("[xllama] session generate: failed to create context\n");
                return res;
            }
            // Apply runtime LoRA after context exists (llama_set_adapters_lora is
            // context-scoped). Must re-apply if context is ever recreated.
            if (m_adapter) {
                llama_adapter_lora* arr[1] = {m_adapter.get()};
                float scales[1] = {m_lora_scale};
                if (llama_set_adapters_lora(m_ctx.get(), arr, 1, scales) != 0) {
                    res.error_msg = "llama_set_adapters_lora failed";
                    log_output("[xllama] session generate: set_adapters_lora failed\n");
                    m_ctx.reset();
                    return res;
                }
                log_output("[xllama] session: runtime LoRA adapter applied\n");
            }
        }
        llama_context* ctx = m_ctx.get();

        // KV-cache reuse: a continuation turn (reuse_kv && !reset_kv) keeps the
        // existing cache and appends the delta; otherwise clear it and re-prefill
        // the full prompt. add_bos only on a full prompt — a delta must not carry
        // a BOS mid-conversation.
        const bool full_prompt = gp.reset_kv || !gp.reuse_kv;
        if (full_prompt) {
            llama_memory_clear(llama_get_memory(ctx), true);
        }

        const llama_vocab* vocab = llama_model_get_vocab(m_model.get());

        // parse_special=true: every Session caller (chat UI, API endpoint, bench)
        // passes a TEMPLATED prompt, whose <|im_start|>/<|im_end|> markers must
        // become the special ids the model was trained on. As plain text the
        // model sees an alien template and its next-token distribution goes flat
        // (LFM2.5: top-2 gap 0.52 vs 3.91 with specials) — flat enough that the
        // Zen2-vs-desktop kernel jitter (max |Δlogit| ~1.6 measured) flipped the
        // greedy token on-device ("User\n\n<|end|>" instead of a greeting).
        int32_t n_tokens =
            llama_tokenize(vocab, gp.prompt.c_str(), static_cast<int32_t>(gp.prompt.size()),
                           nullptr, 0, full_prompt, true);
        if (n_tokens == INT32_MIN) {
            res.error_msg = "tokenize overflow";
            return res;
        }
        // Size query returns the negated required count (see llama.h) — not an error.
        n_tokens = -n_tokens;

        std::vector<llama_token> tokens(static_cast<size_t>(n_tokens));
        n_tokens = llama_tokenize(vocab, gp.prompt.c_str(), static_cast<int32_t>(gp.prompt.size()),
                                  tokens.data(), n_tokens, full_prompt, true);
        if (n_tokens < 0) {
            res.error_msg = "tokenize failed";
            return res;
        }
        tokens.resize(static_cast<size_t>(n_tokens));

        // Prefill: positions continue automatically from the KV cache end, so a
        // reuse turn appends after the retained context.
        const auto t_prefill0 = std::chrono::steady_clock::now();
        llama_batch batch = llama_batch_get_one(tokens.data(), static_cast<int32_t>(tokens.size()));
        if (llama_decode(ctx, batch) != 0) {
            res.error_msg = "prompt decode failed";
            log_output("[xllama] session generate: prompt decode failed\n");
            return res;
        }
        res.t_p_eval_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_prefill0)
                .count();
        res.n_p_eval = static_cast<int>(tokens.size());

        const llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
        LlamaSamplerPtr sampler(llama_sampler_chain_init(sparams));
        // Shared with run_inference — see src/bridge/sampler_chain.h and #125.
        // SamplingConfig::is_greedy() also covers temperature 0, where the full
        // chain must not run (the repetition penalty can flip the argmax).
        add_sampler_stages(sampler.get(), gp.sampling());

        int n_generated = 0;
        bool stopped_by_seq = false;
        const auto t_decode0 = std::chrono::steady_clock::now();

        while (n_generated < gp.n_predict) {
            if (gp.abort_flag && gp.abort_flag->load())
                break;

            llama_token token = llama_sampler_sample(sampler.get(), ctx, -1);
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

            // Stop sequences (shared suffix-match helper).
            if (apply_stop_sequences(res.output_text, gp.stop_sequences)) {
                stopped_by_seq = true;
                break;
            }

            llama_batch next = llama_batch_get_one(&token, 1);
            if (llama_decode(ctx, next) != 0) {
                log_output("[xllama] session generate: decode failed, stopping\n");
                break;
            }
            ++n_generated;
        }

        res.t_eval_ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_decode0)
                .count();
        res.n_eval = n_generated;
        res.ended_with_stop = stopped_by_seq;
        res.success = true;

        char log_buf[256];
        snprintf(log_buf, sizeof(log_buf),
                 "[xllama] session generate: n=%d prefill=%.1fms (%d tok) reuse=%d\n", n_generated,
                 res.t_p_eval_ms, res.n_p_eval, gp.reuse_kv && !gp.reset_kv);
        log_output(log_buf);

        return res;
    }

    int count_tokens(const std::string& prompt) override {
        const llama_vocab* vocab = llama_model_get_vocab(m_model.get());
        // parse_special=true to match generate(): counting template markers as
        // plain text overcounts a chat prompt by ~70% (50 vs 29 tokens on a
        // 2-turn ChatML render), which skews the routing threshold.
        int32_t n = llama_tokenize(vocab, prompt.c_str(), static_cast<int32_t>(prompt.size()),
                                   nullptr, 0, true, true);
        if (n == INT32_MIN)
            return 0;
        return -n;
    }
};

namespace detail {
std::unique_ptr<Session> create_llama(const SessionParams& sp, std::string* err) {
    // resolve_model_path yields a FILE path on Linux (a direct .gguf) but the
    // model DIRECTORY on UWP (catalogue layout LocalState\models\<name>\). llama
    // loads a file, so descend into a directory and pick the single .gguf inside.
    // Exclude LoRA basename so dirs with model.gguf + adapter.gguf load the base.
    std::string lora_base;
    if (!sp.lora_path.empty()) {
        lora_base = std::filesystem::path(sp.lora_path).filename().string();
    }
    std::string abs_path = first_gguf_in_dir(resolve_model_path(sp.model_path), lora_base);
    if (abs_path.empty()) {
        if (err)
            *err = "no .gguf file in model dir: " + resolve_model_path(sp.model_path);
        return nullptr;
    }

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = sp.n_gpu_layers;

    llama_model* raw_model = llama_model_load_from_file(abs_path.c_str(), mparams);
    if (!raw_model) {
        if (err)
            *err = "failed to load model: " + abs_path;
        return nullptr;
    }

    LlamaAdapterLoraPtr adapter;
    if (!sp.lora_path.empty()) {
        llama_adapter_lora* raw_ad = llama_adapter_lora_init(raw_model, sp.lora_path.c_str());
        if (!raw_ad) {
            llama_model_free(raw_model);
            if (err)
                *err = "failed to load LoRA adapter: " + sp.lora_path;
            return nullptr;
        }
        adapter.reset(raw_ad);
        log_output("[xllama] Session: GGUF LoRA adapter loaded (runtime)\n");
    }

    int n_threads = sp.n_threads > 0 ? sp.n_threads : detect_threads_llama();
    int n_ctx = sp.n_ctx > 0 ? sp.n_ctx : 2048;
    log_output("[xllama] Session: GGUF model loaded via llama.cpp (persistent)\n");
    return std::make_unique<LlamaSession>(LlamaModelPtr(raw_model), std::move(adapter),
                                          sp.lora_scale, n_ctx, n_threads, sp.n_batch, sp.n_ubatch);
}
} // namespace detail

} // namespace xllama

#endif // XLLAMA_USE_LLAMA

// ---------------------------------------------------------------------------
// Public factory: dispatch to the compiled backend(s).
// ---------------------------------------------------------------------------
namespace xllama {

std::unique_ptr<Session> Session::create(const SessionParams& sp, std::string* err) {
#if defined(XLLAMA_USE_ORT) && defined(XLLAMA_USE_LLAMA)
    Backend b = sp.backend;
    if (b == Backend::Auto) {
        // Layout-aware: uses suffix fast-path + resolve + directory scan so that
        // bare catalogue names (e.g. "qwen35-0.8b" pointing at a dir with .gguf)
        // are classified correctly. Explicit Backend from manifest (MainPage) or
        // tests still takes precedence.
        b = model_uses_llama_backend(sp.model_path) ? Backend::LlamaCpp : Backend::OrtGenAI;
    }
    return b == Backend::LlamaCpp ? detail::create_llama(sp, err) : detail::create_ort(sp, err);
#elif defined(XLLAMA_USE_ORT)
    return detail::create_ort(sp, err);
#else // XLLAMA_USE_LLAMA
    return detail::create_llama(sp, err);
#endif
}

} // namespace xllama
