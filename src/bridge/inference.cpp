// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "xllama/inference.h"
#include "xllama/chat_prompt.h"
#include "xllama/logit_dump.h"
#include "xllama/path_utils.h"
#include "xllama/platform.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

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

#include "decode_loop_ort.h"
#include "ort_common.h"
#include "ort_sampling.h" // shared ORT search-param builder (#125); needs ort_genai_c.h
    #include "xllama/ort_raii.h"
// clang-format on

    #include <chrono>

namespace xllama {
namespace detail {

// IEEE 754 binary16 → float32. DirectML logit tensors are frequently float16;
// the parity dump normalizes to float32 to match the llama.cpp reference.
static inline float half_to_float(uint16_t h) {
    const uint32_t sign = (h & 0x8000u) << 16;
    const uint32_t exp = (h >> 10) & 0x1Fu;
    const uint32_t mant = h & 0x3FFu;
    uint32_t bits;
    if (exp == 0) {
        if (mant == 0) {
            bits = sign; // ±0
        } else {
            // Subnormal: normalize.
            int e = -1;
            uint32_t m = mant;
            do {
                ++e;
                m <<= 1;
            } while ((m & 0x400u) == 0);
            m &= 0x3FFu;
            bits = sign | ((127 - 15 - e) << 23) | (m << 13);
        }
    } else if (exp == 0x1Fu) {
        bits = sign | 0x7F800000u | (mant << 13); // Inf / NaN
    } else {
        bits = sign | ((exp + (127 - 15)) << 23) | (mant << 13);
    }
    float f;
    std::memcpy(&f, &bits, sizeof(f));
    return f;
}

InferenceResult run_inference_ort(const InferenceParams& params) {
    InferenceResult res;

    // Convert SEH (D3D12/DML OOM/AV) → std::runtime_error so the catch block can log it.
    // Consolidated into ort_common.h to eliminate duplication with session.cpp.
    install_se_translator();

    const std::string model_dir = resolve_model_path(params.model_path);

    if (params.on_status)
        params.on_status("loading model");

    try {
        // Redirect ORT GenAI internal log messages into xllama.log.
        // Consolidated into ort_common.h — eliminates duplication with session.cpp.
        register_oga_logging();

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
        // Size max_length from the ACTUAL prompt, clamped to the context: the old
        // fixed "n_predict + 512" headroom underflowed on long prompts (a 959-tok
        // routed turn exceeded max_length 768 before generating a single token).
        // #130: full ladder lives in resolve_max_length (inference_params.h),
        // shared with Session so the two surfaces cannot drift apart.
        const int max_len = resolve_max_length(params.n_ctx, static_cast<int>(n_prompt_tok),
                                               params.n_predict, params.max_length_override);
        res.max_length = max_len;
        {
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
                                                    static_cast<double>(max_len)),
                  "SetSearchNumber max_length");
        // Temperature + the greedy guard via the shared helper (ort_sampling.h),
        // the same one OrtSession uses — so the two ORT surfaces cannot drift
        // apart again the way they did before this fix. Greedy (incl. temp 0)
        // pins argmax; it is the prerequisite for logit parity.
        apply_ort_sampling(gparams.get(), params.sampling());
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

        // Logit-parity dump: capture the last prompt-token logits.
        //
        // ORT GenAI state machine (verified on the vendored source, identical in
        // v0.14 and 0.15-dev — generators.cpp): AppendTokens runs the prefill
        // forward itself and leaves computed_logits_=true, so GetLogits returns
        // the true last-prompt-token distribution — equal to the last row of
        // GetOutput("logits") and to llama_get_logits_ith(ctx,-1) (canonical
        // pattern: test/c_api_tests.cpp GetLogitsCAPI). After the first
        // GenerateNextToken the flag is false and GetLogits RECOMPUTES a forward
        // over the just-sampled token — the "position after the first generated
        // token" bug observed on-console (golden argmax at rank ~2780).
        //
        // Must be invoked right after AppendTokenSequences, BEFORE the first
        // GenerateNextToken. The C API always copies to a float32 [batch,1,vocab]
        // tensor (ort_genai_c.cpp), but keep the fp16 branch defensive.
        auto dump_prefill_logits = [&]() {
            OgaTensor* raw_logits = nullptr;
            oga_check(OgaGenerator_GetLogits(gen.get(), &raw_logits), "GetLogits");
            OgaTensorPtr logits_t(raw_logits);

            OgaElementType etype = OgaElementType_undefined;
            oga_check(OgaTensorGetType(logits_t.get(), &etype), "TensorGetType");
            size_t rank = 0;
            oga_check(OgaTensorGetShapeRank(logits_t.get(), &rank), "TensorGetShapeRank");
            std::vector<int64_t> shape(rank);
            oga_check(OgaTensorGetShape(logits_t.get(), shape.data(), rank), "TensorGetShape");
            int64_t total = 1;
            for (size_t i = 0; i < rank; ++i)
                total *= shape[i];
            const int vocab =
                rank > 0 ? static_cast<int>(shape[rank - 1]) : static_cast<int>(total);

            // Diagnostic: shape + dtype in the log so an on-device mismatch is
            // attributable without another build cycle.
            {
                std::string s =
                    "[xllama] logits tensor: etype=" + std::to_string(etype) + " shape=[";
                for (size_t i = 0; i < rank; ++i)
                    s += (i ? "," : "") + std::to_string(shape[i]);
                s += "]\n";
                log_output(s);
            }

            void* data = nullptr;
            oga_check(OgaTensorGetData(logits_t.get(), &data), "TensorGetData");

            // Keep only the last token's row (total may cover [batch, seq, vocab]).
            std::vector<float> f32(static_cast<size_t>(vocab));
            const int64_t off = total - vocab; // last row offset
            if (etype == OgaElementType_float32) {
                const float* src = static_cast<const float*>(data) + off;
                for (int i = 0; i < vocab; ++i)
                    f32[static_cast<size_t>(i)] = src[i];
            } else if (etype == OgaElementType_float16) {
                const uint16_t* src = static_cast<const uint16_t*>(data) + off;
                for (int i = 0; i < vocab; ++i)
                    f32[static_cast<size_t>(i)] = half_to_float(src[i]);
            } else {
                log_output("[xllama] WARN: unsupported logit tensor type, dump skipped\n");
                f32.clear();
            }

            if (!f32.empty()) {
                int top1_id = 0;
                float top1_val = f32[0];
                for (int i = 1; i < vocab; ++i)
                    if (f32[static_cast<size_t>(i)] > top1_val) {
                        top1_val = f32[static_cast<size_t>(i)];
                        top1_id = i;
                    }
                const char* piece = nullptr;
                std::string top1_piece;
                if (!OgaTokenizerDecode(tok.get(), &top1_id, 1, &piece) && piece) {
                    top1_piece = piece;
                    OgaDestroyString(piece);
                }
                if (write_logit_dump(params.dump_logits_path, f32.data(), vocab, model_dir,
                                     params.prompt, "ort", params.greedy, top1_piece))
                    log_output("[xllama] logits dumped to " + params.dump_logits_path + "\n");
                else
                    log_output("[xllama] WARN: logit dump failed: " + params.dump_logits_path +
                               "\n");
            }
        };

        // AppendTokenSequences has already run the prefill: computed_logits_ is
        // true and GetLogits here is the last-prompt-token distribution. Any
        // later (post-GenerateNextToken) capture reads the wrong position.
        if (!params.dump_logits_path.empty())
            dump_prefill_logits();

        // Consolidated decode loop — handles stop sequences (previously the
        // stateless path silently ignored them), timing, and metrics.
        run_decode_loop_ort(gen.get(), stream.get(), params, res, t0, n_prompt_tok,
                            params.n_predict);

        auto t_end = std::chrono::steady_clock::now();
        double elapsed_s = std::chrono::duration<double>(t_end - t0).count();
        double prefill_ms = res.t_p_eval_ms;
        double decode_s = res.t_eval_ms / 1000.0;
        int n_decode = res.n_eval;

        res.peak_ws_mb = peak_working_set_mb();

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

    #include "decode_loop.h"
    #include "sampler_chain.h" // shared sampler chain (#125); needs llama.h
    #include "xllama/llama_raii.h"

    #include <vector>

namespace xllama {
namespace detail {

InferenceResult run_inference_llama(const InferenceParams& params) {
    InferenceResult res;

    // Catalogue GGUF entries resolve to the model DIRECTORY; llama loads a file.
    std::string lora_base;
    if (!params.lora_path.empty())
        lora_base = std::filesystem::path(params.lora_path).filename().string();
    const std::string abs_model_path =
        first_gguf_in_dir(resolve_model_path(params.model_path), lora_base);
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

    LlamaAdapterLoraPtr adapter;
    if (!params.lora_path.empty()) {
        llama_adapter_lora* raw_ad = llama_adapter_lora_init(model.get(), params.lora_path.c_str());
        if (!raw_ad) {
            res.error_msg = "failed to load LoRA adapter: " + params.lora_path;
            log_output("[xllama] " + res.error_msg + "\n");
            if (params.on_status)
                params.on_status("error: " + res.error_msg);
            return res;
        }
        adapter.reset(raw_ad);
        log_output("[xllama] LoRA adapter loaded (runtime)\n");
    }

    if (params.on_status)
        params.on_status("decoding");

    const int n_threads = params.n_threads > 0 ? params.n_threads : detect_threads_llama();

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = static_cast<uint32_t>(params.n_ctx);
    cparams.n_threads = n_threads;
    // Prefill runs on n_threads_batch (default 4, independent of n_threads) —
    // see LlamaSession::generate and #168.
    cparams.n_threads_batch = n_threads;
    if (params.n_batch > 0)
        cparams.n_batch = static_cast<uint32_t>(params.n_batch);
    if (params.n_ubatch > 0)
        cparams.n_ubatch = static_cast<uint32_t>(params.n_ubatch);
    if (params.n_batch > 0 || params.n_ubatch > 0)
        log_output("[xllama] prefill batch override: n_batch=" + std::to_string(cparams.n_batch) +
                   " n_ubatch=" + std::to_string(cparams.n_ubatch) + "\n");
    if (params.kv_q8) {
        // #171: q8_0 KV needs flash attention (quantized V throws without it);
        // mirror LlamaSession — force FA, fall back below if the arch refuses.
        cparams.type_k = GGML_TYPE_Q8_0;
        cparams.type_v = GGML_TYPE_Q8_0;
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        log_output("[xllama] KV cache: q8_0 + flash attention (#171)\n");
    }

    llama_context* raw_ctx = llama_init_from_model(model.get(), cparams);
    if (!raw_ctx && params.kv_q8) {
        log_output("[xllama] q8_0 KV context failed — falling back to default cache types\n");
        cparams.type_k = llama_context_default_params().type_k;
        cparams.type_v = llama_context_default_params().type_v;
        cparams.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_AUTO;
        raw_ctx = llama_init_from_model(model.get(), cparams);
    }
    if (!raw_ctx) {
        res.error_msg = "failed to create context";
        log_output("[xllama] failed to create context\n");
        return res;
    }
    LlamaContextPtr ctx(raw_ctx);

    if (adapter) {
        llama_adapter_lora* arr[1] = {adapter.get()};
        float scales[1] = {params.lora_scale};
        if (llama_set_adapters_lora(ctx.get(), arr, 1, scales) != 0) {
            res.error_msg = "llama_set_adapters_lora failed";
            log_output("[xllama] " + res.error_msg + "\n");
            return res;
        }
    }

    const llama_vocab* vocab = llama_model_get_vocab(model.get());

    // parse_special only for --chat: there the prompt is a rendered template
    // whose markers must map to the model's special ids (see session.cpp). A raw
    // -p prompt keeps them as text so user input can't smuggle special tokens.
    const bool parse_special = params.chat_template;
    int32_t n_tokens =
        llama_tokenize(vocab, params.prompt.c_str(), static_cast<int32_t>(params.prompt.size()),
                       nullptr, 0, true, parse_special);
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
                       tokens.data(), n_tokens, true, parse_special);
    if (n_tokens < 0) {
        res.error_msg = "tokenization failed";
        log_output("[xllama] tokenization failed\n");
        return res;
    }
    tokens.resize(static_cast<size_t>(n_tokens));

    log_output(("[xllama] prompt tokens: " + std::to_string(tokens.size()) + "\n").c_str());
    const auto t_prompt0 = std::chrono::steady_clock::now();
    // Prefill chunking and the context guard live in decode_loop.h — see there
    // for why an oversized logical batch aborts rather than erroring.
    const int n_prompt_tokens = static_cast<int>(tokens.size());
    const int n_ctx_active = static_cast<int>(llama_n_ctx(ctx.get()));
    if (n_prompt_tokens + 1 > n_ctx_active) {
        res.error_msg = prompt_too_long_message(n_prompt_tokens, n_ctx_active);
        log_output(("[xllama] " + res.error_msg + "\n").c_str());
        return res;
    }
    if (!prefill_chunked(ctx.get(), tokens.data(), n_prompt_tokens)) {
        res.error_msg = "prompt decode failed";
        log_output("[xllama] prompt decode failed\n");
        return res;
    }
    const double prompt_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_prompt0)
            .count();

    // Logit-parity dump: capture the last prefill-token logits (one deterministic
    // forward pass, no autoregressive drift) before any sampling touches them.
    if (!params.dump_logits_path.empty()) {
        const int n_vocab = llama_vocab_n_tokens(vocab);
        const float* logits = llama_get_logits_ith(ctx.get(), -1);
        if (logits) {
            // Detokenize argmax so the sidecar can catch tokenizer/vocab misalignment.
            int top1_id = 0;
            float top1_val = logits[0];
            for (int i = 1; i < n_vocab; ++i)
                if (logits[i] > top1_val) {
                    top1_val = logits[i];
                    top1_id = i;
                }
            char pbuf[256] = {};
            int plen = llama_token_to_piece(vocab, top1_id, pbuf, sizeof(pbuf) - 1, 0, false);
            std::string top1_piece = plen > 0 ? std::string(pbuf, static_cast<size_t>(plen)) : "";
            if (write_logit_dump(params.dump_logits_path, logits, n_vocab, abs_model_path,
                                 params.prompt, "llama.cpp", params.greedy, top1_piece))
                log_output("[xllama] logits dumped to " + params.dump_logits_path + "\n");
            else
                log_output("[xllama] WARN: logit dump failed: " + params.dump_logits_path + "\n");
        }
    }

    const llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
    LlamaSamplerPtr sampler(llama_sampler_chain_init(sparams));
    // Shared with Session — see src/bridge/sampler_chain.h. Before #125 this
    // path built temp -> dist and nothing else: no top_k, no top_p, no
    // repetition penalty, so a CLI run could not reproduce a GUI generation
    // even given identical flags. Greedy (including temperature 0) is handled
    // inside the builder.
    add_sampler_stages(sampler.get(), params.sampling(), vocab);

    const auto t_gen0 = std::chrono::steady_clock::now();
    // Seed for W2 prompt-lookup: prefill tokens only; the loop appends accepted
    // generated tokens when on_accepted is null (CLI path).
    std::vector<llama_token> gen_history = tokens;
    DecodeLoopParams dlp;
    dlp.ctx = ctx.get();
    dlp.sampler = sampler.get();
    dlp.vocab = vocab;
    dlp.n_predict = params.n_predict;
    dlp.stop_sequences = &params.stop_sequences;
    dlp.abort_flag = params.abort_flag;
    dlp.on_token = params.on_token;
    dlp.echo_stdout = params.echo_stdout;
    dlp.decode_start = t_gen0;
    dlp.prompt_lookup = params.prompt_lookup;
    dlp.token_history = params.prompt_lookup ? &gen_history : nullptr;
    const DecodeLoopResult dlr = decode_loop(dlp, res.output_text);
    const int n_generated = dlr.n_generated;
    res.ended_with_stop = dlr.ended_with_stop;
    res.n_drafted = dlr.n_drafted;
    res.n_spec_accepted = dlr.n_accepted;

    if (params.echo_stdout)
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
    res.t_first_token_ms = dlr.first_token_ms > 0.0 ? prompt_ms + dlr.first_token_ms : 0.0;
    res.n_p_eval = static_cast<int32_t>(tokens.size());
    res.n_eval = n_generated;
    res.peak_ws_mb = peak_working_set_mb();
    if (dlr.rewind_failed) {
        res.success = false;
        res.error_msg =
            "speculative KV rewind unsupported (disable --prompt-lookup for this model)";
        log_output(("[xllama] " + res.error_msg + "\n").c_str());
        return res;
    }
    res.success = true;

    char log_buf[320];
    snprintf(log_buf, sizeof(log_buf),
             "[xllama] done: load=%.0fms prompt=%.1f tok/s decode=%.1f tok/s peak=%zuMB"
             " drafted=%d spec_accept=%d\n",
             res.t_load_ms,
             res.n_p_eval > 0 && res.t_p_eval_ms > 0
                 ? static_cast<double>(res.n_p_eval) / (res.t_p_eval_ms / 1000.0)
                 : 0.0,
             res.n_eval > 0 && res.t_eval_ms > 0
                 ? static_cast<double>(res.n_eval) / (res.t_eval_ms / 1000.0)
                 : 0.0,
             res.peak_ws_mb, res.n_drafted, res.n_spec_accepted);
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
