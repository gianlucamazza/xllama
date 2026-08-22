// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
//
// Consolidated ORT GenAI decode loop. Replaces the two near-identical loops
// in run_inference_ort (inference.cpp:317–349) and OrtSession::run_decode
// (session.cpp:138–190).
//
// Key fix: the stateless path (run_inference_ort) now checks stop sequences
// just like the chat path — previously it silently ignored them.
//
// Header-only, WinRT-free by design (declares OgaGenerator* / OgaTokenizerStream*
// raw pointers; callers create/destroy the objects). Host-testable with mocks.

#pragma once

#include "xllama/chat_prompt.h"      // apply_stop_sequences
#include "xllama/inference.h"        // InferenceResult
#include "xllama/inference_params.h" // GenerateParams
#include "xllama/platform.h"         // peak_working_set_mb

#include <chrono>

#ifdef XLLAMA_USE_ORT
    #include "ort_genai_c.h"
#endif

namespace xllama {
namespace detail {

// Run the ORT GenAI decode loop. Fills InferenceResult fields:
// n_p_eval, t_p_eval_ms, n_eval, t_eval_ms, ended_with_stop, peak_ws_mb, success.
// Does NOT write log_output or GPU mem info — those are caller-specific.
inline void run_decode_loop_ort(OgaGenerator* gen, OgaTokenizerStream* stream,
                                const GenerateParams& gp, InferenceResult& res,
                                std::chrono::steady_clock::time_point t_prefill_start,
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
                    gp.on_token(std::string_view(piece));
            }
        }
        ++n_generated;

        // Stop sequences checked after each full iteration (same as llama.cpp
        // decode_loop.h: the stop-triggering token IS counted).
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

} // namespace detail
} // namespace xllama