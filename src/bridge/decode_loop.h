// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// Internal: the llama.cpp prefill and generation loops, written in exactly one
// place. Needs llama.h, so it stays out of the public include/ tree.
#pragma once

#include "llama.h"
#include "xllama/chat_prompt.h" // apply_stop_sequences
#include "xllama/platform.h"    // log_output

#include <atomic>
#include <cstdio>
#include <functional>
#include <string>
#include <string_view>
#include <vector>

namespace xllama {

// Why this is shared rather than copied: run_inference (CLI, bench) and
// LlamaSession (chat UI, LAN API) ran two hand-maintained copies of the same
// prefill and generation loops, and the copies had already drifted.
//
//   * #193 — a prompt past the logical batch aborted the process, because
//     llama_decode ASSERTS on an oversized batch instead of returning an error.
//     The fix had to be written twice; inference.cpp still carried the comment
//     "same fix as LlamaSession::generate" as evidence.
//   * The stop-sequence token count diverged silently: run_inference counted the
//     token that triggered the stop, LlamaSession did not, so n_eval — and the
//     published decode_tok_s derived from it — differed by one between the two
//     paths for the same generation.
//
// The project had already drawn this conclusion for sampling (#125/#141,
// sampler_chain.h) and stop sequences (chat_prompt.h). This is the piece that
// was left out.

// Feed |n_tokens| tokens through the context, chunked at the model's logical
// batch. Returns false if a chunk fails to decode; the caller owns what that
// means for its cache, since the two callers answer that differently.
//
// The chunking is NOT optional: llama_decode does not return an error for a
// batch larger than n_batch, it trips GGML_ASSERT(n_tokens_all <= n_batch) and
// aborts — in Release too. n_batch defaults to min(n_ctx, 2048) while a 4096-token
// coding session's trimmer ceiling is 3846, so a long paste used to kill the
// process. This is the LOGICAL batch only; the physical ubatch stays at the #172
// optimum of 512, which is what every published prefill rate was measured on.
inline bool prefill_chunked(llama_context* ctx, const llama_token* tokens, int n_tokens) {
    const int n_batch = std::max(1, static_cast<int>(llama_n_batch(ctx)));
    for (int off = 0; off < n_tokens; off += n_batch) {
        llama_batch batch = llama_batch_get_one(const_cast<llama_token*>(tokens) + off,
                                                std::min(n_batch, n_tokens - off));
        if (llama_decode(ctx, batch) != 0)
            return false;
    }
    return true;
}

// The message for a prompt that cannot fit the context at all. Chunking makes an
// oversized BATCH safe, not an oversized CONTEXT — without this the run would
// fail somewhere inside the loop with a bare "decode failed".
inline std::string prompt_too_long_message(int n_tokens, int n_ctx) {
    return "prompt too long: " + std::to_string(n_tokens) +
           " tokens exceed n_ctx=" + std::to_string(n_ctx);
}

struct DecodeLoopParams {
    llama_context* ctx = nullptr;
    llama_sampler* sampler = nullptr;
    const llama_vocab* vocab = nullptr;
    int n_predict = 0;
    const std::vector<std::string>* stop_sequences = nullptr;
    const std::atomic<bool>* abort_flag = nullptr;
    // Streamed piece-by-piece to the caller (chat UI token stream, CLI echo).
    std::function<void(std::string_view)> on_token;
    // CLI only: mirror the stream to stdout as it decodes.
    bool echo_stdout = false;
    // Called after a token has been accepted into the cache. LlamaSession uses it
    // to keep m_kv_tokens in step with the KV cells, which is what the #170a
    // prefix diff and the #170b snapshot fingerprint both read.
    std::function<void(llama_token)> on_accepted;
};

struct DecodeLoopResult {
    int n_generated = 0;
    bool ended_with_stop = false; // a textual stop sequence matched
};

// Generate up to |n_predict| tokens, appending decoded text to |output_text|.
//
// n_generated counts every token the model produced, INCLUDING the one that
// triggered a stop sequence. That token was sampled and rendered — the cost is
// real and t_eval contains it — so leaving it out understates decode_tok_s.
// run_inference already counted it and LlamaSession did not; this unifies on the
// counting version, which is why LlamaSession's figure moves by one token on a
// stop-sequence finish.
inline DecodeLoopResult decode_loop(const DecodeLoopParams& p, std::string& output_text) {
    DecodeLoopResult out;
    while (out.n_generated < p.n_predict) {
        if (p.abort_flag && p.abort_flag->load())
            break;

        llama_token token = llama_sampler_sample(p.sampler, p.ctx, -1);
        if (llama_vocab_is_eog(p.vocab, token)) {
            log_output("[xllama] EOG after " + std::to_string(out.n_generated) + " tokens\n");
            break;
        }

        char buf[256] = {};
        const int len = llama_token_to_piece(p.vocab, token, buf, sizeof(buf) - 1, 0, false);
        if (len > 0) {
            buf[len] = '\0';
            output_text += buf;
            if (p.on_token)
                p.on_token(std::string_view(buf, static_cast<size_t>(len)));
            if (p.echo_stdout) {
                std::fputs(buf, stdout);
                std::fflush(stdout);
            }
        }

        // Stop strings (e.g. Gemma's <end_of_turn>, not an EOG token in every
        // GGUF): the shared suffix matcher trims the trailing match in place.
        if (p.stop_sequences && apply_stop_sequences(output_text, *p.stop_sequences)) {
            out.ended_with_stop = true;
            ++out.n_generated;
            log_output("[xllama] stop sequence after " + std::to_string(out.n_generated) +
                       " tokens\n");
            break;
        }

        llama_batch next = llama_batch_get_one(&token, 1);
        if (llama_decode(p.ctx, next) != 0) {
            log_output("[xllama] decode failed at token, stopping generation\n");
            break;
        }
        if (p.on_accepted)
            p.on_accepted(token);
        ++out.n_generated;
    }
    return out;
}

} // namespace xllama
