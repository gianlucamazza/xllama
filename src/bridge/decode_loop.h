// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// Internal: the llama.cpp prefill and generation loops, written in exactly one
// place. Needs llama.h, so it stays out of the public include/ tree.
#pragma once

#include "llama.h"
#include "xllama/chat_prompt.h" // apply_stop_sequences
#include "xllama/platform.h"    // log_output
#include "xllama/speculative.h" // prompt_lookup_draft (#210)

#include <atomic>
#include <chrono>
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
//
// Phase 15 W2 (#210) adds optional draft-free prompt-lookup inside this same
// loop so CLI and Session cannot drift on speculation either.

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

    // Start of the decode phase, used to expose time-to-first-token.
    std::chrono::steady_clock::time_point decode_start{};

    // Phase 15 W2 (#210): draft-free prompt-lookup speculative decoding.
    // Default OFF. Requires token_history seeded with the prefill tokens.
    // History ownership: when on_accepted is set it must update the same
    // vector (Session: on_accepted pushes m_kv_tokens); when null the loop
    // appends to token_history itself (CLI path).
    bool prompt_lookup = false;
    std::vector<llama_token>* token_history = nullptr;
    int spec_n_gram = kSpecNgramDefault;
    int spec_k = kSpecDraftKDefault;
};

struct DecodeLoopResult {
    int n_generated = 0;
    double first_token_ms = 0.0;
    bool ended_with_stop = false; // a textual stop sequence matched
    // Speculative counters (zero when prompt_lookup is off or never drafted).
    int n_drafted = 0;  // draft tokens proposed (not counting the lead sample)
    int n_accepted = 0; // draft tokens that matched the target sample
    // True when a needed llama_memory_seq_rm refused after a multi-token verify
    // batch. Callers must treat the generation as failed and drop the KV —
    // continuing would desync history from cells (hybrid/LFM caches).
    bool rewind_failed = false;
};

namespace detail {

// Emit a sampled token into the output stream. Returns true if a stop sequence
// ended generation (caller should still count the token).
inline bool emit_token(const DecodeLoopParams& p, llama_token token, std::string& output_text) {
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
    return p.stop_sequences && apply_stop_sequences(output_text, *p.stop_sequences);
}

// Record an accepted token in history / session bookkeeping.
inline void accept_token(const DecodeLoopParams& p, llama_token token) {
    if (p.on_accepted)
        p.on_accepted(token);
    else if (p.token_history)
        p.token_history->push_back(token);
}

// Decode one already-sampled token (classic path).
inline bool decode_one(llama_context* ctx, llama_token token) {
    llama_batch next = llama_batch_get_one(&token, 1);
    return llama_decode(ctx, next) == 0;
}

// Multi-token verify batch: lead sample + draft, all positions request logits
// so the same sampler chain can sample at each batch index (#210 W2.2).
// Positions are explicit from the current seq_pos_max so auto-tracking cannot
// disagree with seq_rm on a partial accept.
inline bool decode_verify_batch(llama_context* ctx, const std::vector<llama_token>& feed) {
    if (feed.empty())
        return true;
    llama_batch batch = llama_batch_init(static_cast<int32_t>(feed.size()), 0, 1);
    llama_memory_t mem = llama_get_memory(ctx);
    const llama_pos pos0 = llama_memory_seq_pos_max(mem, 0) + 1;
    for (size_t i = 0; i < feed.size(); ++i) {
        batch.token[i] = feed[i];
        batch.pos[i] = pos0 + static_cast<llama_pos>(i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = 1; // need a logit row per position for verification
    }
    batch.n_tokens = static_cast<int32_t>(feed.size());
    const int rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    return rc == 0;
}

// Drop KV cells past the first |n_keep| tokens of a verify batch that began
// after |pos_before|. Returns false if the memory refuses (no mutation on
// hybrid caches — #183 / #210 W2.3).
inline bool trim_verify_tail(llama_memory_t mem, llama_pos pos_before, int n_keep, int n_feed) {
    if (n_keep >= n_feed)
        return true;
    const llama_pos keep = pos_before + static_cast<llama_pos>(n_keep) + 1;
    return llama_memory_seq_rm(mem, 0, keep, -1);
}

// History for n-gram search = prefill/prior accepted tokens + the lead sample
// not yet recorded.
inline std::vector<int32_t> history_for_draft(const DecodeLoopParams& p, llama_token lead) {
    std::vector<int32_t> h;
    if (p.token_history) {
        h.reserve(p.token_history->size() + 1);
        for (llama_token t : *p.token_history)
            h.push_back(static_cast<int32_t>(t));
    }
    h.push_back(static_cast<int32_t>(lead));
    return h;
}

// Classic single-token step. Returns false to stop the outer loop.
// Sets |stop| when a stop sequence matched; |decode_ok| false on decode error.
inline bool classic_step(const DecodeLoopParams& p, llama_token token, std::string& output_text,
                         DecodeLoopResult& out, bool& stop, bool& decode_ok) {
    stop = false;
    decode_ok = true;
    if (emit_token(p, token, output_text)) {
        out.ended_with_stop = true;
        ++out.n_generated;
        stop = true;
        log_output("[xllama] stop sequence after " + std::to_string(out.n_generated) + " tokens\n");
        return false;
    }
    if (!decode_one(p.ctx, token)) {
        log_output("[xllama] decode failed at token, stopping generation\n");
        decode_ok = false;
        return false;
    }
    accept_token(p, token);
    ++out.n_generated;
    return true;
}

} // namespace detail

// Generate up to |n_predict| tokens, appending decoded text to |output_text|.
//
// n_generated counts every token the model produced, INCLUDING the one that
// triggered a stop sequence. That token was sampled and rendered — the cost is
// real and t_eval contains it — so leaving it out understates decode_tok_s.
// run_inference already counted it and LlamaSession did not; this unifies on the
// counting version, which is why LlamaSession's figure moves by one token on a
// stop-sequence finish.
//
// When prompt_lookup is on, the lead token is always committed with the same
// classic path as non-spec (so greedy output matches). Drafts are then verified
// with a multi-token batch of *only* the draft tokens (W2.2). Rejected tails
// use llama_memory_seq_rm; a refused rewind sets rewind_failed (W2.3).
//
// Correctness note (2026-08-07): feeding [lead, draft...] in one batch and
// sampling index 0 for the first draft diverged from sequential greedy even
// with n_spec_accepted=0 — batch logits after lead were not equivalent to a
// single-token decode of lead on this stack. Commit lead first, then speculate.
inline DecodeLoopResult decode_loop(const DecodeLoopParams& p, std::string& output_text) {
    DecodeLoopResult out;
    bool spec_enabled = p.prompt_lookup && p.token_history != nullptr && p.spec_k > 0;

    while (out.n_generated < p.n_predict) {
        if (p.abort_flag && p.abort_flag->load())
            break;

        llama_token token = llama_sampler_sample(p.sampler, p.ctx, -1);
        if (llama_vocab_is_eog(p.vocab, token)) {
            log_output("[xllama] EOG after " + std::to_string(out.n_generated) + " tokens\n");
            break;
        }

        // Always commit the sampled token via the classic path first. Spec never
        // changes this step — that is what keeps greedy text identical.
        {
            if (out.n_generated == 0 && out.first_token_ms == 0.0 &&
                p.decode_start != std::chrono::steady_clock::time_point{}) {
                out.first_token_ms = std::chrono::duration<double, std::milli>(
                                         std::chrono::steady_clock::now() - p.decode_start)
                                         .count();
            }
            bool stop = false, decode_ok = true;
            if (!detail::classic_step(p, token, output_text, out, stop, decode_ok))
                break;
        }
        if (out.n_generated >= p.n_predict)
            break;

        if (!spec_enabled)
            continue;

        // Lead is already in token_history via accept_token. Draft from that
        // alone (no extra lead argument).
        std::vector<int32_t> draft32;
        if (p.token_history && static_cast<int>(p.token_history->size()) >= p.spec_n_gram) {
            std::vector<int32_t> hist;
            hist.reserve(p.token_history->size());
            for (llama_token t : *p.token_history)
                hist.push_back(static_cast<int32_t>(t));
            draft32 = prompt_lookup_draft(hist, p.spec_n_gram, p.spec_k);
        }
        if (draft32.empty())
            continue; // decline: no n-gram evidence, no extra cost

        out.n_drafted += static_cast<int>(draft32.size());

        // Verify draft[0] against the logits we already have after the lead
        // (same sample classic would take next). If it disagrees, that sample
        // *is* the true next token — classic_step it and skip the batch.
        const llama_token first = llama_sampler_sample(p.sampler, p.ctx, -1);
        if (llama_vocab_is_eog(p.vocab, first)) {
            log_output("[xllama] EOG after " + std::to_string(out.n_generated) +
                       " tokens (pre-draft)\n");
            break;
        }
        if (first != static_cast<llama_token>(draft32[0])) {
            bool stop = false, decode_ok = true;
            if (!detail::classic_step(p, first, output_text, out, stop, decode_ok))
                break;
            continue;
        }

        // first == draft[0]: at least one draft token is free. Batch-decode all
        // drafts; logits[i] predict the token after draft[i].
        std::vector<llama_token> feed;
        feed.reserve(draft32.size());
        for (int32_t d : draft32)
            feed.push_back(static_cast<llama_token>(d));
        const int n_feed = static_cast<int>(feed.size());

        llama_memory_t mem = llama_get_memory(p.ctx);
        const llama_pos pos_before = llama_memory_seq_pos_max(mem, 0);

        if (!detail::decode_verify_batch(p.ctx, feed)) {
            log_output("[xllama] speculative draft batch failed — classic for first match\n");
            if (!detail::trim_verify_tail(mem, pos_before, /*n_keep=*/0, n_feed)) {
                out.rewind_failed = true;
                break;
            }
            // first already sampled and matches draft[0]; commit it classically.
            bool stop = false, decode_ok = true;
            if (!detail::classic_step(p, first, output_text, out, stop, decode_ok))
                break;
            continue;
        }

        auto fail_rewind = [&]() {
            log_output("[xllama] speculative: seq_rm refused — aborting generation "
                       "(hybrid/SWA caches cannot tail-rewind; disable prompt_lookup "
                       "for this model)\n");
            out.rewind_failed = true;
            spec_enabled = false;
        };

        // Commit draft[0] (already verified equal to |first|).
        int n_keep = 0;
        auto commit_draft = [&](llama_token tok) -> bool {
            // returns false → stop outer loop
            if (llama_vocab_is_eog(p.vocab, tok)) {
                log_output("[xllama] EOG after " + std::to_string(out.n_generated) +
                           " tokens (spec)\n");
                return false;
            }
            if (detail::emit_token(p, tok, output_text)) {
                out.ended_with_stop = true;
                ++out.n_generated;
                return false;
            }
            detail::accept_token(p, tok);
            ++out.n_generated;
            return true;
        };

        ++out.n_accepted;
        ++n_keep;
        if (!commit_draft(first)) {
            if (!detail::trim_verify_tail(mem, pos_before, n_keep, n_feed))
                fail_rewind();
            break;
        }

        bool stop_all = false;
        for (size_t i = 1; i < feed.size(); ++i) {
            if (out.n_generated >= p.n_predict || (p.abort_flag && p.abort_flag->load())) {
                if (!detail::trim_verify_tail(mem, pos_before, n_keep, n_feed))
                    fail_rewind();
                stop_all = true;
                break;
            }
            // Logits after feed[i-1] (batch index i-1) predict feed[i].
            const llama_token cand =
                llama_sampler_sample(p.sampler, p.ctx, static_cast<int32_t>(i - 1));
            const llama_token drafted = feed[i];
            if (cand == drafted) {
                ++out.n_accepted;
                ++n_keep;
                if (!commit_draft(cand)) {
                    if (!detail::trim_verify_tail(mem, pos_before, n_keep, n_feed))
                        fail_rewind();
                    stop_all = true;
                    break;
                }
                continue;
            }
            // Reject: keep accepted drafts, drop the rest, commit |cand|.
            if (!detail::trim_verify_tail(mem, pos_before, n_keep, n_feed)) {
                fail_rewind();
                stop_all = true;
                break;
            }
            if (llama_vocab_is_eog(p.vocab, cand)) {
                log_output("[xllama] EOG after " + std::to_string(out.n_generated) +
                           " tokens (spec reject)\n");
                stop_all = true;
                break;
            }
            if (detail::emit_token(p, cand, output_text)) {
                out.ended_with_stop = true;
                ++out.n_generated;
                stop_all = true;
                break;
            }
            // cand is not yet in the KV (only accepted drafts are).
            if (!detail::decode_one(p.ctx, cand)) {
                log_output("[xllama] decode failed after speculative reject\n");
                stop_all = true;
                break;
            }
            detail::accept_token(p, cand);
            ++out.n_generated;
            break;
        }

        if (out.rewind_failed)
            break;
        if (stop_all)
            break;
        if (n_keep < n_feed) {
            if (!detail::trim_verify_tail(mem, pos_before, n_keep, n_feed)) {
                fail_rewind();
                break;
            }
        }
        // All drafts accepted: logits after the last draft are ready for -1.
    }
    return out;
}

} // namespace xllama
