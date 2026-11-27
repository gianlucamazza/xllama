// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// Internal: the llama.cpp prefill and generation loops, written in exactly one
// place. Needs llama.h, so it stays out of the public include/ tree.
#pragma once

#include "llama.h"
#include "xllama/chat_prompt.h" // apply_stop_sequences
#include "xllama/inference_params.h"
#include "xllama/inference_progress.h" // prefill/decode progress + stall watchdog
#include "xllama/platform.h"           // log_output
#include "xllama/speculative.h"        // prompt_lookup_draft (#210)

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <functional>
#include <string>
#include <string_view>
#include <unordered_set>
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

// Abort inside a prefill batch (#580). While prefill_chunked runs, llama.cpp's
// abort callback returns wd->aborted(): the caller's abort_flag (Stop, API
// disconnect) or a fired stall detector. ggml-cpu polls it between graph nodes,
// and llama_decode returns 2 between ubatches. Before, an abort waited for the
// end of an n_batch chunk (up to 2048 tokens, ~90 s for a 3B model on Series S),
// and a stuck llama_decode could not be ended at all. The ubatches already
// processed stay in the KV, as after any failed chunk, so the caller drops its
// resident record (Session's fail_prefill does). The D3D12 backend has no abort
// hook: offloaded layers stop at the next ubatch. Decode steps are not covered;
// one token is short, and decode_loop checks the flag before each one.
class ScopedPrefillAbort {
  public:
    ScopedPrefillAbort(llama_context* ctx, InferenceWatchdog* wd) : ctx_(wd ? ctx : nullptr) {
        if (ctx_)
            llama_set_abort_callback(
                ctx_, [](void* data) { return static_cast<InferenceWatchdog*>(data)->aborted(); },
                wd);
    }
    ~ScopedPrefillAbort() {
        if (ctx_)
            llama_set_abort_callback(ctx_, nullptr, nullptr);
    }
    ScopedPrefillAbort(const ScopedPrefillAbort&) = delete;
    ScopedPrefillAbort& operator=(const ScopedPrefillAbort&) = delete;

  private:
    llama_context* ctx_;
};

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
// |done_base| is how many prompt tokens of this prefill were already decoded by
// an earlier call (hybrid checkpoint split, #601), so the watchdog's prompt progress
// stays absolute and monotonic across calls.
inline bool prefill_chunked(llama_context* ctx, const llama_token* tokens, int n_tokens,
                            InferenceWatchdog* wd = nullptr, int done_base = 0,
                            int* decode_rc = nullptr) {
    const int n_batch = std::max(1, static_cast<int>(llama_n_batch(ctx)));
    const ScopedPrefillAbort abort_scope(ctx, wd);
    for (int off = 0; off < n_tokens; off += n_batch) {
        if (wd && wd->aborted())
            return false;
        const int chunk = std::min(n_batch, n_tokens - off);
        llama_batch batch = llama_batch_get_one(const_cast<llama_token*>(tokens) + off, chunk);
        if (const int rc = llama_decode(ctx, batch)) {
            if (decode_rc)
                *decode_rc = rc;
            return false;
        }
        if (wd)
            wd->note_prefill_chunk(done_base + off + chunk);
    }
    return true;
}

struct DecodeLoopParams {
    llama_context* ctx = nullptr;
    llama_sampler* sampler = nullptr;
    const llama_vocab* vocab = nullptr;
    int n_predict = 0;
    const std::vector<std::string>* stop_sequences = nullptr;
    // Special tokens emitted as text (GenerateParams::preserved_tokens); null =
    // every special token is dropped from the output.
    const std::unordered_set<llama_token>* preserved_tokens = nullptr;
    // A tool template's grammar (ADR 0005, sampler_chain.h); null = none. It
    // turns speculative drafting off: drafts are accepted into samplers before
    // a rejected tail is rolled back, and a grammar cannot rewind.
    llama_sampler* grammar = nullptr;
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

    // Bench only: keep decoding through end-of-generation so every run decodes
    // exactly n_predict tokens. Two backends whose arithmetic differs (CPU q8
    // activations vs d3d12 f32) reach EOG at different points, and a decode
    // tok/s over a handful of tokens is noise (D2b). Default off.
    bool ignore_eog = false;

    // Optional stall watchdog / decode-progress logger. Shared with prefill.
    InferenceWatchdog* watchdog = nullptr;
};

struct DecodeLoopResult {
    int n_generated = 0;
    double first_token_ms = 0.0;
    DecodeTermination termination = DecodeTermination::TokenLimit;
    int decode_rc = 0;
    int n_drafted = 0;
    int n_accepted = 0;

    void fail_decode(int rc) {
        decode_rc = rc;
        switch (rc) {
        case 1:
            termination = DecodeTermination::NoKvSlot;
            break;
        case 2:
            termination = DecodeTermination::BackendAbort;
            break;
        case -1:
            termination = DecodeTermination::InvalidBatch;
            break;
        default:
            termination = DecodeTermination::BackendFailure;
            break;
        }
    }
};

// Both GGUF callers publish the same partial counts and failure. Only Session
// retains a context and must invalidate it before another request can use it.
inline void finish_decode(const DecodeLoopResult& loop, InferenceResult& res,
                          const InferenceWatchdog& wd) {
    res.termination = loop.termination;
    res.decode_rc = loop.decode_rc;
    res.n_eval = loop.n_generated;
    res.n_drafted = loop.n_drafted;
    res.n_spec_accepted = loop.n_accepted;
    res.ended_with_stop = loop.termination == DecodeTermination::StopSequence;
    if (wd.stalled()) {
        res.termination = DecodeTermination::Stalled;
        res.stalled = true;
        res.error_msg = wd.stall_error();
    } else {
        switch (loop.termination) {
        case DecodeTermination::NoKvSlot:
            res.error_msg = "no KV slot for batch";
            break;
        case DecodeTermination::BackendAbort:
            res.error_msg = "backend aborted batch";
            break;
        case DecodeTermination::InvalidBatch:
            res.error_msg = "invalid input batch";
            break;
        case DecodeTermination::BackendFailure:
            res.error_msg = "backend failure";
            break;
        case DecodeTermination::RewindFailure:
            res.error_msg =
                "speculative KV rewind unsupported (disable prompt lookup for this model)";
            break;
        default:
            break;
        }
    }
    if (loop.decode_rc != 0)
        res.error_msg += " (llama_decode rc=" + std::to_string(loop.decode_rc) + ")";
    res.success = res.error_msg.empty();
}

inline void fail_prefill(InferenceResult& res, int rc, const InferenceWatchdog& wd) {
    DecodeLoopResult loop;
    if (rc) {
        loop.fail_decode(rc);
        finish_decode(loop, res, wd);
    } else if (wd.stalled()) {
        res.termination = DecodeTermination::Stalled;
        res.stalled = true;
        res.error_msg = wd.stall_error();
    } else if (wd.aborted()) {
        res.termination = DecodeTermination::Cancelled;
        res.error_msg = "aborted during prefill";
    } else {
        res.error_msg = "prompt decode failed";
    }
}

namespace detail {

// Emit a sampled token into the output stream. Returns true if a stop sequence
// ended generation (caller should still count the token).
inline bool emit_token(const DecodeLoopParams& p, llama_token token, std::string& output_text) {
    char buf[256] = {};
    const bool special = p.preserved_tokens && p.preserved_tokens->count(token) > 0;
    const int len = llama_token_to_piece(p.vocab, token, buf, sizeof(buf) - 1, 0, special);
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

// One token through the chain, checked against the grammar the way llama-server
// samples (common/sampling.cpp): the chain picks, the grammar checks only that
// token, and only a rejection pays a grammar pass over the whole vocabulary.
// llama_sampler_sample would accept into the chain on its own, so a resample
// would accept twice; here both samplers accept the final token once.
inline llama_token sample_with_grammar(const DecodeLoopParams& p) {
    const float* logits = llama_get_logits_ith(p.ctx, -1);
    const int n_vocab = llama_vocab_n_tokens(p.vocab);
    std::vector<llama_token_data> cur(static_cast<size_t>(n_vocab));
    auto fill = [&] {
        for (llama_token i = 0; i < n_vocab; ++i)
            cur[static_cast<size_t>(i)] = llama_token_data{i, logits[i], 0.0f};
        return llama_token_data_array{cur.data(), cur.size(), -1, false};
    };
    llama_token_data_array all = fill();
    llama_sampler_apply(p.sampler, &all);
    llama_token id = all.data[all.selected].id;

    llama_token_data one_data{id, 1.0f, 0.0f};
    llama_token_data_array one{&one_data, 1, -1, false};
    llama_sampler_apply(p.grammar, &one);
    if (one.data[0].logit == -INFINITY) {
        all = fill();
        llama_sampler_apply(p.grammar, &all);
        llama_sampler_apply(p.sampler, &all);
        id = all.data[all.selected].id;
    }
    llama_sampler_accept(p.grammar, id);
    llama_sampler_accept(p.sampler, id);
    return id;
}

// Record an accepted token in history / session bookkeeping.
inline void accept_token(const DecodeLoopParams& p, llama_token token) {
    if (p.on_accepted)
        p.on_accepted(token);
    else if (p.token_history)
        p.token_history->push_back(token);
}

// The one end-of-generation decision for every branch of decode_loop (classic,
// pre-draft, draft commit, spec reject), so ignore_eog cannot drift between them.
inline bool stops_at_eog(const DecodeLoopParams& p, llama_token token) {
    return llama_vocab_is_eog(p.vocab, token) && !p.ignore_eog;
}

// Decode one already-sampled token (classic path).
inline int decode_one(llama_context* ctx, llama_token token) {
    llama_batch next = llama_batch_get_one(&token, 1);
    return llama_decode(ctx, next);
}

// Multi-token verify batch: lead sample + draft, all positions request logits
// so the same sampler chain can sample at each batch index (#210 W2.2).
// Positions are explicit from the current seq_pos_max so auto-tracking cannot
// disagree with seq_rm on a partial accept.
inline int decode_verify_batch(llama_context* ctx, const std::vector<llama_token>& feed) {
    if (feed.empty())
        return 0;
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
    return rc;
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
inline bool classic_step(const DecodeLoopParams& p, llama_token token, std::string& output_text,
                         DecodeLoopResult& out) {
    ++out.n_generated; // exposed to the caller before the KV write
    const bool stopped = emit_token(p, token, output_text);
    if (const int rc = decode_one(p.ctx, token)) {
        out.fail_decode(rc);
        return false;
    }
    accept_token(p, token);
    if (stopped) {
        out.termination = DecodeTermination::StopSequence;
        log_output("[xllama] stop sequence after " + std::to_string(out.n_generated) + " tokens\n");
        return false;
    }
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
// use llama_memory_seq_rm; a refused rewind ends generation (W2.3).
//
// Correctness note (2026-08-07): feeding [lead, draft...] in one batch and
// sampling index 0 for the first draft diverged from sequential greedy even
// with n_spec_accepted=0 — batch logits after lead were not equivalent to a
// single-token decode of lead on this stack. Commit lead first, then speculate.
inline DecodeLoopResult decode_loop(const DecodeLoopParams& p, std::string& output_text) {
    DecodeLoopResult out;
    const bool spec_enabled =
        p.prompt_lookup && p.token_history != nullptr && p.spec_k > 0 && p.grammar == nullptr;

    auto interrupted = [&]() {
        if ((p.abort_flag && p.abort_flag->load()) || (p.watchdog && p.watchdog->aborted())) {
            out.termination = DecodeTermination::Cancelled;
            return true;
        }
        return false;
    };

    while (out.n_generated < p.n_predict && !interrupted()) {
        llama_token token =
            p.grammar ? detail::sample_with_grammar(p) : llama_sampler_sample(p.sampler, p.ctx, -1);
        if (detail::stops_at_eog(p, token)) {
            out.termination = DecodeTermination::EndOfGeneration;
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
            if (!detail::classic_step(p, token, output_text, out))
                break;
        }
        if (p.watchdog)
            p.watchdog->on_decode_token(out.n_generated);
        if (out.n_generated >= p.n_predict)
            break;

        if (!spec_enabled)
            continue;
        if (interrupted())
            break;

        // Lead is already in token_history via accept_token. Draft from that
        // alone (no extra lead argument).
        std::vector<int32_t> draft32;
        if (p.token_history && static_cast<int>(p.token_history->size()) >= p.spec_n_gram) {
            std::vector<int32_t> hist;
            hist.reserve(p.token_history->size());
            for (llama_token t : *p.token_history)
                hist.push_back(static_cast<int32_t>(t));
            const int room =
                std::min(p.n_predict - out.n_generated, static_cast<int>(llama_n_batch(p.ctx)));
            draft32 = prompt_lookup_draft(hist, p.spec_n_gram, std::min(p.spec_k, room));
        }
        if (draft32.empty())
            continue; // decline: no n-gram evidence, no extra cost

        out.n_drafted += static_cast<int>(draft32.size());

        // Verify draft[0] against the logits we already have after the lead
        // (same sample classic would take next). If it disagrees, that sample
        // *is* the true next token — classic_step it and skip the batch.
        const llama_token first = llama_sampler_sample(p.sampler, p.ctx, -1);
        if (detail::stops_at_eog(p, first)) {
            out.termination = DecodeTermination::EndOfGeneration;
            log_output("[xllama] EOG after " + std::to_string(out.n_generated) +
                       " tokens (pre-draft)\n");
            break;
        }
        if (first != static_cast<llama_token>(draft32[0])) {
            if (!detail::classic_step(p, first, output_text, out))
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

        if (const int rc = detail::decode_verify_batch(p.ctx, feed)) {
            // Abort and fatal errors can leave processed ubatches resident.
            // Do not sample or retry from this context.
            out.fail_decode(rc);
            break;
        }

        auto fail_rewind = [&]() {
            log_output("[xllama] speculative: seq_rm refused — aborting generation "
                       "(hybrid/SWA caches cannot tail-rewind; disable prompt_lookup "
                       "for this model)\n");
            out.termination = DecodeTermination::RewindFailure;
        };

        if (interrupted()) {
            if (!detail::trim_verify_tail(mem, pos_before, 0, n_feed))
                fail_rewind();
            break;
        }

        // Commit draft[0] (already verified equal to |first|).
        int n_keep = 0;
        auto commit_draft = [&](llama_token tok) -> bool {
            // returns false → stop outer loop
            if (detail::stops_at_eog(p, tok)) {
                out.termination = DecodeTermination::EndOfGeneration;
                log_output("[xllama] EOG after " + std::to_string(out.n_generated) +
                           " tokens (spec)\n");
                return false;
            }
            const bool stopped = detail::emit_token(p, tok, output_text);
            detail::accept_token(p, tok); // verified cells already contain this token
            ++n_keep;
            ++out.n_generated;
            if (stopped) {
                out.termination = DecodeTermination::StopSequence;
                return false;
            }
            return true;
        };

        ++out.n_accepted;
        if (!commit_draft(first)) {
            if (!detail::trim_verify_tail(mem, pos_before, n_keep, n_feed))
                fail_rewind();
            break;
        }

        bool stop_all = false;
        for (size_t i = 1; i < feed.size(); ++i) {
            if (interrupted()) {
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
            if (detail::stops_at_eog(p, cand)) {
                out.termination = DecodeTermination::EndOfGeneration;
                log_output("[xllama] EOG after " + std::to_string(out.n_generated) +
                           " tokens (spec reject)\n");
                stop_all = true;
                break;
            }
            stop_all = !detail::classic_step(p, cand, output_text, out);
            break;
        }

        if (stop_all)
            break;
        if (p.watchdog)
            p.watchdog->on_decode_token(out.n_generated);
        // All drafts accepted: logits after the last draft are ready for -1.
    }
    return out;
}

} // namespace xllama
