// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT
// Exercise the production loop and public entry points. Linker wrapping keeps
// the scripted backend and the decode fault entirely outside the product.
#include "../src/bridge/decode_loop.h"
#include "xllama/inference.h"
#include "xllama/session.h"

#include <cstdlib>
#include <doctest/doctest.h>

using namespace xllama;

namespace {
struct Script {
    std::vector<llama_token> samples = {'A', 'B', 'X'};
    std::vector<llama_token> cells = {'A', 'B', 'C'};
    size_t sampled = 0;
    int calls = 0;
    int fail_call = 0;
    int rc = -3;
    bool rewind = true;
    bool pending = false;
};
Script script;
bool real_model = false;
} // namespace

extern "C" {
int32_t __real_llama_decode(llama_context*, llama_batch);
int32_t __wrap_llama_decode(llama_context* ctx, llama_batch batch) {
    if (real_model)
        script.pending = true;
    if (++script.calls == script.fail_call) {
        // llama.cpp can retain processed ubatches on abort/fatal. Exercise
        // callers against an already mutated context on those returns.
        if (script.rc == 2 || script.rc < -1) {
            if (real_model)
                REQUIRE(__real_llama_decode(ctx, batch) == 0);
            else
                script.cells.push_back(batch.token[0]);
        }
        return script.rc;
    }
    if (real_model)
        return __real_llama_decode(ctx, batch);
    script.cells.insert(script.cells.end(), batch.token, batch.token + batch.n_tokens);
    return 0;
}
void __real_llama_synchronize(llama_context*);
void __wrap_llama_synchronize(llama_context* ctx) {
    script.pending = false;
    __real_llama_synchronize(ctx);
}
void __real_llama_memory_clear(llama_memory_t, bool);
void __wrap_llama_memory_clear(llama_memory_t mem, bool data) {
    if (real_model && script.calls == script.fail_call && script.calls > 0)
        CHECK_FALSE(script.pending);
    __real_llama_memory_clear(mem, data);
}
void __real_llama_set_abort_callback(llama_context*, ggml_abort_callback, void*);
void __wrap_llama_set_abort_callback(llama_context* ctx, ggml_abort_callback cb, void* data) {
    if (real_model && !cb)
        CHECK_FALSE(script.pending);
    __real_llama_set_abort_callback(ctx, cb, data);
}
llama_token __wrap_llama_sampler_sample(llama_sampler*, llama_context* ctx, int32_t) {
    if (real_model) {
        const auto* vocab = llama_model_get_vocab(llama_get_model(ctx));
        for (llama_token t = 0; t < llama_vocab_n_tokens(vocab); ++t) {
            if ((llama_vocab_get_attr(vocab, t) & LLAMA_TOKEN_ATTR_NORMAL) &&
                !llama_vocab_is_eog(vocab, t))
                return t;
        }
        FAIL("model has no ordinary token");
    }
    return script.samples.at(script.sampled++);
}
bool __real_llama_vocab_is_eog(const llama_vocab*, llama_token);
bool __wrap_llama_vocab_is_eog(const llama_vocab* vocab, llama_token token) {
    return real_model ? __real_llama_vocab_is_eog(vocab, token) : token == 0;
}
int32_t __real_llama_token_to_piece(const llama_vocab*, llama_token, char*, int32_t, int32_t, bool);
int32_t __wrap_llama_token_to_piece(const llama_vocab* vocab, llama_token token, char* buf,
                                    int32_t size, int32_t strip, bool special) {
    if (real_model)
        return __real_llama_token_to_piece(vocab, token, buf, size, strip, special);
    buf[0] = static_cast<char>(token);
    return 1;
}
uint32_t __real_llama_n_batch(const llama_context*);
uint32_t __wrap_llama_n_batch(const llama_context* ctx) {
    return real_model ? __real_llama_n_batch(ctx) : 8;
}
llama_memory_t __real_llama_get_memory(const llama_context*);
llama_memory_t __wrap_llama_get_memory(const llama_context* ctx) {
    return real_model ? __real_llama_get_memory(ctx) : nullptr;
}
llama_pos __real_llama_memory_seq_pos_max(llama_memory_t, llama_seq_id);
llama_pos __wrap_llama_memory_seq_pos_max(llama_memory_t mem, llama_seq_id seq) {
    return real_model ? __real_llama_memory_seq_pos_max(mem, seq)
                      : static_cast<llama_pos>(script.cells.size()) - 1;
}
bool __real_llama_memory_seq_rm(llama_memory_t, llama_seq_id, llama_pos, llama_pos);
bool __wrap_llama_memory_seq_rm(llama_memory_t mem, llama_seq_id seq, llama_pos start,
                                llama_pos end) {
    if (real_model)
        return __real_llama_memory_seq_rm(mem, seq, start, end);
    if (script.rewind)
        script.cells.resize(static_cast<size_t>(start));
    return script.rewind;
}
}

TEST_CASE("decode failure reaches the shared result from every llama_decode call site") {
    for (int call : {1, 2, 3}) { // lead, verify batch, rejected draft replacement
        for (auto [rc, reason] : {std::pair{1, DecodeTermination::NoKvSlot},
                                  std::pair{2, DecodeTermination::BackendAbort},
                                  std::pair{-1, DecodeTermination::InvalidBatch},
                                  std::pair{-3, DecodeTermination::BackendFailure}}) {
            CAPTURE(call);
            CAPTURE(rc);
            script = Script{};
            script.fail_call = call;
            script.rc = rc;
            auto history = script.cells;
            DecodeLoopParams p;
            p.n_predict = 3;
            p.prompt_lookup = true;
            p.spec_n_gram = 1;
            p.token_history = &history;
            InferenceResult result;
            std::string streamed;
            p.on_token = [&](std::string_view text) { streamed += text; };
            const auto loop = decode_loop(p, result.output_text);
            InferenceWatchdog wd({0, 0, 0}, nullptr);
            finish_decode(loop, result, wd);
            CHECK_FALSE(result.success);
            CHECK(result.termination == reason);
            CHECK(result.decode_rc == rc);
            CHECK(result.error_msg.find(std::to_string(rc)) != std::string::npos);
            CHECK_FALSE(result.hit_token_limit(1));
            CHECK(script.calls == call);
            CHECK(result.output_text == streamed);
            CHECK(streamed == (call == 3 ? "ABX" : "A"));
            CHECK(result.n_eval == static_cast<int>(streamed.size()));
            if (rc == 2 || rc < -1)
                CHECK(history.size() + 1 == script.cells.size());
            else
                CHECK(history == script.cells);
        }
    }
}

TEST_CASE("decode natural stops and speculative rewinds keep history aligned with cells") {
    script = Script{};
    auto history = script.cells;
    DecodeLoopParams p;
    p.n_predict = 3;
    p.prompt_lookup = true;
    p.spec_n_gram = 1;
    p.token_history = &history;
    std::vector<std::string> stops;
    p.stop_sequences = &stops;
    std::atomic<bool> cancel{false};
    p.abort_flag = &cancel;
    std::string expected_text = "ABX";
    auto reason = DecodeTermination::TokenLimit;
    int expected_count = 3;
    SUBCASE("rejected draft replacement stays resident") {}
    SUBCASE("classic EOG skips decode") {
        script.samples = {0};
        expected_text.clear();
        expected_count = 0;
        reason = DecodeTermination::EndOfGeneration;
    }
    SUBCASE("EOG in rejected draft") {
        script.samples.back() = 0;
        expected_text = "AB";
        expected_count = 2;
        reason = DecodeTermination::EndOfGeneration;
    }
    SUBCASE("stop on matching draft commits stop and removes suffix") {
        stops = {"B"};
        expected_text = "A";
        expected_count = 2;
        reason = DecodeTermination::StopSequence;
    }
    SUBCASE("stop on rejected replacement at cap takes precedence") {
        stops = {"X"};
        expected_text = "AB";
        reason = DecodeTermination::StopSequence;
    }
    SUBCASE("cancel during lead before drafting") {
        p.on_token = [&](std::string_view) { cancel = true; };
        expected_text = "A";
        expected_count = 1;
        reason = DecodeTermination::Cancelled;
    }
    SUBCASE("refused rewind after verify leaves state unusable") {
        script.rewind = false;
        expected_text = "AB";
        expected_count = 2;
        reason = DecodeTermination::RewindFailure;
    }
    InferenceResult result;
    const auto loop = decode_loop(p, result.output_text);
    InferenceWatchdog wd({0, 0, 0}, nullptr);
    finish_decode(loop, result, wd);
    CHECK(result.termination == reason);
    CHECK(result.output_text == expected_text);
    CHECK(result.n_eval == expected_count);
    CHECK(result.hit_token_limit(p.n_predict) == (reason == DecodeTermination::TokenLimit));
    CHECK(result.success == (reason != DecodeTermination::RewindFailure));
    if (result.success) {
        CHECK(history == script.cells);
        CHECK(history.size() == 3 + static_cast<size_t>(expected_count));
    }
}

TEST_CASE("failed generation invalidates Session state and fails run_inference (opt-in model)") {
    const char* path = std::getenv("XLLAMA_TEST_DECODE_FAULT_MODEL");
    if (!path)
        path = std::getenv("XLLAMA_TEST_MODEL");
    if (!path) {
        MESSAGE(
            "Set XLLAMA_TEST_DECODE_FAULT_MODEL to run the model-backed cache invalidation case");
        return;
    }
    struct Backend {
        Backend() {
            real_model = true;
        }
        ~Backend() {
            real_model = false;
        }
    } backend;
    SessionParams sp;
    sp.model_path = path;
    sp.n_threads = 1;
    sp.n_ctx = 128;
    std::string err;
    auto session = Session::create(sp, &err);
    REQUIRE_MESSAGE(session, err);
    GenerateParams gp;
    gp.prompt = "abc";
    gp.n_predict = 2;
    gp.reuse_kv = gp.reset_kv = true;
    script = Script{};
    REQUIRE(session->generate(gp).success);
    gp.reset_kv = false;
    gp.expected_kv_epoch = session->kv_epoch();
    script.fail_call = script.calls + 2; // continuation prefill then one emitted token
    auto failed = session->generate(gp);
    CHECK_FALSE(failed.success);
    CHECK_FALSE(failed.retry_full_prompt);
    CHECK_FALSE(failed.completed_reply(false));
    CHECK(failed.decode_rc == -3);
    CHECK(failed.n_eval == 1);
    CHECK(session->kv_epoch() != gp.expected_kv_epoch);
    CHECK_FALSE(session->save_state("unused.gguf-state", &err));
    const auto stale = session->generate(gp);
    CHECK_FALSE(stale.success);
    CHECK(stale.retry_full_prompt);
    gp.reset_kv = true;
    auto recovered = session->generate(gp);
    REQUIRE(recovered.success);
    CHECK(recovered.n_p_eval == recovered.n_prompt);

    InferenceParams ip;
    ip.prompt = gp.prompt;
    ip.model_path = path;
    ip.n_threads = 1;
    ip.n_ctx = 128;
    ip.n_predict = 2;
    script.fail_call = script.calls + 2;
    failed = run_inference(ip);
    CHECK_FALSE(failed.success);
    CHECK_FALSE(failed.retry_full_prompt);
    CHECK_FALSE(failed.completed_reply(false));
    CHECK(failed.decode_rc == -3);
    CHECK(failed.n_eval == 1);
}
