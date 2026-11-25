// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "xllama/path_utils.h"
#include "xllama/platform.h"
#include "xllama/prefix_snapshot.h"
#include "xllama/session.h"
#include "xllama/session_hub.h"

#include <chrono>
#include <doctest/doctest.h>

#include <climits>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>

TEST_CASE("SessionHub: failed create leaves the hub empty and bumps generation") {
    auto& hub = xllama::session_hub();
    std::lock_guard<std::mutex> lk(hub.mtx);
    const uint64_t g0 = hub.generation;
    xllama::SessionParams sp;
    sp.model_path = "/nonexistent/hub-model.gguf";
    std::string err;
    CHECK(hub.ensure_locked("hub-model", sp, &err) == nullptr);
    CHECK(hub.session == nullptr);
    CHECK(hub.model.empty());
    // The attempted swap must invalidate any holder's generation even though
    // creation failed — its old session pointer is gone either way.
    CHECK(hub.generation == g0 + 1);
    hub.reset_locked(); // no-op when already empty
    CHECK(hub.generation == g0 + 1);
}

TEST_CASE("resolve_max_length: the #130 ladder has one home") {
    using xllama::resolve_max_length;
    // override < 0: saturate to n_ctx — what Session always requests.
    CHECK(resolve_max_length(2048, 1289, 256, -1) == 2048);
    CHECK(resolve_max_length(3072, 10, 96, -1) == 3072);
    // override == 0: derive min(n_ctx, prompt + n_predict) — bench default,
    // keeps every historical row's meaning.
    CHECK(resolve_max_length(2048, 1289, 256, 0) == 1545); // the shipping default
                                                           // that landed in the valley
    CHECK(resolve_max_length(2048, 1800, 512, 0) == 2048); // clamped to context
    // override > 0: explicit, clamped to (n_prompt, n_ctx].
    CHECK(resolve_max_length(2048, 1289, 256, 1801) == 1801);
    CHECK(resolve_max_length(2048, 1289, 256, 100) == 1290);  // floor: prompt+1
    CHECK(resolve_max_length(2048, 1289, 256, 9999) == 2048); // ceiling: n_ctx
    // F01: an INT_MAX n_predict saturates instead of overflowing.
    CHECK(resolve_max_length(2048, 1289, INT_MAX, 0) == 2048);
}

TEST_CASE("Session::create rejects non-existent model path") {
    xllama::SessionParams sp;
    sp.model_path = "/nonexistent/path/model.gguf";
    std::string err;
    auto s = xllama::Session::create(sp, &err);
    CHECK(s == nullptr);
    CHECK(!err.empty());
}

TEST_CASE("Session::create rejects empty model path") {
    xllama::SessionParams sp;
    sp.model_path = "";
    std::string err;
    auto s = xllama::Session::create(sp, &err);
    CHECK(s == nullptr);
    CHECK(!err.empty());
}

TEST_CASE("model_uses_llama_backend detects .gguf suffix and dir layout") {
    using xllama::model_uses_llama_backend;

    CHECK(model_uses_llama_backend("foo.gguf"));
    CHECK(model_uses_llama_backend("/abs/path/model.Q4_K.gguf"));
    CHECK_FALSE(model_uses_llama_backend("smollm2-360m"));
    CHECK_FALSE(model_uses_llama_backend("some-ort-model-dir"));

    // Temp dir containing a .gguf file (simulates catalogue layout on disk)
    auto tmp = std::filesystem::temp_directory_path() / "xllama-gguf-layout-test";
    std::filesystem::create_directories(tmp);
    {
        std::ofstream f((tmp / "model.gguf").string());
        f << "dummy";
    }
    CHECK(model_uses_llama_backend(tmp.string()));

    // Dir without .gguf should be false
    auto tmp2 = std::filesystem::temp_directory_path() / "xllama-no-gguf";
    std::filesystem::create_directories(tmp2);
    CHECK_FALSE(model_uses_llama_backend(tmp2.string()));

    // Cleanup
    std::error_code ec;
    std::filesystem::remove_all(tmp, ec);
    std::filesystem::remove_all(tmp2, ec);
}

TEST_CASE("first_gguf_in_dir prefers model.gguf over adapter.gguf") {
    using xllama::first_gguf_in_dir;
    auto tmp = std::filesystem::temp_directory_path() / "xllama-gguf-prefer-test";
    std::error_code ec;
    std::filesystem::remove_all(tmp, ec);
    std::filesystem::create_directories(tmp);
    {
        std::ofstream a((tmp / "adapter.gguf").string());
        a << "adapter";
        std::ofstream m((tmp / "model.gguf").string());
        m << "model";
    }
    const std::string picked = first_gguf_in_dir(tmp.string(), "adapter.gguf");
    CHECK(picked.find("model.gguf") != std::string::npos);
    // Without exclude, model.gguf still preferred by name
    const std::string picked2 = first_gguf_in_dir(tmp.string());
    CHECK(picked2.find("model.gguf") != std::string::npos);
    std::filesystem::remove_all(tmp, ec);
}

TEST_CASE("Session::create fails when LoRA path is invalid (llama)") {
    // Real base not required: create fails at model load first if path is junk.
    // Here we only assert empty model + lora still rejects cleanly.
    xllama::SessionParams sp;
    sp.model_path = "/nonexistent/base.gguf";
    sp.lora_path = "/nonexistent/adapter.gguf";
    sp.backend = xllama::Backend::LlamaCpp;
    std::string err;
    auto s = xllama::Session::create(sp, &err);
    CHECK(s == nullptr);
    CHECK(!err.empty());
}

TEST_CASE("Session::create Auto with explicit Backend::LlamaCpp on GGUF layout") {
    // Even without real weights the dispatch must select the llama path.
    // We only assert that it does not succeed via the ORT path and that an error is produced.
    auto tmp = std::filesystem::temp_directory_path() / "xllama-session-gguf-auto";
    std::filesystem::create_directories(tmp);
    {
        std::ofstream f((tmp / "mini.gguf").string());
        f << "not-a-real-gguf";
    }

    xllama::SessionParams sp;
    sp.model_path = tmp.string();
    sp.backend = xllama::Backend::Auto; // should resolve to Llama via layout

    std::string err;
    auto s = xllama::Session::create(sp, &err);
    // On Linux this build only has llama; we expect a load failure from the llama code path.
    CHECK(s == nullptr);
    CHECK(!err.empty());

    std::error_code ec;
    std::filesystem::remove_all(tmp, ec);
}

// #170a: a full-prompt turn rewinds the resident KV to the common token prefix
// instead of clearing. Opt-in — needs a real GGUF:
//   XLLAMA_TEST_MODEL=/path/to/model.gguf ./xllama-tests
TEST_CASE("Session: KV prefix reuse collapses a repeated prefill (opt-in: XLLAMA_TEST_MODEL)") {
    const char* model_env = std::getenv("XLLAMA_TEST_MODEL");
    if (!model_env)
        return;

    xllama::SessionParams sp;
    sp.model_path = model_env;
    sp.n_ctx = 512;
    std::string err;
    auto sess = xllama::Session::create(sp, &err);
    REQUIRE_MESSAGE(sess, err);

    auto mkgp = [](const std::string& p) {
        xllama::GenerateParams gp;
        gp.prompt = p;
        gp.n_predict = 12;
        gp.temperature = 0.0f; // greedy: outputs must be reproducible
        return gp;
    };
    const std::string prompt = "The capital of France is";

    const auto r_cold = sess->generate(mkgp(prompt));
    REQUIRE(r_cold.success);
    REQUIRE(r_cold.n_p_eval > 1);
    CHECK(r_cold.n_prompt == r_cold.n_p_eval);

    // Regenerate: identical prompt on the same session. Two legal regimes:
    // pure-attention caches rewind and re-prefill exactly 1 token (the logits
    // source); hybrid caches (LFM2) cannot erase a partial tail — seq_rm
    // returns false — so the session Clears and re-prefills the whole prompt
    // (a full-state snapshot restore would skip the last-token logit seed).
    // Byte-identical output is the correctness bar in both regimes.
    const auto r_regen = sess->generate(mkgp(prompt));
    REQUIRE(r_regen.success);
    CHECK(r_regen.n_p_eval <= r_cold.n_p_eval);
    // #335: the prompt is as long as before even when only its tail prefilled;
    // usage.prompt_tokens reads n_prompt, not n_p_eval.
    CHECK(r_regen.n_prompt == r_cold.n_prompt);
    CHECK(r_regen.output_text == r_cold.output_text);

    // Extended prompt (the LAN-API shape). This prompt diverges from the
    // resident generated tail right after the original prompt, so it exercises
    // the tail-removal path in both regimes.
    const bool full_rewind = r_regen.n_p_eval == 1;
    const std::string ext = prompt + " Paris, and the capital of Italy is";
    const auto r_ext = sess->generate(mkgp(ext));
    REQUIRE(r_ext.success);
    CHECK(r_ext.n_prompt == sess->count_tokens(ext));

    auto fresh = xllama::Session::create(sp, &err);
    REQUIRE_MESSAGE(fresh, err);
    const auto r_fresh = fresh->generate(mkgp(ext));
    REQUIRE(r_fresh.success);
    if (full_rewind) {
        // Pure-attention: only the extension past the common prefix re-prefills.
        CHECK(r_ext.n_p_eval < sess->count_tokens(ext));
        // Exact equality is the wrong bar here (it holds for the regenerate
        // case above, where the resident bytes are identical): the resident
        // prefix was accumulated in a different batch shape than the fresh
        // session's single prefill, so the K/V last bits differ and greedy can
        // flip at a NEAR-TIE downstream — inherent to every prefix cache,
        // observed as " Rome." vs " Rome.\n\nThe answer is Paris.". The
        // correctness signal is the leading strong-signal token agreeing.
        REQUIRE(r_fresh.output_text.size() >= 5);
        REQUIRE(r_ext.output_text.size() >= 5);
        CHECK(r_ext.output_text.substr(0, 5) == r_fresh.output_text.substr(0, 5));
    } else {
        // Hybrid (LFM2): seq_rm refuses a tail rewind. A matching prefix
        // snapshot means only the suffix prefills; otherwise this is a full
        // clear + single-batch re-prefill (byte-identical to a fresh session).
        CHECK(r_ext.n_p_eval <= r_fresh.n_p_eval);
        if (r_ext.n_p_eval == r_fresh.n_p_eval)
            CHECK(r_ext.output_text == r_fresh.output_text);
        else if (!r_ext.output_text.empty() && !r_fresh.output_text.empty())
            CHECK(r_ext.output_text.substr(0, 1) == r_fresh.output_text.substr(0, 1));
    }
}

// A prompt longer than the logical batch must be prefilled in chunks. This is
// not a "nicer error" test: llama_decode ASSERTS n_tokens <= n_batch (GGML_ABORT,
// Release included), so before the chunking this test process died. n_batch
// defaults to min(n_ctx, 2048), so the regime only exists above the shipping
// context — which is exactly what the coding catalogue opens (n_ctx 4096, ceiling
// 3846 prompt tokens). Opt-in — needs a real GGUF:
//   XLLAMA_TEST_MODEL=/path/to/model.gguf ./xllama-tests
TEST_CASE("Session: a prompt past n_batch prefills in chunks (opt-in: XLLAMA_TEST_MODEL)") {
    const char* model_env = std::getenv("XLLAMA_TEST_MODEL");
    if (!model_env)
        return;

    // Same regime as a 4096-token coding session with the default 2048 batch,
    // at 1/30th of the tokens: what matters is prompt > n_batch, and n_batch is
    // a knob. A 2100-token prefill in a Debug build takes minutes; this runs in
    // seconds and fails identically (an abort, not a failed assertion) without
    // the chunking.
    xllama::SessionParams sp;
    sp.model_path = model_env;
    sp.n_ctx = 512;
    sp.n_batch = 64;
    std::string err;
    auto sess = xllama::Session::create(sp, &err);
    REQUIRE_MESSAGE(sess, err);

    // Grow past n_batch with the model's own tokenizer, so the test does not
    // depend on one vocabulary's density.
    std::string prompt = "Summarize the following log.\n";
    int n_prompt = 0;
    for (int i = 0; i < 200 && n_prompt <= 96; ++i) {
        prompt += "line " + std::to_string(i) + ": nothing happened\n";
        if ((i % 4) == 3)
            n_prompt = sess->count_tokens(prompt);
    }
    n_prompt = sess->count_tokens(prompt);
    REQUIRE(n_prompt > sp.n_batch); // past the logical batch: the regime under test
    REQUIRE(n_prompt < 400);        // still inside n_ctx with room to generate

    xllama::GenerateParams gp;
    gp.prompt = prompt;
    gp.n_predict = 8;
    gp.temperature = 0.0f; // greedy: the two sessions below must agree
    const auto r = sess->generate(gp);
    REQUIRE_MESSAGE(r.success, r.error_msg); // before the chunking: SIGABRT here
    CHECK(r.n_p_eval == n_prompt);           // every token was prefilled, not just the first chunk

    // And chunking must not change what the model reads: the same prompt in ONE
    // batch (default n_batch, well above this prompt) sees the same context.
    // Only the leading token is compared — a different batch shape moves the
    // last K/V bits, so greedy can flip at a near-tie downstream (same caveat as
    // the #170a prefix-reuse case above).
    sp.n_batch = 0;
    auto whole = xllama::Session::create(sp, &err);
    REQUIRE_MESSAGE(whole, err);
    const auto r_whole = whole->generate(gp);
    REQUIRE(r_whole.success);
    CHECK(r_whole.n_p_eval == n_prompt);
    CHECK(r.n_eval == r_whole.n_eval);
    if (!r.output_text.empty() && !r_whole.output_text.empty())
        CHECK(r.output_text.substr(0, 1) == r_whole.output_text.substr(0, 1));
}

// A full prompt that cannot fit the context has nothing to evict — the trimmer
// upstream owns that — so it must fail fast rather than reach the decode, where
// an over-context batch is an abort. Opt-in:
//   XLLAMA_TEST_MODEL=/path/to/model.gguf ./xllama-tests
TEST_CASE("Session: a prompt past n_ctx fails fast (opt-in: XLLAMA_TEST_MODEL)") {
    const char* model_env = std::getenv("XLLAMA_TEST_MODEL");
    if (!model_env)
        return;

    xllama::SessionParams sp;
    sp.model_path = model_env;
    sp.n_ctx = 512;
    std::string err;
    auto sess = xllama::Session::create(sp, &err);
    REQUIRE_MESSAGE(sess, err);

    std::string prompt;
    for (int i = 0; i < 400; ++i)
        prompt += "token " + std::to_string(i) + " ";
    REQUIRE(sess->count_tokens(prompt) > 512);

    xllama::GenerateParams gp;
    gp.prompt = prompt;
    gp.n_predict = 8;
    const auto r = sess->generate(gp);
    CHECK_FALSE(r.success);
    CHECK(r.error_msg.find("prompt too long") != std::string::npos);
    CHECK(r.n_eval == 0);
}

// #170b: the resident KV round-trips through a file, so a conversation the
// session no longer holds resumes without re-reading its history. Opt-in —
// needs a real GGUF:
//   XLLAMA_TEST_MODEL=/path/to/model.gguf ./xllama-tests
TEST_CASE("Session: KV state round-trips through a file (opt-in: XLLAMA_TEST_MODEL)") {
    const char* model_env = std::getenv("XLLAMA_TEST_MODEL");
    if (!model_env)
        return;

    const auto state_path = std::filesystem::temp_directory_path() / "xllama-kv-state.bin";
    std::error_code ec;
    std::filesystem::remove(state_path, ec);

    xllama::SessionParams sp;
    sp.model_path = model_env;
    sp.n_ctx = 2048; // the shipping context: file size scales with it
    std::string err;
    auto a = xllama::Session::create(sp, &err);
    REQUIRE_MESSAGE(a, err);

    auto mkgp = [](const std::string& p, bool reset) {
        xllama::GenerateParams gp;
        gp.prompt = p;
        gp.n_predict = 16;
        gp.temperature = 0.0f; // greedy: the two sessions must agree exactly
        gp.reuse_kv = true;
        gp.reset_kv = reset;
        return gp;
    };
    // A conversation of realistic length: re-reading exactly this is the cost
    // the file exists to avoid, and its size is what the eviction policy has
    // to budget for.
    std::string filler;
    for (int i = 0; i < 80; ++i)
        filler += "Item " + std::to_string(i) +
                  ": the harbour master filed a report on tide tables and cargo manifests. ";
    const std::string seed_prompt =
        "<|im_start|>system\nYou are terse.<|im_end|>\n<|im_start|>user\n" + filler +
        "The capital of France is Paris. The capital of Italy is Rome. Remember "
        "both.<|im_end|>\n<|im_start|>assistant\n";
    const std::string follow = "<|im_end|>\n<|im_start|>user\nWhich city did I name "
                               "first?<|im_end|>\n<|im_start|>assistant\n";

    const auto r_seed = a->generate(mkgp(seed_prompt, /*reset=*/true));
    REQUIRE_MESSAGE(r_seed.success, r_seed.error_msg);

    const auto t0 = std::chrono::steady_clock::now();
    const bool saved = a->save_state(state_path.string(), &err);
    const double save_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    REQUIRE_MESSAGE(saved, err);
    const auto bytes = std::filesystem::file_size(state_path, ec);
    MESSAGE("state file: " << bytes / 1024 << " KiB for " << r_seed.n_p_eval << " prompt tokens ("
                           << bytes / 1024 / (r_seed.n_p_eval ? r_seed.n_p_eval : 1)
                           << " KiB/token), save " << save_ms << " ms, prefill was "
                           << r_seed.t_p_eval_ms << " ms");

    // The continuation the resident session would produce, for comparison.
    const auto r_cont_resident = a->generate(mkgp(follow, /*reset=*/false));
    REQUIRE_MESSAGE(r_cont_resident.success, r_cont_resident.error_msg);

    // A second session that never saw the conversation: restoring the file
    // must put it in the same state, not merely a similar one.
    auto b = xllama::Session::create(sp, &err);
    REQUIRE_MESSAGE(b, err);
    const auto t1 = std::chrono::steady_clock::now();
    const bool loaded = b->load_state(state_path.string(), &err);
    const double load_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
    REQUIRE_MESSAGE(loaded, err);
    MESSAGE("load " << load_ms << " ms");

    const auto r_cont_restored = b->generate(mkgp(follow, /*reset=*/false));
    REQUIRE_MESSAGE(r_cont_restored.success, r_cont_restored.error_msg);
    // Only the follow-up is prefilled — the history came from the file.
    CHECK(r_cont_restored.n_p_eval == r_cont_resident.n_p_eval);
    CHECK(r_cont_restored.n_p_eval < r_seed.n_p_eval);
    CHECK(r_cont_restored.output_text == r_cont_resident.output_text);

    // A file whose fingerprint does not match must be refused, not fed to the
    // cache: the pin checks shape only, so this is the whole defence against
    // loading another model's KV.
    xllama::SessionParams sp2 = sp;
    sp2.n_ctx = 1024; // same weights, different context configuration
    auto c = xllama::Session::create(sp2, &err);
    REQUIRE_MESSAGE(c, err);
    err.clear();
    CHECK_FALSE(c->load_state(state_path.string(), &err));
    CHECK(!err.empty());
    // Refusal must leave the session usable.
    const auto r_after = c->generate(mkgp("<|im_start|>user\nHi.<|im_end|>\n"
                                          "<|im_start|>assistant\n",
                                          /*reset=*/true));
    CHECK(r_after.success);

    std::filesystem::remove(state_path, ec);
}

// #169: a continuation turn that would overflow n_ctx evicts the oldest
// resident tokens past n_keep (front-drop + RoPE shift) instead of failing,
// so a long chat never leaves the reuse regime. Opt-in — needs a real GGUF:
//   XLLAMA_TEST_MODEL=/path/to/model.gguf ./xllama-tests
TEST_CASE(
    "Session: context shift keeps continuations alive past n_ctx (opt-in: XLLAMA_TEST_MODEL)") {
    const char* model_env = std::getenv("XLLAMA_TEST_MODEL");
    if (!model_env)
        return;

    xllama::SessionParams sp;
    sp.model_path = model_env;
    sp.n_ctx = 256; // small on purpose: a few turns must overflow it
    std::string err;
    auto sess = xllama::Session::create(sp, &err);
    REQUIRE_MESSAGE(sess, err);

    const std::string sys = "<|im_start|>system\nYou are terse.<|im_end|>\n";
    const int n_keep = sess->count_tokens(sys);
    REQUIRE(n_keep > 0);

    // Seed the persistent context (reuse + reset).
    xllama::GenerateParams seed;
    seed.prompt = sys + "<|im_start|>user\nSay ok.<|im_end|>\n<|im_start|>assistant\n";
    seed.n_predict = 32;
    seed.temperature = 0.0f;
    seed.reuse_kv = true;
    seed.reset_kv = true;
    seed.n_keep = n_keep;
    seed.turn_close = "<|im_end|>\n";
    const auto r_seed = sess->generate(seed);
    REQUIRE_MESSAGE(r_seed.success, r_seed.error_msg);

    // Enough continuation turns to overflow n_ctx=256 several times over.
    // Capability decides the contract (known once the lazy context exists):
    // LFM2 and pure-attention archs shift, so every turn must succeed;
    // an arch that cannot shift (Qwen3.5's imrope cache — seq_add would be
    // a GGML_ASSERT abort, which the gate must prevent) keeps the #173
    // fail-fast, so the overflowing turn must fail CLEANLY with n_eval == 0 —
    // the exact signature the UI's full-prefill retry keys on.
    const bool shifts = sess->can_context_shift();
    int total_tokens = sess->count_tokens(seed.prompt) + r_seed.n_eval;
    bool saw_clean_overflow = false;
    for (int i = 0; i < 8; ++i) {
        xllama::GenerateParams cont;
        cont.prompt = "<|im_end|>\n<|im_start|>user\nAnd again, tell me more "
                      "about that, please.<|im_end|>\n<|im_start|>assistant\n";
        cont.n_predict = 48;
        cont.temperature = 0.0f;
        cont.reuse_kv = true;
        cont.reset_kv = false;
        cont.n_keep = n_keep;
        cont.turn_close = "<|im_end|>\n";
        const auto rc = sess->generate(cont);
        if (shifts) {
            REQUIRE_MESSAGE(rc.success, rc.error_msg);
            CHECK(rc.n_p_eval > 0);
        } else if (!rc.success) {
            CHECK(rc.n_eval == 0);
            CHECK(rc.error_msg.find("context full") != std::string::npos);
            saw_clean_overflow = true;
            break;
        }
        total_tokens += rc.n_p_eval + rc.n_eval;
    }
    if (shifts) {
        // The conversation genuinely outgrew the context — the loop above
        // only proves the point if eviction actually had to happen.
        CHECK(total_tokens > sp.n_ctx);
    } else {
        CHECK(saw_clean_overflow);
    }
}

// Hybrid-shaped hook (documented): with XLLAMA_TEST_MODEL, the planner in
// prefix_snapshot.h is exercised through LlamaSession. A matching extension
// keeps the in-memory prefix snapshot (smaller n_p_eval than a cold run). A
// non-extension either restores that snapshot or clears and matches cold.
TEST_CASE(
    "Session: hybrid prefix snapshot reuses a full-prompt suffix (opt-in: XLLAMA_TEST_MODEL)") {
    const char* model = std::getenv("XLLAMA_TEST_MODEL");
    if (!model)
        return;
    xllama::SessionParams sp;
    sp.model_path = model;
    sp.n_ctx = 512;
    sp.n_threads = 2;
    std::string error;
    auto sess = xllama::Session::create(sp, &error);
    REQUIRE_MESSAGE(sess, error);

    auto mkgp = [](const std::string& p) {
        xllama::GenerateParams gp;
        gp.prompt = p;
        gp.n_predict = 8;
        gp.temperature = 0.0f;
        return gp;
    };
    const std::string first = "<|im_start|>user\nName a colour.<|im_end|>\n<|im_start|>assistant\n";
    const auto r1 = sess->generate(mkgp(first));
    REQUIRE_MESSAGE(r1.success, r1.error_msg);
    REQUIRE(r1.n_p_eval > 0);

    // Extension of the saved prefix (the first prompt): not of the generated
    // tail. Hybrid restores the snapshot; non-hybrid rewinds. Either way the
    // extra prefill is the suffix, not the whole second prompt.
    const std::string ext = first + "Blue.<|im_end|>\n<|im_start|>user\nAnd another."
                                    "<|im_end|>\n<|im_start|>assistant\n";
    const auto r_ext = sess->generate(mkgp(ext));
    REQUIRE_MESSAGE(r_ext.success, r_ext.error_msg);
    CHECK(r_ext.n_p_eval < r_ext.n_prompt);
    CHECK(r_ext.n_prompt == sess->count_tokens(ext));

    // Unrelated user text. Hybrid cannot rewind, so this is a full prefill
    // (n_p_eval == n_prompt). Attention-only models share the ChatML wrapper
    // with `first` and rewind that prefix (n_p_eval < n_prompt, at least 1).
    const std::string other = "<|im_start|>user\nSay hi.<|im_end|>\n<|im_start|>assistant\n";
    const auto r_other = sess->generate(mkgp(other));
    REQUIRE_MESSAGE(r_other.success, r_other.error_msg);
    CHECK(r_other.n_p_eval >= 1);
    CHECK(r_other.n_p_eval <= r_other.n_prompt);
}

TEST_CASE("Session: hybrid prefix snapshot slack and exact regenerate "
          "(opt-in: XLLAMA_TEST_MODEL)") {
    const char* model = std::getenv("XLLAMA_TEST_MODEL");
    if (!model)
        return;
    xllama::SessionParams sp;
    sp.model_path = model;
    sp.n_ctx = 512;
    sp.n_threads = 2;
    std::string error;
    auto sess = xllama::Session::create(sp, &error);
    REQUIRE_MESSAGE(sess, error);

    auto mkgp = [](const std::string& p) {
        xllama::GenerateParams gp;
        gp.prompt = p;
        gp.n_predict = 8;
        gp.temperature = 0.0f;
        return gp;
    };
    const std::string stem = "<|im_start|>system\nYou are a helpful assistant. "
                             "Answer with one word only.\n<|im_end|>\n"
                             "<|im_start|>user\nPlease name one item from this category: ";
    const std::string p1 = stem + "colour.<|im_end|>\n<|im_start|>assistant\n";
    const std::string p2 = stem + "fruit.<|im_end|>\n<|im_start|>assistant\n";
    REQUIRE(sess->count_tokens(p1) > static_cast<int>(xllama::kPrefixSnapshotTailSlack));
    REQUIRE(p1 != p2);

    const auto r1 = sess->generate(mkgp(p1));
    REQUIRE_MESSAGE(r1.success, r1.error_msg);
    REQUIRE(r1.n_p_eval > 0);

    const auto r_regen = sess->generate(mkgp(p1));
    REQUIRE_MESSAGE(r_regen.success, r_regen.error_msg);
    CHECK(r_regen.n_prompt == r1.n_prompt);
    CHECK(r_regen.output_text == r1.output_text);
    // Attention rewinds to the last token; a hybrid restores its checkpoint
    // and re-prefills only the tail after it. Neither prefills from scratch.
    CHECK(r_regen.n_p_eval >= 1);
    CHECK(r_regen.n_p_eval < r_regen.n_prompt);

    auto cold1 = xllama::Session::create(sp, &error);
    REQUIRE_MESSAGE(cold1, error);
    const auto c1 = cold1->generate(mkgp(p1));
    REQUIRE_MESSAGE(c1.success, c1.error_msg);
    CHECK(r_regen.output_text == c1.output_text);
    cold1.reset();

    const auto r_div = sess->generate(mkgp(p2));
    REQUIRE_MESSAGE(r_div.success, r_div.error_msg);
    CHECK(r_div.n_p_eval < r_div.n_prompt);
    CHECK(r_div.n_p_eval >= 1);
    CHECK_FALSE(r_div.output_text.empty());

    auto cold2 = xllama::Session::create(sp, &error);
    REQUIRE_MESSAGE(cold2, error);
    const auto c2 = cold2->generate(mkgp(p2));
    REQUIRE_MESSAGE(c2.success, c2.error_msg);
    CHECK(r_div.output_text == c2.output_text);
}

// Snapshot restoration is not a promise that a hybrid recurrent state can
// rewind to edited/re-rendered history. Compare its fallback with a fresh run.
TEST_CASE("Session: divergent restored snapshot matches cold prefill (opt-in: XLLAMA_TEST_MODEL)") {
    const char* model = std::getenv("XLLAMA_TEST_MODEL");
    if (!model)
        return;
    const auto path = std::filesystem::temp_directory_path() / "xllama-divergent-state.bin";
    xllama::SessionParams sp;
    sp.model_path = model;
    sp.n_ctx = 512;
    sp.n_threads = 2;
    std::string error;
    auto source = xllama::Session::create(sp, &error);
    REQUIRE_MESSAGE(source, error);
    xllama::GenerateParams gp;
    gp.prompt = "<|im_start|>user\nSay hello.<|im_end|>\n<|im_start|>assistant\n";
    gp.n_predict = 12;
    gp.temperature = 0;
    REQUIRE(source->generate(gp).success);
    REQUIRE_MESSAGE(source->save_state(path.string(), &error), error);
    source.reset();
    auto restored = xllama::Session::create(sp, &error);
    REQUIRE_MESSAGE(restored, error);
    REQUIRE_MESSAGE(restored->load_state(path.string(), &error), error);
    gp.prompt += "Edited assistant reply.<|im_end|>\n<|im_start|>user\nSay goodbye."
                 "<|im_end|>\n<|im_start|>assistant\n";
    const auto actual = restored->generate(gp);
    REQUIRE_MESSAGE(actual.success, actual.error_msg);
    restored.reset();
    auto cold = xllama::Session::create(sp, &error);
    REQUIRE_MESSAGE(cold, error);
    const auto expected = cold->generate(gp);
    REQUIRE_MESSAGE(expected.success, expected.error_msg);
    CHECK(actual.n_p_eval <= expected.n_p_eval);
    if (actual.n_p_eval == expected.n_p_eval)
        CHECK(actual.output_text == expected.output_text);
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

TEST_CASE("Session: hybrid head/tail prefill matches the opt-out greedy output "
          "(opt-in: XLLAMA_TEST_MODEL)") {
    // With snapshots on, a hybrid full prefill longer than the tail slack runs as
    // head (captured) + tail. That must not change what a cold turn generates
    // compared with hybrid-prefix-snapshots=0 (one prefill).
    const char* model = std::getenv("XLLAMA_TEST_MODEL");
    if (!model)
        return;
    xllama::SessionParams on;
    on.model_path = model;
    on.n_ctx = 1024;
    on.n_threads = 2;
    xllama::SessionParams off = on;
    off.prefix_snapshots = false;

    const std::string stem = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                             "<|im_start|>user\n";
    const std::vector<std::string> prompts = {
        stem + "Name one colour.<|im_end|>\n<|im_start|>assistant\n",
        stem + "Explain in two sentences why the sky looks blue during the day and red at "
               "sunset.<|im_end|>\n<|im_start|>assistant\n",
    };
    for (const std::string& p : prompts) {
        xllama::GenerateParams gp;
        gp.prompt = p;
        gp.n_predict = 32;
        gp.temperature = 0.0f;
        std::string error;
        auto s_on = xllama::Session::create(on, &error);
        REQUIRE_MESSAGE(s_on, error);
        const auto r_on = s_on->generate(gp);
        REQUIRE_MESSAGE(r_on.success, r_on.error_msg);
        s_on.reset();
        auto s_off = xllama::Session::create(off, &error);
        REQUIRE_MESSAGE(s_off, error);
        const auto r_off = s_off->generate(gp);
        REQUIRE_MESSAGE(r_off.success, r_off.error_msg);
        CHECK(r_on.n_p_eval == r_off.n_p_eval);
        CHECK(r_on.output_text == r_off.output_text);
    }
}

namespace {
struct EpochProbeSession final : xllama::Session {
    xllama::InferenceResult generate(const xllama::GenerateParams&) override {
        return {};
    }
    int count_tokens(const std::string&) override {
        return 0;
    }
    void rewrite() {
        touch_kv();
    }
};
} // namespace

TEST_CASE("Session::kv_epoch: unique per instance, renewed on every rewrite") {
    EpochProbeSession a;
    EpochProbeSession b;
    CHECK(a.kv_epoch() != 0);
    CHECK(a.kv_epoch() != b.kv_epoch()); // a reloaded session never matches the old one
    const uint64_t before = a.kv_epoch();
    a.rewrite();
    CHECK(a.kv_epoch() != before);
    CHECK(a.kv_epoch() != b.kv_epoch());
}

namespace {
xllama::GenerateParams greedy_turn(const std::string& prompt, bool reuse, bool reset) {
    xllama::GenerateParams gp;
    gp.prompt = prompt;
    gp.n_predict = 8;
    gp.temperature = 0.0f;
    gp.reuse_kv = reuse;
    gp.reset_kv = reset;
    return gp;
}
} // namespace

TEST_CASE("Session: a continuation on a KV another caller rewrote is refused (opt-in: "
          "XLLAMA_TEST_MODEL)") {
    // The GUI continues its own conversation with a delta; the LAN API sends a
    // stateless full prompt on the same resident session in between. Appending
    // the GUI delta to the API's KV would continue the API conversation.
    const char* model = std::getenv("XLLAMA_TEST_MODEL");
    if (!model)
        return;
    xllama::SessionParams sp;
    sp.model_path = model;
    sp.n_ctx = 1024;
    std::string error;
    auto s = xllama::Session::create(sp, &error);
    REQUIRE_MESSAGE(s, error);

    const std::string sys = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n";
    const std::string gui1 = sys + "<|im_start|>user\nMy name is Ada.<|im_end|>\n"
                                   "<|im_start|>assistant\n";
    const auto r1 = s->generate(greedy_turn(gui1, true, true));
    REQUIRE_MESSAGE(r1.success, r1.error_msg);
    const uint64_t gui_epoch = s->kv_epoch();

    // Same owner: a continuation keeps the epoch.
    const std::string delta = "<|im_end|>\n<|im_start|>user\nSay it again.<|im_end|>\n"
                              "<|im_start|>assistant\n";
    xllama::GenerateParams own = greedy_turn(delta, true, false);
    own.expected_kv_epoch = gui_epoch;
    const auto r_own = s->generate(own);
    REQUIRE_MESSAGE(r_own.success, r_own.error_msg);
    CHECK(s->kv_epoch() == gui_epoch);

    // Another caller's full prompt rewrites the KV.
    const std::string api = sys + "<|im_start|>user\nList three colours.<|im_end|>\n"
                                  "<|im_start|>assistant\n";
    const auto r_api = s->generate(greedy_turn(api, false, false));
    REQUIRE_MESSAGE(r_api.success, r_api.error_msg);
    CHECK(s->kv_epoch() != gui_epoch);

    // The GUI's next continuation is refused before anything is touched.
    xllama::GenerateParams stale = greedy_turn(delta, true, false);
    stale.expected_kv_epoch = gui_epoch;
    const uint64_t api_epoch = s->kv_epoch();
    const auto r_stale = s->generate(stale);
    CHECK_FALSE(r_stale.success);
    CHECK(r_stale.n_eval == 0);
    CHECK(r_stale.error_msg.find("changed since the last turn") != std::string::npos);
    CHECK(s->kv_epoch() == api_epoch);
}

TEST_CASE("Session: a full prompt past n_ctx leaves the resident KV and its epoch intact "
          "(opt-in: XLLAMA_TEST_MODEL)") {
    const char* model = std::getenv("XLLAMA_TEST_MODEL");
    if (!model)
        return;
    xllama::SessionParams sp;
    sp.model_path = model;
    sp.n_ctx = 512;
    std::string error;
    auto s = xllama::Session::create(sp, &error);
    REQUIRE_MESSAGE(s, error);
    const std::string turn = "<|im_start|>user\nHello.<|im_end|>\n<|im_start|>assistant\n";
    REQUIRE(s->generate(greedy_turn(turn, true, true)).success);
    const uint64_t epoch = s->kv_epoch();

    std::string huge;
    for (int i = 0; i < 400; ++i)
        huge += "token " + std::to_string(i) + " ";
    const auto r = s->generate(greedy_turn(huge, false, false));
    CHECK_FALSE(r.success);
    CHECK(r.error_msg.find("prompt too long") != std::string::npos);
    CHECK(s->kv_epoch() == epoch); // rejected before the #170a rewind

    // ...so the conversation still continues on its own KV.
    xllama::GenerateParams cont = greedy_turn(
        "<|im_end|>\n<|im_start|>user\nAgain.<|im_end|>\n<|im_start|>assistant\n", true, false);
    cont.expected_kv_epoch = epoch;
    const auto r_cont = s->generate(cont);
    CHECK_MESSAGE(r_cont.success, r_cont.error_msg);
}

TEST_CASE("Session: a KV file carries the prefix checkpoint, so an edited reply after a load "
          "re-prefills only the tail (opt-in: XLLAMA_TEST_MODEL)") {
    // Without the checkpoint trailer a hybrid model loaded from disk has no
    // checkpoint: the first divergence Clears and prefills everything.
    const char* model = std::getenv("XLLAMA_TEST_MODEL");
    if (!model)
        return;
    const auto path = std::filesystem::temp_directory_path() / "xllama-ckpt-state.bin";
    xllama::SessionParams sp;
    sp.model_path = model;
    sp.n_ctx = 512;
    sp.n_threads = 2;
    std::string error;
    xllama::GenerateParams gp;
    gp.prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n"
                "<|im_start|>user\nSay hello to the reader.<|im_end|>\n<|im_start|>assistant\n";
    gp.n_predict = 12;
    gp.temperature = 0;
    {
        auto source = xllama::Session::create(sp, &error);
        REQUIRE_MESSAGE(source, error);
        REQUIRE(source->generate(gp).success);
        REQUIRE_MESSAGE(source->save_state(path.string(), &error), error);
    }
    xllama::GenerateParams edited = gp;
    edited.prompt += "Edited assistant reply.<|im_end|>\n<|im_start|>user\nSay goodbye."
                     "<|im_end|>\n<|im_start|>assistant\n";
    auto restored = xllama::Session::create(sp, &error);
    REQUIRE_MESSAGE(restored, error);
    REQUIRE_MESSAGE(restored->load_state(path.string(), &error), error);
    const auto actual = restored->generate(edited);
    REQUIRE_MESSAGE(actual.success, actual.error_msg);
    restored.reset();
    auto cold = xllama::Session::create(sp, &error);
    REQUIRE_MESSAGE(cold, error);
    const auto expected = cold->generate(edited);
    REQUIRE_MESSAGE(expected.success, expected.error_msg);
    CHECK(actual.n_p_eval < expected.n_p_eval);
    CHECK(actual.output_text == expected.output_text);
    std::error_code ec;
    std::filesystem::remove(path, ec);
}

TEST_CASE("Session: a shift with no turn boundary to evict fails fast instead of cutting a "
          "message (opt-in: XLLAMA_TEST_MODEL)") {
    const char* model = std::getenv("XLLAMA_TEST_MODEL");
    if (!model)
        return;
    xllama::SessionParams sp;
    sp.model_path = model;
    sp.n_ctx = 256;
    std::string error;
    auto s = xllama::Session::create(sp, &error);
    REQUIRE_MESSAGE(s, error);
    // One long user turn with no terminator before the overflow: nothing whole
    // can be evicted.
    std::string user;
    for (int i = 0; i < 40; ++i)
        user += "word" + std::to_string(i) + " ";
    xllama::GenerateParams seed = greedy_turn("<|im_start|>user\n" + user, true, true);
    seed.turn_close = "<|im_end|>\n";
    REQUIRE(s->generate(seed).success);
    if (!s->can_context_shift())
        return;
    xllama::GenerateParams cont = greedy_turn(user + user + user, true, false);
    cont.turn_close = "<|im_end|>\n";
    cont.expected_kv_epoch = s->kv_epoch();
    const auto r = s->generate(cont);
    CHECK_FALSE(r.success);
    CHECK(r.n_eval == 0);
    CHECK(r.error_msg.find("context full") != std::string::npos);
}

// Measurement, not a gate: does a hybrid's recurrent state still carry a fact
// from a turn the #169 shift evicted? Attention-only models cannot (the cells
// are gone); a hybrid's conv/SSM state absorbed it before the eviction. The
// result is recorded in docs/sdk/session.md and context_shift.h.
TEST_CASE("Session: probe for evicted-turn recall after a context shift (opt-in: "
          "XLLAMA_TEST_MODEL, measurement)") {
    const char* model = std::getenv("XLLAMA_TEST_MODEL");
    if (!model)
        return;
    xllama::SessionParams sp;
    sp.model_path = model;
    sp.n_ctx = 128; // small: the first turn must be evicted
    std::string error;
    auto s = xllama::Session::create(sp, &error);
    REQUIRE_MESSAGE(s, error);
    const std::string sys = "<|im_start|>system\nYou are terse.<|im_end|>\n";
    const int n_keep = s->count_tokens(sys);
    auto turn = [&](const std::string& prompt, bool reset, int n_predict) {
        xllama::GenerateParams gp = greedy_turn(prompt, true, reset);
        gp.n_predict = n_predict;
        gp.n_keep = n_keep;
        gp.turn_close = "<|im_end|>\n";
        if (!reset)
            gp.expected_kv_epoch = s->kv_epoch();
        return s->generate(gp);
    };
    REQUIRE(turn(sys + "<|im_start|>user\nRemember this code word: PAPAYA.<|im_end|>\n"
                       "<|im_start|>assistant\n",
                 true, 8)
                .success);
    for (int i = 0; i < 14; ++i) {
        const auto r = turn("<|im_end|>\n<|im_start|>user\nName a random animal, one word."
                            "<|im_end|>\n<|im_start|>assistant\n",
                            false, 8);
        REQUIRE_MESSAGE(r.success, r.error_msg);
    }
    const auto probe =
        turn("<|im_end|>\n<|im_start|>user\nWhat was the code word I asked you to "
             "remember? Answer with the word only.<|im_end|>\n<|im_start|>assistant\n",
             false, 8);
    REQUIRE_MESSAGE(probe.success, probe.error_msg);
    const bool recalled = probe.output_text.find("PAPAYA") != std::string::npos ||
                          probe.output_text.find("Papaya") != std::string::npos ||
                          probe.output_text.find("papaya") != std::string::npos;
    const std::string verdict = recalled ? "YES" : "no";
    MESSAGE("evicted-turn recall: " << verdict << " — reply '" << probe.output_text << "'");
}

// Private (anonymous) resident memory. The console reads weights into
// allocated buffers (no mmap, docs/uwp-constraints.md §1), so this, not VmRSS
// with the mapped GGUF pages, is the host figure that matches it.
static std::size_t anon_rss_mb() {
    FILE* fp = std::fopen("/proc/self/status", "r");
    if (!fp)
        return 0;
    char line[256];
    std::size_t kb = 0;
    while (std::fgets(line, sizeof(line), fp))
        if (std::strncmp(line, "RssAnon:", 8) == 0) {
            kb = static_cast<std::size_t>(std::strtoul(line + 8, nullptr, 10));
            break;
        }
    std::fclose(fp);
    return kb / 1024;
}

// Calibration for kSessionMemoryMarginMb: the dry run must cover the private
// memory a real load and first turn add. Records both numbers.
TEST_CASE("Session::estimate_memory covers the real load (opt-in: XLLAMA_TEST_MODEL)") {
    const char* model = std::getenv("XLLAMA_TEST_MODEL");
    if (!model)
        return;
    for (int n_ctx : {2048, 8192}) {
        xllama::SessionParams sp;
        sp.model_path = model;
        sp.n_ctx = n_ctx;
        std::string error;
        const auto est = xllama::Session::estimate_memory(sp, &error);
        REQUIRE_MESSAGE(est, error);
        CHECK(est->host_model_mb > 0);
        CHECK(est->host_context_mb > 0);
        const std::size_t ws0 = anon_rss_mb();
        auto s = xllama::Session::create(sp, &error);
        REQUIRE_MESSAGE(s, error);
        REQUIRE(s->generate(greedy_turn("<|im_start|>user\nHi.<|im_end|>\n<|im_start|>assistant\n",
                                        false, false))
                    .success);
        const std::size_t ws1 = anon_rss_mb();
        const std::size_t grown = ws1 > ws0 ? ws1 - ws0 : 0;
        MESSAGE("n_ctx " << n_ctx << ": estimate " << est->host_mb() << " MB (weights "
                         << est->host_model_mb << ", context " << est->host_context_mb
                         << ", compute " << est->host_compute_mb << "), private memory grew "
                         << grown << " MB");
        CHECK(grown <= est->host_mb() + xllama::kSessionMemoryMarginMb);
    }
}

TEST_CASE("Session::estimate_memory: a GGUF n_ctx above the trained context is capped there "
          "(opt-in: XLLAMA_TEST_MODEL)") {
    const char* model = std::getenv("XLLAMA_TEST_MODEL");
    if (!model)
        return;
    // Dry runs only (no_alloc): both requests are far above any catalogue
    // model's trained context, so both open at it and size the same KV (#570).
    // Without the cap the second would size twice the context of the first.
    std::size_t context_mb[2] = {};
    int i = 0;
    for (int n_ctx : {1 << 24, 1 << 25}) {
        xllama::SessionParams sp;
        sp.model_path = model;
        sp.n_ctx = n_ctx;
        std::string error;
        const auto est = xllama::Session::estimate_memory(sp, &error);
        REQUIRE_MESSAGE(est, error);
        context_mb[i++] = est->host_context_mb;
    }
    MESSAGE("context at the trained n_ctx: " << context_mb[0] << " MB");
    CHECK(context_mb[0] > 0);
    CHECK(context_mb[0] == context_mb[1]);
}
