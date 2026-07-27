// Copyright (c) 2024 Gianluca Mazza
// SPDX-License-Identifier: MIT

#include "xllama/path_utils.h"
#include "xllama/session.h"
#include "xllama/session_hub.h"

#include <chrono>
#include <doctest/doctest.h>
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

    // Regenerate: identical prompt on the same session. Two legal regimes:
    // pure-attention caches rewind and re-prefill exactly 1 token (the logits
    // source); hybrid caches (LFM2) cannot erase a partial tail — seq_rm
    // returns false — so the session degrades to a full clear + re-prefill.
    // Byte-identical output is the correctness bar in both regimes.
    const auto r_regen = sess->generate(mkgp(prompt));
    REQUIRE(r_regen.success);
    CHECK(r_regen.n_p_eval <= r_cold.n_p_eval);
    CHECK(r_regen.output_text == r_cold.output_text);

    // Extended prompt (the LAN-API shape). This prompt diverges from the
    // resident generated tail right after the original prompt, so it exercises
    // the tail-removal path in both regimes.
    const bool full_rewind = r_regen.n_p_eval == 1;
    const std::string ext = prompt + " Paris, and the capital of Italy is";
    const auto r_ext = sess->generate(mkgp(ext));
    REQUIRE(r_ext.success);

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
        // Hybrid (LFM2): the divergent tail cannot be erased — seq_rm refuses —
        // so the session must have degraded to a full clear + single-batch
        // re-prefill, which is exactly a fresh session: byte-identical output.
        CHECK(r_ext.output_text == r_fresh.output_text);
    }
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
